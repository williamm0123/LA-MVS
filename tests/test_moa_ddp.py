"""CPU/Gloo two-process tests; DA3 is stubbed, the LAPE training graph is real."""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

import train_moa as train


def test_sharded_batches_cover_unique_samples_and_share_scales():
    samplers = [train.ShardedEpochSampler(train.EpochShuffleSampler(27, 7), 2, r, 2)
                for r in range(2)]
    previous = None
    for epoch in (0, 1):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        orders = [list(s) for s in samplers]
        assert len(orders[0]) == len(orders[1]) == 12
        assert set(orders[0]).isdisjoint(orders[1])
        global_order = samplers[0].global_order()
        buckets = {idx: i // 4 for i, idx in enumerate(global_order)}
        for step in range(6):
            assert {buckets[idx] for order in orders for idx in order[2*step:2*step+2]} == {step}
        if previous is not None:
            assert orders != previous
        previous = orders


class SyntheticDataset(Dataset):
    def __init__(self, cfg, count):
        self.cfg, self.count = cfg, count
        self.epoch, self.buckets = 0, {}

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        self.epoch = epoch

    def reset_scale_plan(self, order, batch_size):
        self.buckets = {idx: i // batch_size for i, idx in enumerate(order)}

    def __getitem__(self, idx):
        # Preserve the model RNG while generating deterministic sample data.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(100 + idx + self.epoch * self.count)
            hw = (64, 80) if self.buckets.get(idx, 0) % 2 == 0 else (80, 96)
            batch = train.synthetic_batch(self.cfg, torch.device('cpu'), 1, hw)
        return {k: v[0] for k, v in batch.items()}


class MetricNetwork(torch.nn.Module):
    def forward(self, batch):
        return {'depth_full': batch['depth_gt'] + batch['error'][:, None, None]}


class MetricDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {'images': torch.zeros(1), 'depth_gt': torch.full((2, 2), 600.),
                'mask': torch.ones(2, 2), 'depth_values': torch.tensor([400., 900.]),
                'error': torch.tensor(float(idx + 1))}


def metric_loss(out, batch, diagnostics=False):
    return torch.tensor(0.), {'loss': batch['error'].mean()}


def worker(root, lape):
    # Use two CPU processes even on a host with a single CUDA device.
    from test_lape_unit import _StubDA3, _tiny_cfg
    from models.network_moa import MoAMVSNet
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    device = torch.device('cpu')
    base = _tiny_cfg(enabled=(lape == 'on'))
    base = dataclasses.replace(base, paths=dataclasses.replace(base.paths, project_path=root),
                               train=dataclasses.replace(base.train, batch_size=1, val_batch_size=1,
                                                         num_views=3, num_workers=0, max_steps=2,
                                                         log_interval=1, ckpt_interval=1,
                                                         val_interval=1, amp=False))
    models = []
    def factory(cfg):
        model = MoAMVSNet(cfg, da3_net=_StubDA3())
        models.append(model)
        return model
    train.MoAMVSNet = factory
    args = train.parse_args(['--profile', 'local', '--name', 'ddp', '--resume', 'off'])
    def datasets(cfg, args):
        return SyntheticDataset(cfg, 5), SyntheticDataset(cfg, 3)
    ckpt_path = Path(root) / 'log/experiments/ddp/model/latest.pth'
    try:
        train.run(args, config_fn=lambda _: base, datasets_fn=datasets)
        # The real model's trainable parameters agree across both ranks.
        for p in models[-1].parameters():
            if p.requires_grad:
                other = p.detach().clone()
                dist.broadcast(other, 0)
                assert torch.allclose(other, p, atol=1e-7), 'parameters diverged'
        ck = train.load_checkpoint(ckpt_path, map_location='cpu')
        assert ck['step'] == 2 and ck['steps_per_epoch'] == 2
        assert ck['world_size'] == 2 and ck['global_batch_size'] == 2
        assert len(ck['rng_by_rank']) == 2
        assert not any(k.startswith(('module.', 'network.', 'da3_sva.da3.')) for k in ck['model'])
        assert ck['optimizer'] is not None
        args.resume = 'auto'
        cfg = dataclasses.replace(base, train=dataclasses.replace(base.train, max_steps=3))
        train.run(args, config_fn=lambda _: cfg, datasets_fn=datasets)
        assert train.load_checkpoint(ckpt_path)['step'] == 3
        # Only rank 1 requests a stop; all ranks checkpoint and return cleanly.
        args.stop_file = str(Path(root) / 'stop')
        if train.rank() == 1:
            Path(args.stop_file).touch()
        dist.barrier()
        cfg = dataclasses.replace(base, train=dataclasses.replace(base.train, max_steps=9))
        train.run(args, config_fn=lambda _: cfg, datasets_fn=datasets)
        assert train.load_checkpoint(ckpt_path)['step'] == 4
        # Exact validation partition: uneven lengths, then a completely empty shard.
        for count in (3, 1):
            loader = DataLoader(MetricDataset(), batch_size=1,
                                sampler=range(train.rank(), count, 2))
            metrics = train.validate(MetricNetwork(), loader, metric_loss, base, device, False)
            assert metrics['pixels'] == count * 4
            assert metrics['abs_err'] == (2. if count == 3 else 1.)
            assert metrics['loss'] == (2. if count == 3 else 1.)
        assert train.collective_flag(train.rank() == 1, device)
        if train.rank() == 0:
            print('DDP_TEST_OK', flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('lape', ['on', 'off'])
def test_two_process_training_resume_validation_and_stop(tmp_path, lape):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    repo = Path(__file__).resolve().parents[1]
    env['PYTHONPATH'] = os.pathsep.join([str(repo), str(repo / 'tests'), env.get('PYTHONPATH', '')])
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                             '--nproc-per-node=2', '--max-restarts=0', str(Path(__file__).resolve()),
                             '--worker', str(tmp_path), lape], cwd=repo, env=env,
                            text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'DDP_TEST_OK' in result.stdout


def test_sbatch_launcher_graceful_stop_and_requeue(tmp_path):
    """Exercise the real shell launch/wait/trap path without submitting a job."""
    repo = Path(__file__).resolve().parents[1]
    script = (repo / 'scripts/train_lape_umhpc.sh').read_text()
    # Only environment activation is omitted; python/GPU/sbatch are local stubs.
    script = script.replace('source ~/.bashrc\nconda activate uprmvs', ':')
    launcher = tmp_path / 'launcher.sh'
    launcher.write_text(script)
    (tmp_path / 'train_moa.py').touch()
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    stubs = {
        'python': '''#!/bin/bash
if [[ "$1" == "-" ]]; then cat >/dev/null; exit 0; fi
printf '%s\\n' "$@" > "$PROJECT_DIR/launch_args"
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--stop-file" ]]; then STOP_FILE=$2; break; fi
    shift
done
kill -USR1 "$PPID"
for i in {1..100}; do
    if [[ -f "$STOP_FILE" ]]; then exit 0; fi
    sleep 0.05
done
exit 9
''',
        'nvidia-smi': '#!/bin/bash\nexit 0\n',
        'sbatch': '#!/bin/bash\nprintf "%s\\n" "$@" > "$PROJECT_DIR/requeue_args"\n',
    }
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    env = dict(os.environ, PROJECT_DIR=str(tmp_path), FRESH='0', CHAIN='0', MAX_CHAIN='4',
               RUN_NAME='test', PER_GPU_BATCH='2', EPOCHS='10', STEPS='0', WARMUP_STEPS='1000',
               LR_REF='3e-4', LR_REF_BATCH='2', PATH=f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    env.pop('LR', None)
    result = subprocess.run(['bash', str(launcher)], env=env, text=True,
                            capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    launch = (tmp_path / 'launch_args').read_text().splitlines()
    assert '--nproc-per-node=2' in launch
    assert launch[launch.index('--lr') + 1] == '0.0004243'
    assert launch[launch.index('--warmup-steps') + 1] == '1000'
    assert launch[launch.index('--epochs') + 1] == '10'
    assert '--max-steps' not in launch
    requeue = (tmp_path / 'requeue_args').read_text()
    assert 'FRESH=0,CHAIN=1' in requeue and 'LR=0.0004243' in requeue
    assert not list((tmp_path / 'logs').glob('lape_stop_*'))


if __name__ == '__main__' and sys.argv[1] == '--worker':
    worker(sys.argv[2], sys.argv[3])
