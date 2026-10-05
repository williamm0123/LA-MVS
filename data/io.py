"""DTU file readers."""

from __future__ import annotations

import re

import numpy as np


def read_pfm(filename: str) -> np.ndarray:
    """Read a PFM file and return a contiguous numpy array."""
    with open(filename, "rb") as file:
        header = file.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
        if not dim_match:
            raise ValueError("Malformed PFM header.")
        width, height = map(int, dim_match.groups())

        scale = float(file.readline().decode("utf-8").rstrip())
        endian = "<" if scale < 0 else ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        return np.flipud(data).copy()



