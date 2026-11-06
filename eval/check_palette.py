#!/usr/bin/env python3
"""Colour-blind separation check for the paper's categorical palette.

The dataviz skill ships a Node validator; this machine has no Node, so the two
checks that actually gate a print figure are reimplemented here rather than
eyeballed: adjacent-pair separation in OKLab under normal vision, and under
simulated dichromacy.

Thresholds follow the skill: CVD dE >= 8 is the target, 6-8 is a floor that needs a
secondary encoding, and a normal-vision floor below 15 is a hard failure.

Dichromat simulation is Vienot, Brettel & Mollon (1999), the standard LMS-plane
projection. Numbers are reported to 1 decimal; what matters is which side of the
threshold they fall.

    python3 eval/check_palette.py "#2a78d6,#eb6834,#1baf7a"
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

# sRGB <-> LMS (Hunt-Pointer-Estevez, normalised to D65) used by the Vienot method.
RGB2LMS = np.array([[0.31399022, 0.63951294, 0.04649755],
                    [0.15537241, 0.75789446, 0.08670142],
                    [0.01775239, 0.10944209, 0.87256922]])
LMS2RGB = np.linalg.inv(RGB2LMS)
SIM = {  # dichromat projection matrices in LMS
    "protanopia": np.array([[0, 1.05118294, -0.05116099], [0, 1, 0], [0, 0, 1]]),
    "deuteranopia": np.array([[1, 0, 0], [0.9513092, 0, 0.04866992], [0, 0, 1]]),
    "tritanopia": np.array([[1, 0, 0], [0, 1, 0], [-0.86744736, 1.86727089, 0]]),
}


def hex2rgb(h: str) -> np.ndarray:
    h = h.strip().lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_oklab(c: np.ndarray) -> np.ndarray:
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return m2 @ np.cbrt(m1 @ c)


def simulate(rgb: np.ndarray, kind: str) -> np.ndarray:
    lin = srgb_to_linear(rgb)
    lms = RGB2LMS @ lin
    return np.clip(LMS2RGB @ (SIM[kind] @ lms), 0, 1)


def de(a: np.ndarray, b: np.ndarray) -> float:
    """OKLab distance x100, the units the skill's thresholds are stated in."""
    return float(np.linalg.norm(linear_to_oklab(a) - linear_to_oklab(b)) * 100)


def main() -> None:
    hexes = [h for h in sys.argv[1].split(",") if h.strip()]
    cols = [hex2rgb(h) for h in hexes]
    ok = True
    print(f"palette: {', '.join(hexes)}\n")
    for i, j in itertools.combinations(range(len(cols)), 2):
        lin_i, lin_j = srgb_to_linear(cols[i]), srgb_to_linear(cols[j])
        normal = de(lin_i, lin_j)
        flag = "FAIL" if normal < 15 else "pass"
        if normal < 15:
            ok = False
        print(f"{hexes[i]} vs {hexes[j]}")
        print(f"   normal vision  dE={normal:5.1f}  [{flag}, floor 15]")
        for kind in SIM:
            d = de(srgb_to_linear(simulate(cols[i], kind)),
                   srgb_to_linear(simulate(cols[j], kind)))
            if d >= 8:
                f = "pass"
            elif d >= 6:
                f = "FLOOR - needs a second encoding"
            else:
                f = "FAIL"
                ok = False
            print(f"   {kind:13s} dE={d:5.1f}  [{f}, target 8]")
        print()
    print("RESULT:", "palette usable" if ok else "palette FAILS - re-step it")


if __name__ == "__main__":
    main()
