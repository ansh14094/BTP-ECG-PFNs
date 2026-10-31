"""Evaluation metrics for imbalanced regression.

Two complementary conventions are implemented:

1. Shot-stratified errors (Yang et al., ICML 2021, "Delving into Deep Imbalanced
   Regression"): bin the target, count TRAIN samples per bin, and label each test
   point many- / medium- / few-shot by the training frequency of its bin. Errors are
   reported per region. This is the convention the rest of this repo already uses.

2. SERA, the Squared Error-Relevance Area (Ribeiro & Moniz, Machine Learning 2020,
   "Imbalanced regression and extreme value prediction"), built on the relevance
   function phi: Y -> [0, 1] of Ribeiro (2011). SERA integrates the squared error
   restricted to increasingly relevant subsets, so it needs no binning and no
   arbitrary shot thresholds.

Both are reported because neither alone is accepted everywhere in this literature.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

# Shot-region thresholds on TRAIN bin counts, following Yang et al. (2021).
MANY_SHOT_MIN = 100
FEW_SHOT_MAX = 20


def relevance_control_points(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boxplot-based control points for the 'extremes' relevance function.

    Mirrors `phi.control(method="extremes")` from the R UBL package: relevance is 1
    at a tail only when that tail actually has boxplot outliers, 0 at the median, and
    the curve is a monotone PCHIP interpolant through those points with zero
    derivative at each control point.

    Returns:
        (x_control, y_control) sorted by x.
    """
    y = np.asarray(y_train, dtype=float)
    q1, med, q3 = np.percentile(y, [25, 50, 75])
    iqr = q3 - q1
    adj_low, adj_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    xs, ys = [], []
    # Low tail: relevant only if the boxplot flags low outliers.
    if y.min() < adj_low:
        xs.append(adj_low)
        ys.append(1.0)
    else:
        xs.append(y.min())
        ys.append(0.0)

    xs.append(med)
    ys.append(0.0)

    if y.max() > adj_high:
        xs.append(adj_high)
        ys.append(1.0)
    else:
        xs.append(y.max())
        ys.append(0.0)

    x_ctrl = np.asarray(xs, dtype=float)
    y_ctrl = np.asarray(ys, dtype=float)
    order = np.argsort(x_ctrl)
    return x_ctrl[order], y_ctrl[order]


class RelevanceFunction:
    """phi: Y -> [0, 1], fitted on the TRAINING targets only.

    Fitting on train (never on test) keeps the metric from leaking test-set
    information into how the metric itself weights errors.
    """

    def __init__(self, y_train: np.ndarray) -> None:
        self.x_ctrl, self.y_ctrl = relevance_control_points(y_train)
        if len(np.unique(self.x_ctrl)) < 2:
            self._interp = None  # degenerate (constant) target
        else:
            self._interp = PchipInterpolator(self.x_ctrl, self.y_ctrl, extrapolate=False)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self._interp is None:
            return np.zeros_like(y)
        phi = self._interp(y)
        # PCHIP does not extrapolate; outside the control range relevance saturates
        # at the nearest control point's value.
        phi = np.where(y < self.x_ctrl[0], self.y_ctrl[0], phi)
        phi = np.where(y > self.x_ctrl[-1], self.y_ctrl[-1], phi)
        return np.clip(phi, 0.0, 1.0)


def sera(y_true: np.ndarray, y_pred: np.ndarray, phi: RelevanceFunction,
         n_steps: int = 1001, normalize: bool = True) -> float:
    """Squared Error-Relevance Area (Ribeiro & Moniz, 2020).

    SER_t = sum of squared errors over {i : phi(y_i) >= t};  SERA = int_0^1 SER_t dt.

    Args:
        normalize: if True, divide by the number of test points so the value is
            comparable across datasets and splits of different size.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rel = phi(y_true)
    sq_err = (y_pred - y_true) ** 2

    ts = np.linspace(0.0, 1.0, n_steps)
    # ser[k] = sum of squared errors of points with relevance >= ts[k].
    ser = np.array([sq_err[rel >= t].sum() for t in ts], dtype=float)
    area = float(np.trapezoid(ser, ts))
    return area / len(y_true) if normalize else area


def geometric_mean_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Geometric mean of absolute errors (the 'GM' column in the DIR literature)."""
    abs_err = np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))
    return float(np.exp(np.mean(np.log(abs_err + eps))))


def shot_regions(y_train: np.ndarray, y_eval: np.ndarray, bin_width: float = 1.0,
                 ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Assign each eval point to many/medium/few-shot by TRAIN frequency of its bin.

    Returns:
        (masks keyed by region name, per-eval-point train bin count).
    """
    y_train = np.asarray(y_train, float)
    y_eval = np.asarray(y_eval, float)

    lo = np.floor(min(y_train.min(), y_eval.min()) / bin_width) * bin_width
    hi = np.ceil(max(y_train.max(), y_eval.max()) / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)

    train_counts, _ = np.histogram(y_train, bins=edges)
    # np.digitize returns 1-based bin indices for values inside the range.
    idx = np.clip(np.digitize(y_eval, edges) - 1, 0, len(train_counts) - 1)
    counts_per_point = train_counts[idx]

    masks = {
        "many": counts_per_point >= MANY_SHOT_MIN,
        "medium": (counts_per_point >= FEW_SHOT_MAX) & (counts_per_point < MANY_SHOT_MIN),
        "few": counts_per_point < FEW_SHOT_MAX,
    }
    return masks, counts_per_point


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray,
             phi: RelevanceFunction, bin_width: float = 1.0) -> dict:
    """Full metric bundle: overall + per-shot errors, plus SERA."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true

    out = {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "gm": geometric_mean_error(y_true, y_pred),
        "sera": sera(y_true, y_pred, phi),
    }

    masks, _ = shot_regions(y_train, y_true, bin_width=bin_width)
    for region, mask in masks.items():
        if mask.sum() == 0:
            out[f"mae_{region}"] = float("nan")
            out[f"rmse_{region}"] = float("nan")
            out[f"n_{region}"] = 0
            continue
        e = err[mask]
        out[f"n_{region}"] = int(mask.sum())
        out[f"mae_{region}"] = float(np.mean(np.abs(e)))
        out[f"rmse_{region}"] = float(np.sqrt(np.mean(e ** 2)))
    return out
