#!/usr/bin/env python3
"""Zero-shot forecasting benchmark: Timestral on PTB-XL ECG signals.

This evaluates Timestral at the task it was actually built for -- predicting future
values of a series -- rather than bending it into age regression, which it has no head
for. Each 10-second 12-lead record is split into a context window and a forecast
horizon, and the model predicts the horizon from the context with no training.

Why ECG is an interesting stress test for a forecasting foundation model: the signal is
quasi-periodic (heartbeats) but the period drifts, and the clinically important part is
a sharp, short-lived QRS complex. A model that merely continues a smooth trend will do
badly on exactly the part that matters, so the naive and seasonal-naive baselines below
are not strawmen -- seasonal naive with a well-estimated cardiac period is a genuinely
strong competitor here.

Metrics
-------
MAE / RMSE  in millivolts, directly interpretable.
MASE        MAE divided by the in-context seasonal-naive MAE, so it is scale-free and
            comparable across records. MASE < 1 means the model beats the "repeat the
            previous heartbeat" rule on that record.
CRPS        Evaluated from Timestral's 21 quantiles via the standard pinball-loss
            approximation, so the probabilistic output is scored rather than discarded.
"""
from __future__ import annotations

import argparse, json, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecg_data  # noqa: E402

SAMPLING_HZ = 100  # PTB-XL at 1000 samples / 10 s


def to_np(x) -> np.ndarray:
    """Timestral returns CUDA torch tensors; bring them to host numpy."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def estimate_period(x: np.ndarray, lo: int = 40, hi: int = 150) -> int:
    """Cardiac period in samples, from the autocorrelation peak of the context.

    Searched over 40-150 samples = 0.4-1.5 s at 100 Hz, i.e. 40-150 bpm. Used only to
    define the seasonal-naive baseline and the MASE denominator.
    """
    x = x - x.mean()
    if np.allclose(x, 0):
        return lo
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    hi = min(hi, len(x) - 1)
    if hi <= lo:
        return lo
    return int(lo + np.argmax(ac[lo:hi]))


def crps_from_quantiles(q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> float:
    """CRPS approximated by the mean pinball loss over the quantile grid.

    CRPS(F, y) = 2 * integral_0^1 pinball_tau(y, F^-1(tau)) dtau; averaging the pinball
    losses over an evenly spaced grid and doubling approximates that integral.

    Args:
        q: quantile forecasts, (..., K)
        y: observations broadcastable to q[..., 0]
        levels: the K quantile levels
    """
    y = np.expand_dims(y, -1)
    diff = y - q
    pin = np.maximum(levels * diff, (levels - 1.0) * diff)
    return float(2.0 * pin.mean())


def baselines(ctx: np.ndarray, horizon: int, period: int) -> dict[str, np.ndarray]:
    """Forecasts that require no model at all.

    naive           repeat the last observed sample
    seasonal_naive  repeat the previous cardiac cycle (tiled to the horizon)
    mean            repeat the context mean
    """
    last = np.repeat(ctx[..., -1:], horizon, axis=-1)
    mean = np.repeat(ctx.mean(axis=-1, keepdims=True), horizon, axis=-1)
    reps = int(np.ceil(horizon / period))
    season = np.tile(ctx[..., -period:], reps)[..., :horizon]
    return {"naive": last, "seasonal_naive": season, "mean": mean}


def mase_denominator(ctx: np.ndarray, period: int) -> float:
    """In-context MAE of the seasonal-naive rule; the MASE scale factor."""
    if ctx.shape[-1] <= period:
        return float(np.mean(np.abs(np.diff(ctx, axis=-1)))) or 1e-8
    d = np.abs(ctx[..., period:] - ctx[..., :-period]).mean()
    return float(d) if d > 1e-8 else 1e-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Timestral-209M")
    ap.add_argument("--split", default="test")
    ap.add_argument("--context", type=int, default=800, help="samples of history (8 s)")
    ap.add_argument("--horizon", type=int, default=200, help="samples to forecast (2 s)")
    ap.add_argument("--n-records", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mode", default="multivariate", choices=["multivariate", "univariate"],
                    help="feed the 12 leads jointly, or each lead as its own series")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from timestral import TimestralForecaster

    X, _ = ecg_data.load_split_arrays()[args.split]
    if args.n_records:
        X = X[: args.n_records]
    N, V, T = X.shape
    C, H = args.context, args.horizon
    assert C + H <= T, f"context+horizon ({C}+{H}) exceeds record length {T}"
    print(f"{args.split}: {N} records x {V} leads x {T} samples "
          f"({T/SAMPLING_HZ:.0f} s @ {SAMPLING_HZ} Hz)", flush=True)
    print(f"context {C} ({C/SAMPLING_HZ:.0f} s) -> horizon {H} ({H/SAMPLING_HZ:.0f} s), "
          f"mode={args.mode}, model={args.model}", flush=True)

    t0 = time.time()
    fc = TimestralForecaster.from_pretrained(args.model, device=args.device)
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    acc = {k: {"abs": 0.0, "sq": 0.0, "mase": [], "crps": []}
           for k in ("timestral", "naive", "seasonal_naive", "mean")}
    n_pts = 0
    t0 = time.time()

    for s in range(0, N, args.batch):
        xb = X[s : s + args.batch].astype(np.float32)
        ctx, truth = xb[..., :C], xb[..., C : C + H]

        if args.mode == "multivariate":
            out = fc.forecast(ctx, prediction_length=H)              # (B, V, H, 21)
            med = to_np(out.median).astype(np.float64)
            qs = to_np(out.quantiles).astype(np.float64)
        else:
            flat = ctx.reshape(-1, 1, C)
            out = fc.forecast(flat, prediction_length=H)
            med = to_np(out.median).astype(np.float64).reshape(xb.shape[0], V, H)
            qs = to_np(out.quantiles).astype(np.float64).reshape(xb.shape[0], V, H, -1)
        levels = to_np(out.quantile_levels).astype(np.float64)

        for i in range(xb.shape[0]):
            # One period per record, estimated on lead II (index 1), the usual rhythm lead.
            p = estimate_period(ctx[i, 1])
            denom = mase_denominator(ctx[i], p)
            preds = {"timestral": med[i], **baselines(ctx[i], H, p)}
            for name, pr in preds.items():
                err = pr - truth[i]
                acc[name]["abs"] += float(np.abs(err).sum())
                acc[name]["sq"] += float((err ** 2).sum())
                acc[name]["mase"].append(float(np.abs(err).mean() / denom))
            acc["timestral"]["crps"].append(crps_from_quantiles(qs[i], truth[i], levels))
        n_pts += xb.shape[0] * V * H

        if (s // args.batch) % 5 == 0:
            done = min(s + args.batch, N)
            print(f"  {done}/{N} records  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'method':16s} {'MAE (mV)':>10s} {'RMSE (mV)':>10s} {'MASE':>8s} {'CRPS':>8s}")
    rows = []
    for name, a in acc.items():
        mae = a["abs"] / n_pts
        rmse = (a["sq"] / n_pts) ** 0.5
        mase = float(np.mean(a["mase"]))
        crps = float(np.mean(a["crps"])) if a["crps"] else float("nan")
        print(f"{name:16s} {mae:10.4f} {rmse:10.4f} {mase:8.3f} "
              f"{crps:8.4f}" if a["crps"] else
              f"{name:16s} {mae:10.4f} {rmse:10.4f} {mase:8.3f} {'—':>8s}")
        rows.append(dict(method=name, mae=mae, rmse=rmse, mase=mase, crps=crps,
                         model=args.model, mode=args.mode, context=C, horizon=H,
                         split=args.split, n_records=int(N), seconds=round(time.time()-t0, 1)))

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tag = args.tag or f"{os.path.basename(args.model)}_{args.mode}_c{C}h{H}"
    out_path = os.path.join(HERE, "results", f"timestral_{tag}.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_path}")
    print("TIMESTRAL_BENCH_COMPLETE")


if __name__ == "__main__":
    main()
