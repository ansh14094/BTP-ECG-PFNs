"""Tabular imbalanced-regression datasets and the DIR-style split protocol.

Protocol follows Yang et al. (ICML 2021): the TRAIN set keeps the natural, skewed
target distribution, while VAL and TEST are made approximately uniform over target
bins. A balanced test set is what makes few-shot error estimates low-variance -- with
a natural random split the tail would contain a handful of points and the few-shot
column would be mostly noise.
"""
from __future__ import annotations

import io
import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.makedirs(CACHE, exist_ok=True)


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    cat_features: list[str]
    bin_width: float  # target bin width used for shot regions
    # Datasets with an official, fixed train/val/test split (PTB-XL) carry it here as
    # (train_idx, val_idx, test_idx). Resplitting those would make results
    # incomparable with the numbers already recorded for this repo's ECG work, so
    # run.py uses this split verbatim under `--protocol fixed`.
    fixed_split: tuple | None = None
    # Per-bin val/test budgets for dir_split. These must be set per dataset: a fixed
    # budget that gives a sensible split on a 7-bin target (wine quality) sends over
    # half of a 50-bin target (communities) into the test set, and vice versa leaves
    # the few-bin dataset with a test set too small to estimate tail error.
    per_bin_val: int = 10
    per_bin_test: int = 20


def _cached_download(url: str, fname: str) -> bytes:
    path = os.path.join(CACHE, fname)
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    with open(path, "rb") as f:
        return f.read()


def load_abalone() -> Dataset:
    """UCI Abalone: predict ring count (age proxy) from physical measurements.

    4177 x 8. Rings are right-skewed: most animals sit at 8-11 rings while the
    high-ring tail is rare, which is the canonical cheap imbalanced-regression
    testbed and appears in the IM-Context tabular suite (arXiv:2405.18202).
    """
    raw = _cached_download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
        "abalone.data",
    )
    cols = ["Sex", "Length", "Diameter", "Height", "WholeWeight",
            "ShuckedWeight", "VisceraWeight", "ShellWeight", "Rings"]
    df = pd.read_csv(io.BytesIO(raw), header=None, names=cols)
    y = df.pop("Rings").to_numpy(dtype=float)
    return Dataset("abalone", df, y, cat_features=["Sex"], bin_width=1.0)


def load_openml(name: str, data_id: int, target: str, bin_width: float) -> Dataset:
    """Generic OpenML loader via scikit-learn (no extra dependency needed)."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=CACHE, parser="auto")
    df = bunch.frame.copy()
    y = df.pop(target).to_numpy(dtype=float)
    cat = [c for c in df.columns if str(df[c].dtype) in ("category", "object")]
    return Dataset(name, df, y, cat_features=cat, bin_width=bin_width)


def load_communities() -> Dataset:
    """UCI Communities and Crime: predict violent crimes per capita (normalised 0-1).

    1994 x 127. The target is strongly right-skewed -- most communities have low
    violent-crime rates and the high-crime tail is sparse -- which is why it recurs
    in the imbalanced-regression literature and in the IM-Context tabular suite
    (arXiv:2405.18202). Missing values are encoded '?'.

    The five leading identifier columns (state, county, community, communityname,
    fold) are non-predictive by the dataset's own documentation and are dropped;
    `fold` in particular is a cross-validation index and would be leakage.
    """
    raw = _cached_download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
        "communities.data",
    )
    df = pd.read_csv(io.BytesIO(raw), header=None, na_values="?")
    df = df.drop(columns=[0, 1, 2, 3, 4])          # non-predictive identifiers
    y = df.pop(df.columns[-1]).to_numpy(dtype=float)
    df.columns = [f"f{i}" for i in range(df.shape[1])]
    return Dataset("communities", df, y, cat_features=[], bin_width=0.05,
                   per_bin_val=8, per_bin_test=16)


def load_wine_white() -> Dataset:
    """UCI Wine Quality (white): predict the 0-10 sensory score.

    4898 x 11. Scores pile up at 5-6 while 3, 4, 8 and 9 are rare -- a severe,
    naturally occurring target imbalance on an ordinal scale.
    """
    raw = _cached_download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        "winequality-white.csv",
    )
    df = pd.read_csv(io.BytesIO(raw), sep=";")
    y = df.pop("quality").to_numpy(dtype=float)
    return Dataset("wine_white", df, y, cat_features=[], bin_width=1.0,
                   per_bin_val=30, per_bin_test=60)


def load_airfoil() -> Dataset:
    """UCI Airfoil Self-Noise: predict scaled sound pressure level in dB.

    1503 x 5. Appears in the IM-Context tabular suite.
    """
    raw = _cached_download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        "airfoil_self_noise.dat",
    )
    cols = ["frequency", "angle", "chord_length", "velocity",
            "displacement_thickness", "sound_pressure"]
    df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, names=cols, engine="python")
    y = df.pop("sound_pressure").to_numpy(dtype=float)
    return Dataset("airfoil", df, y, cat_features=[], bin_width=1.0)


def load_superconduct() -> Dataset:
    """UCI Superconductivity: predict critical temperature in kelvin.

    21263 x 81, heavily right-skewed (most materials have low Tc). Within
    TabPFN-2.5's stated 50k-sample / 2k-feature envelope.
    """
    import zipfile

    raw = _cached_download(
        "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
        "superconduct.zip",
    )
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open("train.csv") as fh:
            df = pd.read_csv(fh)
    y = df.pop("critical_temp").to_numpy(dtype=float)
    return Dataset("superconduct", df, y, cat_features=[], bin_width=4.0)


def load_embedding_npz(name: str, path: str, bin_width: float = 1.0) -> Dataset:
    """Load encoder embeddings dumped by ecg_tabpfn/extract_embeddings.py.

    The npz holds X_/y_ per split; they are concatenated here and the original split
    boundaries are preserved in `fixed_split`, so the official PTB-XL partition is
    used rather than a fresh DIR split.
    """
    d = np.load(path)
    Xs, ys, idx, start = [], [], {}, 0
    for split in ("train", "val", "test"):
        X, y = d[f"X_{split}"], d[f"y_{split}"]
        Xs.append(X); ys.append(y)
        idx[split] = np.arange(start, start + len(y)); start += len(y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(float)
    df = pd.DataFrame(X, columns=[f"e{i}" for i in range(X.shape[1])])
    return Dataset(name, df, y, cat_features=[], bin_width=bin_width,
                   fixed_split=(idx["train"], idx["val"], idx["test"]))


_ECG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "ecg_tabpfn", "results")
_AGEDB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "agedb_tabpfn", "results")


LOADERS = {
    "abalone": load_abalone,
    "communities": load_communities,
    "wine_white": load_wine_white,
    "airfoil": load_airfoil,
    "superconduct": load_superconduct,
    # PTB-XL age prediction via encoder embeddings (see ecg_tabpfn/).
    "ptbxl_tabpfn_enc": lambda: load_embedding_npz(
        "ptbxl_tabpfn_enc", os.path.join(_ECG, "emb_enc_main.npz")),
    # E2.1 / E2.2: encoders trained the same way as E2, but through a frozen TabPFN
    # that has itself been fine-tuned with biased context sampling (Gaussian, and
    # Gaussian+uniform respectively) rather than the stock checkpoint.
    "ptbxl_enc_e21": lambda: load_embedding_npz(
        "ptbxl_enc_e21", os.path.join(_ECG, "emb_enc_e21.npz")),
    "ptbxl_enc_e22": lambda: load_embedding_npz(
        "ptbxl_enc_e22", os.path.join(_ECG, "emb_enc_e22.npz")),
    # Encoders trained through a FROZEN TabICL body, one per prior-level checkpoint.
    # Same backbone and warm start as E1/E2; only the body they were trained through
    # differs, and none of those bodies has ever seen PTB-XL.
    "ptbxl_enc_e3": lambda: load_embedding_npz(
        "ptbxl_enc_e3", os.path.join(_ECG, "emb_enc_e3.npz")),
    "ptbxl_enc_e4": lambda: load_embedding_npz(
        "ptbxl_enc_e4", os.path.join(_ECG, "emb_enc_e4.npz")),
    "ptbxl_enc_e5": lambda: load_embedding_npz(
        "ptbxl_enc_e5", os.path.join(_ECG, "emb_enc_e5.npz")),
    "ptbxl_l1_enc": lambda: load_embedding_npz(
        "ptbxl_l1_enc", os.path.join(_ECG, "emb_baseline.npz")),
    # Timestral's own encoder used in place of ours: the forecasting foundation
    # model's representation, frozen and never trained on age. `_pca` is reduced to
    # 128 dims (PCA fitted on train only) so its width matches E1/E2 exactly.
    "ptbxl_timestral_enc": lambda: load_embedding_npz(
        "ptbxl_timestral_enc", os.path.join(_ECG, "emb_timestral.npz")),
    "ptbxl_timestral_pca": lambda: load_embedding_npz(
        "ptbxl_timestral_pca", os.path.join(_ECG, "emb_timestral_pca.npz")),
    # AgeDB-DIR face-age embeddings. Its official val/test are already balanced over
    # age bins by construction, so `--protocol fixed` is the right protocol here and
    # no DIR rebalancing is needed, unlike PTB-XL.
    "agedb_l1_enc": lambda: load_embedding_npz(
        "agedb_l1_enc", os.path.join(_AGEDB, "emb_l1.npz")),
    "agedb_tabpfn_enc": lambda: load_embedding_npz(
        "agedb_tabpfn_enc", os.path.join(_AGEDB, "emb_tabpfn.npz")),
}


def load(name: str) -> Dataset:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(LOADERS)}")
    return LOADERS[name]()


def dir_split(y: np.ndarray, bin_width: float, seed: int,
              per_bin_val: int = 10, per_bin_test: int = 20,
              min_train_per_bin: int = 1,
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balanced val/test, imbalanced train -- the DIR protocol.

    Each target bin donates up to `per_bin_test` points to test and `per_bin_val` to
    val, so val/test are near-uniform over the target range. Everything left over is
    train and therefore retains (indeed sharpens) the natural skew.

    `min_train_per_bin` keeps at least one training point in a bin rather than letting
    a sparse bin be emptied entirely into the test set, which would turn a few-shot
    evaluation into an extrapolation-only evaluation and overstate the difficulty.

    Returns:
        (train_idx, val_idx, test_idx) index arrays into y.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)

    lo = np.floor(y.min() / bin_width) * bin_width
    hi = np.ceil(y.max() / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    bin_id = np.clip(np.digitize(y, edges) - 1, 0, len(edges) - 2)

    train, val, test = [], [], []
    for b in np.unique(bin_id):
        idx = np.flatnonzero(bin_id == b)
        rng.shuffle(idx)
        budget = max(0, len(idx) - min_train_per_bin)
        n_test = min(per_bin_test, budget)
        n_val = min(per_bin_val, budget - n_test)
        test.extend(idx[:n_test])
        val.extend(idx[n_test:n_test + n_val])
        train.extend(idx[n_test + n_val:])

    return (np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test)))


def natural_split(n: int, seed: int, val_frac: float = 0.1, test_frac: float = 0.2):
    """Plain random split, for reporting results under the natural test distribution."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    return (np.array(sorted(perm[n_test + n_val:])),
            np.array(sorted(perm[n_test:n_test + n_val])),
            np.array(sorted(perm[:n_test])))
