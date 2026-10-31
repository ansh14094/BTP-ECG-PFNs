#!/usr/bin/env python3
"""
Two-stage age prediction: extract penultimate-layer (pre-linear) features from a
trained ResNet1D-Wang / Inception1D backbone for each 12-lead ECG record, treat
those as tabular features, and fit a TabPFN regressor on top of them.

Requires a checkpoint already trained with train.py (--resume points to it).

Usage:
    python tabpfn_feature_experiment.py --resume checkpoint_resnetWang/<run>/ckpt.best.pth.tar
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import PTBXLDataset
from resnet1d_wang_fds import resnet1d_wang_fds
from inception1d_fds import inception1d_fds

BASE = os.path.dirname(os.path.abspath(__file__))


def build_model(model_name, ckpt_path, device):
    if model_name == 'resnet1d_wang_fds':
        model = resnet1d_wang_fds(input_channels=12, fds=False)
    elif model_name == 'inception1d_fds':
        model = inception1d_fds(input_channels=12, fds=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, loader, device):
    # Both backbones expose their final regressor as `self.linear`, applied
    # directly to the pooled encoding. Swapping it for Identity makes eval-mode
    # forward() return that encoding instead of the age prediction, without
    # touching resnet1d_wang_fds.py / inception1d_fds.py.
    orig_linear = model.linear
    model.linear = nn.Identity()
    feats, ages = [], []
    try:
        for ecg, age, _ in loader:
            ecg = ecg.to(device)
            encoding = model(ecg)
            feats.append(encoding.cpu().numpy())
            ages.append(age.numpy().reshape(-1))
    finally:
        model.linear = orig_linear
    return np.concatenate(feats, axis=0), np.concatenate(ages, axis=0)


def shot_metrics(preds, labels):
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    out = {}
    for name, mask in [
        ('many (age<=30)', labels <= 30),
        ('median (30<age<60)', (labels > 30) & (labels < 60)),
        ('low (age>=60)', labels >= 60),
    ]:
        if mask.sum() > 0:
            err = preds[mask] - labels[mask]
            out[name] = {'n': int(mask.sum()), 'mae': float(np.mean(np.abs(err))), 'mse': float(np.mean(err ** 2))}
        else:
            out[name] = {'n': 0, 'mae': float('nan'), 'mse': float('nan')}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, required=True, help='path to trained checkpoint (.pth.tar)')
    parser.add_argument('--model', type=str, default='resnet1d_wang_fds',
                         choices=['resnet1d_wang_fds', 'inception1d_fds'])
    parser.add_argument('--data_dir', type=str, default=os.path.join(BASE, 'data'))
    parser.add_argument('--dataset', type=str, default='1000_timesteps')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default=os.path.join(BASE, 'tabpfn_results'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_csv(os.path.join(args.data_dir, f'{args.dataset}.csv'))
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']

    loaders = {}
    for name, sub in [('train', df_train), ('val', df_val), ('test', df_test)]:
        ds = PTBXLDataset(data_dir=args.data_dir, df=sub, split=name)
        loaders[name] = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

    print(f"Loading {args.model} checkpoint from {args.resume}")
    model = build_model(args.model, args.resume, device)

    print("Extracting penultimate-layer features...")
    X_train, y_train = extract_features(model, loaders['train'], device)
    X_val, y_val = extract_features(model, loaders['val'], device)
    X_test, y_test = extract_features(model, loaders['test'], device)
    print(f"Feature shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    np.savez(os.path.join(args.out_dir, 'features.npz'),
             X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    print("Fitting TabPFN regressor on extracted features...")
    from tabpfn import TabPFNRegressor
    reg = TabPFNRegressor()
    reg.fit(X_train, y_train)

    for split_name, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
        preds = reg.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        mse = float(np.mean((preds - y) ** 2))
        print(f"\n[{split_name}] Overall MAE={mae:.3f} MSE={mse:.3f}")
        for shot_name, m in shot_metrics(preds, y).items():
            print(f"  {shot_name}: n={m['n']}, MAE={m['mae']:.3f}, MSE={m['mse']:.3f}")

        pd.DataFrame({'true_age': y, 'pred_age': preds}).to_csv(
            os.path.join(args.out_dir, f'tabpfn_preds_{split_name}.csv'), index=False)

    print(f"\nDone. Results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
