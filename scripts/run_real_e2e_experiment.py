#!/usr/bin/env python3
"""
End-to-end CBM on REAL Gaia DR3 light curves.

No circular information flow: concepts are extracted from actual
Gaia epoch photometry, not synthesized from features.
"""
import warnings
warnings.filterwarnings('ignore')

import sys, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm_variable_stars.shared.constants import CONCEPT_NAMES_12, LABEL_MAP
from cbm_variable_stars.models import create_model
from cbm_variable_stars.losses.cbm_loss import CBMJointLoss
from cbm_variable_stars.training.trainer import Trainer


def extract_features_from_lightcurve(times, mags, period, color, n_bins=50):
    """Extract 12 physical concepts + phase-folded curve from real light curve."""
    phases = (times % period) / period

    # Fourier fit (3 harmonics)
    design = np.zeros((len(phases), 7))
    design[:, 0] = 1
    for k in range(1, 4):
        design[:, 2*k-1] = np.sin(2*np.pi*k*phases)
        design[:, 2*k] = np.cos(2*np.pi*k*phases)

    coeffs, _, _, _ = np.linalg.lstsq(design, mags, rcond=None)

    A1 = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
    A2 = np.sqrt(coeffs[3]**2 + coeffs[4]**2)
    A3 = np.sqrt(coeffs[5]**2 + coeffs[6]**2)
    R21 = A2 / max(A1, 1e-10)
    R31 = A3 / max(A1, 1e-10)
    phi1 = np.arctan2(coeffs[1], coeffs[2])
    phi2 = np.arctan2(coeffs[3], coeffs[4])
    phi21 = (phi2 - 2*phi1) % (2*np.pi)

    amplitude = np.percentile(mags, 98) - np.percentile(mags, 2)
    mean_mag = np.mean(mags)
    skewness = float(pd.Series(mags).skew())
    kurtosis = float(pd.Series(mags).kurtosis())
    rise_fraction = phases[np.argmin(mags)]

    residuals = (mags - mean_mag) / max(np.std(mags), 0.001)
    stetson_K = np.mean(np.abs(residuals)) / np.sqrt(np.mean(residuals**2))

    # Period SNR from Lomb-Scargle
    from astropy.timeseries import LombScargle
    ls = LombScargle(times, mags)
    freq, power = ls.autopower(minimum_frequency=0.001, maximum_frequency=25.0, samples_per_peak=5)
    period_snr = float(np.max(power) / max(np.median(power), 1e-10))

    features = [period, amplitude, rise_fraction, R21, R31, phi21,
                skewness, kurtosis, stetson_K, period_snr, color, mean_mag]

    # Phase-fold and bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(phases, bin_edges) - 1, 0, n_bins - 1)
    binned = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = bin_idx == b
        if m.any():
            binned[b] = np.median(mags[m])
    valid = ~np.isnan(binned)
    if valid.sum() >= 3:
        binned = np.interp(np.arange(n_bins), np.where(valid)[0], binned[valid])
    else:
        binned = np.full(n_bins, mean_mag)

    return np.array(features, dtype=np.float32), binned.astype(np.float32)


def load_real_data(n_bins=50):
    """Load all real Gaia light curves and extract features."""
    from astropy.timeseries import LombScargle

    ep_dir = Path('data/raw/gaia/epoch_photometry')
    meta_dir = Path('data/raw/gaia/metadata')
    downloaded = set(int(f.stem) for f in ep_dir.glob('*.parquet'))

    class_map = {'RR': None, 'CEP': 'DCEP', 'DSCT_GDOR_SXPHE': 'DSCT_SXPHE',
                 'ECL': 'ECL', 'LPV': 'MIRA_SR'}

    all_features, all_curves, all_labels = [], [], []

    for meta_file in sorted(meta_dir.glob('gaia_metadata_*.parquet')):
        if 'basic' in meta_file.stem or 'periods' in meta_file.stem:
            continue
        cls = meta_file.stem.replace('gaia_metadata_', '')
        if cls not in class_map:
            continue

        df = pd.read_parquet(meta_file)
        df_dl = df[df['source_id'].isin(downloaded)]
        our_label = class_map[cls]

        count = 0
        for _, row in df_dl.iterrows():
            sid = row['source_id']
            lc = pd.read_parquet(ep_dir / f'{sid}.parquet')
            if len(lc) < 20:
                continue

            times, mags = lc['time'].values, lc['mag'].values
            color = row['bp_rp'] if pd.notna(row.get('bp_rp')) else 0.6

            # Compute period via Lomb-Scargle
            try:
                ls = LombScargle(times, mags)
                freq, power = ls.autopower(minimum_frequency=0.001, maximum_frequency=25.0,
                                           samples_per_peak=5)
                period = 1.0 / freq[np.argmax(power)]
                if period < 0.01 or period > 2000:
                    continue
            except:
                continue

            try:
                features, curve = extract_features_from_lightcurve(times, mags, period, color, n_bins)
            except:
                continue

            # Assign label
            if cls == 'RR':
                label_name = 'RRAB' if period > 0.45 else 'RRC'
            else:
                label_name = our_label

            all_features.append(features)
            all_curves.append(curve)
            all_labels.append(LABEL_MAP[label_name])
            count += 1

        print(f'  {cls}: {count} sources processed')

    features_raw = np.array(all_features, dtype=np.float32)
    curves = np.array(all_curves, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    return features_raw, curves, labels


class DictDS(torch.utils.data.Dataset):
    def __init__(self, f, l, c):
        self.f, self.l, self.c = f, l, c
    def __len__(self): return len(self.l)
    def __getitem__(self, i):
        return {'features': self.f[i], 'concept_gt': self.c[i], 'label': self.l[i]}


def main():
    import os
    os.chdir(PROJECT_ROOT)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    n_bins = 50

    # === Step 1: Load and process real data ===
    print('\n=== Step 1: Loading REAL Gaia light curves ===')
    features_raw, curves, labels = load_real_data(n_bins)
    N = len(labels)
    print(f'Total: {N} sources')
    for lbl, name in enumerate(['RRAB', 'RRC', 'DCEP', 'DSCT_SXPHE', 'ECL', 'MIRA_SR']):
        print(f'  {name}: {(labels == lbl).sum()}')

    # Build multi-channel input (4, n_bins)
    multichannel = np.zeros((N, 4, n_bins), dtype=np.float32)
    multichannel[:, 0, :] = curves
    for i in range(N):
        multichannel[:, 1, :] = np.log10(np.clip(features_raw[:, 0], 1e-3, None))[:, None]
        multichannel[:, 2, :] = features_raw[:, 10:11]
        multichannel[:, 3, :] = features_raw[:, 11:12]

    # === Step 2: 5-fold CV ===
    print('\n=== Step 2: 5-fold Cross-Validation ===')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    all_concept_preds, all_concept_gts = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(multichannel, labels)):
        print(f'\n--- Fold {fold+1}/5 (train:{len(train_idx)}, val:{len(val_idx)}) ---')

        scaler = StandardScaler()
        train_gt = scaler.fit_transform(features_raw[train_idx])
        val_gt = scaler.transform(features_raw[val_idx])

        train_ds = DictDS(
            torch.tensor(multichannel[train_idx], dtype=torch.float32),
            torch.tensor(labels[train_idx], dtype=torch.long),
            torch.tensor(train_gt, dtype=torch.float32),
        )
        val_ds = DictDS(
            torch.tensor(multichannel[val_idx], dtype=torch.float32),
            torch.tensor(labels[val_idx], dtype=torch.long),
            torch.tensor(val_gt, dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

        model = create_model('e2e_hard_cbm', n_bins=n_bins, num_classes=6, hidden_dims=[64, 32])

        # Class weights
        train_labels = labels[train_idx]
        n_classes = 6
        counts = np.bincount(train_labels, minlength=n_classes).astype(float)
        counts = np.clip(counts, 1, None)
        cw = torch.tensor(len(train_labels) / (n_classes * counts), dtype=torch.float32)
        cw = cw / cw.mean()

        loss_fn = CBMJointLoss(alpha=1.0, beta=1.0, class_weights=cw, use_concept_loss=True)
        trainer = Trainer(model=model, loss_fn=loss_fn, learning_rate=1e-3, weight_decay=1e-4,
                          max_epochs=200, patience=20, device=device,
                          log_dir='results/e2e_real/logs', checkpoint_dir='results/e2e_real/checkpoints')
        result = trainer.fit(train_loader, val_loader, fold_id=fold)

        metrics = trainer.validate(val_loader)
        fold_results.append({
            'fold': fold, 'accuracy': metrics['val_accuracy'],
            'macro_f1': metrics['val_macro_f1'], 'best_epoch': result['best_epoch'],
        })

        # Collect concept predictions
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['features'].to(device))
                all_concept_preds.append(output['concepts'].cpu().numpy())
                all_concept_gts.append(batch['concept_gt'].numpy())

    # === Step 3: Results ===
    print('\n' + '='*60)
    print('REAL DATA E2E CBM RESULTS')
    print('='*60)

    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['macro_f1'] for r in fold_results]
    print(f'Accuracy: {np.mean(accs):.4f} +/- {np.std(accs, ddof=1):.4f}')
    print(f'Macro F1: {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=1):.4f}')

    # Concept quality
    preds = np.concatenate(all_concept_preds)
    gts = np.concatenate(all_concept_gts)

    print(f'\nConcept Prediction Quality (N={len(preds)}):')
    print(f'{"Concept":20s} {"R2":>8s} {"Pearson":>8s}')
    print('-' * 40)
    concept_results = {}
    for i, name in enumerate(CONCEPT_NAMES_12):
        p, g = preds[:, i], gts[:, i]
        ss_res = np.sum((p - g) ** 2)
        ss_tot = np.sum((g - g.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        r = pearsonr(p, g)[0] if np.std(p) > 1e-6 else 0
        concept_results[name] = {'r2': float(r2), 'pearson_r': float(r)}
        print(f'{name:20s} {r2:8.4f} {r:8.4f}')

    # Save
    output_dir = Path('results/e2e_real')
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'data_source': 'REAL Gaia DR3 epoch photometry',
        'n_sources': int(N),
        'n_bins': n_bins,
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs, ddof=1)),
        'macro_f1_mean': float(np.mean(f1s)),
        'macro_f1_std': float(np.std(f1s, ddof=1)),
        'fold_results': fold_results,
        'concept_quality': concept_results,
    }
    with open(output_dir / 'real_e2e_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nResults saved to {output_dir / "real_e2e_results.json"}')
    print('\n*** NO CIRCULAR INFORMATION FLOW ***')
    print('*** Concepts extracted from REAL Gaia observations ***')


if __name__ == '__main__':
    main()
