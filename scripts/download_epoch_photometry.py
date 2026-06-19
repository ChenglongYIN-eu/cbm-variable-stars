#!/usr/bin/env python3
"""
Download Gaia DR3 epoch photometry for variable star sources.

Downloads in small batches with retries to handle archive instability.
Saves each source as a parquet file with columns: time, mag, flux, flux_error.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from astroquery.gaia import Gaia


def download_one_source(source_id: int, output_dir: Path, max_retries: int = 3) -> bool:
    """Download epoch photometry for a single source."""
    out_path = output_dir / f"{source_id}.parquet"
    if out_path.exists():
        return True  # Already downloaded

    for attempt in range(max_retries):
        try:
            data = Gaia.load_data(
                ids=[source_id],
                data_release='Gaia DR3',
                retrieval_type='EPOCH_PHOTOMETRY',
                format='votable',
                verbose=False,
            )

            if not data:
                return False

            # Parse the result
            for key, value in data.items():
                if hasattr(value, '__iter__') and not isinstance(value, str):
                    for item in value:
                        table = item.to_table() if hasattr(item, 'to_table') else item
                        break
                else:
                    table = value.to_table() if hasattr(value, 'to_table') else value
                break

            # Extract G-band data
            if 'g_transit_time' in table.colnames:
                mask = ~np.isnan(table['g_transit_mag'].data.data) if hasattr(table['g_transit_mag'].data, 'data') else np.ones(len(table), dtype=bool)
                # Also filter rejected
                if 'rejected_by_photometry' in table.colnames:
                    reject = table['rejected_by_photometry']
                    if hasattr(reject, 'data'):
                        try:
                            mask = mask & (~reject.data.astype(bool))
                        except (ValueError, TypeError):
                            pass

                times = np.array(table['g_transit_time'], dtype=np.float64)
                mags = np.array(table['g_transit_mag'], dtype=np.float64)

                # Handle flux errors
                if 'g_transit_flux_error' in table.colnames and 'g_transit_flux' in table.colnames:
                    flux = np.array(table['g_transit_flux'], dtype=np.float64)
                    flux_err = np.array(table['g_transit_flux_error'], dtype=np.float64)
                    mag_err = np.abs(2.5 / np.log(10) * flux_err / np.clip(flux, 1e-10, None))
                else:
                    mag_err = np.full(len(times), 0.01)

                # Filter valid data
                valid = np.isfinite(times) & np.isfinite(mags)
                times = times[valid]
                mags = mags[valid]
                mag_err = mag_err[valid]

                if len(times) < 10:
                    return False

                df = pd.DataFrame({
                    'time': times,
                    'mag': mags,
                    'mag_err': mag_err,
                })
                df.to_parquet(out_path, index=False)
                return True

            return False

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                return False

    return False


def main():
    parser = argparse.ArgumentParser(description="Download Gaia epoch photometry")
    parser.add_argument("--n-per-class", type=int, default=500, help="Sources per class")
    parser.add_argument("--classes", nargs="+", default=["RR", "CEP", "DSCT_GDOR_SXPHE", "ECL", "LPV"])
    parser.add_argument("--output-dir", default="data/raw/gaia/epoch_photometry")
    parser.add_argument("--metadata-dir", default="data/raw/gaia/metadata")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = Path(args.metadata_dir)

    total_downloaded = 0
    total_failed = 0

    for cls in args.classes:
        # Find metadata file
        meta_file = None
        for pattern in [f"gaia_metadata_{cls}.parquet", f"gaia_metadata_{cls.replace('_', '|')}.parquet"]:
            candidate = metadata_dir / pattern
            if candidate.exists():
                meta_file = candidate
                break

        if meta_file is None:
            # Try glob
            matches = list(metadata_dir.glob(f"gaia_metadata_*{cls[:3]}*.parquet"))
            if matches:
                meta_file = matches[0]

        if meta_file is None:
            print(f"[SKIP] No metadata for {cls}")
            continue

        df = pd.read_parquet(meta_file)
        source_ids = df['source_id'].tolist()[:args.n_per_class]
        print(f"\n{'='*60}")
        print(f"Class: {cls} | Sources to download: {len(source_ids)}")
        print(f"{'='*60}")

        cls_downloaded = 0
        cls_failed = 0

        for i, sid in enumerate(source_ids):
            success = download_one_source(sid, output_dir)
            if success:
                cls_downloaded += 1
            else:
                cls_failed += 1

            if (i + 1) % 50 == 0:
                print(f"  [{cls}] {i+1}/{len(source_ids)} "
                      f"(downloaded: {cls_downloaded}, failed: {cls_failed})")

            # Gentle rate limiting
            if (i + 1) % 10 == 0:
                time.sleep(1)

        print(f"  [{cls}] DONE: {cls_downloaded} downloaded, {cls_failed} failed")
        total_downloaded += cls_downloaded
        total_failed += cls_failed

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_downloaded} downloaded, {total_failed} failed")
    print(f"Files saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
