# cbm_variable_stars/data/gaia_download.py
"""
Gaia DR3 variable star data download module.

Downloads metadata via TAP+ ADQL queries and epoch photometry via DataLink.
Covers all 6 variable star types defined in the project.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.constants import CLASS_NAMES, GAIA_LABEL_MAP


# [C8 FIX] ADQL query templates with quality filters:
#   - best_class_score >= 0.5 (high-confidence classifications only)
#   - has_epoch_photometry = 'true' (ensures light curves are available)
#   - num_selected_g_fov >= 12 (minimum photometric measurements)
GAIA_ADQL_TEMPLATES = {
    "RRAB": """
        SELECT s.source_id, s.ra, s.dec, s.phot_g_mean_mag,
               s.bp_rp, s.parallax, s.parallax_error,
               r.best_classification AS best_class_name,
               r.pf AS period, r.peak_to_peak_g AS amplitude
        FROM gaiadr3.gaia_source AS s
        JOIN gaiadr3.vari_rrlyrae AS r ON s.source_id = r.source_id
        JOIN gaiadr3.vari_summary AS vs ON s.source_id = vs.source_id
        WHERE r.best_classification = 'RRab'
          AND r.pf > 0 AND r.peak_to_peak_g > 0
          AND vs.num_selected_g_fov >= 12
          AND s.has_epoch_photometry = 'true'
          AND s.phot_g_mean_mag < {mag_limit}
        ORDER BY s.phot_g_mean_mag ASC
    """,
    "RRC": """
        SELECT s.source_id, s.ra, s.dec, s.phot_g_mean_mag,
               s.bp_rp, s.parallax, s.parallax_error,
               r.best_classification AS best_class_name,
               r.p1_o AS period, r.peak_to_peak_g AS amplitude
        FROM gaiadr3.gaia_source AS s
        JOIN gaiadr3.vari_rrlyrae AS r ON s.source_id = r.source_id
        JOIN gaiadr3.vari_summary AS vs ON s.source_id = vs.source_id
        WHERE r.best_classification = 'RRc'
          AND r.p1_o > 0 AND r.peak_to_peak_g > 0
          AND vs.num_selected_g_fov >= 12
          AND s.has_epoch_photometry = 'true'
          AND s.phot_g_mean_mag < {mag_limit}
        ORDER BY s.phot_g_mean_mag ASC
    """,
    "DCEP": """
        SELECT s.source_id, s.ra, s.dec, s.phot_g_mean_mag,
               s.bp_rp, s.parallax, s.parallax_error,
               c.type_best_classification AS best_class_name,
               c.pf AS period, c.peak_to_peak_g AS amplitude
        FROM gaiadr3.gaia_source AS s
        JOIN gaiadr3.vari_cepheid AS c ON s.source_id = c.source_id
        JOIN gaiadr3.vari_summary AS vs ON s.source_id = vs.source_id
        WHERE c.type_best_classification = 'DCEP'
          AND c.pf > 0 AND c.peak_to_peak_g > 0
          AND vs.num_selected_g_fov >= 12
          AND s.has_epoch_photometry = 'true'
          AND s.phot_g_mean_mag < {mag_limit}
        ORDER BY s.phot_g_mean_mag ASC
    """,
    "DSCT_SXPHE": """
        SELECT s.source_id, s.ra, s.dec, s.phot_g_mean_mag,
               s.bp_rp, s.parallax, s.parallax_error,
               v.best_class_name, v.best_class_score,
               1.0/d.frequency AS period, d.amplitude_estimate AS amplitude
        FROM gaiadr3.gaia_source AS s
        JOIN gaiadr3.vari_classifier_result AS v ON s.source_id = v.source_id
        JOIN gaiadr3.vari_summary AS vs ON s.source_id = vs.source_id
        JOIN gaiadr3.vari_short_timescale AS d ON s.source_id = d.source_id
        WHERE v.best_class_name = 'DSCT|GDOR|SXPHE'
          AND d.frequency > 0
          AND vs.num_selected_g_fov >= 12
          AND s.has_epoch_photometry = 'true'
          AND s.phot_g_mean_mag < {mag_limit}
        ORDER BY s.phot_g_mean_mag ASC
    """,
    "ECL": """
        SELECT s.source_id, s.ra, s.dec, s.phot_g_mean_mag,
               s.bp_rp, s.parallax, s.parallax_error,
               v.best_class_name, v.best_class_score,
               1.0/e.frequency AS period,
               e.geom_model_gaussian1_depth AS amplitude
        FROM gaiadr3.gaia_source AS s
        JOIN gaiadr3.vari_classifier_result AS v ON s.source_id = v.source_id
        JOIN gaiadr3.vari_summary AS vs ON s.source_id = vs.source_id
        JOIN gaiadr3.vari_eclipsing_binary AS e ON s.source_id = e.source_id
        WHERE v.best_class_name = 'ECL'
          AND e.frequency > 0
          AND vs.num_selected_g_fov >= 12
          AND s.has_epoch_photometry = 'true'
          AND s.phot_g_mean_mag < {mag_limit}
        ORDER BY s.phot_g_mean_mag ASC
    """,
    "MIRA_SR": """
        SELECT s.source_id, s.ra, s.dec, s.phot_g_mean_mag,
               s.bp_rp, s.parallax, s.parallax_error,
               v.best_class_name, v.best_class_score,
               1.0/l.frequency AS period, l.amplitude AS amplitude
        FROM gaiadr3.gaia_source AS s
        JOIN gaiadr3.vari_classifier_result AS v ON s.source_id = v.source_id
        JOIN gaiadr3.vari_summary AS vs ON s.source_id = vs.source_id
        JOIN gaiadr3.vari_long_period_variable AS l ON s.source_id = l.source_id
        WHERE v.best_class_name = 'LPV'
          AND l.frequency > 0
          AND vs.num_selected_g_fov >= 12
          AND s.has_epoch_photometry = 'true'
          AND s.phot_g_mean_mag < {mag_limit}
        ORDER BY s.phot_g_mean_mag ASC
    """,
}


def query_gaia_metadata(
    var_type: str,
    cfg: DictConfig,
    mag_limit: float = 20.5,
    max_rows: int = 5000,
) -> pd.DataFrame:
    """
    Query Gaia DR3 metadata for a given variable type via TAP+.

    Parameters
    ----------
    var_type : str
        Variable star type: one of CLASS_NAMES
    cfg : DictConfig
        Configuration object
    mag_limit : float
        Magnitude limit for source selection
    max_rows : int
        Maximum number of rows to retrieve

    Returns
    -------
    pd.DataFrame
        Metadata table with columns: source_id, ra, dec, phot_g_mean_mag,
        bp_rp, parallax, best_class_name, period, amplitude, etc.
        Empty DataFrame if query fails.

    Notes
    -----
    Uses astroquery.gaia TapPlus interface to query the Gaia archive.
    Requires internet connection.
    """
    try:
        from astroquery.gaia import Gaia
    except ImportError:
        logger.error("astroquery not installed. Run: pip install astroquery")
        return pd.DataFrame()

    if var_type not in GAIA_ADQL_TEMPLATES:
        logger.error(f"No ADQL template for var_type: {var_type}")
        return pd.DataFrame()

    # Get target count from config
    target_n = getattr(
        getattr(cfg, "var_types", {}).get(var_type, {}),
        "target_n_gaia",
        max_rows,
    )
    if hasattr(cfg, "var_types") and var_type in cfg.var_types:
        target_n = cfg.var_types[var_type].get("target_n_gaia", max_rows)

    query = GAIA_ADQL_TEMPLATES[var_type].format(mag_limit=mag_limit)

    logger.info(f"[Gaia] Querying {var_type} metadata (target: {target_n})...")

    for attempt in range(3):
        try:
            job = Gaia.launch_job_async(
                query=query,
                name=f"cbm_{var_type.lower()}",
                dump_to_file=False,
                verbose=False,
            )
            result = job.get_results()
            df = result.to_pandas()

            # Truncate to target
            if len(df) > target_n:
                df = df.head(target_n)

            # Standardize columns
            df["label_name"] = var_type
            from cbm_variable_stars.shared.constants import LABEL_TO_IDX
            df["label"] = LABEL_TO_IDX.get(var_type, -1)
            df["source_id"] = df["source_id"].astype(str)

            # ECL/MIRA_SR: period is now computed directly in ADQL
            # (1.0/e.frequency and 1.0/l.frequency respectively)

            logger.info(f"[Gaia/{var_type}] Retrieved {len(df)} sources")
            return df

        except Exception as e:
            logger.warning(f"[Gaia/{var_type}] Query attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(30)

    logger.error(f"[Gaia/{var_type}] All query attempts failed")
    return pd.DataFrame()


def download_epoch_photometry(
    source_ids: list[str] | pd.Series,
    output_dir: str | Path,
    cfg: DictConfig,
    batch_size: int = 100,
) -> dict[str, Path]:
    """
    Download Gaia DR3 epoch photometry via DataLink for a list of source IDs.

    Parameters
    ----------
    source_ids : list of str or pd.Series
        Gaia source IDs to download
    output_dir : str or Path
        Directory to save epoch photometry files (one .parquet per source)
    cfg : DictConfig
        Configuration object
    batch_size : int
        Number of sources per DataLink request batch

    Returns
    -------
    dict[str, Path]
        Mapping {source_id: local_parquet_path} for successfully downloaded sources

    Notes
    -----
    Uses astroquery.gaia DataLink API.
    Each source saved as {source_id}.parquet with columns:
        time, mag, flux, flux_error, band, rejected_by_photometry

    Key fix (M5): Requires 'time' column in output. Will raise if missing.
    """
    try:
        from astroquery.gaia import Gaia
    except ImportError:
        logger.error("astroquery not installed. Run: pip install astroquery")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_ids = list(source_ids)
    downloaded = {}
    n_total = len(source_ids)

    logger.info(f"[Gaia DataLink] Downloading epoch photometry for {n_total} sources...")

    for batch_start in range(0, n_total, batch_size):
        batch = source_ids[batch_start: batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        n_batches = (n_total + batch_size - 1) // batch_size

        logger.info(f"  Batch {batch_num}/{n_batches}: {len(batch)} sources")

        for attempt in range(3):
            try:
                # DataLink retrieval
                datalink_results = Gaia.load_data(
                    ids=batch,
                    data_release="Gaia DR3",
                    retrieval_type="EPOCH_PHOTOMETRY",
                    format="votable",
                    use_names_over_ids=True,
                )

                for source_id_key, table_list in datalink_results.items():
                    sid = str(source_id_key).strip()

                    # Skip already downloaded
                    local_path = output_dir / f"{sid}.parquet"
                    if local_path.exists():
                        downloaded[sid] = local_path
                        continue

                    for table in table_list:
                        try:
                            df = table.to_pandas()

                            # [M5] Time column must exist
                            if "time" not in df.columns:
                                # Try alternate column names
                                for alt_col in ["bjd_time", "tcb_at_gaia", "time_at_gaia"]:
                                    if alt_col in df.columns:
                                        df = df.rename(columns={alt_col: "time"})
                                        break
                                else:
                                    logger.error(
                                        f"[{sid}] Epoch photometry missing 'time' column. "
                                        f"Available: {list(df.columns)}. Skipping."
                                    )
                                    continue

                            # Compute magnitude from flux
                            if "mag" not in df.columns and "flux" in df.columns:
                                valid = df["flux"] > 0
                                df.loc[valid, "mag"] = -2.5 * np.log10(df.loc[valid, "flux"]) + 25.7

                            # Filter to G band if band column exists
                            if "band" in df.columns:
                                df_g = df[df["band"] == "G"].copy()
                            else:
                                df_g = df.copy()
                                df_g["band"] = "G"

                            # Remove rejected observations
                            if "rejected_by_photometry" in df_g.columns:
                                df_g = df_g[~df_g["rejected_by_photometry"]].copy()

                            if len(df_g) >= 10:
                                df_g.to_parquet(local_path, index=False)
                                downloaded[sid] = local_path
                        except Exception as e:
                            logger.debug(f"[{sid}] Error processing DataLink table: {e}")

                break  # Success

            except Exception as e:
                logger.warning(f"  Batch {batch_num} attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(30)
                else:
                    logger.error(f"  Batch {batch_num} permanently failed")

    logger.info(
        f"[Gaia DataLink] Downloaded: {len(downloaded)}/{n_total} sources"
    )
    return downloaded


def download_all_gaia_data(
    cfg: DictConfig,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Main entry point: query metadata and download epoch photometry for all 6 var types.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object with paths.raw_gaia, var_types, etc.
    output_dir : str or Path, optional
        Override output directory. Defaults to cfg.paths.raw_gaia.

    Returns
    -------
    pd.DataFrame
        Combined metadata table for all variable types, with column
        'has_epoch_photometry' (bool) indicating which sources have
        downloaded epoch photometry.

    Example
    -------
    >>> cfg = load_config("configs/default.yaml")
    >>> gaia_metadata = download_all_gaia_data(cfg)
    >>> print(gaia_metadata.groupby("label_name").size())
    """
    if output_dir is None:
        output_dir = Path(cfg.paths.raw_gaia)
    else:
        output_dir = Path(output_dir)

    meta_dir = output_dir / "metadata"
    ep_dir = output_dir / "epoch_photometry"
    meta_dir.mkdir(parents=True, exist_ok=True)
    ep_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    for var_type in CLASS_NAMES:
        # Check if metadata already exists
        meta_path = meta_dir / f"gaia_metadata_{var_type}.parquet"
        if meta_path.exists():
            logger.info(f"[Gaia/{var_type}] Loading cached metadata: {meta_path}")
            df = pd.read_parquet(meta_path)
        else:
            df = query_gaia_metadata(var_type=var_type, cfg=cfg)
            if len(df) > 0:
                df.to_parquet(meta_path, index=False)
                logger.info(f"[Gaia/{var_type}] Saved metadata: {meta_path}")

        if len(df) > 0:
            all_metadata.append(df)

    if not all_metadata:
        logger.error("[Gaia] No metadata retrieved for any var type")
        return pd.DataFrame()

    combined_metadata = pd.concat(all_metadata, ignore_index=True)

    # Save combined metadata
    combined_path = meta_dir / "all_metadata.parquet"
    combined_metadata.to_parquet(combined_path, index=False)
    logger.info(
        f"[Gaia] Combined metadata: {len(combined_metadata)} sources -> {combined_path}"
    )

    # Download epoch photometry
    source_ids = combined_metadata["source_id"].tolist()
    downloaded = download_epoch_photometry(
        source_ids=source_ids,
        output_dir=ep_dir,
        cfg=cfg,
    )

    # Mark which sources have epoch photometry
    combined_metadata["has_epoch_photometry"] = combined_metadata["source_id"].isin(
        downloaded.keys()
    )
    n_ep = combined_metadata["has_epoch_photometry"].sum()
    logger.info(
        f"[Gaia] Epoch photometry available: {n_ep}/{len(combined_metadata)} sources"
    )

    # Re-save with has_epoch_photometry column
    combined_metadata.to_parquet(combined_path, index=False)

    return combined_metadata
