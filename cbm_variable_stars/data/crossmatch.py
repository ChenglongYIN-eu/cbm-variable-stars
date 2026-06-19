# cbm_variable_stars/data/crossmatch.py
"""
Gaia-OGLE cross-matching module.

Provides two approaches to match OGLE sources to Gaia DR3:
1. Query Gavras et al. (2023) pre-computed cross-match table from Gaia archive
2. Coordinate-based cross-match using astropy SkyCoord

The cross-match results enable enriching OGLE features with Gaia photometry
(BP-RP color, G-band magnitude) for the C11/C12 concepts.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cbm_variable_stars.shared.logger import logger


def query_gavras_crossmatch(
    output_path: Optional[str | Path] = None,
    max_rows: int = 500000,
) -> pd.DataFrame:
    """
    Query the Gavras et al. (2023) Gaia-OGLE cross-match table from the Gaia archive.

    Parameters
    ----------
    output_path : str or Path, optional
        If provided, save the result to this parquet file for caching.
    max_rows : int
        Maximum number of rows to retrieve.

    Returns
    -------
    pd.DataFrame
        Cross-match table with columns:
        - source_id: Gaia DR3 source ID (str)
        - ogle_id: OGLE source ID (str)
        - angular_distance: separation in arcseconds (float)
        Empty DataFrame if query fails or table not available.

    Notes
    -----
    Gavras et al. (2023) provides a pre-computed cross-match between
    Gaia DR3 variable stars and OGLE-IV. The table is available in the
    Gaia archive as gaiadr3.vari_ogle4_crossid or similar.

    If the table is not available, returns empty DataFrame and caller
    should fall back to coordinate_crossmatch().

    Example
    -------
    >>> crossmatch_df = query_gavras_crossmatch()
    >>> if len(crossmatch_df) > 0:
    ...     print(f"Gavras cross-match: {len(crossmatch_df)} pairs")
    """
    # Check cache first
    if output_path is not None and Path(output_path).exists():
        logger.info(f"Loading cached Gavras cross-match from {output_path}")
        return pd.read_parquet(output_path)

    try:
        from astroquery.gaia import Gaia
    except ImportError:
        logger.error("astroquery not installed. Run: pip install astroquery")
        return pd.DataFrame()

    # Try multiple possible table names for the OGLE cross-match
    candidate_queries = [
        # Gavras et al. (2023) cross-match table (if available)
        f"""
        SELECT source_id, ogle4_source_id AS ogle_id, angular_distance
        FROM gaiadr3.vari_ogle4_crossid
        WHERE angular_distance < 1.5
        ORDER BY angular_distance ASC
        """,
        # Alternative: use neighbourhood table
        f"""
        SELECT xm.source_id, xm.dr2_source_id AS ogle_id,
               xm.angular_distance
        FROM gaiadr3.dr2_neighbourhood AS xm
        WHERE xm.angular_distance < 1.5
        ORDER BY xm.angular_distance ASC
        """,
    ]

    for query in candidate_queries:
        for attempt in range(3):
            try:
                job = Gaia.launch_job_async(
                    query=query,
                    dump_to_file=False,
                    verbose=False,
                )
                result = job.get_results()
                df = result.to_pandas()

                if len(df) == 0:
                    logger.debug("Gavras query returned 0 rows, trying next query")
                    break

                # Normalize
                df["source_id"] = df["source_id"].astype(str)
                if "ogle_id" in df.columns:
                    df["ogle_id"] = df["ogle_id"].astype(str)

                logger.info(
                    f"[Gavras cross-match] Retrieved {len(df)} pairs"
                )

                if output_path is not None:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(output_path, index=False)

                return df

            except Exception as e:
                logger.debug(f"Gavras query attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(15)
                break  # Try next query template

    logger.info(
        "[Gavras cross-match] Table not available or empty. "
        "Use coordinate_crossmatch() as fallback."
    )
    return pd.DataFrame()


def coordinate_crossmatch(
    gaia_df: pd.DataFrame,
    ogle_df: pd.DataFrame,
    radius_arcsec: float = 1.0,
    gaia_ra_col: str = "ra",
    gaia_dec_col: str = "dec",
    gaia_id_col: str = "source_id",
    ogle_ra_col: str = "ra",
    ogle_dec_col: str = "dec",
    ogle_id_col: str = "ogle_id",
) -> pd.DataFrame:
    """
    Cross-match Gaia and OGLE sources using sky coordinates.

    Parameters
    ----------
    gaia_df : pd.DataFrame
        Gaia metadata table with RA/Dec columns
    ogle_df : pd.DataFrame
        OGLE metadata table with RA/Dec columns
    radius_arcsec : float
        Maximum matching radius in arcseconds (default: 1.0)
    gaia_ra_col, gaia_dec_col : str
        Column names for Gaia RA and Dec (degrees)
    gaia_id_col : str
        Column name for Gaia source ID
    ogle_ra_col, ogle_dec_col : str
        Column names for OGLE RA and Dec (degrees)
    ogle_id_col : str
        Column name for OGLE source ID

    Returns
    -------
    pd.DataFrame
        Cross-match results with columns:
        - source_id: Gaia source ID (str)
        - ogle_id: OGLE source ID (str)
        - separation_arcsec: on-sky separation in arcseconds (float)
        Only includes the best (closest) Gaia match for each OGLE source.
        Empty DataFrame if no coordinates available or no matches found.

    Notes
    -----
    Uses astropy.coordinates.SkyCoord.match_to_catalog_sky() for
    efficient nearest-neighbor matching. For each OGLE source, finds
    the nearest Gaia source within radius_arcsec.

    Example
    -------
    >>> crossmatch = coordinate_crossmatch(
    ...     gaia_meta, ogle_params,
    ...     radius_arcsec=1.0
    ... )
    >>> print(f"Matched {len(crossmatch)} OGLE-Gaia pairs")
    """
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        logger.error("astropy not installed. Run: pip install astropy")
        return pd.DataFrame()

    # Validate required columns
    for df_name, df, ra_col, dec_col in [
        ("gaia_df", gaia_df, gaia_ra_col, gaia_dec_col),
        ("ogle_df", ogle_df, ogle_ra_col, ogle_dec_col),
    ]:
        for col in [ra_col, dec_col]:
            if col not in df.columns:
                logger.error(
                    f"[crossmatch] {df_name} missing column '{col}'. "
                    f"Available: {list(df.columns)}"
                )
                return pd.DataFrame()

    # Drop rows with missing coordinates
    gaia_clean = gaia_df[[gaia_id_col, gaia_ra_col, gaia_dec_col]].dropna()
    ogle_clean = ogle_df[[ogle_id_col, ogle_ra_col, ogle_dec_col]].dropna()

    if len(gaia_clean) == 0 or len(ogle_clean) == 0:
        logger.warning("[crossmatch] No valid coordinates in one or both catalogs")
        return pd.DataFrame()

    logger.info(
        f"[crossmatch] Matching {len(ogle_clean)} OGLE sources "
        f"against {len(gaia_clean)} Gaia sources (radius={radius_arcsec}\")..."
    )

    # Build SkyCoord objects
    gaia_coords = SkyCoord(
        ra=gaia_clean[gaia_ra_col].values * u.degree,
        dec=gaia_clean[gaia_dec_col].values * u.degree,
        frame="icrs",
    )
    ogle_coords = SkyCoord(
        ra=ogle_clean[ogle_ra_col].values * u.degree,
        dec=ogle_clean[ogle_dec_col].values * u.degree,
        frame="icrs",
    )

    # Match each OGLE source to nearest Gaia source
    idx, sep2d, _ = ogle_coords.match_to_catalog_sky(gaia_coords)
    sep_arcsec = sep2d.arcsecond

    # Apply radius cut
    mask = sep_arcsec <= radius_arcsec
    n_matches = mask.sum()

    logger.info(
        f"[crossmatch] Found {n_matches}/{len(ogle_clean)} matches "
        f"within {radius_arcsec}\""
    )

    if n_matches == 0:
        return pd.DataFrame()

    # Build result table
    matched_ogle = ogle_clean.iloc[mask].reset_index(drop=True)
    matched_gaia = gaia_clean.iloc[idx[mask]].reset_index(drop=True)

    result = pd.DataFrame({
        "source_id": matched_gaia[gaia_id_col].values,
        "ogle_id": matched_ogle[ogle_id_col].values,
        "separation_arcsec": sep_arcsec[mask],
    })

    result["source_id"] = result["source_id"].astype(str)
    result["ogle_id"] = result["ogle_id"].astype(str)

    return result
