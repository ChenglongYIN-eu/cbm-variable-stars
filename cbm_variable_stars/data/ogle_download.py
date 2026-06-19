# cbm_variable_stars/data/ogle_download.py
"""
OGLE-IV variable star catalog download module.

Downloads params.dat (stellar parameters) and individual light curve .dat files
for all 6 variable star types.

Key corrections (M4):
- HTTP preferred over FTP (more stable, especially for Asian networks)
- FTP connections reused within a field/type group
- Resume support: skip already-downloaded files
"""

from __future__ import annotations
import ftplib
import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.constants import OGLE_FTP_PATHS, OGLE_SUBTYPE_MAP, RANDOM_SEED


def parse_ogle_lightcurve(lc_content: str) -> pd.DataFrame:
    """
    Parse an OGLE light curve .dat file into a DataFrame.

    Parameters
    ----------
    lc_content : str
        Raw text content of the .dat file.
        Format: space-separated columns HJD  mag  mag_err [optional extra columns]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hjd, mag, mag_err
        Empty DataFrame if parsing fails completely.

    Example
    -------
    >>> content = open("OGLE-BLG-RRLYR-00001.dat").read()
    >>> lc_df = parse_ogle_lightcurve(content)
    >>> print(lc_df.head())
       hjd          mag   mag_err
    0  2451234.5  16.234  0.012
    """
    lines = lc_content.strip().split("\n")
    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                hjd = float(parts[0])
                mag = float(parts[1])
                mag_err = float(parts[2])
                data.append((hjd, mag, mag_err))
            except ValueError:
                continue
    return pd.DataFrame(data, columns=["hjd", "mag", "mag_err"])


def parse_ogle_params(
    content: str,
    field: str,
    var_class: str,
) -> pd.DataFrame:
    """
    Parse OGLE params.dat file into a DataFrame.

    Parameters
    ----------
    content : str
        Raw text content of params.dat
    field : str
        Observation field: "blg", "lmc", "smc"
    var_class : str
        Variable class directory name: "rrlyr", "cep", "ecl", "lpv", "dsct"

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including: ogle_id, period, amplitude,
        ra, dec, subtype, field, var_class
        Exact columns depend on the specific OGLE catalog.

    Notes
    -----
    OGLE params.dat format varies slightly by star type. The first line
    is usually a header. Tab or space delimited. Column 'ID' becomes 'ogle_id'.
    """
    lines = content.strip().split("\n")
    if not lines:
        return pd.DataFrame()

    # Find header line
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            header_line = stripped.lstrip("#").strip()
            data_start = i + 1
        elif stripped and header_line is None:
            # First non-comment, non-empty line might be header
            # Check if it contains alphabetic characters (header)
            if any(c.isalpha() for c in stripped.split()[0]):
                header_line = stripped
                data_start = i + 1
            break

    if header_line is None:
        logger.debug(f"[ogle/{field}/{var_class}] No header found in params.dat")
        return pd.DataFrame()

    # Parse column names
    cols = header_line.split()

    rows = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= len(cols):
            rows.append(parts[: len(cols)])
        elif len(parts) > 0:
            # Pad with None
            padded = parts + [None] * (len(cols) - len(parts))
            rows.append(padded)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Rename common column aliases
    rename_map = {
        "id": "ogle_id",
        "star_id": "ogle_id",
        "p": "period",
        "p_1": "period",
        "per": "period",
        "i_mean": "mean_mag_i",
        "v_mean": "mean_mag_v",
        "i_amp": "amplitude_i",
        "ampl_i": "amplitude_i",
        "ampl": "amplitude",
        "type": "subtype",
        "cl": "subtype",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Add metadata columns
    df["field"] = field
    df["var_class"] = var_class

    # Convert numeric columns
    for col in ["period", "amplitude", "amplitude_i", "mean_mag_i", "mean_mag_v"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure ogle_id is string
    if "ogle_id" in df.columns:
        df["ogle_id"] = df["ogle_id"].astype(str)

    logger.debug(
        f"[ogle/{field}/{var_class}] Parsed {len(df)} entries "
        f"from params.dat (cols: {list(df.columns)})"
    )
    return df


def download_ogle_file_http(
    url: str,
    timeout: int = 60,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Download a single OGLE file via HTTP.

    Parameters
    ----------
    url : str
        Full HTTP URL, e.g.:
        "https://ogledb.astrouw.edu.pl/~ogle/OCVS/blg/rrlyr/phot/OGLE-BLG-RRLYR-00001.dat"
    timeout : int
        Request timeout in seconds
    max_retries : int
        Number of retry attempts

    Returns
    -------
    str or None
        File text content, or None if download failed.

    Example
    -------
    >>> content = download_ogle_file_http(
    ...     "https://ogledb.astrouw.edu.pl/~ogle/OCVS/blg/rrlyr/params.dat"
    ... )
    >>> if content:
    ...     params_df = parse_ogle_params(content, "blg", "rrlyr")
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 10 * (2 ** attempt)
                logger.debug(f"HTTP download failed ({url}): {e}, retrying in {wait}s")
                time.sleep(wait)
            else:
                logger.debug(f"HTTP download permanently failed ({url}): {e}")
                return None


def download_ogle_catalog(
    var_type: str,
    target_n: int,
    ogle_ftp_host: str = "ftp.astrouw.edu.pl",
    ogle_ftp_base: str = "/ogle/ogle4/OCVS/",
    http_base_url: str = "https://ogledb.astrouw.edu.pl/~ogle/OCVS/",
    output_dir: str | Path = "data/raw/ogle/",
    min_observations: int = 50,
    max_retries: int = 3,
    timeout: int = 300,
    prefer_http: bool = True,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """
    Download OGLE params and light curves for a given variable star type.

    Parameters
    ----------
    var_type : str
        Variable star type: "RRAB", "RRC", "DCEP", "DSCT_SXPHE", "ECL", "MIRA_SR"
    target_n : int
        Target number of sources to download
    ogle_ftp_host : str
        OGLE FTP server hostname
    ogle_ftp_base : str
        FTP root path
    http_base_url : str
        [M4] OGLE HTTP base URL
    output_dir : str or Path
        Local save directory
    min_observations : int
        Minimum number of data points in a valid light curve
    max_retries : int
        Retry attempts
    timeout : int
        Download timeout in seconds
    prefer_http : bool
        [M4] True = try HTTP first, fall back to FTP only if HTTP fails

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Path]]
        (params_df, {ogle_id: light_curve_parquet_path})

    Key corrections (M4)
    --------------------
    1. prefer_http=True: HTTP attempted first for params.dat and light curves
    2. FTP fallback uses connection reuse within each field/var_class group
    3. Already-downloaded files are skipped (resume support)

    Example
    -------
    >>> params_df, lc_paths = download_ogle_catalog(
    ...     var_type="RRAB",
    ...     target_n=2000,
    ...     prefer_http=True,
    ... )
    >>> print(f"Got {len(params_df)} RRABs, {len(lc_paths)} light curves")
    """
    output_dir = Path(output_dir)
    lc_dir = output_dir / "light_curves" / var_type
    lc_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    if var_type not in OGLE_FTP_PATHS:
        raise ValueError(
            f"Unsupported var_type: {var_type}. "
            f"Available: {list(OGLE_FTP_PATHS.keys())}"
        )

    all_params = []
    lc_paths = {}

    for field, rel_path in OGLE_FTP_PATHS[var_type].items():
        var_class = rel_path.rstrip("/").split("/")[-1]

        logger.info(f"[OGLE/{var_type}/{field}] Processing field: {rel_path}")

        # ---- Download params.dat ----
        params_content = None

        if prefer_http:
            params_url = f"{http_base_url}{rel_path}params.dat"
            logger.info(f"[OGLE/{var_type}/{field}] HTTP: {params_url}")
            params_content = download_ogle_file_http(params_url, timeout=timeout)

        if params_content is None:
            # HTTP failed, fall back to FTP
            logger.info(
                f"[OGLE/{var_type}/{field}] HTTP unavailable, trying FTP for params.dat"
            )
            for attempt in range(max_retries):
                try:
                    ftp = ftplib.FTP(ogle_ftp_host, timeout=timeout)
                    ftp.login()
                    full_path = ogle_ftp_base + rel_path
                    ftp.cwd(full_path)
                    buf = io.BytesIO()
                    ftp.retrbinary("RETR params.dat", buf.write)
                    params_content = buf.getvalue().decode("utf-8", errors="replace")
                    ftp.quit()
                    break
                except Exception as e:
                    logger.warning(
                        f"[OGLE/{var_type}/{field}] FTP attempt {attempt+1}/{max_retries}: {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(30)

        if not params_content:
            logger.debug(f"[OGLE/{var_type}/{field}] Cannot get params.dat, skipping")
            continue

        params_df = parse_ogle_params(params_content, field, var_class)

        if params_df.empty:
            logger.debug(f"[OGLE/{var_type}/{field}] Empty params table, skipping")
            continue

        # Filter by subtype mapping
        if "subtype" in params_df.columns:
            params_df["mapped_type"] = params_df["subtype"].map(OGLE_SUBTYPE_MAP)
            params_df = params_df[params_df["mapped_type"] == var_type].copy()

        logger.info(f"[OGLE/{var_type}/{field}] Found {len(params_df)} matching sources")
        all_params.append(params_df)

    if not all_params:
        logger.error(f"[OGLE/{var_type}] No data retrieved from any field")
        return pd.DataFrame(), {}

    # Combine all fields and truncate to target
    params_combined = pd.concat(all_params, ignore_index=True)

    if len(params_combined) > target_n:
        # [M4 FIX] Use random sampling instead of period-sorted truncation
        # to avoid biasing toward long-period sources
        params_combined = params_combined.sample(
            n=target_n, random_state=RANDOM_SEED
        ).reset_index(drop=True)

    params_combined["label_name"] = var_type
    from cbm_variable_stars.shared.constants import LABEL_TO_IDX
    params_combined["label"] = LABEL_TO_IDX.get(var_type, -1)

    logger.info(
        f"[OGLE/{var_type}] Selected {len(params_combined)} sources, "
        f"starting light curve download..."
    )

    # ---- Download light curves [M4: connection reuse per field/var_class] ----
    # Group by (field, var_class) to reuse FTP connections
    group_cols = ["field", "var_class"]
    available_cols = [c for c in group_cols if c in params_combined.columns]

    if available_cols:
        groups = params_combined.groupby(available_cols)
    else:
        # Treat entire dataset as one group
        params_combined["_field"] = "unknown"
        params_combined["_var_class"] = "unknown"
        groups = params_combined.groupby(["_field", "_var_class"])

    for group_key, group in groups:
        if isinstance(group_key, str):
            field_val = group_key
            vc_val = "unknown"
        else:
            field_val = group_key[0] if len(group_key) > 0 else "unknown"
            vc_val = group_key[1] if len(group_key) > 1 else "unknown"

        ftp_conn = None  # Lazy FTP connection initialization

        for _, row in group.iterrows():
            ogle_id = row.get("ogle_id", "")
            if not ogle_id:
                continue

            local_path = lc_dir / f"{ogle_id}.parquet"

            # Resume: skip already downloaded
            if local_path.exists():
                lc_paths[ogle_id] = local_path
                continue

            lc_content = None

            if prefer_http:
                lc_url = (
                    f"{http_base_url}{field_val}/{vc_val}/phot/{ogle_id}.dat"
                )
                lc_content = download_ogle_file_http(
                    lc_url, timeout=60, max_retries=2
                )

            if lc_content is None:
                # FTP fallback with connection reuse
                for attempt in range(max_retries):
                    try:
                        if ftp_conn is None:
                            ftp_conn = ftplib.FTP(ogle_ftp_host, timeout=timeout)
                            ftp_conn.login()
                            logger.debug(
                                f"[OGLE/{var_type}/{field_val}] FTP connection established (reuse)"
                            )

                        ftp_lc_path = (
                            f"{ogle_ftp_base}{field_val}/{vc_val}/phot/{ogle_id}.dat"
                        )
                        buf = io.BytesIO()
                        ftp_conn.retrbinary(f"RETR {ftp_lc_path}", buf.write)
                        lc_content = buf.getvalue().decode("utf-8", errors="replace")
                        break

                    except (ftplib.error_perm, ftplib.error_temp, EOFError, OSError) as e:
                        logger.debug(f"[{ogle_id}] FTP download failed: {e}")
                        # Connection may be broken, reset for next attempt
                        ftp_conn = None
                        if attempt < max_retries - 1:
                            time.sleep(5)

            if lc_content:
                lc_df = parse_ogle_lightcurve(lc_content)
                if len(lc_df) >= min_observations:
                    lc_df.to_parquet(local_path, index=False)
                    lc_paths[ogle_id] = local_path

        # Close FTP connection for this group
        if ftp_conn is not None:
            try:
                ftp_conn.quit()
            except Exception:
                pass

    logger.info(
        f"[OGLE/{var_type}] Light curve download complete: "
        f"{len(lc_paths)}/{len(params_combined)}"
    )

    # Save metadata
    params_combined.to_parquet(
        meta_dir / f"ogle_params_{var_type}.parquet", index=False
    )

    return params_combined, lc_paths
