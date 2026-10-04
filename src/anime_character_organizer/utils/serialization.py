"""
Data type coercion and serialization helpers.
"""

import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def safe_bool(value: Any) -> bool:
    """Coerce boolean-like inputs (True/False, 1/0, 'true'/'false', 'yes'/'no') to bool."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_bool_series(series: pd.Series) -> pd.Series:
    """Vectorized coercion of a pandas Series to boolean."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def safe_float(value: Any, default: float = np.nan) -> float:
    """Safely convert value to float, returning default on failure or NaN."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer, returning default on failure or NaN."""
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_json_dumps(value: Any) -> str:
    """Serialize object to JSON string with UTF-8 support and deterministic key sorting."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_dict_like(value: Any) -> Dict[str, Any]:
    """Parse a dictionary or JSON string representation of a dictionary."""
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def dataframe_to_json_records(df: pd.DataFrame, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convert dataframe rows to JSON-serializable list of dicts with NaN replaced by None."""
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return df.replace({np.nan: None}).to_dict(orient="records")


def safe_read_csv(path: Any, **kwargs: Any) -> pd.DataFrame:
    """Read CSV safely, returning an empty DataFrame if the file is missing, empty, or unparseable."""
    from pathlib import Path
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, **kwargs)
    except Exception:
        return pd.DataFrame()
