"""
Data type coercion and serialization helpers.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

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
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def safe_json_dumps(value: Any) -> str:
    """Serialize object to JSON string with UTF-8 support and deterministic key sorting."""
    return json.dumps(json_safe_value(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


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


def safe_read_csv(path: Any, **kwargs: Any) -> pd.DataFrame:
    """Read optional CSVs; missing/empty files yield an empty frame, malformed files raise."""
    from .paths import checked_path

    p = checked_path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def json_safe_value(value: Any) -> Any:
    """Convert nested NumPy scalars/non-finite metrics to strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def require_columns(frame: pd.DataFrame, columns: set[str], context: str) -> None:
    """Reject malformed stage tables with an actionable schema error."""
    from ..exceptions import ConfigurationError

    missing = columns - set(frame.columns)
    if missing:
        raise ConfigurationError(f"{context} missing required columns: {sorted(missing)}")


def write_json(path: Union[str, Path], payload: Any) -> Path:
    """Serialize payload as strict UTF-8 JSON with non-finite values converted to null."""
    from .paths import checked_path

    target = checked_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(json_safe_value(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
    return target
