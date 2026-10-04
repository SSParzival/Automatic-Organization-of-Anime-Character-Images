"""
Timestamp utilities.
"""

from datetime import datetime


def now_iso() -> str:
    """Return the current time formatted as ISO 8601 with timezone."""
    return datetime.now().astimezone().isoformat(timespec="seconds")
