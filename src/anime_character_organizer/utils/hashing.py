"""
Cryptographic, perceptual, and identifier hashing functions.
"""

import hashlib
from pathlib import Path
from typing import Any, Union


def sha256_file(path: Union[str, Path], chunk_size: int = 1_048_576) -> str:
    """Compute the SHA-256 hex digest of a file in streaming chunks."""
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_short_hash(value: Any, length: int = 12) -> str:
    """Compute a deterministic short SHA-256 hash string for a value."""
    encoded = str(value).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()[:length]


def imagehash_to_hex(hash_obj: Any) -> str:
    """Format an imagehash object as a clean lowercase hex string."""
    return str(hash_obj).strip().lower()


def hash_hex_to_int(hash_hex: str) -> int:
    """Convert a hexadecimal hash string to an integer for fast bitwise Hamming operations."""
    return int(hash_hex, 16)


def hamming_int(a: int, b: int) -> int:
    """Compute Hamming distance between two integers using population count."""
    return (a ^ b).bit_count()
