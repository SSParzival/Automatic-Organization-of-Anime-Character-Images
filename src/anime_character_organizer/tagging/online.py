"""
Online character identification: Danbooru (file hash, pixiv ID, IQDB reverse search) and SauceNAO.

Danbooru tag lookups need no account. Reverse image search needs DANBOORU_LOGIN and DANBOORU_API_KEY
(IQDB) or SAUCENAO_API_KEY; only a downscaled JPEG of the image is uploaded.
"""

import hashlib
import io
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import requests
from PIL import Image

from ..exceptions import TaggingError

USER_AGENT = "anime-character-organizer/0.1"
DANBOORU_URL = "https://danbooru.donmai.us"
SAUCENAO_URL = "https://saucenao.com/search.php"
UPLOAD_MAX_SIDE = 768
POST_FIELDS = "id,md5,source,pixiv_id,tag_string_character"
PIXIV_FILENAME = re.compile(r"^(\d{5,10})_p(\d+)")
MD5_FILENAME = re.compile(r"([0-9a-f]{32})$")


class ServiceLimitReached(TaggingError):
    """Raised when a service quota is exhausted for the rest of the run."""


class RateLimiter:
    """Keep at least `interval` seconds between consecutive calls."""

    def __init__(self, interval: float):
        self.interval = interval
        self.last = 0.0

    def wait(self) -> None:
        delay = self.last + self.interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


def send(
    session: requests.Session, limiter: RateLimiter, method: str, url: str, retries: int = 3, **kwargs: Any
) -> requests.Response:
    """Send a request, retrying timeouts, 429 and 5xx responses with backoff."""
    for attempt in range(retries):
        limiter.wait()
        try:
            response = session.request(method, url, timeout=60, **kwargs)
        except requests.RequestException as exc:
            failure = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code != 429 and response.status_code < 500:
                return response
            failure = f"HTTP {response.status_code}"
            if attempt == retries - 1:
                return response
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                time.sleep(min(float(retry_after), 300.0))
        time.sleep(float(2 ** (attempt + 1)))
    raise TaggingError(f"{method} {url} failed after {retries} attempts ({failure}).")


def file_md5(path: Union[str, Path]) -> str:
    """MD5 of a file; Danbooru indexes posts by the MD5 of the original file."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_bytes(path: Union[str, Path], max_side: int = UPLOAD_MAX_SIDE) -> bytes:
    """Downscaled RGB JPEG of an image, enough for reverse image search."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def post_characters(post: Dict[str, Any]) -> List[str]:
    return str(post.get("tag_string_character") or "").split()


class DanbooruClient:
    """Danbooru JSON API; reverse search (IQDB) requires a login and API key."""

    def __init__(self, login: Optional[str] = None, api_key: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.can_search_images = bool(login and api_key)
        if self.can_search_images:
            self.session.auth = (str(login), str(api_key))
        # Danbooru allows 10 reads and 1 write per second; IQDB searches are sent at the write rate.
        self.limiter = RateLimiter(0.5)
        self.search_limiter = RateLimiter(1.0)

    def get(self, endpoint: str, **params: Any) -> List[Dict[str, Any]]:
        response = send(self.session, self.limiter, "GET", f"{DANBOORU_URL}/{endpoint}.json", params=params)
        if not response.ok:
            raise TaggingError(f"Danbooru {endpoint} returned HTTP {response.status_code}.")
        payload = response.json()
        return payload if isinstance(payload, list) else []

    def posts(self, query: str, limit: int = 200) -> List[Dict[str, Any]]:
        return self.get("posts", tags=query, limit=limit, only=POST_FIELDS)

    def posts_by_md5(self, md5s: Iterable[str], batch_size: int = 100) -> Dict[str, Dict[str, Any]]:
        """Posts whose original file has one of these MD5 digests."""
        unique = sorted(set(md5s))
        found: Dict[str, Dict[str, Any]] = {}
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            found |= {post["md5"]: post for post in self.posts("md5:" + ",".join(batch)) if post.get("md5")}
        return found

    def iqdb(self, image: bytes) -> List[Dict[str, Any]]:
        """Visually similar posts as {'score', 'post'} sorted by descending score."""
        response = send(
            self.session,
            self.search_limiter,
            "POST",
            f"{DANBOORU_URL}/iqdb_queries.json",
            files={"search[file]": ("image.jpg", image, "image/jpeg")},
        )
        if not response.ok:
            raise TaggingError(f"Danbooru IQDB returned HTTP {response.status_code}: {response.text[:200]}")
        matches = [m for m in response.json() if isinstance(m, dict) and isinstance(m.get("post"), dict)]
        return sorted(matches, key=lambda m: -float(m.get("score", 0)))


class SauceNaoClient:
    """SauceNAO reverse image search across Danbooru, Gelbooru, pixiv, Twitter and other indexes."""

    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.api_key = api_key
        self.limiter = RateLimiter(0.0)

    def search(self, image: bytes) -> List[Dict[str, Any]]:
        """Results sorted by descending similarity; waits out the 30 s window once if it is exhausted."""
        for _ in range(2):
            response = send(
                self.session,
                self.limiter,
                "POST",
                SAUCENAO_URL,
                retries=1,
                params={"output_type": 2, "api_key": self.api_key, "db": 999, "numres": 8},
                files={"file": ("image.jpg", image, "image/jpeg")},
            )
            header = response.json().get("header", {}) if "json" in response.headers.get("content-type", "") else {}
            message = str(header.get("message", ""))
            if response.status_code == 429:
                if "daily" in message.lower():
                    raise ServiceLimitReached(f"SauceNAO daily search limit reached: {message}")
                time.sleep(30)
                continue
            if not response.ok or int(header.get("status", 0)) != 0:
                raise TaggingError(f"SauceNAO returned HTTP {response.status_code}: {message[:200]}")
            self.limiter.interval = 30.0 if int(header.get("short_remaining", 1)) <= 0 else 0.0
            results = response.json().get("results") or []
            return sorted(results, key=lambda r: -float(r["header"]["similarity"]))
        raise ServiceLimitReached("SauceNAO short-term limit did not reset.")


def clients_from_environment() -> Dict[str, Any]:
    """Danbooru client (always) and SauceNAO client (when SAUCENAO_API_KEY is set)."""
    saucenao_key = os.environ.get("SAUCENAO_API_KEY")
    return {
        "danbooru": DanbooruClient(os.environ.get("DANBOORU_LOGIN"), os.environ.get("DANBOORU_API_KEY")),
        "saucenao": SauceNaoClient(saucenao_key) if saucenao_key else None,
    }
