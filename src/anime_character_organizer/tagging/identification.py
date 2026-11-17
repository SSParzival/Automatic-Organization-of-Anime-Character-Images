"""
Per-image online identification with a persistent cache.

Exact file matches on Danbooru (MD5, pixiv ID in the file name) are tried first, then reverse image
search (Danbooru IQDB, then SauceNAO). The best result of every service is cached by file MD5, so an
image is sent to each service at most once and similarity thresholds can change without new queries.
"""

import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from ..exceptions import TaggingError
from .online import (
    MD5_FILENAME,
    PIXIV_FILENAME,
    DanbooruClient,
    SauceNaoClient,
    ServiceLimitReached,
    file_md5,
    post_characters,
    upload_bytes,
)

PROVIDERS = ("danbooru_md5", "danbooru_pixiv", "danbooru_iqdb", "saucenao")
EXACT_SIMILARITY = 100.0
MAX_ERRORS = 5
STATUS_URL = re.compile(r"(?:twitter|x)\.com/.+/status/(\d+)")
SAUCENAO_FLOOR = 60.0  # weaker SauceNAO results are unrelated images; they are not followed to Danbooru


def post_result(post: Dict[str, Any], similarity: float = EXACT_SIMILARITY) -> Dict[str, Any]:
    return {
        "similarity": float(similarity),
        "characters": post_characters(post),
        "reference": f"danbooru:{post.get('id')}",
    }


class OnlineIdentifier:
    """Identify the characters of an image from the services that are configured."""

    def __init__(
        self,
        danbooru: DanbooruClient,
        saucenao: Optional[SauceNaoClient] = None,
        cache_path: Optional[Path] = None,
        min_similarity: float = 85.0,
    ):
        self.danbooru = danbooru
        self.saucenao_client = saucenao
        self.cache_path = cache_path
        self.min_similarity = min_similarity
        self.cache: Dict[str, Dict[str, Any]] = {}
        if cache_path is not None and cache_path.is_file():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.md5: Dict[str, str] = {}
        self.disabled: set = set()
        self.queries: Counter = Counter()
        self.errors: Counter = Counter()

    @property
    def can_search_images(self) -> bool:
        return self.danbooru.can_search_images or self.saucenao_client is not None

    def key(self, path: Union[str, Path]) -> str:
        path = str(path)
        if path not in self.md5:
            self.md5[path] = file_md5(path)
        return self.md5[path]

    def results(self, path: Union[str, Path]) -> Dict[str, Any]:
        return self.cache.setdefault(self.key(path), {})

    def accepted(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """First cached result, in provider order, that reaches the similarity threshold."""
        results = self.results(path)
        for provider in PROVIDERS:
            result = results.get(provider)
            if result and result["characters"] and result["similarity"] >= self.min_similarity:
                return {"provider": provider, **result}
        return None

    def prefetch(self, paths: Iterable[Union[str, Path]]) -> None:
        """Look up the MD5 of many images on Danbooru in a few batched requests."""
        pending = [str(p) for p in paths if "danbooru_md5" not in self.results(p)]
        if not pending:
            return
        names = {p: MD5_FILENAME.search(Path(p).stem) for p in pending}
        digests = [self.key(p) for p in pending] + [m.group(1) for m in names.values() if m]
        try:
            posts = self.danbooru.posts_by_md5(digests)
        except TaggingError as exc:
            self.errors["danbooru_md5"] += 1
            self.disabled |= {"danbooru_md5", "danbooru_pixiv", "danbooru_iqdb"}
            warnings.warn(f"Danbooru disabled for this run: {exc}", RuntimeWarning, stacklevel=2)
            return
        self.queries["danbooru_md5"] += 1
        for path in pending:
            named = names[path]
            post = posts.get(self.key(path)) or (posts.get(named.group(1)) if named else None)
            self.results(path)["danbooru_md5"] = post_result(post) if post else None
        self.save()

    def identify(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Best accepted identification, querying services that have not seen this image yet."""
        results = self.results(path)
        for provider in PROVIDERS:
            found = self.accepted(path)
            if found:
                return found
            if provider in results or provider in self.disabled or not self.applies(provider, path):
                continue
            try:
                results[provider] = getattr(self, provider)(str(path))
            except ServiceLimitReached as exc:
                self.disabled.add(provider)
                warnings.warn(f"{provider} disabled for this run: {exc}", RuntimeWarning, stacklevel=2)
            except (TaggingError, OSError, ValueError) as exc:
                self.errors[provider] += 1
                if self.errors[provider] >= MAX_ERRORS:
                    self.disabled.add(provider)
                    warnings.warn(f"{provider} disabled after {MAX_ERRORS} errors: {exc}", RuntimeWarning, stacklevel=2)
            else:
                self.queries[provider] += 1
                if sum(self.queries.values()) % 25 == 0:
                    self.save()
        return self.accepted(path)

    def applies(self, provider: str, path: Union[str, Path]) -> bool:
        if provider == "danbooru_pixiv":
            return bool(PIXIV_FILENAME.match(Path(path).name))
        if provider == "danbooru_iqdb":
            return self.danbooru.can_search_images
        if provider == "saucenao":
            return self.saucenao_client is not None
        return True

    def danbooru_md5(self, path: str) -> Optional[Dict[str, Any]]:
        self.prefetch([path])
        return self.results(path).get("danbooru_md5")

    def danbooru_pixiv(self, path: str) -> Optional[Dict[str, Any]]:
        match = PIXIV_FILENAME.match(Path(path).name)
        return self.pixiv_post(match.group(1), int(match.group(2))) if match else None

    def pixiv_post(self, pixiv_id: Union[str, int], page: Optional[int] = None) -> Optional[Dict[str, Any]]:
        posts = self.danbooru.posts(f"pixiv:{pixiv_id}")
        same_page = [p for p in posts if page is not None and f"_p{page}." in str(p.get("source"))]
        chosen = (same_page or posts)[:1]
        return post_result(chosen[0]) if chosen else None

    def danbooru_iqdb(self, path: str) -> Optional[Dict[str, Any]]:
        matches = self.danbooru.iqdb(upload_bytes(path))
        return post_result(matches[0]["post"], matches[0]["score"]) if matches else None

    def saucenao(self, path: str) -> Optional[Dict[str, Any]]:
        """Highest-similarity SauceNAO result that names characters, directly or through Danbooru."""
        if self.saucenao_client is None:
            return None
        for result in self.saucenao_client.search(upload_bytes(path)):
            similarity = float(result["header"]["similarity"])
            if similarity < SAUCENAO_FLOOR:
                break
            found = self.saucenao_characters(result.get("data", {}))
            if found:
                return {"similarity": similarity, "characters": found[0], "reference": found[1]}
        return None

    def saucenao_characters(self, data: Dict[str, Any]) -> Optional[tuple]:
        """Characters of a SauceNAO result: its Danbooru post, the Danbooru post of its pixiv or X source, or its own list."""
        status = data.get("tweet_id") or next(
            (m.group(1) for url in data.get("ext_urls") or [] if (m := STATUS_URL.search(str(url)))), None
        )
        queries = [
            f"id:{int(data['danbooru_id'])}" if data.get("danbooru_id") else None,
            f"pixiv:{int(data['pixiv_id'])}" if data.get("pixiv_id") else None,
            f"source:*/status/{int(status)}" if status else None,
        ]
        for query in filter(None, queries):
            posts = [p for p in self.danbooru.posts(query) if post_characters(p)]
            if posts:
                return post_characters(posts[0]), f"danbooru:{posts[0].get('id')}"
        names = [n.strip() for n in str(data.get("characters") or "").split(",") if n.strip()]
        return (names, (data.get("ext_urls") or [""])[0]) if names else None

    def save(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.cache_path)
