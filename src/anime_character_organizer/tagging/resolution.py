"""
Canonical character names: one Danbooru character tag per character, whatever spelling a source used.

Spellings are matched to Danbooru tags, tag aliases (old names) are replaced by their current tag,
and costume or form tags are followed through tag implications to the base character, so
`saber`, `saber_(fate)` and `artoria_pendragon_(fate)` all resolve to `artoria_pendragon_(fate)`.
Answers are cached on disk; without network access names fall back to their normalized spelling.
"""

import json
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from ..exceptions import TaggingError
from .online import DanbooruClient

CHARACTER_CATEGORY = 4
BATCH_SIZE = 100
MAX_IMPLICATION_DEPTH = 4


def name_key(name: str) -> str:
    """Spelling-insensitive key: NFKC, lowercase, whitespace runs replaced by underscores."""
    text = unicodedata.normalize("NFKC", str(name)).strip().lower().replace("\\", "")
    return re.sub(r"[\s_]+", "_", text).strip("_")


def base_words(tag: str) -> frozenset:
    """Words of a tag without its trailing qualifiers, e.g. rem_(re:zero) -> {'rem'}."""
    return frozenset(w for w in re.sub(r"(_\([^)]*\))+$", "", tag).split("_") if w)


def batches(items: List[str], size: int = BATCH_SIZE) -> Iterable[List[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class CharacterResolver:
    """Map character names to canonical Danbooru character tags (None for non-character tags)."""

    def __init__(self, client: Optional[DanbooruClient], cache_path: Optional[Path] = None):
        self.client = client
        self.cache_path = cache_path
        self.cache: Dict[str, Optional[str]] = {}
        if cache_path is not None and cache_path.is_file():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def resolve(self, names: Iterable[str]) -> Dict[str, Optional[str]]:
        keys = {name: name_key(name) for name in names}
        pending = sorted({k for k in keys.values() if k and k not in self.cache})
        if pending and self.client is not None:
            try:
                self.cache |= self.lookup(pending)
            except TaggingError as exc:
                warnings.warn(f"Character names left unresolved: {exc}", RuntimeWarning, stacklevel=2)
                self.client = None
            else:
                self.save()
        return {name: self.cache.get(key, key or None) for name, key in keys.items()}

    @property
    def api(self) -> DanbooruClient:
        if self.client is None:
            raise TaggingError("No Danbooru client.")
        return self.client

    def save(self) -> None:
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def tag_info(self, names: Iterable[str]) -> Dict[str, int]:
        """Category of each name that is a Danbooru tag in use."""
        info: Dict[str, int] = {}
        for batch in batches(sorted(set(names))):
            for tag in self.api.get("tags", **{"search[name_comma]": ",".join(batch), "limit": 1000}):
                if int(tag.get("post_count", 0)) > 0 and not tag.get("is_deprecated"):
                    info[tag["name"]] = int(tag["category"])
        return info

    def relations(self, endpoint: str, names: Iterable[str]) -> Dict[str, List[str]]:
        """Active alias or implication consequents per antecedent."""
        found: Dict[str, List[str]] = {}
        for batch in batches(sorted(set(names))):
            params = {"search[antecedent_name_comma]": ",".join(batch), "search[status]": "active", "limit": 1000}
            for row in self.api.get(endpoint, **params):
                found.setdefault(row["antecedent_name"], []).append(row["consequent_name"])
        return found

    def autocomplete(self, key: str) -> Optional[str]:
        """Character tag whose name, ignoring qualifiers and word order, is the free-text key."""
        words = base_words(key)
        rows = self.api.get("autocomplete", **{"search[query]": key.replace("_", " "), "search[type]": "tag_query"})
        matches = sorted(
            (
                (int(r.get("post_count", 0)), r["value"])
                for r in rows
                if r.get("category") == CHARACTER_CATEGORY and base_words(r["value"]) == words
            ),
            reverse=True,
        )
        if not matches or (len(matches) > 1 and matches[0][0] < 5 * matches[1][0]):
            return None
        return matches[0][1]

    def lookup(self, keys: List[str]) -> Dict[str, Optional[str]]:
        info = self.tag_info(keys)
        aliases = self.relations("tag_aliases", [k for k in keys if k not in info])
        tags: Dict[str, Optional[str]] = {k: k if k in info else next(iter(aliases.get(k, [])), None) for k in keys}
        info |= self.tag_info(t for t in tags.values() if t and t not in info)
        for key, tag in list(tags.items()):
            if tag is None or tag not in info:
                tags[key] = found = self.autocomplete(key)
                if found:
                    info[found] = CHARACTER_CATEGORY
        characters = {t for t in tags.values() if t is not None and info.get(t) == CHARACTER_CATEGORY}
        bases = self.base_characters(characters)
        return {k: bases[t] if t in characters else None for k, t in tags.items()}

    def base_characters(self, tags: Iterable[str]) -> Dict[str, str]:
        """Follow implications that lead to exactly one other character tag (costumes, forms)."""
        base = {t: t for t in tags}
        frontier = set(base.values())
        for _ in range(MAX_IMPLICATION_DEPTH):
            implied = self.relations("tag_implications", frontier)
            categories = self.tag_info(c for targets in implied.values() for c in targets)
            step = {}
            for tag, targets in implied.items():
                characters = [t for t in targets if categories.get(t) == CHARACTER_CATEGORY]
                if len(characters) == 1:
                    step[tag] = characters[0]
            if not step:
                break
            base = {t: step.get(b, b) for t, b in base.items()}
            frontier = set(step.values())
        return base


def canonical_tags(tags: Mapping[str, float], names: Mapping[str, Optional[str]]) -> Dict[str, float]:
    """Merge tagger scores under canonical names (highest score wins); unresolved names keep their spelling."""
    merged: Dict[str, float] = {}
    for tag, score in tags.items():
        name = names.get(tag) or name_key(tag)
        merged[name] = max(float(score), merged.get(name, 0.0))
    return dict(sorted(merged.items(), key=lambda item: -item[1]))
