"""In-memory stand-in for the Danbooru endpoints used by online identification and name resolution.

The tags, aliases and implications are fixtures shaped like Danbooru data, not a copy of it.
"""

TAGS = {
    "hatsune_miku": 4,
    "hatsune_miku_(append)": 4,
    "saber_(fate)": 4,
    "artoria_pendragon_(fate)": 4,
    "rem_(re:zero)": 4,
    "original": 3,
    "vocaloid": 3,
}
ALIASES = {"saber": "saber_(fate)"}
IMPLICATIONS = {
    "saber_(fate)": ["artoria_pendragon_(fate)"],
    "hatsune_miku_(append)": ["hatsune_miku"],
    "hatsune_miku": ["vocaloid"],
}
AUTOCOMPLETE = {"miku hatsune": [{"value": "hatsune_miku", "category": 4, "post_count": 150000}]}


class FakeDanbooru:
    """Answers the endpoints the resolver and identifier use; records every call."""

    can_search_images = True

    def __init__(self, posts=(), iqdb=()):
        self.posts_list = list(posts)
        self.iqdb_matches = list(iqdb)
        self.calls = []

    def get(self, endpoint, **params):
        self.calls.append(endpoint)
        if endpoint == "tags":
            names = params["search[name_comma]"].split(",")
            return [{"name": n, "category": TAGS[n], "post_count": 10} for n in names if n in TAGS]
        if endpoint in {"tag_aliases", "tag_implications"}:
            table = ALIASES if endpoint == "tag_aliases" else IMPLICATIONS
            rows = []
            for name in params["search[antecedent_name_comma]"].split(","):
                targets = table.get(name, [])
                for target in [targets] if isinstance(targets, str) else targets:
                    rows.append({"antecedent_name": name, "consequent_name": target})
            return rows
        return AUTOCOMPLETE.get(params["search[query]"], [])

    def posts(self, query, limit=200):
        self.calls.append(query)
        key, _, value = query.partition(":")
        if key == "source":
            return [p for p in self.posts_list if str(p.get("source", "")).endswith(value.lstrip("*"))]
        return [p for p in self.posts_list if str(p.get(key if key != "pixiv" else "pixiv_id")) == value]

    def posts_by_md5(self, md5s):
        self.calls.append("md5")
        return {p["md5"]: p for p in self.posts_list if p["md5"] in set(md5s)}

    def iqdb(self, image):
        self.calls.append("iqdb")
        return self.iqdb_matches
