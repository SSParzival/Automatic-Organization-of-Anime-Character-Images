"""Online identification and name resolution against in-memory stand-ins of Danbooru and SauceNAO."""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from anime_character_organizer.exceptions import TaggingError
from anime_character_organizer.tagging.identification import OnlineIdentifier
from anime_character_organizer.tagging.online import (
    DanbooruClient,
    SauceNaoClient,
    ServiceLimitReached,
    clients_from_environment,
    file_md5,
    send,
    upload_bytes,
)
from anime_character_organizer.tagging.resolution import CharacterResolver, base_words, canonical_tags, name_key
from tests.fixtures.online_substitutes import FakeDanbooru


def test_name_key_and_base_words():
    assert name_key("  Hatsune  Miku ") == "hatsune_miku"
    assert name_key("Ｒｅｍ (Re:Zero)") == "rem_(re:zero)"
    assert base_words("rem_(re:zero)") == {"rem"}
    assert base_words("aria_(human)_(zenless_zone_zero)") == {"aria"}


def test_resolver_merges_spellings_aliases_and_costumes(tmp_path):
    client = FakeDanbooru()
    resolver = CharacterResolver(client, tmp_path / "names.json")
    names = resolver.resolve(["Saber", "saber_(fate)", "hatsune_miku_(append)", "miku hatsune", "original", "x_y"])
    assert names == {
        "Saber": "artoria_pendragon_(fate)",
        "saber_(fate)": "artoria_pendragon_(fate)",
        "hatsune_miku_(append)": "hatsune_miku",  # the copyright implication is not followed
        "miku hatsune": "hatsune_miku",
        "original": None,
        "x_y": None,
    }
    calls = len(client.calls)
    reloaded = CharacterResolver(client, tmp_path / "names.json")
    assert reloaded.resolve(["SABER"]) == {"SABER": "artoria_pendragon_(fate)"}
    assert len(client.calls) == calls  # answered from the cache file


def test_resolver_falls_back_to_spelling_when_offline():
    client = MagicMock()
    client.get.side_effect = TaggingError("down")
    resolver = CharacterResolver(client)
    with pytest.warns(RuntimeWarning, match="unresolved"):
        assert resolver.resolve(["Rem (Re:Zero)"]) == {"Rem (Re:Zero)": "rem_(re:zero)"}
    assert resolver.client is None and resolver.cache == {}


def test_canonical_tags_keeps_best_score_per_character():
    names = {"hatsune_miku": "hatsune_miku", "hatsune_miku_(append)": "hatsune_miku", "Retired Tag": None}
    merged = canonical_tags({"hatsune_miku_(append)": 0.9, "hatsune_miku": 0.6, "Retired Tag": 0.8}, names)
    assert merged == {"hatsune_miku": 0.9, "retired_tag": 0.8}  # tagger names Danbooru dropped keep their spelling


def make_image(path, color=(10, 20, 30)):
    Image.new("RGB", (1200, 600), color).save(path)
    return path


def test_identifier_prefers_exact_match_and_caches_every_service(tmp_path):
    exact = make_image(tmp_path / "exact.png")
    other = make_image(tmp_path / "other.png", (200, 0, 0))
    client = FakeDanbooru(
        posts=[{"id": 1, "md5": file_md5(exact), "tag_string_character": "hatsune_miku"}],
        iqdb=[{"score": 92.5, "post": {"id": 2, "tag_string_character": "rem_(re:zero) ram_(re:zero)"}}],
    )
    cache = tmp_path / "cache.json"
    identifier = OnlineIdentifier(client, cache_path=cache, min_similarity=85)
    identifier.prefetch([exact, other])
    assert client.calls == ["md5"]
    assert identifier.identify(exact) == {
        "provider": "danbooru_md5",
        "similarity": 100.0,
        "characters": ["hatsune_miku"],
        "reference": "danbooru:1",
    }
    assert identifier.identify(other)["characters"] == ["rem_(re:zero)", "ram_(re:zero)"]
    assert client.calls == ["md5", "iqdb"]
    identifier.save()

    stricter = OnlineIdentifier(client, cache_path=cache, min_similarity=95)
    assert stricter.identify(other) is None  # cached IQDB score below the new threshold, nothing re-sent
    assert client.calls == ["md5", "iqdb"]
    assert json.loads(cache.read_text())[file_md5(other)]["danbooru_iqdb"]["similarity"] == 92.5


def test_identifier_uses_pixiv_file_names_and_md5_file_names(tmp_path):
    pixiv = make_image(tmp_path / "123456_p1.png")
    renamed = make_image(tmp_path / f"{'a' * 32}.png", (1, 2, 3))
    client = FakeDanbooru(
        posts=[
            {"id": 7, "md5": "x", "pixiv_id": 123456, "source": "https://i.pximg.net/123456_p0.png"},
            {"id": 8, "md5": "y", "pixiv_id": 123456, "source": "https://i.pximg.net/123456_p1.png"},
            {"id": 9, "md5": "a" * 32, "tag_string_character": "rem_(re:zero)"},
        ]
    )
    client.posts_list[1]["tag_string_character"] = "hatsune_miku"
    client.can_search_images = False
    identifier = OnlineIdentifier(client)
    identifier.prefetch([pixiv, renamed])
    assert identifier.identify(renamed)["reference"] == "danbooru:9"
    found = identifier.identify(pixiv)
    assert (found["provider"], found["reference"], found["characters"]) == (
        "danbooru_pixiv",
        "danbooru:8",
        ["hatsune_miku"],
    )


def test_identifier_falls_back_to_saucenao_and_stops_at_quota(tmp_path):
    first = make_image(tmp_path / "first.png")
    second = make_image(tmp_path / "second.png", (5, 5, 5))
    client = FakeDanbooru(posts=[{"id": 42, "md5": "z", "tag_string_character": "hatsune_miku"}])
    client.can_search_images = False
    saucenao = MagicMock()
    saucenao.search.side_effect = [
        [
            {"header": {"similarity": "95.1"}, "data": {"ext_urls": ["https://x"], "title": "no characters"}},
            {"header": {"similarity": "91.0"}, "data": {"danbooru_id": 42}},
        ],
        ServiceLimitReached("daily"),
    ]
    identifier = OnlineIdentifier(client, saucenao)
    assert identifier.identify(first) == {
        "provider": "saucenao",
        "similarity": 91.0,
        "characters": ["hatsune_miku"],
        "reference": "danbooru:42",
    }
    with pytest.warns(RuntimeWarning, match="saucenao disabled"):
        assert identifier.identify(second) is None
    assert "saucenao" in identifier.disabled
    assert "saucenao" not in identifier.results(second)  # retried in a later run


def test_saucenao_x_and_pixiv_results_are_read_from_danbooru():
    client = FakeDanbooru(
        posts=[
            {"id": 5, "source": "https://x.com/artist/status/777", "tag_string_character": "hatsune_miku"},
            {"id": 6, "pixiv_id": 888, "source": "https://i.pximg.net/888_p0.png", "tag_string_character": ""},
        ]
    )
    identifier = OnlineIdentifier(client)
    assert identifier.saucenao_characters({"tweet_id": "777"}) == (["hatsune_miku"], "danbooru:5")
    by_url = {"ext_urls": ["https://twitter.com/i/web/status/777"]}
    assert identifier.saucenao_characters(by_url) == (["hatsune_miku"], "danbooru:5")
    assert identifier.saucenao_characters({"pixiv_id": 888}) is None  # a post without characters names nobody


def test_saucenao_free_text_characters():
    identifier = OnlineIdentifier(FakeDanbooru())
    data = {"characters": "Hatsune Miku, Kagamine Rin", "ext_urls": ["https://gelbooru.com/1"]}
    assert identifier.saucenao_characters(data) == (["Hatsune Miku", "Kagamine Rin"], "https://gelbooru.com/1")
    assert identifier.saucenao_characters({"title": "x"}) is None


def test_failed_hash_lookup_disables_danbooru_for_the_run(tmp_path):
    image = make_image(tmp_path / "a.png")
    client = FakeDanbooru()
    client.posts_by_md5 = MagicMock(side_effect=TaggingError("down"))
    identifier = OnlineIdentifier(client)
    with pytest.warns(RuntimeWarning, match="Danbooru disabled"):
        identifier.prefetch([image])
    assert identifier.identify(image) is None
    assert client.posts_by_md5.call_count == 1 and "iqdb" not in client.calls


def test_upload_is_a_small_jpeg(tmp_path):
    data = upload_bytes(make_image(tmp_path / "big.png"))
    with Image.open(__import__("io").BytesIO(data)) as image:
        assert image.format == "JPEG" and max(image.size) == 768
    assert file_md5(tmp_path / "big.png") == hashlib.md5((tmp_path / "big.png").read_bytes()).hexdigest()


def response(status, payload=None):
    result = MagicMock(status_code=status, ok=status < 400, headers={"content-type": "application/json"})
    result.json.return_value = payload or {}
    return result


def test_send_retries_server_errors_then_returns_last_response():
    session = MagicMock()
    session.request.side_effect = [response(503), response(200, [1])]
    limiter = MagicMock()
    with patch("anime_character_organizer.tagging.online.time.sleep") as sleep:
        assert send(session, limiter, "GET", "https://example").json() == [1]
    assert session.request.call_count == 2 and sleep.call_count == 1


def test_send_gives_up_after_repeated_network_errors():
    import requests

    session = MagicMock()
    session.request.side_effect = requests.ConnectionError("unreachable")
    with patch("anime_character_organizer.tagging.online.time.sleep"), pytest.raises(TaggingError, match="3 attempts"):
        send(session, MagicMock(), "GET", "https://example")
    assert session.request.call_count == 3


def test_saucenao_daily_limit_raises_and_short_limit_slows_down():
    client = SauceNaoClient("key")
    client.session = MagicMock()
    client.session.request.return_value = response(429, {"header": {"message": "Daily Search Limit Exceeded."}})
    with pytest.raises(ServiceLimitReached):
        client.search(b"jpeg")
    ok = {"header": {"status": 0, "short_remaining": 0}, "results": [{"header": {"similarity": "50"}, "data": {}}]}
    client.session.request.return_value = response(200, ok)
    assert len(client.search(b"jpeg")) == 1
    assert client.limiter.interval == 30.0


def test_saucenao_key_is_sent_in_the_body_and_an_exhausted_day_stops_searches():
    client = SauceNaoClient("secret")
    client.session = MagicMock()
    last = {"header": {"status": 0, "long_remaining": 0}, "results": []}
    client.session.request.return_value = response(200, last)
    assert client.search(b"jpeg") == []
    sent = client.session.request.call_args.kwargs
    assert sent["data"]["api_key"] == "secret" and "params" not in sent
    with pytest.raises(ServiceLimitReached):
        client.search(b"jpeg")
    assert client.session.request.call_count == 1


@pytest.mark.parametrize(
    "outcome, problem",
    [(response(200), None), (response(401), "is wrong"), (response(403), "restricted"), (TaggingError("down"), None)],
)
def test_danbooru_credential_check(outcome, problem):
    client = DanbooruClient("user", "key")
    with patch("anime_character_organizer.tagging.online.send", side_effect=[outcome]):
        found = client.check_credentials()
    assert found == problem if problem is None else problem in found
    assert DanbooruClient().check_credentials() is None  # no credentials, no request


def test_rejected_danbooru_credentials_fall_back_to_anonymous_lookups(monkeypatch):
    monkeypatch.setenv("DANBOORU_LOGIN", "user")
    monkeypatch.setenv("DANBOORU_API_KEY", "key")
    monkeypatch.delenv("SAUCENAO_API_KEY", raising=False)
    with patch.object(DanbooruClient, "check_credentials", return_value="rejected"), pytest.warns(RuntimeWarning):
        clients = clients_from_environment()
    assert not clients["danbooru"].can_search_images and clients["danbooru"].session.auth is None
    assert clients["saucenao"] is None


def test_saucenao_keeps_partial_results_and_rejects_failed_searches():
    client = SauceNaoClient("key")
    client.session = MagicMock()
    partial = {"header": {"status": 1}, "results": [{"header": {"similarity": "90"}, "data": {}}]}
    client.session.request.return_value = response(200, partial)
    assert len(client.search(b"jpeg")) == 1
    for header in ({"status": 1}, {"status": -2, "message": "bad image"}):
        client.session.request.return_value = response(200, {"header": header})
        with pytest.raises(TaggingError, match="status"):
            client.search(b"jpeg")
