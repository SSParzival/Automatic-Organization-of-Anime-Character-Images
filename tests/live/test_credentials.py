"""
Live checks of the online services and the credentials in .env. They send real requests, so they run only with
`pytest -m online`; each search uploads a synthetic gradient and SauceNAO's counts against the daily quota.
"""

import os
import secrets

import pytest
from PIL import Image

from anime_character_organizer.tagging.online import (
    DANBOORU_URL,
    DanbooruClient,
    ServiceLimitReached,
    clients_from_environment,
    send,
    upload_bytes,
)
from anime_character_organizer.utils.paths import load_dotenv

pytestmark = pytest.mark.online


@pytest.fixture(scope="module", autouse=True)
def environment():
    load_dotenv()


@pytest.fixture(scope="module")
def clients():
    return clients_from_environment()


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "gradient.png"
    Image.linear_gradient("L").convert("RGB").save(path)
    return upload_bytes(path)


def require(*names):
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.skip(f"{', '.join(missing)} not set in .env or the environment.")


def test_danbooru_anonymous_md5_lookup():
    assert DanbooruClient().posts_by_md5([secrets.token_hex(16)]) == {}


def test_danbooru_credentials_are_accepted():
    require("DANBOORU_LOGIN", "DANBOORU_API_KEY")
    danbooru = DanbooruClient(os.environ["DANBOORU_LOGIN"], os.environ["DANBOORU_API_KEY"])
    assert danbooru.check_credentials() is None
    response = send(danbooru.session, danbooru.limiter, "GET", f"{DANBOORU_URL}/profile.json")
    assert response.json().get("name", "").lower() == os.environ["DANBOORU_LOGIN"].lower()


def test_danbooru_iqdb_search_runs(clients, image):
    require("DANBOORU_LOGIN", "DANBOORU_API_KEY")
    assert isinstance(clients["danbooru"].iqdb(image), list)


def test_saucenao_search_runs(clients, image):
    require("SAUCENAO_API_KEY")
    try:
        results = clients["saucenao"].search(image)
    except ServiceLimitReached:
        pytest.skip("SauceNAO's daily quota for this IP is spent; the key cannot be checked until it resets.")
    assert isinstance(results, list)
