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
    clients_from_environment,
    send,
    upload_bytes,
)
from anime_character_organizer.utils.paths import load_dotenv

pytestmark = pytest.mark.online


@pytest.fixture(scope="module")
def clients():
    load_dotenv()
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


def test_danbooru_credentials_are_accepted(clients):
    require("DANBOORU_LOGIN", "DANBOORU_API_KEY")
    danbooru = clients["danbooru"]
    response = send(danbooru.session, danbooru.limiter, "GET", f"{DANBOORU_URL}/profile.json")
    reasons = {
        401: "DANBOORU_LOGIN or DANBOORU_API_KEY is wrong",
        403: "the API key is valid but restricted; clear its permissions and IP addresses on Danbooru",
    }
    assert response.status_code == 200, reasons.get(response.status_code, f"HTTP {response.status_code}")
    assert response.json().get("name", "").lower() == os.environ["DANBOORU_LOGIN"].lower()


def test_danbooru_iqdb_search_runs(clients, image):
    require("DANBOORU_LOGIN", "DANBOORU_API_KEY")
    assert isinstance(clients["danbooru"].iqdb(image), list)


def test_saucenao_search_runs(clients, image):
    require("SAUCENAO_API_KEY")
    assert isinstance(clients["saucenao"].search(image), list)
