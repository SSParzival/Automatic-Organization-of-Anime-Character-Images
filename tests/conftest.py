"""Shared test configuration: online services are unreachable unless a test provides its own clients."""

from unittest.mock import patch

import pytest

from tests.fixtures.model_substitutes import offline_clients


@pytest.fixture(autouse=True)
def no_online_services():
    with patch("anime_character_organizer.workflows.naming.clients_from_environment", side_effect=offline_clients):
        yield
