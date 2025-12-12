import sys
sys.path.insert(0, '../custom_components/smart_circadian_lighting')
import sys
sys.path.insert(0, '..')
"""Pytest fixtures for Smart Circadian Lighting integration tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components.light import DOMAIN as LIGHT_DOMAIN, ATTR_BRIGHTNESS
from homeassistant.components.button import DOMAIN as BUTTON_DOMAIN
from homeassistant.const import STATE_ON, STATE_OFF
from homeassistant.core import HomeAssistant, State
from pytest_homeassistant_custom_component.common import MockConfigEntry

DOMAIN = "smart_circadian_lighting"


class MockState:
    """A State-like object that isn't affected by global mocking."""
    def __init__(self, entity_id: str, state: str, attributes: dict | None = None):
        if attributes is None:
            attributes = {}
        self.entity_id = entity_id
        self.state = state
        self.attributes = MockAttributes(attributes)

class MockAttributes:
    """Attributes dict that behaves correctly."""
    def __init__(self, attributes: dict):
        self._attributes = attributes.copy()

    def get(self, key, default=None):
        return self._attributes.get(key, default)


@pytest.fixture
def mock_state_factory():
    """Factory fixture for creating mock State objects without global patching."""
    def create_mock_state(entity_id: str, state: str, attributes: dict | None = None) -> MockState:
        """Create a State-like object that behaves correctly for testing."""
        return MockState(entity_id, state, attributes)
    return create_mock_state


@pytest.fixture
def mock_hass_with_services():
    """Mock HASS instance with properly configured services."""
    hass = MagicMock(spec=HomeAssistant)
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock()
    hass.states.async_set = MagicMock()
    hass.bus = MagicMock()
    hass.bus.async_fire = AsyncMock()
    hass.data = {}
    return hass


@pytest.fixture
def mock_config_entry_factory():
    """Factory for creating mock config entries."""
    def create_entry(unique_id="test", data=None):
        if data is None:
            data = {
                "lights": ["light.test_light"],
                "day_brightness": 100,
                "night_brightness": 10,
                "morning_start_time": "06:00:00",
                "morning_end_time": "07:00:00",
                "evening_start_time": "20:00:00",
                "evening_end_time": "21:00:00",
                "color_temp_enabled": False,
            }
        return MockConfigEntry(
            domain=DOMAIN,
            unique_id=unique_id,
            data=data,
        )
    return create_entry

@pytest.fixture
async def single_light_entry(hass: HomeAssistant):
    """Config entry for single light."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="test_single_light",
        data={
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
            "color_temp_enabled": False,
        },
    )
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    return entry

@pytest.fixture
async def multiple_lights_entry(hass: HomeAssistant):
    """Config entry for multiple lights."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="test_multiple_lights",
        data={
            "lights": ["light.test_light1", "light.test_light2"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
            "color_temp_enabled": False,
        },
    )
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    return entry

@pytest.fixture
def mock_config_entry():
    """Mock config entry for repro tests."""
    import tempfile
    from homeassistant.core import HomeAssistant
    from unittest.mock import MagicMock, AsyncMock

    # Create a mock hass instance
    hass = MagicMock()
    hass.config_entries = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.async_block_till_done = AsyncMock()
    hass.states = MagicMock()
    hass.states.async_set = MagicMock()
    hass.states.get = MagicMock()
    hass.bus = MagicMock()
    hass.bus.async_fire = AsyncMock()
    hass.data = {}

    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="mock_repro",
        data={
            "lights": ["light.physical"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
        },
    )
    entry.async_update_entry = AsyncMock()

    return hass, entry
