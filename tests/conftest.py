import sys
sys.path.insert(0, '../custom_components/smart_circadian_lighting')
import sys
sys.path.insert(0, '..')
"""Pytest fixtures for Smart Circadian Lighting integration tests."""

import pytest

from homeassistant.components.light import DOMAIN as LIGHT_DOMAIN
from homeassistant.components.button import DOMAIN as BUTTON_DOMAIN

from homeassistant.const import STATE_ON

from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from unittest.mock import AsyncMock

DOMAIN = "smart_circadian_lighting"

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
