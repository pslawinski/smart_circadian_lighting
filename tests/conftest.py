import sys

sys.path.insert(0, '../custom_components/smart_circadian_lighting')
sys.path.insert(0, '..')

"""Pytest fixtures for Smart Circadian Lighting integration tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.const import STATE_ON
from homeassistant.core import HomeAssistant
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
        self.name = entity_id

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
    from unittest.mock import AsyncMock, MagicMock

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


def create_states_get_side_effect(entity_states: dict) -> callable:
    """Create a side_effect function for states.get() that handles multiple entity IDs.
    
    Args:
        entity_states: Dict mapping entity_id to State/MockState object
    
    Returns:
        A side_effect function that returns the appropriate state for each entity_id
    """
    def states_get_side_effect(entity_id):
        if entity_id in entity_states:
            return entity_states[entity_id]
        return None
    return states_get_side_effect


@pytest.fixture
def mock_states_manager():
    """Factory for creating properly configured mock states with side_effect.
    
    Usage:
        import asyncio
        mock_hass = MagicMock()
        mock_hass.loop = asyncio.get_event_loop()
        states = mock_states_manager(mock_hass)
        states.set_light_state("light.test", STATE_ON, brightness=100)
    """
    def setup_states(mock_hass):
        import asyncio

        entity_states = {}
        mock_hass.states = MagicMock()
        mock_hass.states.get = MagicMock(side_effect=create_states_get_side_effect(entity_states))

        if not hasattr(mock_hass, 'loop') or mock_hass.loop is None:
            mock_hass.loop = asyncio.get_event_loop()

        class StateManager:
            def set_entity_state(self, entity_id: str, state):
                entity_states[entity_id] = state

            def set_light_state(self, entity_id: str, state: str, brightness: int | None = None):
                attrs = {}
                if brightness is not None:
                    attrs[ATTR_BRIGHTNESS] = brightness
                entity_states[entity_id] = MockState(entity_id, state, attrs)

            def set_sun_state(self, elevation: float = 0):
                entity_states["sun.sun"] = MockState("sun.sun", STATE_ON, {"elevation": elevation})

            def get_entity(self, entity_id: str):
                return entity_states.get(entity_id)

        return StateManager()

    return setup_states
