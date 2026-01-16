
import asyncio
import logging
from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from freezegun import freeze_time
from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.const import STATE_OFF, STATE_ON
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_circadian_lighting import DOMAIN
from custom_components.smart_circadian_lighting.circadian_logic import _convert_percent_to_255
from custom_components.smart_circadian_lighting.light import CircadianLight

_LOGGER = logging.getLogger(__name__)

@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock()
    hass.states.async_set = MagicMock()
    hass.bus = MagicMock()
    hass.bus.async_fire = AsyncMock()
    hass.data = {
        "entity_registry": MagicMock(),
        DOMAIN: {
            "test_entry": {
                "manual_overrides_enabled": True,
                "config": {
                    "day_brightness": 100,
                    "night_brightness": 10,
                    "morning_start_time": "06:00:00",
                    "morning_end_time": "08:00:00",
                    "evening_start_time": "18:00:00",
                    "evening_end_time": "20:00:00",
                }
            }
        }
    }
    hass.loop = asyncio.new_event_loop()

    def async_create_task_mock(coro):
        """Mock async_create_task to avoid 'never awaited' warnings."""
        if asyncio.iscoroutine(coro):
            coro.close()
        return MagicMock()

    hass.async_create_task = MagicMock(side_effect=async_create_task_mock)

    # Mock entity registry async_get
    with patch("homeassistant.helpers.entity_registry.async_get", return_value=hass.data["entity_registry"]):
        yield hass

@pytest.fixture
def config_entry():
    """Create a mock config entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "lights": ["light.test_bulb"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "08:00:00",
            "evening_start_time": "18:00:00",
            "evening_end_time": "20:00:00",
            "manual_override_threshold": 5,
            "morning_override_clear_time": "05:00:00",
            "evening_override_clear_time": "17:00:00",
        },
        entry_id="test_entry",
    )
    return entry

@pytest.mark.asyncio
async def test_power_on_during_day_restores_night_brightness_no_override(mock_hass, config_entry):
    """Test that a bulb turning on during the day with night brightness is NOT overridden.

    Scenario:
    1. Light was last set to night brightness (25) during the night.
    2. It is now noon (day brightness should be 255).
    3. Light turns on and reports brightness 25 (its power-on state).
    4. SYSTEM SHOULD NOT FLAG OVERRIDE and should update it to 255.
    """
    light_entity_id = "light.test_bulb"
    config = config_entry.data

    # Initialize light
    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value={})
    mock_store.async_save = AsyncMock()
    with patch("custom_components.smart_circadian_lighting.light.Store", return_value=mock_store):
        light = CircadianLight(mock_hass, light_entity_id, config, config_entry, mock_store)
        light.hass = mock_hass
        light.entity_id = "light.circadian_test"

    # Set the 'last set' state to night brightness
    night_brightness = _convert_percent_to_255(10) # 25
    light._last_set_brightness = night_brightness
    light._first_update_done = True

    # Set current internal brightness to day level
    light._brightness = 255

    # Enable color temp to trigger turn-on update
    light._color_temp_schedule = {"something": "here"}

    # Mock current time to noon
    noon = datetime.combine(datetime.now().date(), time(12, 0, 0))

    # Mock light state change: OFF -> ON with brightness 25
    old_state = MagicMock(state=STATE_OFF, attributes={})
    new_state = MagicMock(state=STATE_ON, attributes={ATTR_BRIGHTNESS: night_brightness})
    event = MagicMock(data={"old_state": old_state, "new_state": new_state})

    mock_hass.states.get.return_value = new_state

    # We expect this to NOT trigger an override because it matches _last_set_brightness
    # But we ALSO expect it to trigger a circadian update because it's a fresh turn-on
    with freeze_time(noon):
        await light._async_entity_state_changed(event)

    # Check if overridden
    assert not light._is_overridden, "Light should NOT be overridden when turning on with last commanded state"
    # Check if force update was called
    mock_hass.async_create_task.assert_called()

@pytest.mark.asyncio
async def test_evening_transition_start_false_soft_override(mock_hass, config_entry):
    """Test that a bulb already at night brightness at evening transition start is NOT soft overridden.

    Scenario:
    1. Light is at night brightness (25) - maybe it was just turned on or manually set earlier.
    2. Evening transition starts (18:00).
    3. Transition logic checks if light is "ahead" of transition.
    4. Since night brightness < day brightness, it might think it's a soft override.
    5. BUT if it matches _last_set_brightness, it should be ignored.
    """
    light_entity_id = "light.test_bulb"
    config = config_entry.data

    # Initialize light
    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value={})
    mock_store.async_save = AsyncMock()
    with patch("custom_components.smart_circadian_lighting.light.Store", return_value=mock_store):
        light = CircadianLight(mock_hass, light_entity_id, config, config_entry, mock_store)
        light.hass = mock_hass
        light.entity_id = "light.circadian_test"

    night_brightness = _convert_percent_to_255(10) # 25
    light._last_set_brightness = night_brightness
    light._brightness = _convert_percent_to_255(100) # Day brightness 255
    light._is_on = True
    light._is_online = True

    # Mock current time to 18:00:01 (just started evening transition)
    transition_start = datetime.combine(datetime.now().date(), time(18, 0, 1))

    # Mock light state
    current_state = MagicMock(state=STATE_ON, attributes={ATTR_BRIGHTNESS: night_brightness})
    mock_hass.states.get.return_value = current_state

    # Mock refresh to return night brightness
    with patch.object(light, "_get_current_brightness_with_refresh", AsyncMock(return_value=night_brightness)):
        with freeze_time(transition_start):
            await light.async_update_brightness(force_update=True)

    # Check if overridden
    assert not light._is_overridden, "Light should NOT be soft-overridden at transition start if it matches last set brightness"

@pytest.mark.asyncio
async def test_manual_override_still_works(mock_hass, config_entry):
    """Verify that ACTUAL manual overrides still work.

    Scenario:
    1. Light was last set to 100 during evening transition.
    2. User manually sets it to 250 (brightens against transition).
    3. It SHOULD be overridden.
    """
    light_entity_id = "light.test_bulb"
    config = config_entry.data

    # Initialize light
    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value={})
    mock_store.async_save = AsyncMock()
    with patch("custom_components.smart_circadian_lighting.light.Store", return_value=mock_store):
        light = CircadianLight(mock_hass, light_entity_id, config, config_entry, mock_store)
        light.hass = mock_hass
        light.entity_id = "light.circadian_test"

    light._last_set_brightness = 100
    light._brightness = 100
    light._first_update_done = True

    # Mock current time to 18:30 (middle of evening transition)
    mid_transition = datetime.combine(datetime.now().date(), time(18, 30, 0))

    # User manual change: 100 -> 250 (against direction of evening transition)
    old_state = MagicMock(state=STATE_ON, attributes={ATTR_BRIGHTNESS: 100})
    new_state = MagicMock(state=STATE_ON, attributes={ATTR_BRIGHTNESS: 250})
    event = MagicMock(data={"old_state": old_state, "new_state": new_state})

    mock_hass.states.get.return_value = new_state

    with patch.object(light, "_get_current_brightness_with_refresh", AsyncMock(return_value=250)):
        with freeze_time(mid_transition):
            await light._async_entity_state_changed(event)

    assert light._is_overridden, "Actual manual override should still be detected"
