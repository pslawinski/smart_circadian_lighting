"""Improved unit tests for override bugs."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import State, HomeAssistant
from unittest.mock import patch, MagicMock, AsyncMock

from pytest_homeassistant_custom_component.common import MockConfigEntry

DOMAIN = "smart_circadian_lighting"
from homeassistant.components.light import DOMAIN as LIGHT_DOMAIN, ATTR_BRIGHTNESS
from homeassistant.const import STATE_ON

@pytest.mark.asyncio
async def test_evening_transition_start_stale_state_sends_command():
    """Repro: Evening pre-transition dim not detected, sends high target."""
    from unittest.mock import MagicMock, AsyncMock

    hass = MagicMock()
    hass.data = {}
    hass.states = MagicMock()
    hass.states.async_set = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()

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

    # Mock the component data
    light = MagicMock()
    light.async_update_light = AsyncMock()
    light.circadian_mode = MagicMock(return_value="evening_transition")
    light._day_brightness_255 = 255
    light._async_calculate_and_apply_brightness = AsyncMock()
    hass.data = {DOMAIN: {entry.entry_id: {"circadian_lights": [light]}}}
    await entry.async_update_entry(hass, {
        "lights": ["light.physical"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "06:00:00",
        "morning_end_time": "07:00:00",
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
    })

    light = hass.data[DOMAIN][entry.entry_id]["circadian_lights"][0]

    # Mock physical light
    hass.states.async_set("light.physical", STATE_ON, {"brightness": 255})

    # Manual dim to 25% (~64)
    await hass.services.async_call(LIGHT_DOMAIN, "turn_on", {
        "entity_id": "light.physical",
        ATTR_BRIGHTNESS: 64
    })
    hass.states.async_set("light.physical", STATE_ON, {"brightness": 64})

    # Mock stale state for special check, fresh after refresh
    hass.states.get = MagicMock(side_effect=[
        State("light.physical", STATE_ON, {"brightness": 255}),  # Stale high in special check
        State("light.physical", STATE_ON, {"brightness": 64}),   # Fresh low after refresh
    ])

    # Mock evening transition
    light.circadian_mode = MagicMock(return_value="evening_transition")
    # Mock day brightness
    light._day_brightness_255 = 255
    await light._async_calculate_and_apply_brightness(force_update=True)

    # FAIL: Should skip (target > current), but if stale refresh, sends
    # light._hass.services.has_call(LIGHT_DOMAIN, "turn_on", service_data={"brightness": 255})
    # Check if update_light called with high brightness
    light.async_update_light.assert_called()
    service_data = light.async_update_light.call_args.kwargs.get('service_data', {})
    assert service_data.get(ATTR_BRIGHTNESS) == 255  # Bug repro: sent high despite low current


@pytest.mark.asyncio
async def test_morning_alarm_dim_near_end_catchup():
    """Repro: Morning near end dim sets override but instant catch-up clears."""
    from unittest.mock import MagicMock, AsyncMock

    hass = MagicMock()
    hass.data = {}
    hass.states = MagicMock()
    hass.states.async_set = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()

    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="mock_repro",
        data={
            "lights": ["light.physical"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "05:00:00",
            "morning_end_time": "06:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
        },
    )
    entry.async_update_entry = AsyncMock()

    # Mock the component data
    light = MagicMock()
    light.circadian_mode = MagicMock(return_value="morning_transition")
    light._brightness = 243
    light._async_calculate_and_apply_brightness = AsyncMock()
    light._async_entity_state_changed = AsyncMock()
    light.extra_state_attributes = {"is_overridden": False}
    hass.data = {DOMAIN: {entry.entry_id: {"circadian_lights": [light]}}}
    await entry.async_update_entry(hass, {
        "lights": ["light.physical"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "05:00:00",
        "morning_end_time": "06:00:00",
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
    })

    light = hass.data[DOMAIN][entry.entry_id]["circadian_lights"][0]

    # Near end (5:55, end 6:00), target high ~95%
    hass.states.async_set("light.physical", STATE_ON, {"brightness": 200})

    # Alarm dim to 30% (~77)
    await hass.services.async_call(LIGHT_DOMAIN, "turn_on", {
        "entity_id": "light.physical",
        ATTR_BRIGHTNESS: 77
    })
    hass.states.async_set("light.physical", STATE_ON, {"brightness": 77})

    # State change triggers override detect
    old_state = State("light.physical", STATE_ON, {"brightness": 200})
    new_state = State("light.physical", STATE_ON, {"brightness": 77})
    event = MagicMock(data={"old_state": old_state, "new_state": new_state})

    await light._async_entity_state_changed(event)

    assert light.extra_state_attributes["is_overridden"]  # Override set

    # Next update: catch-up clears (target high >77)
    light.circadian_mode = MagicMock(return_value="morning_transition")
    light._brightness = 243  # High target near end
    await light._async_calculate_and_apply_brightness()

    assert not light.extra_state_attributes["is_overridden"]  # Cleared
    # Update sent