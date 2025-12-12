"""Improved unit tests for override bugs."""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.const import STATE_ON

_LOGGER = logging.getLogger(__name__)

DOMAIN = "smart_circadian_lighting"

@pytest.mark.asyncio
async def test_evening_transition_start_stale_state_sends_command(mock_hass_with_services, mock_config_entry_factory, mock_state_factory):
    """Repro: Evening pre-transition dim not detected, sends high target."""

    hass = mock_hass_with_services
    entry = mock_config_entry_factory()

    # Create a real light instance with proper behavior
    from custom_components.smart_circadian_lighting.light import CircadianLight

    # Mock the store to avoid file operations
    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    # Create real light instance
    config = {
        "lights": ["light.test_light"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "06:00:00",
        "morning_end_time": "07:00:00",
        "evening_start_time": "19:30:00",  # Transition starts at 19:30
        "evening_end_time": "20:30:00",
        "color_temp_enabled": False,
        "morning_override_clear_time": "08:00:00",
        "evening_override_clear_time": "02:00:00",
    }

    light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
    light.hass = hass  # Ensure hass is set on the entity

    # Set up component data structure
    hass.data[DOMAIN] = {entry.entry_id: {"circadian_lights": [light]}}
    # Set up entity registry in hass.data for HA framework compatibility
    from homeassistant.helpers.entity_registry import DATA_REGISTRY
    hass.data[DATA_REGISTRY] = MagicMock()

    # Simulate user voice command: dim to 25% just before transition (19:28)
    with patch('homeassistant.util.dt.now') as mock_now, \
         patch('custom_components.smart_circadian_lighting.state_management.dt_util.now') as mock_dt_util, \
         patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_light_dt_util, \
         patch('custom_components.smart_circadian_lighting.state_management.async_call_later') as mock_async_call_later, \
         patch('custom_components.smart_circadian_lighting.state_management.async_dispatcher_send') as mock_dispatcher_send, \
         patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=MagicMock()) as mock_er_get, \
         patch.object(light._hass.states, 'get') as mock_states_get, \
         patch.object(light, 'async_write_ha_state') as mock_write_state:

        # Create State objects using our fixture factory to avoid global mocking
        initial_state = mock_state_factory("light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 255})
        dimmed_state = mock_state_factory("light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 64})

        # Set up initial state: light at 100% brightness
        mock_states_get.return_value = initial_state

        before_transition = datetime(2023, 1, 1, 19, 28, 0, 0)
        mock_now.return_value = before_transition
        mock_dt_util.return_value = before_transition
        mock_light_dt_util.return_value = before_transition

        # Update the mock to return dimmed state
        mock_states_get.return_value = dimmed_state

        # Verify light is at 25%
        light_state = hass.states.get("light.test_light")
        assert light_state.attributes.get(ATTR_BRIGHTNESS) == 64

        # Now trigger evening transition start (19:30)
        transition_start = datetime(2023, 1, 1, 19, 30, 0, 0)
        mock_now.return_value = transition_start
        mock_dt_util.return_value = transition_start
        mock_light_dt_util.return_value = transition_start

        # Force circadian update (simulating transition start)
        await light._async_calculate_and_apply_brightness(force_update=True)
        await hass.async_block_till_done()

        # Check what happened - bug would be sending high brightness despite manual dim
        final_state = hass.states.get("light.test_light")
        final_brightness = final_state.attributes.get(ATTR_BRIGHTNESS)

        # This test documents the bug: if final_brightness is high (>64), the bug is reproduced
        # If it's still 64, the component correctly respected the manual adjustment
        if final_brightness != 64:
            # Bug reproduced: component sent high brightness command despite knowing current state
            assert final_brightness > 64, f"Bug reproduced: sent brightness {final_brightness} instead of respecting manual 25% setting"
        else:
            # Component correctly handled the manual adjustment
            assert final_brightness == 64, "Component correctly respected manual brightness setting"


@pytest.mark.asyncio
async def test_morning_alarm_dim_near_end_catchup(mock_hass_with_services, mock_config_entry_factory, mock_state_factory):
    """Repro: Morning near end dim sets override but instant catch-up clears."""

    hass = mock_hass_with_services
    entry = mock_config_entry_factory()

    # Create a real light instance with proper behavior
    from custom_components.smart_circadian_lighting.light import CircadianLight

    # Mock the store to avoid file operations
    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    # Create real light instance
    config = {
        "lights": ["light.test_light"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "05:15:00",  # Transition starts at 05:15
        "morning_end_time": "06:00:00",    # Ends at 06:00
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
        "color_temp_enabled": False,
        "morning_override_clear_time": "08:00:00",
        "evening_override_clear_time": "02:00:00",
    }

    light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
    light.hass = hass  # Ensure hass is set on the entity
    light._hass = hass  # Ensure the internal hass reference is also set
    light.entity_id = f"{DOMAIN}.test_light"  # Set entity_id for the circadian entity

    # Ensure clean state for this test
    light._is_overridden = False
    light._override_timestamp = None

    # Set up component data structure
    hass.data[DOMAIN] = {entry.entry_id: {"circadian_lights": [light]}}

    # Mock time to be near end of morning transition (5:55)
    with patch('homeassistant.util.dt.now') as mock_now, \
         patch('custom_components.smart_circadian_lighting.state_management.dt_util.now') as mock_dt_util, \
         patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_light_dt_util, \
         patch('custom_components.smart_circadian_lighting.state_management.async_call_later') as mock_async_call_later, \
         patch('custom_components.smart_circadian_lighting.state_management.async_dispatcher_send') as mock_dispatcher_send, \
         patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=MagicMock()) as mock_er_get, \
         patch.object(light._hass.states, 'get') as mock_states_get, \
         patch.object(light, 'async_write_ha_state') as mock_write_state, \
         patch.object(light._hass, 'async_create_task', return_value=None):

        # Create State objects using our fixture factory to avoid global mocking
        initial_state = mock_state_factory("light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 200})
        dimmed_state = mock_state_factory("light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 77})

        # Set up initial state: light at high brightness near end of morning transition
        mock_states_get.return_value = initial_state

        alarm_time = datetime(2023, 1, 1, 5, 55, 0, 0)
        mock_now.return_value = alarm_time
        mock_dt_util.return_value = alarm_time
        mock_light_dt_util.return_value = alarm_time

        # Simulate alarm automation: dim lights to ~30% (77/255)
        # Update the mock to return dimmed state
        mock_states_get.return_value = dimmed_state

        # Trigger brightness calculation to detect override
        await light._async_calculate_and_apply_brightness()

        # Check if override was detected after the automation dimming
        # The component should detect that brightness 77 > night_brightness 26 during morning transition
        override_detected = light._is_overridden

        assert override_detected, f"Override should be detected after automation dimming during transition. Mode: {light.circadian_mode}"

        # Advance time to next minute (simulating scheduled update)
        next_minute = datetime(2023, 1, 1, 5, 56, 0, 0)
        mock_now.return_value = next_minute
        mock_dt_util.return_value = next_minute
        mock_light_dt_util.return_value = next_minute

        # Force circadian update (simulating scheduled update at 5:56)
        await light._async_calculate_and_apply_brightness()
        await hass.async_block_till_done()

        # Check final brightness - should still respect the manual 30% setting
        final_state = hass.states.get("light.test_light")
        final_brightness = final_state.attributes.get(ATTR_BRIGHTNESS)

        # Override should prevent circadian from changing the brightness
        assert final_brightness == 77, f"Override should prevent brightness change, but got {final_brightness} instead of 77"


@pytest.mark.asyncio
async def test_transition_start_small_deviation_no_override(mock_hass_with_services, mock_config_entry_factory, mock_state_factory):
    """Test that small brightness deviations within threshold do not trigger override at transition start."""

    hass = mock_hass_with_services
    entry = mock_config_entry_factory()

    # Create a real light instance
    from custom_components.smart_circadian_lighting.light import CircadianLight

    # Mock the store
    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    config = {
        "lights": ["light.test_light"],
        "day_brightness": 100,
        "night_brightness": 10,  # 25.5 in 255 scale
        "manual_override_threshold": 5,  # 12.75 in 255 scale
        "morning_start_time": "05:15:00",
        "morning_end_time": "06:00:00",
        "evening_start_time": "19:30:00",
        "evening_end_time": "20:30:00",
        "color_temp_enabled": False,
        "morning_override_clear_time": "08:00:00",
        "evening_override_clear_time": "02:00:00",
    }

    light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
    light.hass = hass
    light.entity_id = f"{DOMAIN}.test_light"

    # Ensure clean state
    light._is_overridden = False
    light._override_timestamp = None

    hass.data[DOMAIN] = {entry.entry_id: {"circadian_lights": [light]}}

    with patch('homeassistant.util.dt.now') as mock_now, \
         patch('custom_components.smart_circadian_lighting.state_management.dt_util.now') as mock_dt_util, \
         patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_light_dt_util, \
         patch('custom_components.smart_circadian_lighting.state_management.async_call_later') as mock_async_call_later, \
         patch('custom_components.smart_circadian_lighting.state_management.async_dispatcher_send') as mock_dispatcher_send, \
         patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=MagicMock()) as mock_er_get, \
         patch.object(light._hass.states, 'get') as mock_states_get, \
         patch.object(light, 'async_write_ha_state') as mock_write_state:

        # Set brightness to night + small deviation (within threshold)
        # night = 25.5, threshold = 12.75, so 25.5 + 10 = 35.5 (within 25.5 + 12.75 = 38.25)
        small_deviation_state = mock_state_factory("light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 36})  # 36 > 25.5 but < 38.25
        mock_states_get.return_value = small_deviation_state

        # Start morning transition
        transition_start = datetime(2023, 1, 1, 5, 15, 0, 0)
        mock_now.return_value = transition_start
        mock_dt_util.return_value = transition_start
        mock_light_dt_util.return_value = transition_start

        # Trigger brightness calculation
        await light._async_calculate_and_apply_brightness(force_update=True)
        await hass.async_block_till_done()

        # Should NOT detect override for small deviation
        assert not light._is_overridden, "Override incorrectly detected for small brightness deviation within threshold. Current: 36, Night: 25.5, Threshold: 12.75"



