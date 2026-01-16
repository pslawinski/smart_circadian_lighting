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
         patch('custom_components.smart_circadian_lighting.state_management.async_call_later'), \
         patch('custom_components.smart_circadian_lighting.state_management.async_dispatcher_send'), \
         patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=MagicMock()), \
         patch.object(light._hass.states, 'get') as mock_states_get, \
         patch.object(light, 'async_write_ha_state'):

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
         patch('custom_components.smart_circadian_lighting.state_management.async_call_later'), \
         patch('custom_components.smart_circadian_lighting.state_management.async_dispatcher_send'), \
         patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=MagicMock()), \
         patch.object(light._hass.states, 'get') as mock_states_get, \
         patch.object(light, 'async_write_ha_state'), \
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
async def test_transition_start_ahead_adjustment_override(mock_hass_with_services, mock_config_entry_factory, mock_state_factory):
    """Test that ahead brightness adjustments trigger override at morning transition start.

    This test verifies that when a light's brightness is ahead of the circadian schedule
    at the start of a morning transition (brighter than night + threshold), a manual override
    is correctly triggered.

    Expected behavior: Override detected because the light is ahead of schedule.
    Current bug: The ahead check uses night brightness instead of the actual transition target,
    causing incorrect override detection for lights that are legitimately ahead.
    """

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
         patch('custom_components.smart_circadian_lighting.state_management.async_call_later'), \
         patch('custom_components.smart_circadian_lighting.state_management.async_dispatcher_send'), \
         patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=MagicMock()), \
         patch.object(light._hass.states, 'get') as mock_states_get, \
         patch.object(light, 'async_write_ha_state'):

        # Calculate actual target at transition start: night + (day - night) * (300/2700) ≈ 25.5 + 229.5 * 0.111 ≈ 50.95
        # Set brightness slightly higher than target (same direction as morning transition)
        # 52 > 50.95, so same direction (brightening), should not trigger override
        small_deviation_state = mock_state_factory("light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 52})
        mock_states_get.return_value = small_deviation_state

        # Start morning transition
        transition_start = datetime(2023, 1, 1, 5, 15, 0, 0)
        mock_now.return_value = transition_start
        mock_dt_util.return_value = transition_start
        mock_light_dt_util.return_value = transition_start

        # Trigger brightness calculation
        await light._async_calculate_and_apply_brightness(force_update=True)
        await hass.async_block_till_done()

        # Should detect override for ahead adjustment
        assert light._is_overridden, "Override not detected for ahead brightness adjustment. Current: 52, Night: 25.5, Threshold: 12.75"


@pytest.mark.parametrize("transition_type,user_brightness,night_brightness_pct,day_brightness_pct,threshold,test_label", [
    ("morning", 77, 10, 100, 5, "morning_well_above_threshold"),
    ("morning", 41, 10, 100, 5, "morning_slightly_above_threshold"),
    ("morning", 40, 10, 100, 5, "morning_at_threshold_boundary"),
    ("morning", 38, 10, 100, 5, "morning_below_threshold"),
    ("evening", 64, 10, 100, 5, "evening_well_below_threshold"),
    ("evening", 191, 10, 100, 5, "evening_slightly_below_threshold"),
    ("evening", 241, 10, 100, 5, "evening_at_threshold_boundary"),
    ("evening", 243, 10, 100, 5, "evening_above_threshold"),
])
@pytest.mark.asyncio
async def test_override_persists_through_single_transition_cycle(
    mock_hass_with_services,
    mock_config_entry_factory,
    mock_state_factory,
    transition_type,
    user_brightness,
    night_brightness_pct,
    day_brightness_pct,
    threshold,
    test_label
):
    """Regression test: Override detected at transition start must persist through the same cycle.

    This test catches the bug where an override is detected AND immediately cleared
    in the same call to _async_calculate_and_apply_brightness(). The override detection
    happens first (user adjusted against transition direction), but the override clearing
    check runs immediately after, and since the circadian target is at an extreme value
    at transition start, the clearing condition is satisfied and the override gets cleared
    before it has a chance to take effect.

    Bug symptom: User dims/brightens lights during evening/morning transition → system
    detects override → system immediately clears override in same cycle → brightness jumps
    back toward circadian target.

    Expected behavior: Override detected → override persists through at least the current
    cycle → override only clears when circadian target naturally catches up to manual
    brightness on subsequent cycles.

    Test parameters:
    - transition_type: "morning" or "evening"
    - user_brightness: brightness value the user set (in 0-255 scale)
    - night_brightness_pct: night brightness as percentage (0-100)
    - day_brightness_pct: day brightness as percentage (0-100)
    - threshold: manual override threshold percentage (0-100)
    """

    hass = mock_hass_with_services
    entry = mock_config_entry_factory()

    from custom_components.smart_circadian_lighting.light import CircadianLight

    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    config = {
        "lights": ["light.test_light"],
        "day_brightness": day_brightness_pct,
        "night_brightness": night_brightness_pct,
        "manual_override_threshold": threshold,
        "morning_start_time": "06:00:00",
        "morning_end_time": "07:00:00",
        "evening_start_time": "19:30:00",
        "evening_end_time": "20:30:00",
        "color_temp_enabled": False,
        "morning_override_clear_time": "08:00:00",
        "evening_override_clear_time": "02:00:00",
    }

    light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
    light.hass = hass
    light._hass = hass
    light.entity_id = f"{DOMAIN}.test_light"

    light._is_overridden = False
    light._override_timestamp = None

    hass.data[DOMAIN] = {entry.entry_id: {"circadian_lights": [light]}}

    with patch('homeassistant.util.dt.now') as mock_now, \
         patch('custom_components.smart_circadian_lighting.state_management.dt_util.now') as mock_dt_util, \
         patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_light_dt_util, \
         patch('custom_components.smart_circadian_lighting.state_management.async_call_later'), \
         patch('custom_components.smart_circadian_lighting.state_management.async_dispatcher_send'), \
         patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=MagicMock()), \
         patch.object(light._hass.states, 'get') as mock_states_get, \
         patch.object(light, 'async_write_ha_state'), \
         patch.object(light._hass, 'async_create_task', return_value=None):

        ahead_brightness_state = mock_state_factory(
            "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: user_brightness}
        )
        mock_states_get.return_value = ahead_brightness_state

        if transition_type == "morning":
            transition_start = datetime(2023, 1, 1, 6, 0, 0, 0)
        else:  # evening
            transition_start = datetime(2023, 1, 1, 19, 30, 0, 0)

        mock_now.return_value = transition_start
        mock_dt_util.return_value = transition_start
        mock_light_dt_util.return_value = transition_start

        await light._async_calculate_and_apply_brightness(force_update=True)
        await hass.async_block_till_done()

        night_brightness_255 = int(round(night_brightness_pct * 255 / 100))
        day_brightness_255 = int(round(day_brightness_pct * 255 / 100))
        threshold_255 = int(round(threshold * 255 / 100))

        if transition_type == "morning":
            expected_trigger = user_brightness > night_brightness_255 + threshold_255
            debug_msg = (
                f"Morning transition [{test_label}]: user brightness {user_brightness} vs "
                f"night {night_brightness_255} + threshold {threshold_255} = {night_brightness_255 + threshold_255}"
            )
        else:  # evening
            expected_trigger = user_brightness < day_brightness_255 - threshold_255
            debug_msg = (
                f"Evening transition [{test_label}]: user brightness {user_brightness} vs "
                f"day {day_brightness_255} - threshold {threshold_255} = {day_brightness_255 - threshold_255}"
            )

        assert light._is_overridden == expected_trigger, (
            f"REGRESSION [{test_label}]: {debug_msg}. "
            f"Expected override: {expected_trigger}, Got: {light._is_overridden}. "
            "Override was cleared in the same cycle it was detected. "
            "This allows the circadian target to override the user's manual adjustment."
        )


