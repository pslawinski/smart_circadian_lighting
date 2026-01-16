
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import State
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_circadian_lighting import circadian_logic, state_management
from custom_components.smart_circadian_lighting.const import DOMAIN
from custom_components.smart_circadian_lighting.light import CircadianLight


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock()
    hass.states.async_set = MagicMock()
    hass.data = {}
    return hass

@pytest.mark.asyncio
async def test_zwave_soft_override_behavior(mock_hass):
    """Test that Z-Wave in-direction overrides trigger soft overrides and pin parameter 18."""
    # Setup Z-Wave light in entity registry
    mock_ent_reg = MagicMock()
    mock_entity_entry = MagicMock()
    mock_entity_entry.platform = "zwave_js"
    mock_entity_entry.device_id = "test_device_id"
    mock_ent_reg.async_get.return_value = mock_entity_entry

    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
    config = {
        "lights": ["light.test_zwave"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "06:00:00",
        "morning_end_time": "07:00:00",
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
        "manual_override_threshold": 5,
        "morning_override_clear_time": "08:00:00",
        "evening_override_clear_time": "02:00:00",
    }

    mock_hass.data[DOMAIN] = {entry.entry_id: {"manual_overrides_enabled": True}}

    with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_ent_reg):
        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)
        light.hass = mock_hass
        light.entity_id = "light.circadian_test"
        light._first_update_done = True

        # Mock async_update_brightness to avoid RuntimeWarning when called via async_create_task
        light.async_update_brightness = MagicMock()

        # Set a circadian setpoint for evening transition (e.g., 200 / ~78%)
        light._brightness = 200

        # 1. Trigger in-direction override during evening transition (dimming)
        now = dt_util.as_utc(datetime(2026, 1, 1, 20, 30, 0))

        # Simulate manual adjustment from 200 (circadian) to 64 (25%)
        with patch("custom_components.smart_circadian_lighting.state_management.async_save_override_state", AsyncMock()):
            await state_management._check_for_manual_override(
                light,
                old_brightness=190,
                new_brightness=64,
                old_color_temp=None,
                new_color_temp=None,
                now=now,
                was_online=True
            )

        assert light._is_overridden is True
        assert light._is_soft_override is True
        assert light._soft_override_value == 64
        assert light._brightness == 64

        # 2. Verify Z-Wave parameter 18 is pinned to manual value (64)
        mock_hass.services.async_call.reset_mock()
        with patch("custom_components.smart_circadian_lighting.light.dt_util.now", return_value=now):
            await light._async_calculate_and_apply_brightness(force_update=True)

        expected_zwave_value = int(64 * 99 / 255) # 24
        mock_hass.services.async_call.assert_any_call(
            "zwave_js",
            "set_config_parameter",
            {
                "device_id": "test_device_id",
                "parameter": 18,
                "value": expected_zwave_value,
            }
        )

        # 3. Verify turning light off does NOT clear soft override
        event = MagicMock()
        event.data = {
            "old_state": State("light.test_zwave", STATE_ON, {"brightness": 64}),
            "new_state": State("light.test_zwave", STATE_OFF, {})
        }

        with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_ent_reg), \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now", return_value=now):
            await light._async_entity_state_changed(event)

        assert light._is_overridden is True
        assert light._is_soft_override is True

        # 4. Verify turning light ON skips immediate circadian update if soft override active
        mock_hass.states.get.return_value = State("light.test_zwave", STATE_OFF, {})
        new_state = State("light.test_zwave", STATE_ON, {ATTR_BRIGHTNESS: 64})
        event = MagicMock()
        event.data = {
            "old_state": State("light.test_zwave", STATE_OFF, {}),
            "new_state": new_state
        }

        with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_ent_reg):
            with patch.object(light, "_async_calculate_and_apply_brightness", AsyncMock()) as mock_update:
                await light._async_entity_state_changed(event)
                # Should NOT call update because it's a Z-Wave soft override
                mock_update.assert_not_called()

@pytest.mark.asyncio
async def test_zwave_hard_override_does_not_pin_parameter_18(mock_hass):
    """Verify that against-direction (hard) overrides do NOT pin parameter 18."""
    # Setup Z-Wave light
    mock_ent_reg = MagicMock()
    mock_entity_entry = MagicMock()
    mock_entity_entry.platform = "zwave_js"
    mock_entity_entry.device_id = "test_device_id"
    mock_ent_reg.async_get.return_value = mock_entity_entry

    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
    config = {
        "lights": ["light.test_zwave"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "06:00:00",
        "morning_end_time": "07:00:00",
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
        "manual_override_threshold": 5,
    }

    mock_hass.data[DOMAIN] = {entry.entry_id: {"manual_overrides_enabled": True}}

    with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_ent_reg):
        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)
        light.hass = mock_hass
        light.entity_id = "light.circadian_test"
        light._first_update_done = True

        # Mock async_update_brightness to avoid RuntimeWarning when called via async_create_task
        light.async_update_brightness = MagicMock()
        light._brightness = 100

        now = dt_util.as_utc(datetime(2026, 1, 1, 20, 30, 0))

        # Trigger AGAINST-direction override (brightening during evening)
        with patch("custom_components.smart_circadian_lighting.state_management.async_save_override_state", AsyncMock()):
            await state_management._check_for_manual_override(
                light,
                old_brightness=90,
                new_brightness=150,
                old_color_temp=None,
                new_color_temp=None,
                now=now,
                was_online=True
            )

        assert light._is_overridden is True
        assert light._is_soft_override is False

        # Verify parameter 18 sync is SKIPPED during a hard override
        mock_hass.services.async_call.reset_mock()
        with patch("custom_components.smart_circadian_lighting.light.dt_util.now", return_value=now):
            # Circadian target at 20:30 is approx 55 (midway between 100 and 10)
            # 150 > 55, so it's a hard override (brightened in evening)
            # but it shouldn't trigger catch-up yet because evening catch-up is when circadian <= manual
            # Wait, 55 <= 150 IS True.

            # Ah, catch-up for evening is: circadian <= manual
            # If I want to avoid catch-up in evening, I need circadian > manual.
            # But hard override in evening is manual > circadian.
            # So hard overrides in evening will ALWAYS be "caught up" by definition if we use this logic?

            # Let's re-read the catch-up logic:
            # if mode == "evening_transition" and target_brightness_255 <= current_brightness:
            #    should_clear_override = True

            # Evening transition is decreasing brightness.
            # If user brightens (hard override), current_brightness > target_brightness_255.
            # So it will always be caught up immediately.

            # Wait, is that right?
            # If it's 8:30pm and the light should be at 50%, but I set it to 100% (hard override).
            # The system says "well, 50% <= 100%, so I've caught up".
            # This means hard overrides in the evening are cleared immediately?

            # Let's check morning.
            # if mode == "morning_transition" and target_brightness_255 >= current_brightness:
            #    should_clear_override = True
            # Morning transition is increasing.
            # If user dims (hard override), current_brightness < target_brightness_255.
            # So target_brightness_255 >= current_brightness will be True immediately.

            # This means hard overrides are ALWAYS caught up immediately by this logic!
            # This logic only makes sense for SOFT overrides where we are "ahead" of the transition.

            # For hard overrides, we are "behind" (dimmed in morning) or "ahead" in a way that the transition will NEVER catch up (brightened in evening).

            # If I'm dimmed in morning (hard), the transition will keep increasing, so I'll stay "behind".
            # If I'm brightened in evening (hard), the transition will keep decreasing, so I'll stay "ahead".

            # So the catch-up logic should probably ONLY apply to soft overrides?
            # Or at least, it should be aware of the direction.

            # Morning: soft is brightening (ahead). Transition catches up when it reaches the brightened level.
            # Morning: hard is dimming (behind). Transition will NEVER catch up because it's moving away from the dimmed level.

            # Evening: soft is dimming (ahead). Transition catches up when it reaches the dimmed level.
            # Evening: hard is brightening (behind). Transition will NEVER catch up because it's moving away from the brightened level.

            # So the catch-up logic:
            # Morning: clear if target >= current AND it's a soft override?
            # Actually, if it's a hard override (dimmed in morning), and the target is already > current, it should stay overridden.

            # Let's look at the logic again:
            # if mode == "morning_transition" and target_brightness_255 >= current_brightness:
            #    should_clear_override = True

            # If I dim to 50 in morning while target is 100. target (100) >= current (50) is True.
            # So it clears immediately.

            # This explains why the test is failing. Hard overrides are being cleared immediately by the catch-up logic.

            # I should probably restrict catch-up clearing to soft/in-direction overrides.
            # For hard overrides, the clearing event is usually turning off/on (for Z-Wave) or scheduled clear times.

            await light._async_calculate_and_apply_brightness(force_update=True)

        # Ensure it was NOT called for parameter 18 at all
        for call in mock_hass.services.async_call.call_args_list:
            if call[0][0] == "zwave_js" and call[0][1] == "set_config_parameter":
                if call[0][2].get("parameter") == 18:
                    pytest.fail("Z-Wave parameter 18 was synced during a hard override, but it should have been skipped.")

@pytest.mark.asyncio
async def test_soft_override_persistence_restore(mock_hass):
    """Test that soft override state is correctly restored from store."""
    mock_store = MagicMock()
    mock_store.async_save = AsyncMock()
    # Mock saved state with soft override
    saved_state = {
        "light.test_zwave": {
            "is_overridden": True,
            "is_soft_override": True,
            "soft_override_value": 42,
            "timestamp": datetime.now().isoformat()
        }
    }
    mock_store.async_load = AsyncMock(return_value=saved_state)

    entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
    config = {
        "lights": ["light.test_zwave"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_override_clear_time": "08:00:00",
        "evening_override_clear_time": "02:00:00",
    }

    light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)

    await state_management.async_load_override_state(light)

    assert light._is_overridden is True
    assert light._is_soft_override is True
    assert light._soft_override_value == 42

@pytest.mark.asyncio
async def test_zwave_manual_adjustment_before_transition(mock_hass):
    """Test what happens when a Z-Wave light is adjusted BEFORE a transition starts."""
    # Setup Z-Wave light
    mock_ent_reg = MagicMock()
    mock_entity_entry = MagicMock()
    mock_entity_entry.platform = "zwave_js"
    mock_entity_entry.device_id = "test_device_id"
    mock_ent_reg.async_get.return_value = mock_entity_entry

    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
    config = {
        "lights": ["light.test_zwave"],
        "day_brightness": 100, # 255
        "night_brightness": 10, # 26
        "morning_start_time": "06:00:00",
        "morning_end_time": "07:00:00",
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
        "manual_override_threshold": 5,
        "morning_override_clear_time": "08:00:00",
        "evening_override_clear_time": "02:00:00",
    }

    mock_hass.data[DOMAIN] = {entry.entry_id: {"manual_overrides_enabled": True}}

    with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_ent_reg):
        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)
        light.hass = mock_hass
        light.entity_id = "light.circadian_test"
        light._first_update_done = True

        # Mock async_update_brightness to avoid RuntimeWarning when called via async_create_task
        light.async_update_brightness = MagicMock()

        # Mock brightness refresh to return current value
        light._get_current_brightness_with_refresh = AsyncMock(side_effect=lambda: light._last_reported_brightness)

        # 1. Adjust light at 5:00 PM (Before evening transition)
        now_pre = dt_util.as_utc(datetime(2026, 1, 1, 17, 0, 0))
        with patch("custom_components.smart_circadian_lighting.state_management.dt_util.now", return_value=now_pre):
            light._brightness = circadian_logic.calculate_brightness_for_time(
                now_pre, {}, config, light._day_brightness_255, light._night_brightness_255
            )

            light._manual_override_threshold = 5

            with patch("custom_components.smart_circadian_lighting.state_management.async_save_override_state", AsyncMock()):
                event = MagicMock()
                event.data = {
                    "old_state": State("light.test_zwave", STATE_ON, {"brightness": 255}),
                    "new_state": State("light.test_zwave", STATE_ON, {"brightness": 150})
                }
                await light._async_entity_state_changed(event)

        # Override is NOT detected yet because we are not in transition
        assert light._is_overridden is False

        # 2. Transition starts at 8:30 PM.
        # Detection should happen when we run the update.
        now_transition = dt_util.as_utc(datetime(2026, 1, 1, 20, 30, 0))
        mock_hass.services.async_call.reset_mock()

        # Ensure the physical entity is reported as 150
        mock_hass.states.get.return_value = State("light.test_zwave", STATE_ON, {"brightness": 150})
        light._last_reported_brightness = 150

        with patch("custom_components.smart_circadian_lighting.light.dt_util.now", return_value=now_transition):
            await light._async_calculate_and_apply_brightness(force_update=True)

        assert light._is_overridden is True
        assert light._is_soft_override is True
        assert light._soft_override_value == 150

        expected_zwave_value = int(150 * 99 / 255) # 58
        mock_hass.services.async_call.assert_any_call(
            "zwave_js",
            "set_config_parameter",
            {
                "device_id": "test_device_id",
                "parameter": 18,
                "value": expected_zwave_value,
            }
        )

@pytest.mark.asyncio
async def test_zwave_soft_override_persistence_through_toggle(mock_hass):
    """Verify that Parameter 18 is updated even when the light is OFF, ensuring it comes back on at the overridden value."""
    # Setup Z-Wave light
    mock_ent_reg = MagicMock()
    mock_entity_entry = MagicMock()
    mock_entity_entry.platform = "zwave_js"
    mock_entity_entry.device_id = "test_device_id"
    mock_ent_reg.async_get.return_value = mock_entity_entry

    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
    config = {
        "lights": ["light.test_zwave"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "06:00:00",
        "morning_end_time": "07:00:00",
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
        "manual_override_threshold": 5,
        "morning_override_clear_time": "08:00:00",
        "evening_override_clear_time": "02:00:00",
    }

    mock_hass.data[DOMAIN] = {entry.entry_id: {"manual_overrides_enabled": True}}

    with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_ent_reg):
        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)
        light.hass = mock_hass
        light.entity_id = "light.circadian_test"
        light._first_update_done = True

        # Mock async_update_brightness to avoid RuntimeWarning when called via async_create_task
        light.async_update_brightness = MagicMock()

        # 1. Trigger soft override during evening transition (dimming to 64/25%)
        now = dt_util.as_utc(datetime(2026, 1, 1, 20, 30, 0))
        light._brightness = 200 # Circadian setpoint

        with patch("custom_components.smart_circadian_lighting.state_management.async_save_override_state", AsyncMock()):
            await state_management._check_for_manual_override(
                light, old_brightness=200, new_brightness=64, old_color_temp=None, new_color_temp=None, now=now, was_online=True
            )
            await asyncio.sleep(0) # Let immediate sync task run to avoid RuntimeWarning

        assert light._is_soft_override is True
        assert light._soft_override_value == 64

        # Verify param 18 is set
        mock_hass.services.async_call.reset_mock()
        with patch("custom_components.smart_circadian_lighting.light.dt_util.now", return_value=now):
            await light._async_calculate_and_apply_brightness(force_update=True)

        expected_zwave_value = int(64 * 99 / 255) # 24
        mock_hass.services.async_call.assert_any_call(
            "zwave_js", "set_config_parameter", {"device_id": "test_device_id", "parameter": 18, "value": expected_zwave_value}
        )

        # 2. Turn light OFF
        mock_hass.states.get.return_value = State("light.test_zwave", STATE_OFF, {})
        event = MagicMock()
        event.data = {
            "old_state": State("light.test_zwave", STATE_ON, {"brightness": 64}),
            "new_state": State("light.test_zwave", STATE_OFF, {})
        }
        with patch("custom_components.smart_circadian_lighting.state_management.dt_util.now", return_value=now):
            await light._async_entity_state_changed(event)

        # 3. Simulate time passing within transition while OFF
        now_later = now + timedelta(minutes=15)
        mock_hass.services.async_call.reset_mock()
        with patch("custom_components.smart_circadian_lighting.light.dt_util.now", return_value=now_later):
            # This would normally be called by the timer
            await light._async_calculate_and_apply_brightness(force_update=True)

        # Verify Parameter 18 is STILL set to the pinned manual value (64) even though light is OFF
        mock_hass.services.async_call.assert_any_call(
            "zwave_js", "set_config_parameter", {"device_id": "test_device_id", "parameter": 18, "value": expected_zwave_value}
        )

        # 4. Turn light ON
        # It should come back at 64 because Parameter 18 was kept pinned.
        mock_hass.states.get.return_value = State("light.test_zwave", STATE_ON, {"brightness": 64})
        event = MagicMock()
        event.data = {
            "old_state": State("light.test_zwave", STATE_OFF, {}),
            "new_state": State("light.test_zwave", STATE_ON, {"brightness": 64})
        }
        with patch.object(light, "_async_calculate_and_apply_brightness", AsyncMock()) as mock_update, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now", return_value=now_later):
            await light._async_entity_state_changed(event)
            # Should skip update because it's a Z-Wave soft override
            mock_update.assert_not_called()

        assert light._brightness == 64

@pytest.mark.asyncio
async def test_zwave_in_direction_adjustment_during_transition(mock_hass):
    """Scenario 2: User adjusts light down to 25% DURING the evening transition.
    Verify soft override is detected, target setpoint is changed to 25%,
    and parameter 18 is updated.
    """
    # Setup Z-Wave light
    mock_ent_reg = MagicMock()
    mock_entity_entry = MagicMock()
    mock_entity_entry.platform = "zwave_js"
    mock_entity_entry.device_id = "test_device_id"
    mock_ent_reg.async_get.return_value = mock_entity_entry

    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
    config = {
        "lights": ["light.test_zwave"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "06:00:00",
        "morning_end_time": "07:00:00",
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
        "manual_override_threshold": 5,
    }

    mock_hass.data[DOMAIN] = {entry.entry_id: {"manual_overrides_enabled": True}}

    with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_ent_reg):
        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)
        light.hass = mock_hass
        light.entity_id = "light.circadian_test"
        light._first_update_done = True

        # Mock async_update_brightness to avoid RuntimeWarning when called via async_create_task
        light.async_update_brightness = MagicMock()

        # Set initial circadian state (evening transition started)
        now = dt_util.as_utc(datetime(2026, 1, 1, 20, 15, 0)) # 15 mins into transition
        light._brightness = 200 # Current circadian setpoint
        light._last_set_brightness = 200

        # Mock brightness refresh to return what we set
        light._get_current_brightness_with_refresh = AsyncMock(return_value=64)

        # User adjusts light down to 25% (64)
        with patch("custom_components.smart_circadian_lighting.state_management.async_save_override_state", AsyncMock()), \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now", return_value=now):

            event = MagicMock()
            event.data = {
                "old_state": State("light.test_zwave", STATE_ON, {"brightness": 200}),
                "new_state": State("light.test_zwave", STATE_ON, {"brightness": 64})
            }

            # This should trigger in-direction soft override
            await light._async_entity_state_changed(event)

        assert light._is_overridden is True
        assert light._is_soft_override is True
        assert light._soft_override_value == 64
        assert light._brightness == 64

        # Verify parameter 18 update was triggered
        expected_zwave_value = int(64 * 99 / 255) # 24

        # The update might be async via async_create_task, so we might need to wait or check calls
        # In state_management.py: light.hass.async_create_task(light.async_update_brightness(force_update=True))
        # Since we are in a test and using mock_hass, we can check if async_create_task was called
        # OR if we manually trigger the update logic.

        with patch("custom_components.smart_circadian_lighting.light.dt_util.now", return_value=now):
            await light._async_calculate_and_apply_brightness(force_update=True)

        mock_hass.services.async_call.assert_any_call(
            "zwave_js",
            "set_config_parameter",
            {
                "device_id": "test_device_id",
                "parameter": 18,
                "value": expected_zwave_value,
            }
        )
