
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.const import STATE_OFF, STATE_ON
from custom_components.smart_circadian_lighting.light import CircadianLight
from custom_components.smart_circadian_lighting.const import DOMAIN

@pytest.mark.asyncio
async def test_transition_completes_without_override_when_off(mock_hass_with_services, mock_config_entry_factory, mock_state_factory):
    hass = mock_hass_with_services
    import asyncio
    hass.loop = asyncio.get_running_loop()
    entry = mock_config_entry_factory()

    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    config = {
        "lights": ["light.test_kasa"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "05:15:00",
        "morning_end_time": "06:00:00",
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
        "color_temp_enabled": False,
        "manual_override_threshold": 5,
        "morning_override_clear_time": "08:00:00",
        "evening_override_clear_time": "02:00:00",
    }

    light = CircadianLight(hass, "light.test_kasa", config, entry, mock_store)
    light.hass = hass
    light.entity_id = f"{DOMAIN}.test_kasa"
    light._is_testing = True # Use MIN_UPDATE_INTERVAL (5s)
    
    hass.data[DOMAIN] = {entry.entry_id: {"circadian_lights": [light], "config": config}}

    mock_entity_entry = MagicMock()
    mock_entity_entry.platform = "kasa_smart_dim"
    
    with patch('homeassistant.helpers.entity_registry.async_get') as mock_er_get, \
         patch('homeassistant.util.dt.now') as mock_now, \
         patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_light_dt_util, \
         patch('custom_components.smart_circadian_lighting.circadian_logic.datetime') as mock_datetime, \
         patch('custom_components.smart_circadian_lighting.state_management.async_call_later') as mock_call_later:

        mock_registry = MagicMock()
        mock_registry.async_get.return_value = mock_entity_entry
        mock_er_get.return_value = mock_registry
        
        # Start just before transition
        current_time = datetime(2026, 1, 1, 5, 14, 55)
        
        # Initial state: Light is OFF, brightness is at Night level (10% = 26/255)
        current_light_brightness = 26

        def get_state(entity_id):
            if entity_id == "light.test_kasa":
                return mock_state_factory("light.test_kasa", STATE_OFF, {ATTR_BRIGHTNESS: current_light_brightness})
            return None

        hass.states.get = MagicMock(side_effect=get_state)
        
        mock_now.return_value = current_time
        mock_light_dt_util.return_value = current_time
        mock_datetime.now.return_value = current_time
        mock_datetime.combine = datetime.combine
        mock_datetime.today = datetime.today

        # First update to establish baseline
        await light._async_calculate_and_apply_brightness(force_update=True)
        assert not light._is_overridden
        assert light._brightness == 26

        # Step through the transition until the end
        # 45 minute transition (05:15 to 06:00)
        # We'll step in 1-minute increments to keep the test reasonably fast but thorough
        for minute in range(15, 61):
            current_time = datetime(2026, 1, 1, 5, minute, 0) if minute < 60 else datetime(2026, 1, 1, 6, 0, 0)
            mock_now.return_value = current_time
            mock_light_dt_util.return_value = current_time
            mock_datetime.now.return_value = current_time
            
            # Update the reported brightness of the light to match the last calculated target
            # (Simulating Kasa reporting its "preloaded" state)
            current_light_brightness = light._brightness
            
            await light._async_calculate_and_apply_brightness()
            
            if minute < 60:
                assert not light._is_overridden, f"Override triggered at {current_time.time()} when light is OFF"
            else:
                # At the end of transition, it should reach Day brightness (255)
                assert not light._is_overridden, "Override triggered at end of transition"
                assert light._brightness == 255, f"Transition did not reach 100% brightness (got {light._brightness})"
