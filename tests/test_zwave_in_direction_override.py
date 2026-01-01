
import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from homeassistant.const import STATE_ON, STATE_OFF, STATE_UNAVAILABLE
from homeassistant.core import State
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_circadian_lighting.light import CircadianLight
from custom_components.smart_circadian_lighting.const import DOMAIN, LIGHT_UPDATE_TIMEOUT
from custom_components.smart_circadian_lighting import circadian_logic, state_management

@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock()
    hass.states.async_set = AsyncMock()
    hass.data = {}
    return hass

@pytest.mark.asyncio
async def test_zwave_in_direction_override_persistence(mock_hass):
    """Test that Z-Wave in-direction overrides persist through toggles and update parameter 18."""
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
        
        # Set a circadian setpoint for evening transition (e.g., 200 / ~78%)
        light._brightness = 200
        
        # 1. Trigger in-direction override during evening transition (dimming)
        # Evening transition is 20:00 - 21:00.
        now = datetime(2026, 1, 1, 20, 30, 0)
        
        # Simulate manual adjustment from 200 (circadian) to 64 (25%)
        # Set old_brightness different enough from light._brightness to avoid optimistic update filter
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
        assert light._is_in_direction_override is True
        assert light._brightness == 64
        
        # 2. Verify Z-Wave parameter 18 is set to manual value (64)
        mock_hass.services.async_call.reset_mock()
        with patch("custom_components.smart_circadian_lighting.light.dt_util.now", return_value=now):
            # This will calculate circadian target (~141) but should use manual value (64) for param 18
            await light._async_calculate_and_apply_brightness(force_update=True)
        
        # expected_zwave_value = int(64 * 99 / 255) = 24
        expected_zwave_value = 24
        mock_hass.services.async_call.assert_any_call(
            "zwave_js",
            "set_config_parameter",
            {
                "device_id": "test_device_id",
                "parameter": 18,
                "value": expected_zwave_value,
            }
        )
        
        # 3. Verify turning light off does NOT clear override for Z-Wave in-direction
        # Simulate state change event to OFF
        event = MagicMock()
        event.data = {
            "old_state": State("light.test_zwave", STATE_ON, {"brightness": 64}),
            "new_state": State("light.test_zwave", STATE_OFF, {})
        }
        
        with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_ent_reg):
            await light._async_entity_state_changed(event)
        
        assert light._is_overridden is True
        assert light._is_in_direction_override is True
        
        # 4. Verify parameter 18 is still updated with manual value while light is off
        mock_hass.services.async_call.reset_mock()
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

@pytest.mark.asyncio
async def test_zwave_against_direction_override_clears_on_off(mock_hass):
    """Verify that against-direction overrides STILL clear on off (standard behavior)."""
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
        light._brightness = 100 # Evening transition setpoint
        
        now = datetime(2026, 1, 1, 20, 30, 0)
        
        # Trigger AGAINST-direction override (brightening during evening)
        with patch("custom_components.smart_circadian_lighting.state_management.async_save_override_state", AsyncMock()):
            await state_management._check_for_manual_override(
                light, 
                old_brightness=90, 
                new_brightness=250, # Brightened significantly
                old_color_temp=None, 
                new_color_temp=None, 
                now=now, 
                was_online=True
            )
        
        assert light._is_overridden is True
        assert light._is_in_direction_override is False
        
        # Verify turning light off DOES clear override for against-direction
        event = MagicMock()
        event.data = {
            "old_state": State("light.test_zwave", STATE_ON, {"brightness": 250}),
            "new_state": State("light.test_zwave", STATE_OFF, {})
        }
        
        with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_ent_reg):
            await light._async_entity_state_changed(event)
        
        assert light._is_overridden is False

@pytest.mark.asyncio
async def test_non_zwave_does_not_trigger_in_direction_override(mock_hass):
    """Verify that non-Z-Wave lights do not trigger in-direction overrides."""
    # Setup Kasa light (non-Z-Wave)
    mock_ent_reg = MagicMock()
    mock_entity_entry = MagicMock()
    mock_entity_entry.platform = "kasa_smart_dim"
    mock_ent_reg.async_get.return_value = mock_entity_entry

    mock_store = MagicMock()
    mock_store.async_load = AsyncMock(return_value=None)
    mock_store.async_save = AsyncMock()

    entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
    config = {
        "lights": ["light.test_kasa"],
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
        light = CircadianLight(mock_hass, "light.test_kasa", config, entry, mock_store)
        light.hass = mock_hass
        light.entity_id = "light.circadian_test"
        light._first_update_done = True
        light._brightness = 200
        
        now = datetime(2026, 1, 1, 20, 30, 0)
        
        # Adjustment in-direction for Kasa (dimming during evening)
        await state_management._check_for_manual_override(
            light, 
            old_brightness=190, 
            new_brightness=64, 
            old_color_temp=None, 
            new_color_temp=None, 
            now=now, 
            was_online=True
        )
        
        # Should NOT be overridden (standard behavior for non-Z-Wave)
        assert light._is_overridden is False
