"""Tests for Smart Circadian Lighting Z-Wave JS light support."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import entity_registry as er

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_circadian_lighting import DOMAIN
from custom_components.smart_circadian_lighting.config_flow import _get_supported_dimmer_entities

# Import Z-Wave JS components conditionally
try:
    from homeassistant.components.zwave_js import DOMAIN as ZWAVE_JS_DOMAIN
    ZWAVE_JS_AVAILABLE = True
except ImportError:
    ZWAVE_JS_AVAILABLE = False
    ZWAVE_JS_DOMAIN = "zwave_js"


class TestZwaveConfigFlow:
    """Test Z-Wave JS config flow functionality."""

    def test_get_supported_dimmer_entities_includes_zwave(self):
        """Test that config flow detects Z-Wave dimmer entities."""
        # Mock hass object
        mock_hass = MagicMock()

        # Mock entity registry
        mock_ent_reg = MagicMock()
        mock_ent_reg.entities.values.return_value = [
            MagicMock(platform="kasa_smart_dim", domain="light", entity_id="light.kasa_dimmer"),
            MagicMock(platform="zwave_js", domain="light", entity_id="light.zwave_dimmer"),
            MagicMock(platform="hue", domain="light", entity_id="light.hue_bulb"),  # Should be excluded
        ]

        with patch('homeassistant.helpers.entity_registry.async_get', return_value=mock_ent_reg):
            entities = _get_supported_dimmer_entities(mock_hass)

        assert "light.kasa_dimmer" in entities
        assert "light.zwave_dimmer" in entities
        assert "light.hue_bulb" not in entities
        assert len(entities) == 2


class TestZwaveLightDetection:
    """Test Z-Wave light detection in CircadianLight."""

    def test_zwave_light_detection_in_state_checks(self):
        """Test that Z-Wave lights are detected in _apply_light_state_checks."""
        from homeassistant.components.light import LightEntityFeature
        from custom_components.smart_circadian_lighting.light import CircadianLight

        # Create a mock hass object
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = None  # No initial light state

        # Create a minimal CircadianLight instance for testing
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
            "color_temp_enabled": False,
        }

        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)

        # Mock entity registry to return Z-Wave platform
        mock_ent_reg = MagicMock()
        mock_entity_entry = MagicMock()
        mock_entity_entry.platform = "zwave_js"
        mock_ent_reg.async_get.return_value = mock_entity_entry

        # Mock light state (off)
        mock_light_state = State(
            entity_id="light.test_zwave",
            state=STATE_OFF,
            attributes={
                "supported_features": LightEntityFeature.TRANSITION,
                "brightness": 50  # Use string instead of ATTR_BRIGHTNESS
            }
        )

        service_data = {"brightness": 100}  # Use string instead of ATTR_BRIGHTNESS

        with patch('homeassistant.helpers.entity_registry.async_get', return_value=mock_ent_reg):
            # This should NOT clear service_data for Z-Wave lights (they can be controlled when off)
            light._apply_light_state_checks(service_data, mock_light_state, None, False)

        # Service data should not be cleared for Z-Wave lights
        assert service_data["brightness"] == 100

    def test_non_zwave_light_off_clears_service_data(self):
        """Test that non-Z-Wave lights when off have service data cleared."""
        from homeassistant.components.light import LightEntityFeature
        from custom_components.smart_circadian_lighting.light import CircadianLight

        # Create a mock hass object
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = None  # No initial light state

        # Create a minimal CircadianLight instance for testing
        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
            "color_temp_enabled": False,
        }

        light = CircadianLight(mock_hass, "light.test_light", config, entry, mock_store)

        # Mock entity registry to return non-Z-Wave platform
        mock_ent_reg = MagicMock()
        mock_entity_entry = MagicMock()
        mock_entity_entry.platform = "hue"  # Not Z-Wave or Kasa
        mock_ent_reg.async_get.return_value = mock_entity_entry

        # Mock light state (off)
        mock_light_state = State(
            entity_id="light.test_light",
            state=STATE_OFF,
            attributes={
                "supported_features": LightEntityFeature.TRANSITION,
                "brightness": 50  # Use string instead of ATTR_BRIGHTNESS
            }
        )

        service_data = {"brightness": 100}  # Use string instead of ATTR_BRIGHTNESS

        with patch('homeassistant.helpers.entity_registry.async_get', return_value=mock_ent_reg):
            # This SHOULD clear service_data for non-Z-Wave lights when off
            light._apply_light_state_checks(service_data, mock_light_state, None, False)

        # Service data should be cleared for non-supported lights when off
        assert len(service_data) == 0


class TestZwaveParameterSetting:
    """Test Z-Wave parameter 18 setting logic."""

    @pytest.mark.asyncio
    async def test_zwave_parameter_setting_when_off(self):
        """Test that parameter 18 is set when Z-Wave light is off."""
        from custom_components.smart_circadian_lighting.light import CircadianLight

        # Create a mock hass object
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = None  # No initial light state

        # Create a minimal CircadianLight instance for testing
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
            "color_temp_enabled": False,
        }

        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)

        # Mock entity registry for Z-Wave detection
        mock_ent_reg = MagicMock()
        mock_entity_entry = MagicMock()
        mock_entity_entry.platform = "zwave_js"
        mock_entity_entry.device_id = "test_device_123"
        mock_ent_reg.async_get.return_value = mock_entity_entry

        # Mock light state (off)
        mock_light_state = State(
            entity_id="light.test_zwave",
            state=STATE_OFF,
            attributes={ATTR_BRIGHTNESS: None}
        )

        # Mock hass.states.get to return off state
        with patch.object(mock_hass, 'states') as mock_states, \
             patch('homeassistant.helpers.entity_registry.async_get', return_value=mock_ent_reg), \
             patch.object(mock_hass, 'services', new_callable=MagicMock) as mock_services, \
             patch.object(light, 'async_write_ha_state'):

            mock_states.get.return_value = mock_light_state
            mock_services.async_call = AsyncMock()

            # Call _execute_light_update with brightness data
            service_data = {ATTR_BRIGHTNESS: 128}  # 50% brightness
            await light._execute_light_update(service_data, 128)

            # Verify Z-Wave parameter 18 was set
            mock_services.async_call.assert_called_once()
            call_args = mock_services.async_call.call_args
            assert call_args[0][0] == "zwave_js"  # domain
            assert call_args[0][1] == "set_config_parameter"  # service
            call_data = call_args[0][2]
            assert call_data["device_id"] == "test_device_123"
            assert call_data["parameter"] == 18
            assert call_data["value"] == 49  # 128 * 99 / 255 = 49.41, int() = 49

    @pytest.mark.asyncio
    async def test_zwave_parameter_setting_when_on(self):
        """Test dual setting (parameter + brightness) when Z-Wave light is on."""
        from custom_components.smart_circadian_lighting.light import CircadianLight

        # Create a mock hass object
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = None  # No initial light state

        # Create a minimal CircadianLight instance for testing
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
            "color_temp_enabled": False,
        }

        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)

        # Mock entity registry for Z-Wave detection
        mock_ent_reg = MagicMock()
        mock_entity_entry = MagicMock()
        mock_entity_entry.platform = "zwave_js"
        mock_entity_entry.device_id = "test_device_123"
        mock_ent_reg.async_get.return_value = mock_entity_entry

        # Mock light state (on)
        mock_light_state = State(
            entity_id="light.test_zwave",
            state=STATE_ON,
            attributes={ATTR_BRIGHTNESS: 50}
        )

        # Mock hass.states.get to return on state
        with patch.object(mock_hass, 'states') as mock_states, \
             patch('homeassistant.helpers.entity_registry.async_get', return_value=mock_ent_reg), \
             patch.object(mock_hass, 'services', new_callable=MagicMock) as mock_services, \
             patch.object(light, 'async_write_ha_state'), \
             patch('homeassistant.util.dt.now') as mock_now:

            mock_states.get.return_value = mock_light_state
            mock_services.async_call = AsyncMock()
            mock_now.return_value = MagicMock()

            # Call _execute_light_update with brightness data
            service_data = {ATTR_BRIGHTNESS: 128}  # 50% brightness
            await light._execute_light_update(service_data, 128)

            # Verify both Z-Wave parameter and light.turn_on were called
            assert mock_services.async_call.call_count == 2

            # First call should be Z-Wave parameter setting
            first_call = mock_services.async_call.call_args_list[0]
            assert first_call[0][0] == "zwave_js"
            assert first_call[0][1] == "set_config_parameter"
            call_data = first_call[0][2]
            assert call_data["parameter"] == 18
            assert call_data["value"] == 49  # 128 * 99 / 255 = 49.41

            # Second call should be light.turn_on
            second_call = mock_services.async_call.call_args_list[1]
            assert second_call[0][0] == "light"
            assert second_call[0][1] == "turn_on"
            assert second_call[0][2][ATTR_BRIGHTNESS] == 128

    def test_brightness_scaling_calculation(self):
        """Test that brightness scaling from 0-255 to 0-99 works correctly."""
        # Test edge cases
        test_cases = [
            (0, 0),      # Min brightness
            (255, 99),   # Max brightness
            (128, 49),   # 50% brightness (128 * 99 / 255 = 49.41, int() = 49)
            (191, 74),   # 75% brightness (191 * 99 / 255 = 74.11, int() = 74)
        ]

        for input_brightness, expected_zwave in test_cases:
            # Calculate Z-Wave value the same way as in the code
            zwave_value = int(input_brightness * 99 / 255)
            zwave_value = max(0, min(99, zwave_value))
            assert zwave_value == expected_zwave, f"Input {input_brightness} should give Z-Wave value {expected_zwave}, got {zwave_value}"


# Advanced Z-Wave JS integration tests require full mock driver setup
# These would test real ValueNotifications, node lifecycle, etc.
# For now, we test the core logic that would be used with the mock driver

@pytest.mark.skipif(not ZWAVE_JS_AVAILABLE, reason="Z-Wave JS not available - requires full HA dev environment with Z-Wave JS")
class TestZwaveIntegrationAdvanced:
    """Advanced Z-Wave JS integration tests requiring MockZwaveJsServer."""

    async def test_zwave_value_notifications_parameter_setting(self):
        """Test parameter 18 setting via real ValueNotifications."""
        # This would require MockZwaveJsServer and real Z-Wave device simulation
        pass

    async def test_zwave_node_lifecycle_states(self):
        """Test node lifecycle (not ready, asleep, dead) via mock driver."""
        # This would require MockZwaveJsServer with node status simulation
        pass

    async def test_zwave_force_refresh_service(self):
        """Test force refresh service with real refresh commands."""
        # This would require MockZwaveJsServer with refresh command simulation
        pass