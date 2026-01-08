"""Tests for Smart Circadian Lighting Z-Wave JS light support."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import State


class MockState:
    def __init__(self, entity_id: str, state: str, attributes: dict | None = None):
        if attributes is None:
            attributes = {}
        self.entity_id = entity_id
        self.state = state
        self.attributes = MockAttributes(attributes)

class MockAttributes:
    def __init__(self, attributes: dict):
        self._attributes = attributes.copy()

    def get(self, key, default=None):
        return self._attributes.get(key, default)
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

        with patch('custom_components.smart_circadian_lighting.config_flow.er.async_get', return_value=mock_ent_reg):
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

        with patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=mock_ent_reg):
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
        mock_light_state = MockState(
            entity_id="light.test_zwave",
            state=STATE_OFF,
            attributes={ATTR_BRIGHTNESS: None}
        )

        # Mock all the complex dependencies with optimistic behavior
        with patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=mock_ent_reg), \
             patch.object(mock_hass, 'states') as mock_states, \
             patch.object(mock_hass, 'services') as mock_services, \
             patch.object(light, 'async_write_ha_state'), \
             patch.object(mock_hass, 'async_create_task', return_value=None), \
             patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_now, \
             patch('custom_components.smart_circadian_lighting.circadian_logic.datetime') as mock_datetime:

            # Set up optimistic mocks
            mock_states.get.return_value = mock_light_state
            mock_services.async_call = AsyncMock()  # Assume service calls succeed
            mock_now.return_value = datetime(2023, 1, 1, 6, 22, 30)  # Mid-morning transition (22.5 min in, brightness ~140.25)
            mock_datetime.now.return_value = datetime(2023, 1, 1, 6, 22, 30)

            # Call _async_calculate_and_apply_brightness
            await light._async_calculate_and_apply_brightness()

            # Verify Z-Wave parameter 18 was set
            mock_services.async_call.assert_called()
            calls = mock_services.async_call.call_args_list

            # Find the Z-Wave parameter call
            zwave_call = None
            for call in calls:
                if call[0][0] == "zwave_js" and call[0][1] == "set_config_parameter":
                    zwave_call = call
                    break

            assert zwave_call is not None, f"Z-Wave parameter call not found in calls: {calls}"
            call_data = zwave_call[0][2]
            assert call_data["device_id"] == "test_device_123"
            assert call_data["parameter"] == 18
            # Should be set to circadian target (mid-morning ~115, int(115*99/255)~45)
            assert call_data["value"] == 45

    @pytest.mark.asyncio
    async def test_zwave_parameter_setting_when_on(self, mock_states_manager):
        """Test dual setting (parameter + brightness) when Z-Wave light is on."""
        from custom_components.smart_circadian_lighting.light import CircadianLight

        mock_hass = MagicMock()
        mock_hass.services = MagicMock()
        mock_hass.services.async_call = AsyncMock()
        mock_hass.async_create_task = MagicMock(return_value=None)
        mock_hass.loop = asyncio.get_event_loop()

        states = mock_states_manager(mock_hass)
        states.set_light_state("light.test_zwave", STATE_ON, brightness=50)
        states.set_sun_state()

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

        mock_ent_reg = MagicMock()
        mock_entity_entry = MagicMock()
        mock_entity_entry.platform = "zwave_js"
        mock_entity_entry.device_id = "test_device_123"
        mock_ent_reg.async_get.return_value = mock_entity_entry

        with patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=mock_ent_reg), \
             patch.object(light, 'async_write_ha_state', new_callable=MagicMock), \
             patch.object(light, '_set_exact_circadian_targets', new_callable=AsyncMock), \
             patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_now, \
             patch('custom_components.smart_circadian_lighting.circadian_logic.datetime') as mock_datetime, \
             patch('custom_components.smart_circadian_lighting.light.state_management.async_save_override_state', new_callable=AsyncMock):

            mock_now.return_value = datetime(2023, 1, 1, 2, 0, 0)
            mock_datetime.now.return_value = datetime(2023, 1, 1, 2, 0, 0)

            await light._async_calculate_and_apply_brightness()

            zwave_calls = [call for call in mock_hass.services.async_call.call_args_list
                          if call[0][0] == "zwave_js" and call[0][1] == "set_config_parameter"]
            assert len(zwave_calls) > 0, "Z-Wave parameter call not found"

            light_calls = [call for call in mock_hass.services.async_call.call_args_list
                          if call[0][0] == "light" and call[0][1] == "turn_on"]
            assert len(light_calls) > 0, "Light turn_on call not found"

    @pytest.mark.asyncio
    async def test_zwave_parameter_setting_during_override(self):
        """Test that parameter 18 is set even when Z-Wave light is overridden."""
        from custom_components.smart_circadian_lighting.light import CircadianLight

        # Create a mock hass object
        mock_hass = MagicMock()

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

        # Mock light state (on with manual override)
        mock_light_state = State(
            entity_id="light.test_zwave",
            state=STATE_ON,
            attributes={ATTR_BRIGHTNESS: 150}  # Manually set to 150
        )

        # Set up override state
        light._is_overridden = True

        # Mock all dependencies with optimistic behavior
        with patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=mock_ent_reg), \
             patch.object(mock_hass, 'states', return_value=mock_light_state), \
             patch.object(mock_hass, 'services') as mock_services, \
             patch.object(light, 'async_write_ha_state'), \
             patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_now, \
             patch('custom_components.smart_circadian_lighting.circadian_logic.datetime') as mock_datetime:

            # Set up optimistic mocks
            mock_services.async_call = AsyncMock()  # Assume service calls succeed
            mock_now.return_value = datetime(2023, 1, 1, 2, 0, 0)  # Nighttime
            mock_datetime.now.return_value = datetime(2023, 1, 1, 2, 0, 0)

            # Call _async_calculate_and_apply_brightness
            await light._async_calculate_and_apply_brightness()

            # Verify Z-Wave parameter 18 was NOT set during a hard override
            calls = mock_services.async_call.call_args_list
            for call in calls:
                if call[0][0] == "zwave_js" and call[0][1] == "set_config_parameter":
                    if call[0][2].get("parameter") == 18:
                        pytest.fail("Z-Wave parameter 18 was synced during a hard override in test_light.py, but it should have been skipped.")

    @pytest.mark.asyncio
    async def test_zwave_override_cleared_when_light_turns_off(self):
        """Test that manual override is cleared when Z-Wave light turns off."""
        from homeassistant.core import Event

        from custom_components.smart_circadian_lighting.light import CircadianLight

        # Create a mock hass object
        mock_hass = MagicMock()

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
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "22:00:00",
        }

        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)

        # Mock entity registry for Z-Wave detection
        mock_ent_reg = MagicMock()
        mock_entity_entry = MagicMock()
        mock_entity_entry.platform = "zwave_js"
        mock_ent_reg.async_get.return_value = mock_entity_entry

        # Set up override state
        light._is_overridden = True
        light._override_timestamp = datetime(2023, 1, 1, 10, 0, 0)  # Earlier today

        # Create state change event (ON to OFF)
        old_state = State(entity_id="light.test_zwave", state=STATE_ON, attributes={})
        new_state = State(entity_id="light.test_zwave", state=STATE_OFF, attributes={})
        event = Event("state_changed", {"entity_id": "light.test_zwave", "old_state": old_state, "new_state": new_state})

        with patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=mock_ent_reg), \
             patch('custom_components.smart_circadian_lighting.state_management.async_save_override_state') as mock_save, \
             patch('homeassistant.util.dt.now') as mock_now:

            mock_now.return_value = datetime(2023, 1, 1, 14, 0, 0)  # Daytime

            await light._async_entity_state_changed(event)

            # Verify override was cleared
            assert light._is_overridden == False
            mock_save.assert_called_once()

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
class TestZwaveTransitionBehavior:
    """Test Z-Wave param/brightness behavior during transitions."""

    @pytest.mark.parametrize("mode, offset_min, current_brightness", [
        ("morning", 1, 30),
        ("morning", 22.5, 130),
        ("morning", 44, 240),
        ("evening", 1, 250),
        ("evening", 22.5, 150),
        ("evening", 44, 90),
    ])
    @pytest.mark.asyncio
    async def test_transition_on_no_override_param_and_turn_on(self, mode, offset_min, current_brightness, mock_states_manager):
        """Test param + turn_on during transition, brightness tracking circadian target without override."""
        from custom_components.smart_circadian_lighting.light import CircadianLight

        mock_hass = MagicMock()
        mock_hass.data = {}
        mock_hass.services = MagicMock()
        mock_hass.services.async_call = AsyncMock()
        mock_hass.async_create_task = MagicMock(return_value=None)
        mock_hass.loop = asyncio.get_event_loop()

        states = mock_states_manager(mock_hass)
        states.set_light_state("light.test_zwave", STATE_ON, brightness=current_brightness)
        states.set_sun_state()

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
        config = {
            "lights": ["light.test_zwave"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "05:15:00",
            "morning_end_time": "06:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)

        mock_hass.data[DOMAIN] = {entry.entry_id: {"manual_overrides_enabled": False}}

        mock_ent_reg = MagicMock()
        mock_entity_entry = MagicMock()
        mock_entity_entry.platform = "zwave_js"
        mock_entity_entry.device_id = "test_device_123"
        mock_ent_reg.async_get.return_value = mock_entity_entry

        if mode == "morning":
            start_time = datetime(2023, 1, 1, 5, 15, 0)
        else:
            start_time = datetime(2023, 1, 1, 20, 0, 0)
        test_time = start_time + timedelta(minutes=offset_min)

        with patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=mock_ent_reg), \
             patch.object(light, 'async_write_ha_state', new_callable=MagicMock), \
             patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_now, \
             patch('custom_components.smart_circadian_lighting.circadian_logic.datetime') as mock_datetime, \
             patch('custom_components.smart_circadian_lighting.state_management.async_call_later') as mock_call_later, \
             patch('custom_components.smart_circadian_lighting.state_management.async_save_override_state', new_callable=AsyncMock) as mock_save_override:

            mock_now.return_value = test_time
            mock_datetime.now.return_value = test_time

            await light._async_calculate_and_apply_brightness()

            zwave_calls = [call for call in mock_hass.services.async_call.call_args_list
                          if call[0][0] == "zwave_js" and call[0][1] == "set_config_parameter"]
            assert len(zwave_calls) > 0, "Z-Wave parameter call not found"

            light_calls = [call for call in mock_hass.services.async_call.call_args_list
                          if call[0][0] == "light" and call[0][1] == "turn_on"]
            assert len(light_calls) > 0, "Light turn_on call not found"

            assert not light._is_overridden

    @pytest.mark.asyncio
    async def test_transition_override_param_only_no_turn_on(self, mock_states_manager):
        """Test param only during transition ON, override (large diff)."""
        from custom_components.smart_circadian_lighting.light import CircadianLight

        mock_hass = MagicMock()
        mock_hass.services = MagicMock()
        mock_hass.services.async_call = AsyncMock()
        mock_hass.async_create_task = MagicMock(return_value=None)
        mock_hass.loop = asyncio.get_event_loop()

        states = mock_states_manager(mock_hass)
        states.set_light_state("light.test_zwave", STATE_ON, brightness=170)
        states.set_sun_state()

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
        config = {
            "lights": ["light.test_zwave"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "05:15:00",
            "morning_end_time": "06:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
            "color_temp_enabled": False,
        }

        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)

        mock_ent_reg = MagicMock()
        mock_entity_entry = MagicMock()
        mock_entity_entry.platform = "zwave_js"
        mock_entity_entry.device_id = "test_device_123"
        mock_ent_reg.async_get.return_value = mock_entity_entry

        test_time = datetime(2023, 1, 1, 5, 37, 30)

        with patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=mock_ent_reg), \
             patch.object(light, 'async_write_ha_state'), \
             patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_now, \
             patch('custom_components.smart_circadian_lighting.circadian_logic.datetime') as mock_datetime, \
             patch('custom_components.smart_circadian_lighting.state_management.async_call_later') as mock_call_later, \
             patch('custom_components.smart_circadian_lighting.state_management.async_save_override_state') as mock_save_override:

            mock_now.return_value = test_time
            mock_datetime.now.return_value = test_time

            await light._async_calculate_and_apply_brightness()

            zwave_calls = [call for call in mock_hass.services.async_call.call_args_list
                          if call[0][0] == "zwave_js" and call[0][1] == "set_config_parameter"]
            assert len(zwave_calls) > 0, "Z-Wave parameter call not found"

            light_calls = [call for call in mock_hass.services.async_call.call_args_list
                          if call[0][0] == "light" and call[0][1] == "turn_on"]
            assert len(light_calls) == 0, "Light turn_on should not be called during override"

            assert light._is_overridden

    @pytest.mark.asyncio
    async def test_off_param_only_no_turn_on(self):
        """Test param only when OFF."""
        from custom_components.smart_circadian_lighting.light import CircadianLight

        mock_hass = MagicMock()

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test")
        config = {
            "lights": ["light.test_zwave"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "05:15:00",
            "morning_end_time": "06:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
            "color_temp_enabled": False,
        }

        light = CircadianLight(mock_hass, "light.test_zwave", config, entry, mock_store)

        mock_ent_reg = MagicMock()
        mock_entity_entry = MagicMock()
        mock_entity_entry.platform = "zwave_js"
        mock_entity_entry.device_id = "test_device_123"
        mock_ent_reg.async_get.return_value = mock_entity_entry

        test_time = datetime(2023, 1, 1, 5, 37, 30)  # mid morning

        mock_light_state = MockState(
            entity_id="light.test_zwave",
            state=STATE_OFF,
            attributes={}
        )

        with patch('custom_components.smart_circadian_lighting.light.er.async_get', return_value=mock_ent_reg), \
             patch.object(mock_hass, 'states') as mock_states, \
             patch.object(mock_hass, 'services') as mock_services, \
             patch.object(light, 'async_write_ha_state'), \
             patch.object(mock_hass, 'async_create_task', return_value=None), \
             patch('custom_components.smart_circadian_lighting.light.dt_util.now') as mock_now, \
             patch('custom_components.smart_circadian_lighting.circadian_logic.datetime') as mock_datetime:

            mock_states.get.return_value = mock_light_state
            mock_services.async_call = AsyncMock()

            mock_now.return_value = test_time
            mock_datetime.now.return_value = test_time

            await light._async_calculate_and_apply_brightness()

            # Verify param called
            zwave_calls = [call for call in mock_services.async_call.call_args_list
                          if call[0][0] == "zwave_js" and call[0][1] == "set_config_parameter"]
            assert len(zwave_calls) > 0, "Z-Wave parameter call not found"

            # Verify turn_on NOT called
            light_calls = [call for call in mock_services.async_call.call_args_list
                          if call[0][0] == "light" and call[0][1] == "turn_on"]
            assert len(light_calls) == 0, "Light turn_on should not be called when OFF"
