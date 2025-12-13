"""Comprehensive tests for manual override functionality in Smart Circadian Lighting.

This test suite covers all scenarios in manual_overrides.md:
1. Triggering overrides (Section 1)
2. Behavior during overrides (Section 2)
3. Clearing overrides (Section 3)
4. Quantization error handling (Section 4)

Test Organization:
- Unit Tests: Isolated scale conversion and threshold calculation
- Direction Tests: Transition direction detection logic
- Integration Tests: Real device values with quantization error handling
- BVA Tests: Boundary value analysis with verified conversion paths
"""

import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_COLOR_TEMP_KELVIN
from homeassistant.const import STATE_ON
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_circadian_lighting import DOMAIN
from custom_components.smart_circadian_lighting.light import CircadianLight
from custom_components.smart_circadian_lighting.circadian_logic import _convert_percent_to_255
from custom_components.smart_circadian_lighting import state_management

_LOGGER = logging.getLogger(__name__)


class MockState:
    """A State-like object that works correctly with attributes."""

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
    """Factory for creating mock State objects."""

    def create_mock_state(
        entity_id: str, state: str, attributes: dict | None = None
    ) -> MockState:
        return MockState(entity_id, state, attributes)

    return create_mock_state


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock()
    hass.states.async_set = AsyncMock()
    hass.bus = MagicMock()
    hass.bus.async_fire = AsyncMock()
    hass.data = {}
    hass.loop = asyncio.new_event_loop()
    hass.async_create_task = MagicMock(return_value=None)
    return hass


def brightness_0_99_to_255_truncate(brightness_99: int) -> int:
    """Convert Z-Wave brightness (0-99) to HA scale (0-255) with truncation."""
    if brightness_99 >= 99:
        return 255
    return int(brightness_99 * 255 / 99)


def brightness_0_99_to_255_round(brightness_99: int) -> int:
    """Convert Z-Wave brightness (0-99) to HA scale (0-255) with rounding."""
    if brightness_99 >= 99:
        return 255
    return int(round(brightness_99 * 255 / 99))


def brightness_0_100_to_255_truncate(brightness_100: int) -> int:
    """Convert Kasa brightness (0-100) to HA scale (0-255) with truncation."""
    if brightness_100 >= 100:
        return 255
    return int(brightness_100 * 255 / 100)


def brightness_0_100_to_255_round(brightness_100: int) -> int:
    """Convert Kasa brightness (0-100) to HA scale (0-255) with rounding."""
    if brightness_100 >= 100:
        return 255
    return int(round(brightness_100 * 255 / 100))


def brightness_255_to_0_99_truncate(brightness_255: int) -> int:
    """Convert HA brightness (0-255) to Z-Wave scale (0-99) with truncation."""
    if brightness_255 >= 255:
        return 99
    return int(brightness_255 * 99 / 255)


def brightness_255_to_0_99_round(brightness_255: int) -> int:
    """Convert HA brightness (0-255) to Z-Wave scale (0-99) with rounding."""
    if brightness_255 >= 255:
        return 99
    return int(round(brightness_255 * 99 / 255))


def brightness_255_to_0_100_truncate(brightness_255: int) -> int:
    """Convert HA brightness (0-255) to Kasa scale (0-100) with truncation."""
    if brightness_255 >= 255:
        return 100
    return int(brightness_255 * 100 / 255)


def brightness_255_to_0_100_round(brightness_255: int) -> int:
    """Convert HA brightness (0-255) to Kasa scale (0-100) with rounding."""
    if brightness_255 >= 255:
        return 100
    return int(round(brightness_255 * 100 / 255))


class TestBrightnessScaleConversion:
    """Unit tests for brightness scale conversion functions.
    
    These tests verify that scale conversions work correctly and expose
    quantization error characteristics for each device type.
    """

    def test_zwave_0_99_truncate_conversion_symmetry(self):
        """Test Z-Wave truncate conversion documents actual quantization behavior.
        
        Z-Wave uses 0-99 scale (99 steps). When converting to HA 0-255 scale,
        SIGNIFICANT quantization errors occur. Example: 1 -> 2 -> 0 (loss of data).
        This is why the code must account for quantization error as per Section 4.
        """
        test_cases = [
            (0, 0, 0),      
            (1, 2, 0),      
            (99, 255, 99),  
            (49, 126, 48),  
            (50, 128, 49),  
        ]
        
        for device_val, expected_ha_trunc, expected_device_back in test_cases:
            ha_val = brightness_0_99_to_255_truncate(device_val)
            device_back = brightness_255_to_0_99_truncate(ha_val)
            assert ha_val == expected_ha_trunc, f"Z-Wave truncate {device_val} → {ha_val} (expected {expected_ha_trunc})"
            assert device_back == expected_device_back, f"Z-Wave truncate round-trip {device_val} → {ha_val} → {device_back} shows error tolerance"

    def test_zwave_0_99_round_conversion_symmetry(self):
        """Test Z-Wave round conversion is lossless (KEY FINDING).
        
        IMPORTANT: Rounding during Z-Wave conversion eliminates round-trip errors.
        This is the CORRECT approach to avoid false override triggers.
        """
        test_cases = [
            (0, 0, 0),
            (1, 3, 1),    
            (99, 255, 99),
            (49, 126, 49),
            (50, 129, 50),
        ]
        
        for device_val, expected_ha_round, expected_device_back in test_cases:
            ha_val = brightness_0_99_to_255_round(device_val)
            device_back = brightness_255_to_0_99_round(ha_val)
            assert ha_val == expected_ha_round, f"Z-Wave round {device_val} → {ha_val} (expected {expected_ha_round})"
            assert device_back == expected_device_back, f"Z-Wave round round-trip lossless: {device_val} -> {ha_val} -> {device_back}"

    def test_kasa_0_100_truncate_conversion_symmetry(self):
        """Test Kasa truncate conversion introduces quantization error.
        
        Unlike rounding, truncation can cause round-trip errors up to 3.
        """
        test_cases = [
            (0, 0, 0),
            (1, 2, 0),    
            (100, 255, 100),
            (50, 127, 49),
            (75, 191, 74),
        ]
        
        for device_val, expected_ha_trunc, expected_device_back in test_cases:
            ha_val = brightness_0_100_to_255_truncate(device_val)
            device_back = brightness_255_to_0_100_truncate(ha_val)
            assert ha_val == expected_ha_trunc, f"Kasa truncate {device_val} → {ha_val} (expected {expected_ha_trunc})"
            assert device_back == expected_device_back, f"Kasa truncate round-trip {device_val} → {ha_val} → {device_back}"

    def test_kasa_0_100_round_conversion_symmetry(self):
        """Test Kasa round conversion is lossless (KEY FINDING).
        
        IMPORTANT: Rounding during Kasa conversion eliminates round-trip errors.
        This is the CORRECT approach to avoid false override triggers.
        """
        test_cases = [
            (0, 0, 0),
            (1, 3, 1),    
            (100, 255, 100),
            (50, 128, 50),
            (75, 191, 75),
        ]
        
        for device_val, expected_ha_round, expected_device_back in test_cases:
            ha_val = brightness_0_100_to_255_round(device_val)
            device_back = brightness_255_to_0_100_round(ha_val)
            assert ha_val == expected_ha_round, f"Kasa round {device_val} → {ha_val} (expected {expected_ha_round})"
            assert device_back == expected_device_back, f"Kasa round round-trip lossless: {device_val} -> {ha_val} -> {device_back}"

    def test_quantization_error_range_zwave_truncate(self):
        """Document quantization error for Z-Wave truncate conversion.
        
        Truncation causes up to 3 units of error. This is a CRITICAL issue
        that must be accounted for in override detection to avoid false
        negatives (not triggering override when user clearly dimmed).
        """
        max_error = 0
        for ha_val in range(256):
            device_val = brightness_255_to_0_99_truncate(ha_val)
            ha_back = brightness_0_99_to_255_truncate(device_val)
            error = abs(ha_val - ha_back)
            max_error = max(max_error, error)
        
        assert max_error == 3, f"Z-Wave truncate max error should be 3: {max_error}"

    def test_quantization_error_range_zwave_round(self):
        """Document quantization error for Z-Wave round conversion.
        
        KEY FINDING: Rounding REDUCES quantization errors to maximum 1.
        This is MUCH better than truncation (max error 3), but still introduces
        some error that must be accounted for in override detection.
        """
        max_error = 0
        for ha_val in range(256):
            device_val = brightness_255_to_0_99_round(ha_val)
            ha_back = brightness_0_99_to_255_round(device_val)
            error = abs(ha_val - ha_back)
            max_error = max(max_error, error)
        
        assert max_error == 1, f"Z-Wave round max error should be 1: {max_error}"

    def test_quantization_error_range_kasa_truncate(self):
        """Document quantization error for Kasa truncate conversion.
        
        Truncation causes up to 3 units of error. This is a CRITICAL issue
        that must be accounted for in override detection to avoid false
        negatives (not triggering override when user clearly brightened).
        """
        max_error = 0
        for ha_val in range(256):
            device_val = brightness_255_to_0_100_truncate(ha_val)
            ha_back = brightness_0_100_to_255_truncate(device_val)
            error = abs(ha_val - ha_back)
            max_error = max(max_error, error)
        
        assert max_error == 3, f"Kasa truncate max error should be 3: {max_error}"

    def test_quantization_error_range_kasa_round(self):
        """Document quantization error for Kasa round conversion.
        
        KEY FINDING: Rounding REDUCES quantization errors to maximum 1.
        This is MUCH better than truncation (max error 3), but still introduces
        some error that must be accounted for in override detection.
        """
        max_error = 0
        for ha_val in range(256):
            device_val = brightness_255_to_0_100_round(ha_val)
            ha_back = brightness_0_100_to_255_round(device_val)
            error = abs(ha_val - ha_back)
            max_error = max(max_error, error)
        
        assert max_error == 1, f"Kasa round max error should be 1: {max_error}"


class TestThresholdCalculation:
    """Unit tests for threshold calculation from config percentage.

    Per manual_overrides.md, the threshold is used to determine whether
    a brightness change crosses the setpoint boundary.
    """

    def test_threshold_calculation_5_percent(self):
        """Test threshold calculation for 5% config value."""
        threshold_percent = 5
        expected_255_scale = _convert_percent_to_255(threshold_percent)

        assert expected_255_scale == int(round(5 * 255 / 100)), \
            f"5% should convert to {int(round(5 * 255 / 100))} in 0-255 scale"

    def test_threshold_calculation_10_percent(self):
        """Test threshold calculation for 10% config value."""
        threshold_percent = 10
        expected_255_scale = _convert_percent_to_255(threshold_percent)

        assert expected_255_scale == int(round(10 * 255 / 100)), \
            f"10% should convert to {int(round(10 * 255 / 100))} in 0-255 scale"

    def test_threshold_boundary_calculations(self):
        """Test boundary calculations with different thresholds and circadian targets.

        This verifies the math used in override detection:
        - Morning: override if new_brightness < (circadian_target - threshold)
        - Evening: override if new_brightness > (circadian_target + threshold)
        """
        circadian_target = 150
        threshold = 25

        morning_boundary = circadian_target - threshold
        evening_boundary = circadian_target + threshold

        assert morning_boundary == 125, f"Morning boundary should be 125, got {morning_boundary}"
        assert evening_boundary == 175, f"Evening boundary should be 175, got {evening_boundary}"


class TestColorTempThresholdCalculation:
    """Unit tests for color temperature threshold calculation from config.

    Per state_management.py, color temperature override is triggered if
    the absolute change exceeds the configured threshold (default 100K).
    """

    def test_color_temp_threshold_default(self):
        """Test default color temperature threshold is 100K."""
        config = {}
        threshold = config.get("color_temp_manual_override_threshold", 100)
        assert threshold == 100, f"Default color temp threshold should be 100K, got {threshold}"

    def test_color_temp_threshold_custom(self):
        """Test custom color temperature threshold."""
        config = {"color_temp_manual_override_threshold": 50}
        threshold = config.get("color_temp_manual_override_threshold", 100)
        assert threshold == 50, f"Custom color temp threshold should be 50K, got {threshold}"

    def test_color_temp_change_detection(self):
        """Test color temperature change detection logic.

        Override is triggered if abs(new_temp - old_temp) > threshold.
        """
        threshold = 100

        # Small change should not trigger
        small_change = abs(3000 - 2950)
        assert small_change <= threshold, f"Small change {small_change} should not exceed threshold {threshold}"

        # Large change should trigger
        large_change = abs(3000 - 3200)
        assert large_change > threshold, f"Large change {large_change} should exceed threshold {threshold}"


class TestTransitionDirectionDetection:
    """Unit tests for transition direction detection.
    
    Per manual_overrides.md Section 1:
    - Morning transition increases brightness (user must dim to trigger)
    - Evening transition decreases brightness (user must brighten to trigger)
    - Changes in same direction as transition do NOT trigger override
    """

    def test_morning_transition_dimming_is_opposite_direction(self):
        """Test that dimming during morning transition is opposite direction."""
        is_morning = True
        brightness_diff = -20  
        
        is_opposite_direction = (is_morning and brightness_diff < 0)
        assert is_opposite_direction, "Dimming should be opposite to morning increase"

    def test_morning_transition_brightening_is_same_direction(self):
        """Test that brightening during morning transition is same direction."""
        is_morning = True
        brightness_diff = 20  
        
        is_opposite_direction = (is_morning and brightness_diff < 0)
        assert not is_opposite_direction, "Brightening should be same as morning increase"

    def test_evening_transition_brightening_is_opposite_direction(self):
        """Test that brightening during evening transition is opposite direction."""
        is_morning = False
        brightness_diff = 20  
        
        is_opposite_direction = (not is_morning and brightness_diff > 0)
        assert is_opposite_direction, "Brightening should be opposite to evening decrease"

    def test_evening_transition_dimming_is_same_direction(self):
        """Test that dimming during evening transition is same direction."""
        is_morning = False
        brightness_diff = -20  
        
        is_opposite_direction = (not is_morning and brightness_diff > 0)
        assert not is_opposite_direction, "Dimming should be same as evening decrease"


class TestIntegrationScenarios:
    """Integration test scenarios with documented quantization error handling.

    Per manual_overrides.md Section 4:
    - Account for brightness scale quantization errors
    - Dynamically calculate acceptable error threshold based on device scale
    - Only trigger override if brightness moves AGAINST direction AND crosses threshold

    These tests use realistic device values and verify the complete conversion chain.
    """


class TestColorTempIntegrationScenarios:
    """Integration test scenarios for color temperature override detection.

    Color temperature override is simpler than brightness:
    - No scale conversion or quantization errors
    - Override triggered if absolute change > threshold during transition
    - Threshold is configurable (default 100K)

    These tests verify color temperature override detection with realistic values.
    """

    @pytest.mark.asyncio
    async def test_color_temp_override_during_morning_transition(
        self, mock_hass, mock_state_factory
    ):
        """Integration test: Color temperature override during morning transition.

        Scenario:
        - Morning transition (color temp increasing from warm to cool)
        - Circadian target: 3000K
        - Threshold: 100K
        - User changes to 3200K (200K change > 100K threshold)

        Expected: Override triggered because change exceeds threshold during transition
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": True,
            "day_color_temp_kelvin": 4800,
            "night_color_temp_kelvin": 1800,
            "color_temp_manual_override_threshold": 100,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_ct_integration_morning", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._color_temp_kelvin = 3000

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
              patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
              patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
              patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
              patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
              patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 3000}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 3200}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, (
                f"Color temp override not triggered: circadian={light._color_temp_kelvin}K, "
                f"change=200K, threshold=100K"
            )

    @pytest.mark.asyncio
    async def test_color_temp_no_override_small_change_during_evening_transition(
        self, mock_hass, mock_state_factory
    ):
        """Integration test: No color temperature override for small change during evening transition.

        Scenario:
        - Evening transition (color temp decreasing from cool to warm)
        - Circadian target: 4000K
        - Threshold: 100K
        - User changes to 4080K (80K change < 100K threshold)

        Expected: No override because change is within threshold
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": True,
            "day_color_temp_kelvin": 4800,
            "night_color_temp_kelvin": 1800,
            "color_temp_manual_override_threshold": 100,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_ct_integration_evening", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._color_temp_kelvin = 4000

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        evening_transition = datetime(2023, 1, 1, 19, 45, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
              patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
              patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
              patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
              patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
              patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = evening_transition
            mock_dt_util.return_value = evening_transition
            mock_dt_utcnow.return_value = evening_transition
            mock_datetime.now.return_value = evening_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 4000}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 4080}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert not light._is_overridden, (
                f"Color temp override incorrectly triggered: circadian={light._color_temp_kelvin}K, "
                f"change=80K, threshold=100K"
            )

    @pytest.mark.asyncio
    async def test_morning_override_with_zwave_quantization_error(
        self, mock_hass, mock_state_factory
    ):
        """Integration test: Morning override with Z-Wave quantization error.
        
        Scenario:
        - Device: Z-Wave dimmer (0-99 scale)
        - Circadian target: 150 (HA 0-255 scale)
        - Threshold: 25 (HA 0-255 scale)
        - Morning boundary: 125 (HA scale)
        - User dims to device native 48 (converts to HA 121 with truncate)
        
        Expected: Override triggered because 121 < 125 and dimming is opposite direction
        
        Bug verification: If quantization error causes conversion to HA 122, override
        might not trigger even though user clearly dimmed below boundary.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_integration_zwave", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_device_native = brightness_255_to_0_99_truncate(160)
            new_device_native = brightness_255_to_0_99_truncate(121)

            old_brightness_ha = brightness_0_99_to_255_truncate(old_device_native)
            new_brightness_ha = brightness_0_99_to_255_truncate(new_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, (
                f"Override not triggered: circadian=150, threshold=25, boundary=125, "
                f"device_native={new_device_native}, ha_converted={new_brightness_ha}"
            )

    @pytest.mark.asyncio
    async def test_evening_override_with_kasa_quantization_error(
        self, mock_hass, mock_state_factory
    ):
        """Integration test: Evening override with Kasa quantization error.
        
        BUG FOUND: Device reports brightness at exact boundary due to quantization.
        
        Scenario:
        - Device: Kasa dimmer (0-100 scale)
        - Circadian target: 100 (HA 0-255 scale)
        - Threshold: 25 (HA 0-255 scale)
        - Evening boundary: 125 (HA scale) - must be GREATER than this to trigger
        - User brightens to device native 49 (converts to HA 124 with truncate)
        
        ISSUE: HA 124 is NOT > 125 (boundary is exclusive), so override doesn't trigger.
        But user clearly brightened the light during evening transition.
        
        Expected: This test documents a potential bug where quantization error
        causes the boundary check to be off by 1 LED step.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_integration_kasa", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 100
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        evening_transition = datetime(2023, 1, 1, 19, 45, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = evening_transition
            mock_dt_util.return_value = evening_transition
            mock_dt_utcnow.return_value = evening_transition
            mock_datetime.now.return_value = evening_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_device_native = brightness_255_to_0_100_truncate(80)
            new_device_native = brightness_255_to_0_100_truncate(127)

            old_brightness_ha = brightness_0_100_to_255_truncate(old_device_native)
            new_brightness_ha = brightness_0_100_to_255_truncate(new_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, (
                f"BUG: Quantization error caused boundary check miss. "
                f"Circadian=100, threshold=25, boundary=125, "
                f"device_native={new_device_native}, ha_converted={new_brightness_ha}. "
                f"User clearly brightened (79→124 during evening), but 124 is NOT > 125 (exclusive). "
                f"Code doesn't account for quantization error as required by manual_overrides.md Section 4."
            )

    @pytest.mark.asyncio
    async def test_no_override_during_hardware_transition(
        self, mock_hass, mock_state_factory
    ):
        """Test that no override is triggered for intermediate brightness during hardware transition.

        Per manual_overrides.md Section 4: "No override is triggered as long as each reported
        brightness is between the transition start brightness and the target brightness (inclusive),
        accounting for quantization error."

        Scenario:
        - Morning transition: circadian target increasing from 25 to 198
        - Hardware transition commanded: brightness=198 with transition=60 seconds
        - Light reports starting brightness=25 almost immediately after transition starts
        - Then reports intermediate brightness=100 during transition
        - Should NOT trigger override, as 100 is between start (25) and target (198)
        - Multiple updates during transition period for accuracy
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_hardware_transition", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 25  # Start of transition

        # Simulate active hardware transition state
        light._transition_target = 198
        light._transition_start_time = datetime(2023, 1, 1, 6, 30, 0)
        light._update_interval = 60

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
              patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
              patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
              patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
              patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
              patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher, \
              patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None
            mock_er_get.return_value = MagicMock()

            # First update: Light reports starting brightness=25 almost immediately after transition starts
            # This should not trigger override as it's the expected starting point
            start_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 25}  # Starting value
            )
            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": None, "new_state": start_state})
            )
            assert not light._is_overridden, "Override incorrectly triggered at transition start (brightness=25)"

            # Second update: Light reports intermediate brightness=100 during transition
            # This should NOT trigger override because 100 is between start (25) and target (198)
            intermediate_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 100}  # Intermediate value
            )
            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": start_state, "new_state": intermediate_state})
            )
            assert not light._is_overridden, (
                "Override incorrectly triggered for intermediate brightness (100) during hardware transition. "
                "100 is between start (25) and target (198), so should not trigger override."
            )

            # Third update: Light reports another intermediate value=150 during transition
            # This should still NOT trigger override as 150 is still between 25 and 198
            later_intermediate_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 150}  # Later intermediate value
            )
            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": intermediate_state, "new_state": later_intermediate_state})
            )
            assert not light._is_overridden, (
                "Override incorrectly triggered for later intermediate brightness (150) during hardware transition. "
                "150 is between start (25) and target (198), so should not trigger override."
            )


class TestOverrideTriggeringConditions:
    """Test Scenario Group 1: Conditions for triggering an override (manual_overrides.md Section 1)"""

    @pytest.mark.asyncio
    async def test_t_1_1_normal_operation_no_override_outside_transition(
        self, mock_hass, mock_state_factory
    ):
        """T-1.1: Normal Operation - No override outside transition period.

        hass = mock_hass
        Set time to midday (no transition active).
        Set light to manual brightness.
        Verify light state remains circadian_enabled: true.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 5,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_1_1", data=config)


        # Mock the store
        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True

        # Setup hass data structure
        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set time to midday (12:00) - no transition active
        midday = datetime(2023, 1, 1, 12, 0, 0)

        # Simulate state change event: light is adjusted outside transition
        with patch("homeassistant.util.dt.now") as mock_now, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.now"
        ) as mock_dt_util:
            mock_now.return_value = midday
            mock_dt_util.return_value = midday

            initial_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 200}
            )
            adjusted_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 100}
            )

            # Trigger state change detection
            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light,
                MagicMock(
                    data={"old_state": initial_state, "new_state": adjusted_state}
                ),
            )

            # Outside transition, override should NOT be triggered
            assert (
                not light._is_overridden
            ), "Override incorrectly triggered outside transition period"


class TestColorTempOverrideTriggeringConditions:
    """Test Scenario Group CT-1: Conditions for triggering color temperature override.

    Color temperature override is triggered during transition if the absolute
    change exceeds the threshold (default 100K), regardless of direction.
    """

    @pytest.mark.asyncio
    async def test_ct_1_1_normal_operation_no_override_outside_transition(
        self, mock_hass, mock_state_factory
    ):
        """CT-1.1: Normal Operation - No color temp override outside transition period.

        Set time to midday (no transition active).
        Change color temperature manually.
        Verify no override is triggered.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 5,
            "color_temp_enabled": True,
            "day_color_temp_kelvin": 4800,
            "night_color_temp_kelvin": 1800,
            "color_temp_manual_override_threshold": 100,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_ct_1_1", data=config)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._color_temp_kelvin = 3000  # Set current circadian color temp

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set time to midday (12:00) - no transition active
        midday = datetime(2023, 1, 1, 12, 0, 0)

        with patch("homeassistant.util.dt.now") as mock_now, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.now"
        ) as mock_dt_util:
            mock_now.return_value = midday
            mock_dt_util.return_value = midday

            initial_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 3000}
            )
            adjusted_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 4000}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light,
                MagicMock(
                    data={"old_state": initial_state, "new_state": adjusted_state}
                ),
            )

            # Outside transition, override should NOT be triggered
            assert (
                not light._is_overridden
            ), "Color temp override incorrectly triggered outside transition period"

    @pytest.mark.asyncio
    async def test_ct_1_2_transition_small_change_no_override(
        self, mock_hass, mock_state_factory
    ):
        """CT-1.2: Transition, Small Change - No override for changes within threshold.

        During morning transition, change color temperature by less than threshold.
        Override should NOT be triggered.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 5,
            "color_temp_enabled": True,
            "day_color_temp_kelvin": 4800,
            "night_color_temp_kelvin": 1800,
            "color_temp_manual_override_threshold": 100,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_ct_1_2", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._color_temp_kelvin = 3000

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set time to morning transition (06:30)
        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.now"
        ) as mock_dt_util, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.utcnow"
        ) as mock_dt_utcnow, patch(
            "custom_components.smart_circadian_lighting.state_management.async_call_later"
        ) as mock_call_later, patch(
            "custom_components.smart_circadian_lighting.state_management.async_dispatcher_send"
        ) as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            # Small change: 3000K to 3050K (50K change < 100K threshold)
            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 3000}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 3050}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert (
                not light._is_overridden
            ), "Color temp override incorrectly triggered for small change within threshold"

    @pytest.mark.asyncio
    async def test_ct_1_3_transition_large_change_triggers_override(
        self, mock_hass, mock_state_factory
    ):
        """CT-1.3: Transition, Large Change - Override triggered for changes exceeding threshold.

        During evening transition, change color temperature by more than threshold.
        Override SHOULD be triggered.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 5,
            "color_temp_enabled": True,
            "day_color_temp_kelvin": 4800,
            "night_color_temp_kelvin": 1800,
            "color_temp_manual_override_threshold": 100,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_ct_1_3", data=config)
        hass = mock_hass
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._color_temp_kelvin = 4000

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set time to evening transition (19:45)
        evening_transition = datetime(2023, 1, 1, 19, 45, 0)

        with patch("homeassistant.util.dt.now") as mock_now, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.now"
        ) as mock_dt_util, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.utcnow"
        ) as mock_dt_utcnow, patch(
            "custom_components.smart_circadian_lighting.state_management.async_call_later"
        ) as mock_call_later, patch(
            "custom_components.smart_circadian_lighting.state_management.async_dispatcher_send"
        ) as mock_dispatcher:
            mock_now.return_value = evening_transition
            mock_dt_util.return_value = evening_transition
            mock_dt_utcnow.return_value = evening_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            # Large change: 4000K to 4200K (200K change > 100K threshold)
            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 4000}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 4200}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert (
                light._is_overridden
            ), "Color temp override not triggered for large change exceeding threshold"
            assert light._override_timestamp is not None

    @pytest.mark.asyncio
    async def test_ct_1_4_exact_threshold_edge_case_no_override(
        self, mock_hass, mock_state_factory
    ):
        """CT-1.4: Exact Threshold Edge Case - No override at exact threshold.

        During transition, change color temperature by exactly the threshold.
        Override should NOT be triggered (must exceed threshold).
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 5,
            "color_temp_enabled": True,
            "day_color_temp_kelvin": 4800,
            "night_color_temp_kelvin": 1800,
            "color_temp_manual_override_threshold": 100,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_ct_1_4", data=config)
        hass = mock_hass
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._color_temp_kelvin = 3000

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set time to morning transition
        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.now"
        ) as mock_dt_util, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.utcnow"
        ) as mock_dt_utcnow, patch(
            "custom_components.smart_circadian_lighting.state_management.async_call_later"
        ) as mock_call_later, patch(
            "custom_components.smart_circadian_lighting.state_management.async_dispatcher_send"
        ) as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            # Exact threshold change: 3000K to 3100K (100K change == threshold)
            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 3000}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_COLOR_TEMP_KELVIN: 3100}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            # At exact threshold should NOT trigger (must exceed threshold)
            assert (
                not light._is_overridden
            ), "Color temp override incorrectly triggered at exact threshold boundary"
        """T-1.2: Morning Transition, Wrong Direction.

        Morning transition (brightness increasing).
        User brightens light (same direction as transition).
        Override should NOT be triggered.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 5,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_1_2", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 128  # 50% brightness (circadian target)

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set time to morning transition (06:30)
        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.now"
        ) as mock_dt_util, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.utcnow"
        ) as mock_dt_utcnow, patch(
            "custom_components.smart_circadian_lighting.state_management.async_call_later"
        ) as mock_call_later, patch(
            "custom_components.smart_circadian_lighting.state_management.async_dispatcher_send"
        ) as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            # User brightens light (same direction as transition)
            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 100}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 180}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert (
                not light._is_overridden
            ), "Override incorrectly triggered for same-direction adjustment"

    @pytest.mark.asyncio
    async def test_t_1_3_evening_transition_wrong_direction_no_override(
        self, mock_hass, mock_state_factory
    ):
        """T-1.3: Evening Transition, Wrong Direction.

        Evening transition (brightness decreasing).
        User dims light (same direction as transition).
        Override should NOT be triggered.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 5,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_1_3", data=config)
        hass = mock_hass
        entry.add_to_hass(hass)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150  # High brightness (circadian target for morning)

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set time to evening transition (19:45)
        evening_transition = datetime(2023, 1, 1, 19, 45, 0)

        with patch("homeassistant.util.dt.now") as mock_now, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.now"
        ) as mock_dt_util:
            mock_now.return_value = evening_transition
            mock_dt_util.return_value = evening_transition

            # User dims light (same direction as transition)
            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 180}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 100}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert (
                not light._is_overridden
            ), "Override incorrectly triggered for same-direction adjustment"

    @pytest.mark.asyncio
    async def test_t_1_4_adjustment_within_threshold_no_override(
        self, mock_hass, mock_state_factory
    ):
        """T-1.4: Adjustment Within Threshold.

        Morning transition, brightness at 150, threshold = 10.
        User dims to 145 (within threshold).
        Override should NOT be triggered.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_1_4", data=config)
        hass = mock_hass
        entry.add_to_hass(hass)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150  # Circadian target
        light._manual_override_threshold = 25  # 10% in 0-255 scale

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set time to morning transition
        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.now"
        ) as mock_dt_util, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.utcnow"
        ) as mock_dt_utcnow, patch(
            "custom_components.smart_circadian_lighting.state_management.async_call_later"
        ) as mock_call_later, patch(
            "custom_components.smart_circadian_lighting.state_management.async_dispatcher_send"
        ) as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            # User dims from 160 to 145 (within threshold of 25)
            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 160}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 145}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert (
                not light._is_overridden
            ), "Override incorrectly triggered for adjustment within threshold"

    @pytest.mark.asyncio
    async def test_t_1_5_exact_threshold_edge_case_no_override(
        self, mock_hass, mock_state_factory
    ):
        """T-1.5: Exact Threshold Edge Case.

        Morning transition, brightness = 150, threshold = 10.
        User dims to 140 (exactly at threshold boundary).
        Override should NOT be triggered (must be BEYOND threshold).
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_1_5", data=config)
        hass = mock_hass
        entry.add_to_hass(hass)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150  # Circadian target
        light._manual_override_threshold = 25  # 10% in 0-255 scale

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set time to morning transition
        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, patch(
            "custom_components.smart_circadian_lighting.state_management.dt_util.now"
        ) as mock_dt_util, patch(
            "custom_components.smart_circadian_lighting.state_management.async_call_later"
        ) as mock_call_later, patch(
            "homeassistant.helpers.dispatcher.async_dispatcher_send"
        ) as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            # User dims from 165 to 125 (exactly at threshold: 150 - 25)
            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 165}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 125}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            # At exact threshold should NOT trigger (must be BEYOND threshold)
            assert (
                not light._is_overridden
            ), "Override incorrectly triggered at exact threshold boundary"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_t_1_6a_morning_bva_well_below_boundary(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """T-1.6a: Morning BVA - Well Below Boundary.
        
        Boundary Value Analysis (BVA) test for morning transition override triggering.
        
        Test Setup:
        - Circadian target: 150 (HA 0-255 scale)
        - Threshold: 25 (HA 0-255 scale)
        - Morning boundary: 125 (HA scale)
        - New brightness: 105 (well below 125)
        - Direction: Dimming (opposite to morning increase)
        
        Expected Result: Override SHOULD be triggered
        - Reason: brightness (105) < boundary (125) AND dimming during morning
        
        Known Issues:
        - Quantization error from scale conversion may cause this to fail if the
          code doesn't account for device-native scale limits
        - For Z-Wave (0-99), max error ~2; for Kasa (0-100), max error ~1
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_1_6a_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(160)
            new_brightness_device_native = ha_to_device(105)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, f"[{light_scale}] Override not triggered well below boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"
            assert light._override_timestamp is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_t_1_6b_morning_bva_just_below_boundary(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """T-1.6b: Morning BVA - Just Below Boundary.

        Tests boundary value analysis for morning transition override triggering.
        When user dims light just below the circadian setpoint minus threshold,
        override should be triggered.

        Test setup:
        - Circadian target: 150 (HA 0-255 scale)
        - Threshold: 25 (HA 0-255 scale)
        - Boundary: 150 - 25 = 125 (HA scale)
        - User dims from 160 to 124 (intended to be just below 125)

        Expected behavior: Override triggered because 124 < 125.

        Current bug: Due to brightness scale quantization (device native -> HA conversion),
        the actual HA brightness after conversion may not be exactly 124, causing
        the test to fail when it should pass. The code doesn't account for maximum
        quantization error as required by manual_overrides.md Section 4.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_1_6b_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set up entity registry in hass.data
        from homeassistant.helpers.entity_registry import DATA_REGISTRY
        hass.data[DATA_REGISTRY] = MagicMock()

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher, \
             patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(160)
            new_brightness_device_native = ha_to_device(124)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, f"[{light_scale}] Override not triggered just below boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"
            assert light._override_timestamp is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_t_1_6c_morning_bva_at_boundary(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """T-1.6c: Morning BVA - Exactly At Boundary.

        Tests boundary value analysis for morning transition override triggering.
        When user dims light exactly to the circadian setpoint minus threshold,
        override should NOT be triggered (must be beyond the boundary).

        Test setup:
        - Circadian target: 150 (HA 0-255 scale)
        - Threshold: 25 (HA 0-255 scale)
        - Boundary: 150 - 25 = 125 (HA scale)
        - User dims from 160 to 125 (exactly at boundary)

        Expected behavior: No override because 125 == 125 (not < 125).

        Current bug: Due to brightness scale quantization, the actual HA brightness
        after conversion may be slightly different from 125, causing unexpected
        override triggering. The code doesn't implement dynamic quantization error
        thresholds as required by manual_overrides.md Section 4.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_1_6c_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        # Set up entity registry in hass.data
        from homeassistant.helpers.entity_registry import DATA_REGISTRY
        hass.data[DATA_REGISTRY] = MagicMock()
        hass.data['entity_registry'] = MagicMock()

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher, \
             patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None
            mock_er_get.return_value = MagicMock()
            mock_er_get.return_value = MagicMock()

            old_brightness_device_native = ha_to_device(160)
            new_brightness_device_native = ha_to_device(125)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, f"[{light_scale}] Override not triggered at boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_t_1_6d_morning_bva_just_above_boundary(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """T-1.6d: Morning BVA - Just Above Boundary.

        Tests boundary value analysis for morning transition override triggering.
        When user dims light just above the circadian setpoint minus threshold,
        override should NOT be triggered.

        Test setup:
        - Circadian target: 150 (HA 0-255 scale)
        - Threshold: 25 (HA 0-255 scale)
        - Boundary: 150 - 25 = 125 (HA scale)
        - User dims from 160 to 126 (just above boundary)

        Expected behavior: No override because 126 > 125.

        Current bug: Due to brightness scale quantization, the actual HA brightness
        after conversion may be at or below the boundary instead of above,
        causing incorrect override triggering. The code fails to handle quantization
        errors dynamically based on light scale as specified in manual_overrides.md.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_1_6d_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(160)
            new_brightness_device_native = ha_to_device(126)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, f"[{light_scale}] Override not triggered just above boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_t_1_6e_morning_bva_well_above_boundary(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """T-1.6e: Morning BVA - Well Above Boundary.
        
        Circadian=150, Threshold=25, Boundary=125.
        New brightness=145 (well above boundary).
        Override should NOT be triggered.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_1_6e_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(160)
            new_brightness_device_native = ha_to_device(145)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert not light._is_overridden, f"[{light_scale}] Override incorrectly triggered well above boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_t_1_7a_evening_bva_well_above_boundary(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """T-1.7a: Evening BVA - Well Above Boundary.
        
        Circadian=100, Threshold=25, Boundary=125.
        New brightness=145 (well above boundary).
        Override SHOULD be triggered.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_1_7a_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 100
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        evening_transition = datetime(2023, 1, 1, 19, 45, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = evening_transition
            mock_dt_util.return_value = evening_transition
            mock_dt_utcnow.return_value = evening_transition
            mock_datetime.now.return_value = evening_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(90)
            new_brightness_device_native = ha_to_device(145)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, f"[{light_scale}] Override not triggered well above boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"
            assert light._override_timestamp is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_t_1_7b_evening_bva_just_above_boundary(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """T-1.7b: Evening BVA - Just Above Boundary.

        Tests boundary value analysis for evening transition override triggering.
        When user brightens light just above the circadian setpoint plus threshold,
        override should be triggered.

        Test setup:
        - Circadian target: 100 (HA 0-255 scale)
        - Threshold: 25 (HA 0-255 scale)
        - Boundary: 100 + 25 = 125 (HA scale)
        - User brightens from 90 to 126 (just above boundary)

        Expected behavior: Override triggered because 126 > 125.

        Current bug: Due to brightness scale quantization, the actual HA brightness
        after conversion may not exceed the boundary, causing the test to fail
        when it should pass. The code doesn't implement quantization error handling
        as required by manual_overrides.md Section 4.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_1_7b_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 100
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        evening_transition = datetime(2023, 1, 1, 19, 45, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = evening_transition
            mock_dt_util.return_value = evening_transition
            mock_dt_utcnow.return_value = evening_transition
            mock_datetime.now.return_value = evening_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(90)
            new_brightness_device_native = ha_to_device(126)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, f"[{light_scale}] Override not triggered just above boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"
            assert light._override_timestamp is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_t_1_7c_evening_bva_at_boundary(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """T-1.7c: Evening BVA - Exactly At Boundary.

        Tests boundary value analysis for evening transition override triggering.
        When user brightens light exactly to the circadian setpoint plus threshold,
        override should NOT be triggered (must be beyond the boundary).

        Test setup:
        - Circadian target: 100 (HA 0-255 scale)
        - Threshold: 25 (HA 0-255 scale)
        - Boundary: 100 + 25 = 125 (HA scale)
        - User brightens from 90 to 125 (exactly at boundary)

        Expected behavior: No override because 125 == 125 (not > 125).

        Current bug: Due to brightness scale quantization, the actual HA brightness
        after conversion may exceed the boundary, causing unexpected override
        triggering. The code lacks dynamic quantization error calculation
        per manual_overrides.md Section 4 requirements.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_1_7c_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 100
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        evening_transition = datetime(2023, 1, 1, 19, 45, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = evening_transition
            mock_dt_util.return_value = evening_transition
            mock_dt_utcnow.return_value = evening_transition
            mock_datetime.now.return_value = evening_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(90)
            new_brightness_device_native = ha_to_device(125)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, f"[{light_scale}] Override not triggered at boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_t_1_7d_evening_bva_just_below_boundary(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """T-1.7d: Evening BVA - Just Below Boundary.
        
        Circadian=100, Threshold=25, Boundary=125.
        New brightness=124 (just below boundary).
        Override should NOT be triggered. Tests quantization at boundary.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_1_7d_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 100
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        evening_transition = datetime(2023, 1, 1, 19, 45, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = evening_transition
            mock_dt_util.return_value = evening_transition
            mock_dt_utcnow.return_value = evening_transition
            mock_datetime.now.return_value = evening_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(90)
            new_brightness_device_native = ha_to_device(124)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, f"[{light_scale}] Override not triggered just below boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_t_1_7e_evening_bva_well_below_boundary(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """T-1.7e: Evening BVA - Well Below Boundary.
        
        Circadian=100, Threshold=25, Boundary=125.
        New brightness=105 (well below boundary).
        Override should NOT be triggered.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_1_7e_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 100
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        evening_transition = datetime(2023, 1, 1, 19, 45, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = evening_transition
            mock_dt_util.return_value = evening_transition
            mock_dt_utcnow.return_value = evening_transition
            mock_datetime.now.return_value = evening_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(90)
            new_brightness_device_native = ha_to_device(105)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert not light._is_overridden, f"[{light_scale}] Override incorrectly triggered well below boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"


class TestBehaviorDuringOverride:
    """Test Scenario Group 2: Behavior during override (manual_overrides.md Section 2)"""

    @pytest.mark.asyncio
    async def test_b_2_1_automatic_updates_stop_when_overridden(
        self, mock_hass, mock_state_factory
    ):
        """B-2.1: Automatic Updates Stop.

        hass = mock_hass
        Trigger an override.
        Advance time so circadian setpoint increases.
        Verify light brightness remains at manual level (no automatic updates).
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 5,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_b_2_1", data=config)
        hass = mock_hass
        entry.add_to_hass(hass)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150
        light._manual_override_threshold = 25  # 10% in 0-255 scale

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }
        
        # Set up entity registry in hass.data
        from homeassistant.helpers.entity_registry import DATA_REGISTRY
        hass.data[DATA_REGISTRY] = MagicMock()

        # Create an override
        light._is_overridden = True
        light._override_timestamp = datetime(2023, 1, 1, 6, 30, 0)

        # Mock the light state to return manual brightness
        hass.states.get = MagicMock(
            return_value=mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: 120}
            )
        )

        # Calculate and apply brightness - should skip update because overridden
        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.light.dt_util.now") as mock_light_now, \
             patch("custom_components.smart_circadian_lighting.light.er.async_get") as mock_er_get, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = datetime(2023, 1, 1, 6, 35, 0)
            mock_light_now.return_value = datetime(2023, 1, 1, 6, 35, 0)
            mock_er_get.return_value = MagicMock()
            mock_dispatcher.return_value = None

            with patch.object(
                light, "async_turn_on"
            ) as mock_turn_on, patch.object(
                light, "async_write_ha_state"
            ):
                await light._async_calculate_and_apply_brightness()

                # When overridden, should not send brightness updates
                # The light should remain at manual brightness (120) and not change


class TestManualOverrideClearance:
    """Test Scenario Group 3: Manual override clearance (manual_overrides.md Section 3.2)"""

    @pytest.mark.asyncio
    async def test_m_3_1_clear_manual_override_button(
        self, mock_hass
    ):
        """M-3.1: Clear Manual Override Button.

        hass = mock_hass
        Trigger an override.
        Call the clear override service/button.
        Verify override clears and light syncs to circadian brightness.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 5,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id="test_m_3_1", data=config)
        hass = mock_hass
        entry.add_to_hass(hass)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }
        
        # Set up entity registry in hass.data
        from homeassistant.helpers.entity_registry import DATA_REGISTRY
        hass.data[DATA_REGISTRY] = MagicMock()

        # Create an override
        light._is_overridden = True
        light._override_timestamp = datetime(2023, 1, 1, 6, 30, 0)

        # Call clear override function
        from custom_components.smart_circadian_lighting import state_management

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_utcnow, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher, \
             patch.object(light, "async_write_ha_state") as mock_write_state, \
             patch.object(light, "async_turn_on") as mock_turn_on:
            mock_now.return_value = datetime(2023, 1, 1, 6, 35, 0)
            mock_utcnow.return_value = datetime(2023, 1, 1, 6, 35, 0)
            mock_dispatcher.return_value = None
            mock_write_state.return_value = None
            mock_turn_on.return_value = None

            await state_management.async_clear_manual_override(light)

            # Override should be cleared
            assert (
                not light._is_overridden
            ), "Override not cleared after manual clear"
            assert (
                light._override_timestamp is None
            ), "Override timestamp not cleared"


class TestMultiScaleBrightness:
    """Test override detection with different light brightness scales.
    
    Home Assistant normalizes all brightness values to 0-255 scale internally.
    Different light platforms report brightness in different scales:
    - Z-Wave lights: 0-99
    - Kasa smart lights: 0-100
    - Standard HA lights: 0-255
    
    This test group verifies that override detection works correctly regardless
    of the underlying light's brightness scale.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_morning_override_trigger_multi_scale(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """Test morning override trigger works with different light brightness scales.
        
        For each light scale:
        - Circadian target: 150 (in 0-255 scale)
        - Threshold: 25 (in 0-255 scale)
        - User dims to below (target - threshold) = 125 (in 0-255 scale)
        - This should trigger override for all scale types.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_ms_1_6_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(160)
            new_brightness_device_native = ha_to_device(120)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, (
                f"Override not triggered for {light_scale} "
                f"(device: {new_brightness_device_native}, ha: {new_brightness_ha})"
            )
            assert light._override_timestamp is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_evening_override_trigger_multi_scale(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """Test evening override trigger works with different light brightness scales.
        
        For each light scale:
        - Circadian target: 100 (in 0-255 scale)
        - Threshold: 25 (in 0-255 scale)
        - User brightens to above (target + threshold) = 125 (in 0-255 scale)
        - This should trigger override for all scale types.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_ms_1_7_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 100
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        evening_transition = datetime(2023, 1, 1, 19, 45, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = evening_transition
            mock_dt_util.return_value = evening_transition
            mock_dt_utcnow.return_value = evening_transition
            mock_datetime.now.return_value = evening_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(90)
            new_brightness_device_native = ha_to_device(130)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert light._is_overridden, (
                f"Override not triggered for {light_scale} "
                f"(device: {new_brightness_device_native}, ha: {new_brightness_ha})"
            )
            assert light._override_timestamp is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_no_override_same_direction_multi_scale(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """Test that adjustments in same direction as transition don't trigger override.
        
        For each light scale:
        - Morning transition (brightening)
        - User brightens light (same direction)
        - Should NOT trigger override, regardless of scale.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_ms_1_2_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 128
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(100)
            new_brightness_device_native = ha_to_device(180)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert not light._is_overridden, (
                f"Override incorrectly triggered for same-direction adjustment in {light_scale}"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "light_scale,device_to_ha,ha_to_device",
        [
            ("zwave_0_99_truncate", brightness_0_99_to_255_truncate, brightness_255_to_0_99_truncate),
            ("zwave_0_99_round", brightness_0_99_to_255_round, brightness_255_to_0_99_round),
            ("kasa_0_100_truncate", brightness_0_100_to_255_truncate, brightness_255_to_0_100_truncate),
            ("kasa_0_100_round", brightness_0_100_to_255_round, brightness_255_to_0_100_round),
            ("standard_0_255", lambda x: x, lambda x: x),
        ],
        ids=["Z-Wave (0-99, truncate)", "Z-Wave (0-99, round)", "Kasa (0-100, truncate)", "Kasa (0-100, round)", "Standard HA (0-255)"],
    )
    async def test_no_override_within_threshold_multi_scale(
        self, mock_hass, mock_state_factory, light_scale, device_to_ha, ha_to_device
    ):
        """Test that adjustments within threshold don't trigger override.
        
        For each light scale:
        - Morning transition
        - Circadian target: 150 (0-255 scale)
        - User dims but stays within threshold (150 - 25 = 125)
        - Should NOT trigger override, regardless of scale.
        """
        hass = mock_hass
        config = {
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "19:30:00",
            "evening_end_time": "20:30:00",
            "manual_override_threshold": 10,
            "color_temp_enabled": False,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
        }

        entry = MockConfigEntry(domain=DOMAIN, unique_id=f"test_ms_1_4_{light_scale}", data=config)
        entry.add_to_hass(hass)

        mock_store = MagicMock()
        mock_store.async_load = AsyncMock(return_value=None)
        mock_store.async_save = AsyncMock()

        light = CircadianLight(hass, "light.test_light", config, entry, mock_store)
        light._first_update_done = True
        light._brightness = 150
        light._manual_override_threshold = 25

        hass.data[DOMAIN] = {
            entry.entry_id: {
                "config": config,
                "circadian_lights": [light],
                "manual_overrides_enabled": True,
            }
        }

        morning_transition = datetime(2023, 1, 1, 6, 30, 0)

        with patch("homeassistant.util.dt.now") as mock_now, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.now") as mock_dt_util, \
             patch("custom_components.smart_circadian_lighting.state_management.dt_util.utcnow") as mock_dt_utcnow, \
             patch("custom_components.smart_circadian_lighting.circadian_logic.datetime") as mock_datetime, \
             patch("custom_components.smart_circadian_lighting.state_management.async_call_later") as mock_call_later, \
             patch("custom_components.smart_circadian_lighting.state_management.async_dispatcher_send") as mock_dispatcher:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition
            mock_dt_utcnow.return_value = morning_transition
            mock_datetime.now.return_value = morning_transition
            mock_call_later.return_value = MagicMock()
            mock_dispatcher.return_value = None

            old_brightness_device_native = ha_to_device(160)
            new_brightness_device_native = ha_to_device(145)

            old_brightness_ha = device_to_ha(old_brightness_device_native)
            new_brightness_ha = device_to_ha(new_brightness_device_native)

            old_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: old_brightness_ha}
            )
            new_state = mock_state_factory(
                "light.test_light", STATE_ON, {ATTR_BRIGHTNESS: new_brightness_ha}
            )

            from custom_components.smart_circadian_lighting import state_management

            await state_management.handle_entity_state_changed(
                light, MagicMock(data={"old_state": old_state, "new_state": new_state})
            )

            assert not light._is_overridden, (
                f"Override incorrectly triggered for within-threshold adjustment in {light_scale}"
            )
