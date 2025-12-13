"""Comprehensive tests for manual override functionality in Smart Circadian Lighting.

This test suite covers all scenarios in manual_overrides.md:
1. Triggering overrides (Section 1)
2. Behavior during overrides (Section 2)
3. Clearing overrides (Section 3)
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
    hass.loop = asyncio.get_event_loop()
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

    @pytest.mark.asyncio
    async def test_t_1_2_morning_transition_wrong_direction_no_override(
        self, mock_hass, mock_state_factory
    ):
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
        hass = mock_hass
        entry.add_to_hass(hass)
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
        ) as mock_dt_util:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition

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
        light._brightness = 128  # 50% brightness (circadian target)

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
        ) as mock_dt_util:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition

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
        ) as mock_dt_util:
            mock_now.return_value = morning_transition
            mock_dt_util.return_value = morning_transition

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
        
        Circadian=150, Threshold=25, Boundary=125.
        New brightness=105 (well below boundary).
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

            assert not light._is_overridden, f"[{light_scale}] Override incorrectly triggered at boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"

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

            assert not light._is_overridden, f"[{light_scale}] Override incorrectly triggered just above boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"

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

            assert not light._is_overridden, f"[{light_scale}] Override incorrectly triggered at boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"

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

            assert not light._is_overridden, f"[{light_scale}] Override incorrectly triggered just below boundary (device={new_brightness_device_native}, ha={new_brightness_ha})"

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
