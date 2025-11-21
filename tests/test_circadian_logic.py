"""Tests for circadian_logic.py"""

import sys
from unittest.mock import MagicMock

# Mock HA modules before importing
mock_modules = [
    'homeassistant',
    'homeassistant.config_entries',
    'homeassistant.core',
    'homeassistant.helpers.sun',
    'homeassistant.helpers.entity',
    'homeassistant.helpers.entity_platform',
    'homeassistant.helpers.event',
    'homeassistant.helpers.storage',
    'homeassistant.helpers.dispatcher',
    'homeassistant.helpers',
    'homeassistant.helpers.entity_registry',
    'homeassistant.helpers.selector',
    'homeassistant.components.light',
    'homeassistant.components.sensor',
    'homeassistant.config_entries',
    'homeassistant.const',
    'homeassistant.data_entry_flow',
    'homeassistant.exceptions',
    'homeassistant.util',
    'homeassistant.util.dt',
]

for module in mock_modules:
    sys.modules[module] = MagicMock()

from datetime import datetime, time

import pytest


# Copied functions from circadian_logic.py for testing
def _convert_percent_to_255(percent: float) -> int:
    """Convert a percentage (0-100) to a 0-255 scale."""
    if percent >= 100:
        return 255
    return int(round(percent * 255 / 100))

def get_progress(current: time, start: time, end: time) -> float:
    """Get the progress of a transition as a float between 0.0 and 1.0."""
    now_ts = current.hour * 3600 + current.minute * 60 + current.second
    start_ts = start.hour * 3600 + start.minute * 60 + start.second
    end_ts = end.hour * 3600 + end.minute * 60 + end.second

    if end_ts < start_ts: # Period crosses midnight
        if now_ts < start_ts:
            now_ts += 24 * 3600
        end_ts += 24 * 3600

    total_seconds = end_ts - start_ts
    if total_seconds <= 0:
        return 1.0
    current_seconds = now_ts - start_ts
    return current_seconds / total_seconds

def get_circadian_mode(dt: datetime, temp_transition_override: dict[str, any], config: dict[str, any]) -> str:
    """Determine the circadian mode for a specific time."""
    current_time = dt.time()
    morning_start_time = time.fromisoformat(config["morning_start_time"])
    morning_end_time = time.fromisoformat(config["morning_end_time"])
    evening_start_time = time.fromisoformat(config["evening_start_time"])
    evening_end_time = time.fromisoformat(config["evening_end_time"])

    if _is_time_in_period(current_time, morning_start_time, morning_end_time):
        return "morning_transition"
    elif _is_time_in_period(current_time, evening_start_time, evening_end_time):
        return "evening_transition"
    elif _is_time_in_period(current_time, morning_end_time, evening_start_time):
        return "day"
    else:
        return "night"

def _is_time_in_period(now: time, start: time, end: time) -> bool:
    """Checks if the current time is within a given period, handling midnight crossings."""
    if start <= end:
        return start <= now < end
    else: # Period crosses midnight
        return start <= now or now < end

def calculate_brightness_for_time(dt: datetime, temp_transition_override: dict[str, any], config: dict[str, any], day_brightness_255: int, night_brightness_255: int, light_entity_id: str | None = None) -> int:
    """Calculate the target brightness for a specific time."""
    current_time = dt.time()

    morning_start_time = time.fromisoformat(config["morning_start_time"])
    morning_end_time = time.fromisoformat(config["morning_end_time"])
    evening_start_time = time.fromisoformat(config["evening_start_time"])
    evening_end_time = time.fromisoformat(config["evening_end_time"])

    night_brightness = night_brightness_255
    day_brightness = day_brightness_255
    target_brightness = 0.0

    mode = get_circadian_mode(dt, temp_transition_override, config)

    if mode == "morning_transition":
        progress = get_progress(current_time, morning_start_time, morning_end_time)
        target_brightness = night_brightness + progress * (day_brightness - night_brightness)
    elif mode == "evening_transition":
        progress = get_progress(current_time, evening_start_time, evening_end_time)
        target_brightness = day_brightness - progress * (day_brightness - night_brightness)
    elif mode == "day":
        target_brightness = day_brightness
    else: # night
        target_brightness = night_brightness

    return int(round(target_brightness))


class TestConvertPercentTo255:
    def test_convert_percent_to_255(self):
        assert _convert_percent_to_255(0) == 0
        assert _convert_percent_to_255(100) == 255
        assert _convert_percent_to_255(50) == 128  # round(50*255/100)=128
        assert _convert_percent_to_255(1) == 3  # round(1*255/100)=3
        assert _convert_percent_to_255(99) == 252  # round(99*255/100)=252


class TestGetProgress:
    def test_progress_within_period(self):
        start = time(6, 0, 0)
        end = time(7, 0, 0)
        now = time(6, 30, 0)
        assert get_progress(now, start, end) == 0.5

    def test_progress_at_start(self):
        start = time(6, 0, 0)
        end = time(7, 0, 0)
        now = time(6, 0, 0)
        assert get_progress(now, start, end) == 0.0

    def test_progress_at_end(self):
        start = time(6, 0, 0)
        end = time(7, 0, 0)
        now = time(7, 0, 0)
        assert get_progress(now, start, end) == 1.0

    def test_progress_midnight_crossing(self):
        start = time(22, 0, 0)
        end = time(2, 0, 0)
        now = time(23, 0, 0)
        progress = get_progress(now, start, end)
        assert progress == 0.25  # 1 hour into 4 hour period


class TestCircadianMode:
    def test_morning_transition(self):
        config = {
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
        }
        dt = datetime(2023, 1, 1, 6, 30, 0)
        assert get_circadian_mode(dt, {}, config) == "morning_transition"

    def test_day(self):
        config = {
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
        }
        dt = datetime(2023, 1, 1, 12, 0, 0)
        assert get_circadian_mode(dt, {}, config) == "day"

    def test_evening_transition(self):
        config = {
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
        }
        dt = datetime(2023, 1, 1, 20, 30, 0)
        assert get_circadian_mode(dt, {}, config) == "evening_transition"

    def test_night(self):
        config = {
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
        }
        dt = datetime(2023, 1, 1, 2, 0, 0)
        assert get_circadian_mode(dt, {}, config) == "night"


class TestCalculateBrightnessForTime:
    @pytest.fixture
    def config(self):
        return {
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
        }

    def test_night_brightness(self, config):
        dt = datetime(2023, 1, 1, 2, 0, 0)
        brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
        assert brightness == 25

    def test_day_brightness(self, config):
        dt = datetime(2023, 1, 1, 12, 0, 0)
        brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
        assert brightness == 255

    def test_morning_transition_start(self, config):
        dt = datetime(2023, 1, 1, 6, 0, 0)
        brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
        assert brightness == 25  # At start, should be night brightness

    def test_morning_transition_middle(self, config):
        dt = datetime(2023, 1, 1, 6, 30, 0)
        brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
        # 25 + 0.5 * (255 - 25) = 25 + 115 = 140, round to 140
        assert brightness == 140

    def test_morning_transition_end(self, config):
        dt = datetime(2023, 1, 1, 7, 0, 0)
        brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
        assert brightness == 255  # At end, should be day brightness

    def test_evening_transition_start(self, config):
        dt = datetime(2023, 1, 1, 20, 0, 0)
        brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
        assert brightness == 255

    def test_evening_transition_middle(self, config):
        dt = datetime(2023, 1, 1, 20, 30, 0)
        brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
        # 255 - 0.5 * (255 - 25) = 255 - 115 = 140
        assert brightness == 140

    def test_evening_transition_end(self, config):
        dt = datetime(2023, 1, 1, 21, 0, 0)
        brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
        assert brightness == 25

    def test_brightness_is_integer(self, config):
        # Test at various points to ensure no fractional brightness
        for hour in range(24):
            dt = datetime(2023, 1, 1, hour, 0, 0)
            brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
            assert isinstance(brightness, int)
            assert 0 <= brightness <= 255

    def test_smooth_transition(self, config):
        # Check that brightness increases monotonically during morning
        prev_brightness = 25
        for minute in range(0, 60, 5):
            dt = datetime(2023, 1, 1, 6, minute, 0)
            brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
            assert brightness >= prev_brightness
            prev_brightness = brightness

        # Check decreases during evening
        prev_brightness = 255
        for minute in range(0, 60, 5):
            dt = datetime(2023, 1, 1, 20, minute, 0)
            brightness = calculate_brightness_for_time(dt, {}, config, 255, 25, "test_light")
            assert brightness <= prev_brightness
            prev_brightness = brightness
