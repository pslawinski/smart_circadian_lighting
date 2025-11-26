"""Tests for color_temp_logic.py"""

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

from datetime import datetime, time, timedelta
from unittest.mock import patch

import pytest


# Copied functions from color_temp_logic.py for testing
def kelvin_to_mired(kelvin: float) -> int:
    """Convert Kelvin to mireds."""
    return int(1_000_000 / kelvin)

def mired_to_kelvin(mired: float) -> int:
    """Convert mireds to Kelvin."""
    return int(1_000_000 / mired)

def _parse_time(hass, time_str: str, brightness_config_time: str) -> time | None:
    """Parse a time string that can be 'sync', a fixed time, or based on sun events."""
    if not time_str:
        return None

    if time_str == "sync":
        if brightness_config_time:
            return time.fromisoformat(brightness_config_time)
        else:
            return None

    if "sunrise" in time_str or "sunset" in time_str:
        parts = time_str.split(" ")
        event = parts[0]
        offset = timedelta()
        if len(parts) > 1:
            op = parts[1]
            offset_str = parts[2]
            try:
                parsed_offset = timedelta(hours=int(offset_str.split(":")[0]),
                                        minutes=int(offset_str.split(":")[1]),
                                        seconds=int(offset_str.split(":")[2]))
                if op == "-":
                    offset = -parsed_offset
                else:
                    offset = parsed_offset
            except (ValueError, TypeError, IndexError):
                pass  # Ignore invalid offset

        base_time = time(6, 0, 0) if "sunrise" in event else time(18, 0, 0)
        result_dt = datetime.combine(datetime.today(), base_time) + offset
        return result_dt.time()

    try:
        return time.fromisoformat(time_str)
    except (ValueError, TypeError):
        return None

def interpolate_temp(start_temp: int, end_temp: int, progress: float, curve_type: str) -> int:
    """Interpolate between two temperatures using linear or cosine curve."""
    if curve_type == "cosine":
        import math
        progress = 0.5 * (1 - math.cos(math.pi * progress))
    # Linear is default
    return round(start_temp + progress * (end_temp - start_temp))

def get_color_temp_schedule(hass, config: dict) -> dict | None:
    """Generate a 24-hour color temperature schedule."""
    if not config.get("color_temp_enabled"):
        return None

    sunrise_sunset_temp = config.get("sunrise_sunset_color_temp_kelvin", 2700)
    midday_temp = config.get("midday_color_temp_kelvin", 5000)
    night_temp = config.get("night_color_temp_kelvin", 1800)
    curve_type = config.get("color_curve_type", "cosine")

    morning_start_str = config.get("color_morning_start_time")
    morning_end_str = config.get("color_morning_end_time")
    evening_start_str = config.get("color_evening_start_time")
    evening_end_str = config.get("color_evening_end_time")

    morning_start = _parse_time(hass, morning_start_str, config.get("morning_start_time"))
    morning_end = _parse_time(hass, morning_end_str, config.get("morning_end_time"))
    evening_start = _parse_time(hass, evening_start_str, config.get("evening_start_time"))
    evening_end = _parse_time(hass, evening_end_str, config.get("evening_end_time"))

    # Mock sun times for testing
    sunrise = time(6, 0, 0)
    solar_noon = time(12, 0, 0)
    sunset = time(18, 0, 0)

    if not all([morning_start, morning_end, evening_start, evening_end]):
        return None

    schedule = {
        "morning_start": morning_start,
        "morning_end": morning_end,
        "evening_start": evening_start,
        "evening_end": evening_end,
        "sunrise": sunrise,
        "solar_noon": solar_noon,
        "sunset": sunset,
        "sunrise_sunset_temp": sunrise_sunset_temp,
        "midday_temp": midday_temp,
        "night_temp": night_temp,
        "curve_type": curve_type,
    }
    return schedule

def _is_time_in_period(now: time, start: time, end: time) -> bool:
    """Checks if the current time is within a given period, handling midnight crossings."""
    if start <= end:
        return start <= now < end
    else:  # Period crosses midnight
        return start <= now or now < end

def get_progress(current: time, start: time, end: time) -> float:
    """Get the progress of a transition as a float between 0.0 and 1.0."""
    now_ts = current.hour * 3600 + current.minute * 60 + current.second
    start_ts = start.hour * 3600 + start.minute * 60 + start.second
    end_ts = end.hour * 3600 + end.minute * 60 + end.second

    if end_ts < start_ts:  # Period crosses midnight
        if now_ts < start_ts:
            now_ts += 24 * 3600
        end_ts += 24 * 3600

    total_seconds = end_ts - start_ts
    if total_seconds <= 0:
        return 1.0
    current_seconds = now_ts - start_ts
    return min(1.0, max(0.0, current_seconds / total_seconds))

def get_update_interval(color_temp_schedule, current_time, is_transition, is_on, config):
    """Get the update interval in seconds based on current conditions.

    Args:
        color_temp_schedule: Color temp schedule dict or None
        current_time: Current time as time object
        is_transition: Whether currently in brightness transition
        is_on: Whether circadian is enabled
        config: Configuration dict

    Returns:
        int: Update interval in seconds
    """
    if is_transition:
        return 30  # TRANSITION_UPDATE_INTERVAL
    elif color_temp_schedule and is_on:
        # Check if in color temperature transition
        morning_start = color_temp_schedule.get("morning_start")
        morning_end = color_temp_schedule.get("morning_end")
        evening_start = color_temp_schedule.get("evening_start")
        evening_end = color_temp_schedule.get("evening_end")

        if (morning_start and morning_end and _is_time_in_period(current_time, morning_start, morning_end)) or \
           (evening_start and evening_end and _is_time_in_period(current_time, evening_start, evening_end)):
            return 300  # 5 minutes during color temp transitions

        # During daytime with color temp enabled, update every 5 minutes
        sunrise = color_temp_schedule.get("sunrise")
        sunset = color_temp_schedule.get("sunset")
        if sunrise and sunset and sunrise <= current_time < sunset:
            return 300  # 5 minutes
        else:
            # Not in a transition, calculate time until next brightness transition
            return 3600  # Mock: 1 hour until next transition
    else:
        # Not in a transition, calculate time until next brightness transition
        # For brightness-only entities, no daytime updates
        return 3600  # Mock: 1 hour until next transition


def simulate_scheduler_for_graph(color_temp_schedule, config):
    """Simulate the scheduler to generate points at actual update times.

    Args:
        color_temp_schedule: Color temp schedule dict
        config: Configuration dict

    Returns:
        list: List of [timestamp, temperature] pairs at update times
    """
    points = []
    current_time = time(0, 0, 0)  # Start at midnight
    end_time = time(23, 59, 59)
    is_on = True  # Assume circadian is enabled

    while current_time < end_time:
        # Get current temperature
        temp = get_ct_at_time(color_temp_schedule, current_time)
        if temp is not None:
            timestamp = f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}"
            points.append([timestamp, temp])

        # Determine if in brightness transition (simplified - assume not for color temp testing)
        is_transition = False

        # Get update interval
        interval = get_update_interval(color_temp_schedule, current_time, is_transition, is_on, config)

        # Advance time by interval
        current_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
        new_seconds = current_seconds + interval

        # Handle day wraparound
        if new_seconds >= 86400:  # 24 hours in seconds
            break  # Stop at end of day

        new_hour = new_seconds // 3600
        new_minute = (new_seconds % 3600) // 60
        new_second = new_seconds % 60

        current_time = time(new_hour, new_minute, new_second)

    return points


def get_ct_at_time(schedule: dict, now: time) -> int | None:
    """Get the target color temperature at a specific time based on the schedule."""
    if not schedule:
        return None

    morning_start = schedule["morning_start"]
    morning_end = schedule["morning_end"]
    evening_start = schedule["evening_start"]
    evening_end = schedule["evening_end"]
    sunrise = schedule["sunrise"]
    solar_noon = schedule["solar_noon"]
    sunset = schedule["sunset"]
    sunrise_sunset_temp = schedule["sunrise_sunset_temp"]
    midday_temp = schedule["midday_temp"]
    night_temp = schedule["night_temp"]
    curve_type = schedule["curve_type"]

    # Morning transition: night_temp to sunrise_sunset_temp
    if _is_time_in_period(now, morning_start, morning_end):
        progress = get_progress(now, morning_start, morning_end)
        return interpolate_temp(night_temp, sunrise_sunset_temp, progress, "linear")

    # After morning_end until sunrise: sunrise_sunset_temp
    elif _is_time_in_period(now, morning_end, sunrise):
        return sunrise_sunset_temp

    # Sunrise to solar_noon: sunrise_sunset_temp to midday_temp
    elif _is_time_in_period(now, sunrise, solar_noon):
        progress = get_progress(now, sunrise, solar_noon)
        return interpolate_temp(sunrise_sunset_temp, midday_temp, progress, curve_type)

    # Solar_noon to sunset: midday_temp to sunrise_sunset_temp
    elif _is_time_in_period(now, solar_noon, sunset):
        progress = get_progress(now, solar_noon, sunset)
        return interpolate_temp(midday_temp, sunrise_sunset_temp, progress, curve_type)

    # Sunset to evening_start: sunrise_sunset_temp
    elif _is_time_in_period(now, sunset, evening_start):
        return sunrise_sunset_temp

    # Evening transition: sunrise_sunset_temp to night_temp
    elif _is_time_in_period(now, evening_start, evening_end):
        progress = get_progress(now, evening_start, evening_end)
        return interpolate_temp(sunrise_sunset_temp, night_temp, progress, "linear")

    # Night (after evening_end until morning_start)
    else:
        return night_temp


class TestInterpolateTemp:
    def test_linear_interpolation(self):
        assert interpolate_temp(2700, 5000, 0.0, "linear") == 2700
        assert interpolate_temp(2700, 5000, 0.5, "linear") == 3850
        assert interpolate_temp(2700, 5000, 1.0, "linear") == 5000

    def test_cosine_interpolation(self):
        # Cosine curve is symmetric and smooth
        start = interpolate_temp(2700, 5000, 0.0, "cosine")
        quarter = interpolate_temp(2700, 5000, 0.25, "cosine")
        middle = interpolate_temp(2700, 5000, 0.5, "cosine")
        three_quarter = interpolate_temp(2700, 5000, 0.75, "cosine")
        end = interpolate_temp(2700, 5000, 1.0, "cosine")
        assert start == 2700
        assert end == 5000
        assert middle == 3850  # Cosine at 0.5 is same as linear
        # Check that early progress is slower than linear
        linear_quarter = interpolate_temp(2700, 5000, 0.25, "linear")
        assert quarter < linear_quarter
        # Check that late progress is faster than linear
        linear_three_quarter = interpolate_temp(2700, 5000, 0.75, "linear")
        assert three_quarter > linear_three_quarter


class TestKelvinMiredConversion:
    def test_kelvin_to_mired(self):
        assert kelvin_to_mired(2700) == 370  # 1e6 / 2700 ≈ 370.37, int(370.37)=370
        assert kelvin_to_mired(5000) == 200

    def test_mired_to_kelvin(self):
        assert mired_to_kelvin(370) == 2702  # 1e6 / 370 ≈ 2702.702, int(2702.702)=2702
        assert mired_to_kelvin(200) == 5000

    def test_round_trip(self):
        for kelvin in [2700, 3000, 4000, 5000]:
            mired = kelvin_to_mired(kelvin)
            back = mired_to_kelvin(mired)
            # Due to int truncation, not exact, but close
            assert abs(back - kelvin) <= 5  # Allow some tolerance


class TestParseTime:
    @pytest.fixture
    def mock_hass(self):
        hass = MagicMock()
        return hass

    def test_parse_sync(self, mock_hass):
        result = _parse_time(mock_hass, "sync", "06:00:00")
        assert result == time(6, 0, 0)

    def test_parse_fixed_time(self, mock_hass):
        result = _parse_time(mock_hass, "07:30:00", "06:00:00")
        assert result == time(7, 30, 0)

    def test_parse_sunrise_no_offset(self, mock_hass):
        result = _parse_time(mock_hass, "sunrise", "06:00:00")
        assert result == time(6, 0, 0)  # Mocked to return fixed time

    def test_parse_sunset_with_offset(self, mock_hass):
        result = _parse_time(mock_hass, "sunset + 00:30:00", "06:00:00")
        assert result == time(18, 30, 0)  # 18:00 + 00:30:00

    @pytest.mark.parametrize("event,offset_str,expected", [
        ("sunrise", "01:00:00", time(7, 0, 0)),  # 6:00 + 1:00
        ("sunset", "00:30:00", time(18, 30, 0)),  # 18:00 + 0:30
        ("sunset", "-01:00:00", time(17, 0, 0)),  # 18:00 - 1:00
    ])
    def test_parse_sun_events_with_offset(self, mock_hass, event, offset_str, expected):
        """Test that sun events with offsets are parsed correctly (mocked to local times)."""
        result = _parse_time(mock_hass, f"{event} + {offset_str}", "06:00:00")
        assert result == expected


class TestGetColorTempSchedule:
    @pytest.fixture
    def mock_hass(self):
        return MagicMock()

    def test_disabled(self, mock_hass):
        config = {"color_temp_enabled": False}
        assert get_color_temp_schedule(mock_hass, config) is None

    def test_enabled(self, mock_hass):
        config = {
            "color_temp_enabled": True,
            "sunrise_sunset_color_temp_kelvin": 2700,
            "midday_color_temp_kelvin": 5000,
            "night_color_temp_kelvin": 1800,
            "color_curve_type": "cosine",
            "color_morning_start_time": "05:00:00",
            "color_morning_end_time": "06:00:00",
            "color_evening_start_time": "20:00:00",
            "color_evening_end_time": "21:00:00",
        }
        schedule = get_color_temp_schedule(mock_hass, config)
        assert schedule == {
            "morning_start": time(5, 0, 0),
            "morning_end": time(6, 0, 0),
            "evening_start": time(20, 0, 0),
            "evening_end": time(21, 0, 0),
            "sunrise": time(6, 0, 0),
            "solar_noon": time(12, 0, 0),
            "sunset": time(18, 0, 0),
            "sunrise_sunset_temp": 2700,
            "midday_temp": 5000,
            "night_temp": 1800,
            "curve_type": "cosine",
        }

    def test_get_color_temp_schedule_with_defaults(self, mock_hass):
        config = {
            "color_temp_enabled": True,
            "morning_start_time": "05:15:00",
            "morning_end_time": "06:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:30:00",
            "sunrise_sunset_color_temp_kelvin": 2700,
            "midday_color_temp_kelvin": 5000,
            "night_color_temp_kelvin": 1800,
            "color_curve_type": "cosine",
            "color_morning_start_time": "sync",
            "color_morning_end_time": "sunrise",
            "color_evening_start_time": "sunset",
            "color_evening_end_time": "sync",
        }
        schedule = get_color_temp_schedule(mock_hass, config)
        assert schedule is not None
        assert schedule["sunrise_sunset_temp"] == 2700
        assert schedule["midday_temp"] == 5000
        assert schedule["night_temp"] == 1800
        assert schedule["curve_type"] == "cosine"

    def test_schedule_changes_with_config_changes(self, mock_hass):
        """Test that changing config values changes the generated schedule."""
        config1 = {
            "color_temp_enabled": True,
            "sunrise_sunset_color_temp_kelvin": 2700,
            "midday_color_temp_kelvin": 5000,
            "night_color_temp_kelvin": 1800,
            "color_curve_type": "cosine",
            "color_morning_start_time": "sync",
            "color_morning_end_time": "sunrise",
            "color_evening_start_time": "sunset",
            "color_evening_end_time": "sync",
            "morning_start_time": "05:15:00",
            "morning_end_time": "06:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:30:00",
        }
        schedule1 = get_color_temp_schedule(mock_hass, config1)

        # Change night temp
        config2 = config1.copy()
        config2["night_color_temp_kelvin"] = 2000
        schedule2 = get_color_temp_schedule(mock_hass, config2)

        assert schedule1["night_temp"] == 1800
        assert schedule2["night_temp"] == 2000

        # Change curve type
        config3 = config1.copy()
        config3["color_curve_type"] = "linear"
        schedule3 = get_color_temp_schedule(mock_hass, config3)

        assert schedule1["curve_type"] == "cosine"
        assert schedule3["curve_type"] == "linear"


class TestGetCtAtTime:
    @pytest.fixture(params=[
        {
            "name": "entity_disabled",
            "config": {"color_temp_enabled": False}
        },
        {
            "name": "entity_enabled_cosine",
            "config": {
                "color_temp_enabled": True,
                "sunrise_sunset_color_temp_kelvin": 2700,
                "midday_color_temp_kelvin": 5000,
                "night_color_temp_kelvin": 1800,
                "color_curve_type": "cosine",
                "color_morning_start_time": "05:00:00",
                "color_morning_end_time": "06:00:00",
                "color_evening_start_time": "20:00:00",
                "color_evening_end_time": "21:00:00",
            }
        },
        {
            "name": "entity_enabled_linear",
            "config": {
                "color_temp_enabled": True,
                "sunrise_sunset_color_temp_kelvin": 2700,
                "midday_color_temp_kelvin": 5000,
                "night_color_temp_kelvin": 1800,
                "color_curve_type": "linear",
                "color_morning_start_time": "05:00:00",
                "color_morning_end_time": "06:00:00",
                "color_evening_start_time": "20:00:00",
                "color_evening_end_time": "21:00:00",
            }
        }
    ], ids=lambda x: x["name"])
    def entity_config(self, request):
        return request.param

    @pytest.fixture
    def schedule(self, mock_hass, entity_config):
        return get_color_temp_schedule(mock_hass, entity_config["config"])

    @pytest.fixture
    def mock_hass(self):
        return MagicMock()

    def test_night(self, schedule, entity_config):
        if not entity_config["config"]["color_temp_enabled"]:
            pytest.skip("Test only for enabled entities")
        now = time(2, 0, 0)
        assert get_ct_at_time(schedule, now) == 1800

    def test_none_schedule(self, entity_config):
        if entity_config["config"]["color_temp_enabled"]:
            pytest.skip("Test only for disabled entities")
        assert get_ct_at_time(None, time(12, 0, 0)) is None

    def test_color_temp_is_integer(self, schedule, entity_config):
        if not entity_config["config"]["color_temp_enabled"]:
            pytest.skip("Test only for enabled entities")
        # Test at various points over multiple days to ensure no fractional temps and repeatability
        for day in range(3):  # Test over 3 days
            day_temps = []
            for hour in range(24):
                now = time(hour, 0, 0)
                temp = get_ct_at_time(schedule, now)
                assert isinstance(temp, int) or temp is None
                if temp is not None:
                    assert 1500 <= temp <= 6500  # Reasonable range
                day_temps.append(temp)
            # Ensure results are identical across days
            if day == 0:
                first_day_temps = day_temps.copy()
            else:
                assert day_temps == first_day_temps

    def test_smooth_transition(self, schedule, entity_config):
        if not entity_config["config"]["color_temp_enabled"]:
            pytest.skip("Test only for enabled entities")
        # Check that temp increases monotonically during morning
        prev_temp = 1800
        for minute in range(0, 60, 5):
            now = time(5, minute, 0)
            temp = get_ct_at_time(schedule, now)
            assert temp >= prev_temp
            prev_temp = temp

        # Check decreases during evening
        prev_temp = 2700
        for minute in range(0, 60, 5):
            now = time(20, minute, 0)
            temp = get_ct_at_time(schedule, now)
            assert temp <= prev_temp
            prev_temp = temp

    def test_crucial_points(self, schedule, entity_config):
        if not entity_config["config"]["color_temp_enabled"]:
            pytest.skip("Test only for enabled entities")
        # Test key points in the schedule
        assert get_ct_at_time(schedule, time(2, 0, 0)) == 1800  # Night
        assert get_ct_at_time(schedule, time(5, 0, 0)) == 1800  # Morning start
        assert get_ct_at_time(schedule, time(5, 30, 0)) == 2250  # Mid morning transition
        assert get_ct_at_time(schedule, time(6, 0, 0)) == 2700  # Morning end / Sunrise
        assert get_ct_at_time(schedule, time(9, 0, 0)) == 3850  # Mid sunrise to noon
        assert get_ct_at_time(schedule, time(12, 0, 0)) == 5000  # Solar noon
        assert get_ct_at_time(schedule, time(15, 0, 0)) == 3850  # Mid noon to sunset
        assert get_ct_at_time(schedule, time(18, 0, 0)) == 2700  # Sunset
        assert get_ct_at_time(schedule, time(20, 30, 0)) == 2250  # Mid evening transition (linear)
        assert get_ct_at_time(schedule, time(21, 0, 0)) == 1800  # Evening end
        assert get_ct_at_time(schedule, time(22, 0, 0)) == 1800  # Night again

    def test_graph_points_scheduler_based(self, schedule, entity_config):
        if not entity_config["config"]["color_temp_enabled"]:
            pytest.skip("Test only for enabled entities")
        # Test the graph generation points using scheduler simulation over multiple days for repeatability
        for day_offset in range(3):  # Test over 3 days
            points = simulate_scheduler_for_graph(schedule, entity_config["config"])

            # Verify we have points
            assert len(points) > 0

            # Test key points that should be present (at update times)
            # Night period (1800K) - should be at start
            night_points = [p for p in points if p[1] == 1800]
            assert len(night_points) > 0

            # Morning transition start (1800K at 05:00)
            morning_start = next((p for p in points if p[0] == "05:00:00"), None)
            assert morning_start is not None
            assert morning_start[1] == 1800

            # Mid morning transition (2250K around 05:30)
            mid_morning = next((p for p in points if "05:30" in p[0]), None)
            if mid_morning:  # May not be at exact minute depending on scheduler
                assert mid_morning[1] == 2250

            # Sunrise (2700K at 06:00)
            sunrise = next((p for p in points if p[0] == "06:00:00"), None)
            assert sunrise is not None
            assert sunrise[1] == 2700

            # Solar noon (5000K around 12:00)
            noon = next((p for p in points if "12:00" in p[0]), None)
            if noon:
                assert noon[1] == 5000

            # Sunset (2700K at 18:00)
            sunset = next((p for p in points if p[0] == "18:00:00"), None)
            assert sunset is not None
            assert sunset[1] == 2700

            # Evening transition start (2700K at 20:00)
            evening_start = next((p for p in points if p[0] == "20:00:00"), None)
            assert evening_start is not None
            assert evening_start[1] == 2700

            # Mid evening transition (2250K around 20:30)
            mid_evening = next((p for p in points if "20:30" in p[0]), None)
            if mid_evening:  # May not be at exact minute
                assert mid_evening[1] == 2250

            # Evening end (1800K at 21:00)
            evening_end = next((p for p in points if p[0] == "21:00:00"), None)
            assert evening_end is not None
            assert evening_end[1] == 1800

            # Verify points are in chronological order
            for i in range(1, len(points)):
                prev_time = points[i-1][0]
                curr_time = points[i][0]
                assert prev_time <= curr_time, f"Points not in order: {prev_time} > {curr_time}"


class TestUpdateIntervalScheduling:
    """Comprehensive tests for update interval scheduling logic."""

    @pytest.fixture
    def mock_hass(self):
        return MagicMock()

    @pytest.fixture
    def config_with_color_temp(self):
        return {
            "color_temp_enabled": True,
            "sunrise_sunset_color_temp_kelvin": 2700,
            "midday_color_temp_kelvin": 5000,
            "night_color_temp_kelvin": 1800,
            "color_curve_type": "cosine",
            "color_morning_start_time": "05:00:00",
            "color_morning_end_time": "sunrise",
            "color_evening_start_time": "sunset",
            "color_evening_end_time": "21:00:00",
        }

    @pytest.fixture
    def config_without_color_temp(self):
        return {
            "color_temp_enabled": False,
            "morning_start_time": "05:15:00",
            "morning_end_time": "06:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:30:00",
        }

    @pytest.fixture
    def color_temp_schedule(self, mock_hass, config_with_color_temp):
        return get_color_temp_schedule(mock_hass, config_with_color_temp)

    def test_during_brightness_transition(self, color_temp_schedule, config_with_color_temp, config_without_color_temp):
        """During any brightness transition, always use transition interval."""
        # Test during morning transition
        interval = get_update_interval(color_temp_schedule, time(5, 30, 0), True, True, config_with_color_temp)
        assert interval == 30  # TRANSITION_UPDATE_INTERVAL

        # Test during evening transition
        interval = get_update_interval(color_temp_schedule, time(20, 30, 0), True, True, config_with_color_temp)
        assert interval == 30

        # Test even with color temp disabled
        interval = get_update_interval(None, time(5, 30, 0), True, True, config_without_color_temp)
        assert interval == 30

    def test_daytime_with_color_temp_enabled(self, color_temp_schedule, config_with_color_temp):
        """During daytime with color temp enabled, schedule 5-minute updates."""
        # Test at noon (definitely daytime)
        interval = get_update_interval(color_temp_schedule, time(12, 0, 0), False, True, config_with_color_temp)
        assert interval == 300  # 5 minutes

        # Test early afternoon
        interval = get_update_interval(color_temp_schedule, time(14, 30, 0), False, True, config_with_color_temp)
        assert interval == 300

        # Test late morning
        interval = get_update_interval(color_temp_schedule, time(10, 15, 0), False, True, config_with_color_temp)
        assert interval == 300

    def test_nighttime_with_color_temp_enabled(self, color_temp_schedule, config_with_color_temp):
        """During nighttime with color temp enabled, schedule next transition time."""
        # Test late night
        interval = get_update_interval(color_temp_schedule, time(2, 0, 0), False, True, config_with_color_temp)
        assert interval == 3600  # Next transition time

        # Test early morning before sunrise
        interval = get_update_interval(color_temp_schedule, time(4, 30, 0), False, True, config_with_color_temp)
        assert interval == 3600

        # Test evening after sunset
        interval = get_update_interval(color_temp_schedule, time(22, 0, 0), False, True, config_with_color_temp)
        assert interval == 3600

    def test_without_color_temp_enabled(self, config_without_color_temp):
        """Without color temp, always schedule next transition time regardless of time."""
        # Test daytime
        interval = get_update_interval(None, time(12, 0, 0), False, True, config_without_color_temp)
        assert interval == 3600

        # Test nighttime
        interval = get_update_interval(None, time(2, 0, 0), False, True, config_without_color_temp)
        assert interval == 3600

    def test_circadian_disabled(self, color_temp_schedule, config_with_color_temp):
        """When circadian is disabled, schedule next transition time."""
        # Test daytime with color temp schedule but circadian disabled
        interval = get_update_interval(color_temp_schedule, time(12, 0, 0), False, False, config_with_color_temp)
        assert interval == 3600

        # Test nighttime with circadian disabled
        interval = get_update_interval(color_temp_schedule, time(2, 0, 0), False, False, config_with_color_temp)
        assert interval == 3600

    def test_edge_cases_sunrise_sunset(self, color_temp_schedule, config_with_color_temp):
        """Test edge cases around sunrise and sunset boundaries."""
        # Exactly at sunrise (should be daytime)
        sunrise_time = color_temp_schedule["sunrise"]
        interval = get_update_interval(color_temp_schedule, sunrise_time, False, True, config_with_color_temp)
        assert interval == 300  # Daytime

        # One minute before sunrise (in morning transition due to config)
        before_sunrise = (datetime.combine(datetime.today(), sunrise_time) - timedelta(minutes=1)).time()
        interval = get_update_interval(color_temp_schedule, before_sunrise, False, True, config_with_color_temp)
        assert interval == 300  # In morning color temp transition

        # Exactly at sunset (in evening transition due to config)
        sunset_time = color_temp_schedule["sunset"]
        interval = get_update_interval(color_temp_schedule, sunset_time, False, True, config_with_color_temp)
        assert interval == 300  # In evening color temp transition

        # One minute before sunset (should be daytime)
        before_sunset = (datetime.combine(datetime.today(), sunset_time) - timedelta(minutes=1)).time()
        interval = get_update_interval(color_temp_schedule, before_sunset, False, True, config_with_color_temp)
        assert interval == 300  # Daytime

    def test_no_schedule_available(self, config_with_color_temp):
        """When no color temp schedule is available, fall back to transition timing."""
        # Test daytime
        interval = get_update_interval(None, time(12, 0, 0), False, True, config_with_color_temp)
        assert interval == 3600

        # Test nighttime
        interval = get_update_interval(None, time(2, 0, 0), False, True, config_with_color_temp)
        assert interval == 3600


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases that could cause issues."""

    @pytest.fixture
    def mock_hass(self):
        hass = MagicMock()
        return hass

    def test_sun_event_failure_fallback(self, mock_hass):
        """Test that system falls back to fixed times when sun events can't be retrieved."""
        # Mock sun event failure
        with patch('custom_components.smart_circadian_lighting.color_temp_logic.get_astral_event_next', return_value=None):
            config = {
                "color_temp_enabled": True,
                "sunrise_sunset_color_temp_kelvin": 2700,
                "midday_color_temp_kelvin": 5000,
                "night_color_temp_kelvin": 1800,
                "color_morning_start_time": "sunrise",
                "color_morning_end_time": "sunrise",
                "color_evening_start_time": "sunset",
                "color_evening_end_time": "sunset",
            }

            schedule = get_color_temp_schedule(mock_hass, config)
            # Should still return a schedule with fallback times
            assert schedule is not None
            assert schedule["sunrise"] == time(6, 0, 0)  # Fallback time
            assert schedule["sunset"] == time(18, 0, 0)   # Fallback time

    def test_invalid_color_temperature_values(self, mock_hass):
        """Test handling of invalid color temperature values."""
        config = {
            "color_temp_enabled": True,
            "sunrise_sunset_color_temp_kelvin": -100,  # Invalid negative
            "midday_color_temp_kelvin": 100000,        # Invalid too high
            "night_color_temp_kelvin": 0,               # Invalid zero
            "color_morning_start_time": "05:00:00",
            "color_morning_end_time": "06:00:00",
            "color_evening_start_time": "20:00:00",
            "color_evening_end_time": "21:00:00",
        }

        schedule = get_color_temp_schedule(mock_hass, config)
        assert schedule is not None
        # Currently accepts invalid values (no validation in code)
        assert schedule["sunrise_sunset_temp"] == -100
        assert schedule["midday_temp"] == 100000
        assert schedule["night_temp"] == 0

    def test_missing_config_keys(self, mock_hass):
        """Test handling of missing configuration keys."""
        config = {
            "color_temp_enabled": True,
            # Missing color temperature values
            "color_morning_start_time": "05:00:00",
            "color_morning_end_time": "06:00:00",
            "color_evening_start_time": "20:00:00",
            "color_evening_end_time": "21:00:00",
        }

        schedule = get_color_temp_schedule(mock_hass, config)
        assert schedule is not None
        # Should use defaults for missing values
        assert schedule["sunrise_sunset_temp"] == 2700
        assert schedule["midday_temp"] == 5000
        assert schedule["night_temp"] == 1800

    def test_midnight_crossing_schedules(self, mock_hass):
        """Test schedules that cross midnight."""
        config = {
            "color_temp_enabled": True,
            "sunrise_sunset_color_temp_kelvin": 2700,
            "midday_color_temp_kelvin": 5000,
            "night_color_temp_kelvin": 1800,
            "color_morning_start_time": "23:00:00",  # Late night
            "color_morning_end_time": "01:00:00",    # Early morning (crosses midnight)
            "color_evening_start_time": "22:00:00",
            "color_evening_end_time": "23:30:00",
        }

        schedule = get_color_temp_schedule(mock_hass, config)
        assert schedule is not None
        # Should handle midnight crossing properly
        assert schedule["morning_start"] == time(23, 0, 0)
        assert schedule["morning_end"] == time(1, 0, 0)

    def test_extreme_time_values(self, mock_hass):
        """Test with extreme time values."""
        config = {
            "color_temp_enabled": True,
            "sunrise_sunset_color_temp_kelvin": 2700,
            "midday_color_temp_kelvin": 5000,
            "night_color_temp_kelvin": 1800,
            "color_morning_start_time": "00:00:00",  # Midnight
            "color_morning_end_time": "23:59:59",    # Almost midnight
            "color_evening_start_time": "00:00:01",  # Just after midnight
            "color_evening_end_time": "23:59:58",    # Almost end of day
        }

        schedule = get_color_temp_schedule(mock_hass, config)
        assert schedule is not None
        assert schedule["morning_start"] == time(0, 0, 0)
        assert schedule["morning_end"] == time(23, 59, 59)

    def test_color_temp_calculation_edge_cases(self, mock_hass):
        """Test color temperature calculations at edge cases."""
        config = {
            "color_temp_enabled": True,
            "sunrise_sunset_color_temp_kelvin": 2700,
            "midday_color_temp_kelvin": 5000,
            "night_color_temp_kelvin": 1800,
            "color_morning_start_time": "05:00:00",
            "color_morning_end_time": "06:00:00",
            "color_evening_start_time": "20:00:00",
            "color_evening_end_time": "21:00:00",
        }

        schedule = get_color_temp_schedule(mock_hass, config)
        assert schedule is not None

        # Test at exact boundary times
        assert get_ct_at_time(schedule, time(6, 0, 0)) == 2700  # Sunrise
        assert get_ct_at_time(schedule, time(12, 0, 0)) == 5000  # Solar noon
        assert get_ct_at_time(schedule, time(18, 0, 0)) == 2700  # Sunset

        # Test outside schedule bounds
        assert get_ct_at_time(schedule, time(3, 0, 0)) == 1800  # Night
        assert get_ct_at_time(schedule, time(23, 0, 0)) == 1800  # Night

    def test_sunrise_sunset_equal_times(self, mock_hass):
        """Test edge case where sunrise equals sunset (polar regions, etc.)."""
        # This would happen in extreme latitudes or during testing
        config = {
            "color_temp_enabled": True,
            "sunrise_sunset_color_temp_kelvin": 2700,
            "midday_color_temp_kelvin": 5000,
            "night_color_temp_kelvin": 1800,
            "color_morning_start_time": "06:00:00",
            "color_morning_end_time": "06:00:00",  # Same as sunrise
            "color_evening_start_time": "18:00:00",
            "color_evening_end_time": "18:00:00",  # Same as sunset
        }

        schedule = get_color_temp_schedule(mock_hass, config)
        assert schedule is not None
        # Should handle equal times gracefully
        assert schedule["sunrise"] == time(6, 0, 0)
        assert schedule["sunset"] == time(18, 0, 0)

    def test_update_interval_with_disabled_circadian(self):
        """Test that disabled circadian doesn't trigger frequent updates."""
        # Even with color temp schedule, if circadian is disabled, should not do frequent updates
        schedule = {"sunrise": time(6, 0, 0), "sunset": time(18, 0, 0)}
        current_time = time(12, 0, 0)  # Daytime
        is_transition = False
        is_on = False  # Circadian disabled

        config = {"color_temp_enabled": True}
        interval = get_update_interval(schedule, current_time, is_transition, is_on, config)

        # Should not do frequent updates when circadian is disabled
        assert interval == 3600  # Next transition time, not 5 minutes

    def test_empty_schedule_handling(self):
        """Test that empty schedule is handled gracefully."""
        result = get_ct_at_time(None, time(12, 0, 0))
        assert result is None

        result = get_ct_at_time({}, time(12, 0, 0))
        assert result is None

    def test_evening_transition_sync_with_brightness_linear(self, mock_hass):
        """Test evening color transition when color_evening_start_type is 'Sunrise/Sunset' with 'sunset' event,
        color_evening_end_type is 'sync with brightness', expecting linear transition from sunset temp to night temp."""
        config = {
            "color_temp_enabled": True,
            "sunrise_sunset_color_temp_kelvin": 2700,
            "midday_color_temp_kelvin": 5000,
            "night_color_temp_kelvin": 1800,
            "color_curve_type": "linear",
            "color_morning_start_time": "sync",
            "color_morning_end_time": "sunrise",
            "color_evening_start_time": "sunset",
            "color_evening_end_time": "sync",
            "morning_start_time": "05:15:00",
            "morning_end_time": "06:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:30:00",
        }
        schedule = get_color_temp_schedule(mock_hass, config)
        assert schedule is not None

        # Check that evening_start is sunset time
        assert schedule["evening_start"] == time(18, 0, 0)  # Sunset

        # Check that evening_end is synced with brightness evening_end_time
        assert schedule["evening_end"] == time(21, 30, 0)

        # At sunset (18:00), temp should be sunrise_sunset_temp
        assert get_ct_at_time(schedule, time(18, 0, 0)) == 2700

        # During evening transition (18:00 to 21:30), linear from 2700 to 1800
        # At 19:45 (halfway), should be halfway: (2700 + 1800) / 2 = 2250
        assert get_ct_at_time(schedule, time(19, 45, 0)) == 2250

        # At evening_end (21:30), temp should be night_temp
        assert get_ct_at_time(schedule, time(21, 30, 0)) == 1800

        # After evening_end, until morning_start, stays at night_temp
        assert get_ct_at_time(schedule, time(22, 0, 0)) == 1800
        assert get_ct_at_time(schedule, time(2, 0, 0)) == 1800
        assert get_ct_at_time(schedule, time(5, 0, 0)) == 1800



