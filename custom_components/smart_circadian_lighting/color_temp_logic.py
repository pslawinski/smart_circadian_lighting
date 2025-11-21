"""Core logic for color temperature adjustments in Smart Circadian Lighting."""

import logging
from datetime import time, timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.sun import get_astral_event_next
from homeassistant.util import dt as dt_util

from .const import (
    CONF_COLOR_CURVE_TYPE,
    CONF_COLOR_EVENING_END_TIME,
    CONF_COLOR_EVENING_START_TIME,
    CONF_COLOR_MORNING_END_TIME,
    CONF_COLOR_MORNING_START_TIME,
    CONF_EVENING_END_TIME,
    CONF_EVENING_START_TIME,
    CONF_MIDDAY_COLOR_TEMP_KELVIN,
    CONF_MORNING_END_TIME,
    CONF_MORNING_START_TIME,
    CONF_NIGHT_COLOR_TEMP_KELVIN,
    CONF_SUNRISE_SUNSET_COLOR_TEMP_KELVIN,
)

# Ensure constant is defined to fix NameError
CONF_SUNRISE_SUNSET_COLOR_TEMP_KELVIN = "sunrise_sunset_color_temp_kelvin"

_LOGGER = logging.getLogger(__name__)


def kelvin_to_mired(kelvin: float) -> int:
    """Convert Kelvin to mireds."""
    return int(1_000_000 / kelvin)

def mired_to_kelvin(mired: float) -> int:
    """Convert mireds to Kelvin."""
    return int(1_000_000 / mired)

def _parse_time(
    hass: HomeAssistant,
    time_str: str,
    brightness_config_time: str,
    default_time_str: str,
) -> time | None:
    """Parse a time string that can be 'sync', a fixed time, or based on sun events."""
    if not time_str:
        return None

    if time_str == "sync":
        try:
            return dt_util.parse_time(brightness_config_time)
        except (ValueError, TypeError):
             _LOGGER.warning(f"Invalid brightness time '{brightness_config_time}' for sync, using default '{default_time_str}'")
             return dt_util.parse_time(default_time_str)

    if "sunrise" in time_str or "sunset" in time_str:
        parts = time_str.split(" ")
        event = parts[0] # "sunrise" or "sunset"
        offset = timedelta()
        if len(parts) > 1:
            op = parts[1] # "+" or "-"
            offset_str = parts[2] # "HH:MM:SS"
            try:
                parsed_offset = dt_util.parse_duration(offset_str)
                if op == "-":
                    offset = -parsed_offset
                else:
                    offset = parsed_offset
            except (ValueError, TypeError):
                _LOGGER.error(f"Invalid offset format in '{time_str}'")
                return None


        try:
            now = dt_util.now()
            # Calculate for today. We only care about the time part.
            event_today = get_astral_event_next(hass, event, now.replace(hour=0, minute=0, second=0, microsecond=0))
            if event_today:
                event_today = dt_util.as_local(event_today)
                return (event_today + offset).time()
            else:
                # Should not happen if sun integration is configured
                _LOGGER.warning(f"Could not get next sun event for '{event}'")
                return None
        except Exception as e:
            _LOGGER.error(f"Error getting sun-based time for '{time_str}': {e}")
            return None

    try:
        return dt_util.parse_time(time_str)
    except (ValueError, TypeError):
        _LOGGER.warning(f"Invalid time format '{time_str}' for color temperature, using default '{default_time_str}'")
        return dt_util.parse_time(default_time_str)

def get_color_temp_schedule(
    hass: HomeAssistant,
    config: dict[str, Any],
) -> dict[str, time | int] | None:
    """Generate a 24-hour color temperature schedule."""
    if not config.get("color_temp_enabled"):
        return None

    sunrise_sunset_temp = config.get("sunrise_sunset_color_temp_kelvin") or 2700
    midday_temp = config.get(CONF_MIDDAY_COLOR_TEMP_KELVIN) or 5000
    night_temp = config.get(CONF_NIGHT_COLOR_TEMP_KELVIN) or 1800
    curve_type = config.get(CONF_COLOR_CURVE_TYPE, "cosine")

    morning_start_str = config.get(CONF_COLOR_MORNING_START_TIME) or "sync"
    morning_end_str = config.get(CONF_COLOR_MORNING_END_TIME) or "06:00:00"
    evening_start_str = config.get(CONF_COLOR_EVENING_START_TIME) or "20:00:00"
    evening_end_str = config.get(CONF_COLOR_EVENING_END_TIME) or "sync"

    morning_start = _parse_time(
        hass,
        morning_start_str,
        config.get(CONF_MORNING_START_TIME, "05:15:00"),
        "05:15:00",
    )
    morning_end = _parse_time(
        hass,
        morning_end_str,
        config.get(CONF_MORNING_END_TIME, "06:00:00"),
        "06:00:00",
    )
    evening_start = _parse_time(
        hass,
        evening_start_str,
        config.get(CONF_EVENING_START_TIME, "20:00:00"),
        "20:00:00",
    )
    evening_end = _parse_time(
        hass,
        evening_end_str,
        config.get(CONF_EVENING_END_TIME, "21:30:00"),
        "21:30:00",
    )

    _LOGGER.debug(f"Parsed times: morning_start={morning_start}, morning_end={morning_end}, evening_start={evening_start}, evening_end={evening_end}")
    _LOGGER.debug(f"Temp values: sunrise_sunset={sunrise_sunset_temp}, midday={midday_temp}, night={night_temp}")

    # Get sunrise, solar_noon, sunset
    now = dt_util.now()
    try:
        sunrise = get_astral_event_next(hass, "sunrise", now.replace(hour=0, minute=0, second=0, microsecond=0))
        sunset = get_astral_event_next(hass, "sunset", now.replace(hour=0, minute=0, second=0, microsecond=0))
        sunrise_local = dt_util.as_local(sunrise) if sunrise else None
        sunset_local = dt_util.as_local(sunset) if sunset else None
        _LOGGER.debug(f"Sunrise UTC: {sunrise}, Local: {sunrise_local}")
        _LOGGER.debug(f"Sunset UTC: {sunset}, Local: {sunset_local}")
        if sunrise and sunset:
            # Convert to local time
            sunrise_local = dt_util.as_local(sunrise)
            sunset_local = dt_util.as_local(sunset)
            # Calculate solar_noon as midpoint between sunrise and sunset
            solar_noon_local = sunrise_local + (sunset_local - sunrise_local) / 2
            sunrise_time = sunrise_local.time()
            sunset_time = sunset_local.time()
            solar_noon_time = solar_noon_local.time()
        else:
            _LOGGER.warning("Could not get sun events for color temperature schedule, using fixed times")
            # Fallback to fixed times if sun events unavailable
            sunrise_local = dt_util.parse_datetime("2023-01-01 06:00:00").replace(year=now.year, month=now.month, day=now.day)
            sunset_local = dt_util.parse_datetime("2023-01-01 18:00:00").replace(year=now.year, month=now.month, day=now.day)
            sunrise_time = sunrise_local.time()
            sunset_time = sunset_local.time()
            solar_noon_time = time(12, 0, 0)  # Midday
    except Exception as e:
        _LOGGER.error(f"Error getting sun events: {e}")
        return None


    schedule = {
        "morning_start": morning_start,
        "morning_end": morning_end,
        "evening_start": evening_start,
        "evening_end": evening_end,
        "sunrise": sunrise_time,
        "solar_noon": solar_noon_time,
        "sunset": sunset_time,
        "sunrise_sunset_temp": sunrise_sunset_temp,
        "midday_temp": midday_temp,
        "night_temp": night_temp,
        "curve_type": curve_type,
    }
    _LOGGER.debug(f"Generated color temp schedule: {schedule}")
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


def interpolate_temp(start_temp: int, end_temp: int, progress: float, curve_type: str) -> int:
    """Interpolate between two temperatures using linear or cosine curve."""
    if curve_type == "cosine":
        # Cosine interpolation for smoother transitions
        import math
        progress = 0.5 * (1 - math.cos(math.pi * progress))
    # Linear is default
    return round(start_temp + progress * (end_temp - start_temp))


def get_ct_at_time(
    schedule: dict[str, time | int],
    now: time,
    debug_enable: bool = True,
) -> int | None:
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
        temp = interpolate_temp(night_temp, sunrise_sunset_temp, progress, "linear")
        if debug_enable:
            _LOGGER.debug(f"Time {now}: Morning transition, progress {progress:.2f}, temp {temp}K")
        return temp

    # After morning_end until sunrise: sunrise_sunset_temp
    elif _is_time_in_period(now, morning_end, sunrise):
        if debug_enable:
            _LOGGER.debug(f"Time {now}: Morning stable, temp {sunrise_sunset_temp}K")
        return sunrise_sunset_temp

    # Sunrise to solar_noon: sunrise_sunset_temp to midday_temp
    elif _is_time_in_period(now, sunrise, solar_noon):
        progress = get_progress(now, sunrise, solar_noon)
        temp = interpolate_temp(sunrise_sunset_temp, midday_temp, progress, curve_type)
        if debug_enable:
            _LOGGER.debug(f"Time {now}: Day AM, progress {progress:.2f}, temp {temp}K")
        return temp

    # Solar_noon to sunset: midday_temp to sunrise_sunset_temp
    elif _is_time_in_period(now, solar_noon, sunset):
        progress = get_progress(now, solar_noon, sunset)
        temp = interpolate_temp(midday_temp, sunrise_sunset_temp, progress, curve_type)
        if debug_enable:
            _LOGGER.debug(f"Time {now}: Day PM, progress {progress:.2f}, temp {temp}K")
        return temp

    # Sunset to evening_start: sunrise_sunset_temp
    elif _is_time_in_period(now, sunset, evening_start):
        if debug_enable:
            _LOGGER.debug(f"Time {now}: Evening stable, temp {sunrise_sunset_temp}K")
        return sunrise_sunset_temp

    # Evening transition: sunrise_sunset_temp to night_temp
    elif _is_time_in_period(now, evening_start, evening_end):
        progress = get_progress(now, evening_start, evening_end)
        temp = interpolate_temp(sunrise_sunset_temp, night_temp, progress, "linear")
        if debug_enable:
            _LOGGER.debug(f"Time {now}: Evening transition, progress {progress:.2f}, temp {temp}K")
        return temp

    # Night (after evening_end until morning_start)
    else:
        if debug_enable:
            _LOGGER.debug(f"Time {now}: Night, temp {night_temp}K")
        return night_temp
