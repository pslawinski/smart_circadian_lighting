from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Any

from .const import (
    MIN_BRIGHTNESS_CHANGE_FOR_UPDATE,
    MIN_UPDATE_INTERVAL,
    TRANSITION_SAFETY_BUFFER,
)

_LOGGER = logging.getLogger(__name__)


def _convert_percent_to_255(percent: float) -> int:
    """Convert a percentage (0-100) to a 0-255 scale.

    Args:
        percent: Brightness percentage (0-100)

    Returns:
        Brightness value scaled to 0-255 range
    """
    if percent >= 100:
        return 255
    return int(round(percent * 255 / 100))


def get_transition_times(
    mode: str, temp_transition_override: dict[str, Any], config: dict[str, Any]
) -> tuple[time, time]:
    """Get the start and end times for a transition, considering overrides."""
    if temp_transition_override and temp_transition_override.get("mode") == mode:
        override = temp_transition_override
        start_time = override.get("start_time")
        end_time = override.get("end_time")
        duration = override.get("duration")

        if start_time and duration:
            start_dt = datetime.combine(datetime.today(), start_time)
            end_dt = start_dt + timedelta(seconds=duration)
            end_time = end_dt.time()
            return start_time, end_time
        elif start_time and end_time:
            return start_time, end_time

    # Default times
    start_time = time.fromisoformat(config[f"{mode}_start_time"])
    end_time = time.fromisoformat(config[f"{mode}_end_time"])
    return start_time, end_time


def _is_time_in_period(now: time, start: time, end: time) -> bool:
    """Checks if the current time is within a given period, handling midnight crossings."""
    if start <= end:
        return start <= now < end
    else:  # Period crosses midnight
        return start <= now or now < end


def get_circadian_mode(
    dt: datetime, temp_transition_override: dict[str, Any], config: dict[str, Any]
) -> str:
    """Determine the circadian mode for a specific time."""
    current_time = dt.time()
    morning_start_time, morning_end_time = get_transition_times(
        "morning", temp_transition_override, config
    )
    evening_start_time, evening_end_time = get_transition_times(
        "evening", temp_transition_override, config
    )

    if _is_time_in_period(current_time, morning_start_time, morning_end_time):
        return "morning_transition"
    elif _is_time_in_period(current_time, evening_start_time, evening_end_time):
        return "evening_transition"
    elif _is_time_in_period(current_time, morning_end_time, evening_start_time):
        return "day"
    else:
        return "night"


def is_morning_transition(
    dt: datetime, temp_transition_override: dict[str, Any], config: dict[str, Any]
) -> bool:
    """Check if the current time is during the morning transition."""
    return get_circadian_mode(dt, temp_transition_override, config) == "morning_transition"


def is_evening_transition(
    dt: datetime, temp_transition_override: dict[str, Any], config: dict[str, Any]
) -> bool:
    """Check if the current time is during the evening transition."""
    return get_circadian_mode(dt, temp_transition_override, config) == "evening_transition"


def get_progress(current: time, start: time, end: time) -> float:
    """Get the progress of a transition as a float between 0.0 and 1.0.

    Handles midnight crossings by adjusting timestamps.

    Args:
        current: Current time
        start: Transition start time
        end: Transition end time

    Returns:
        Progress as fraction (0.0 to 1.0)
    """
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
    return current_seconds / total_seconds


def calculate_brightness_for_time(
    dt: datetime,
    temp_transition_override: dict[str, Any],
    config: dict[str, Any],
    day_brightness_255: int,
    night_brightness_255: int,
    light_entity_id: str | None = None,
    debug_enable: bool = True,
) -> int:
    """Calculate the target brightness for a specific time.

    Args:
        dt: DateTime to calculate for
        temp_transition_override: Temporary transition settings
        config: Integration configuration
        day_brightness_255: Day brightness (0-255)
        night_brightness_255: Night brightness (0-255)
        light_entity_id: Light entity ID for logging

    Returns:
        Target brightness (0-255)
    """
    current_time = dt.time()

    morning_start_time, morning_end_time = get_transition_times(
        "morning", temp_transition_override, config
    )
    evening_start_time, evening_end_time = get_transition_times(
        "evening", temp_transition_override, config
    )

    night_brightness = night_brightness_255
    day_brightness = day_brightness_255
    target_brightness = 0.0

    log_prefix = f"[{light_entity_id}] " if light_entity_id else ""
    mode = get_circadian_mode(dt, temp_transition_override, config)

    if mode == "morning_transition":
        progress = get_progress(current_time, morning_start_time, morning_end_time)
        target_brightness = night_brightness + progress * (day_brightness - night_brightness)
        if debug_enable:
            _LOGGER.debug(
                f"{log_prefix}Morning transition in progress ({progress:.2f}). Brightness: {target_brightness:.1f}"
            )
    elif mode == "evening_transition":
        progress = get_progress(current_time, evening_start_time, evening_end_time)
        target_brightness = day_brightness - progress * (day_brightness - night_brightness)
        if debug_enable:
            _LOGGER.debug(
                f"{log_prefix}Evening transition in progress ({progress:.2f}). Brightness: {target_brightness:.1f}"
            )
    elif mode == "day":
        target_brightness = day_brightness
        if debug_enable:
            _LOGGER.debug(f"{log_prefix}Daytime. Brightness: {target_brightness:.1f}")
    else:  # night
        target_brightness = night_brightness
        if debug_enable:
            _LOGGER.debug(f"{log_prefix}Nighttime. Brightness: {target_brightness:.1f}")

    return int(round(target_brightness))


def calculate_brightness(
    time_offset_seconds: int,
    temp_transition_override: dict[str, Any],
    config: dict[str, Any],
    day_brightness_255: int,
    night_brightness_255: int,
    light_entity_id: str | None = None,
    debug_enable: bool = True,
) -> int:
    """Calculate the target brightness based on the time of day."""
    now = datetime.now()
    if time_offset_seconds > 0:
        now += timedelta(seconds=time_offset_seconds)

    return calculate_brightness_for_time(
        now,
        temp_transition_override,
        config,
        day_brightness_255,
        night_brightness_255,
        light_entity_id,
        debug_enable,
    )


def get_seconds_until_next_transition(
    temp_transition_override: dict[str, Any],
    config: dict[str, Any],
    light_entity_id: str | None = None,
) -> int:
    """Calculate the number of seconds until the next transition begins."""
    now = datetime.now()
    current_time = now.time()

    morning_start_time, _ = get_transition_times("morning", temp_transition_override, config)
    evening_start_time, _ = get_transition_times("evening", temp_transition_override, config)

    morning_start_dt = now.replace(
        hour=morning_start_time.hour,
        minute=morning_start_time.minute,
        second=morning_start_time.second,
        microsecond=0,
    )
    evening_start_dt = now.replace(
        hour=evening_start_time.hour,
        minute=evening_start_time.minute,
        second=evening_start_time.second,
        microsecond=0,
    )

    # Find the next transition time
    next_transition_dt = None
    if current_time < morning_start_time:
        next_transition_dt = morning_start_dt
        transition_name = "morning"
    elif current_time < evening_start_time:
        next_transition_dt = evening_start_dt
        transition_name = "evening"
    else:
        # Next transition is tomorrow morning
        next_transition_dt = morning_start_dt + timedelta(days=1)
        transition_name = "morning"

    # Calculate the delta and return the total seconds
    delta = next_transition_dt - now
    seconds_until = max(0, int(delta.total_seconds()))

    log_prefix = f"[{light_entity_id}] " if light_entity_id else ""
    _LOGGER.debug(
        f"{log_prefix}Next transition ({transition_name}) in {seconds_until} seconds at {next_transition_dt.isoformat()}"
    )

    return seconds_until


# Dictionary to store the last update time for each light
last_update_times: dict[str, datetime] = {}


def get_circadian_update_info(
    hass: any,
    light_entity_id: str,
    config: dict[str, Any],
    temp_transition_override: dict[str, Any],
    day_brightness_255: int,
    night_brightness_255: int,
) -> dict[str, Any] | None:
    """Determine the update parameters for a light based on circadian logic."""
    target_light_state = hass.states.get(light_entity_id)
    if not target_light_state:
        _LOGGER.warning(f"Could not find state for {light_entity_id}")
        return None

    physical_is_on = target_light_state.attributes.get("physical_is_on", True)
    current_brightness = target_light_state.attributes.get("brightness")
    if current_brightness is None:
        _LOGGER.debug(f"[{light_entity_id}] No brightness attribute, skipping update.")
        return None

    # Calculate the ideal circadian brightness
    circadian_brightness = calculate_brightness(
        0,  # No offset
        temp_transition_override,
        config,
        day_brightness_255,
        night_brightness_255,
        light_entity_id,
    )

    # Skip update if brightness change is negligible
    if abs(current_brightness - circadian_brightness) < MIN_BRIGHTNESS_CHANGE_FOR_UPDATE:
        _LOGGER.debug(f"[{light_entity_id}] Brightness change too small, skipping update.")
        return None

    now = datetime.now()
    in_morning_transition = is_morning_transition(now, temp_transition_override, config)
    in_evening_transition = is_evening_transition(now, temp_transition_override, config)

    # Default transition for non-circadian updates
    transition = 0 if not physical_is_on else 1.8

    if in_morning_transition or in_evening_transition:
        # Get transition start and end times
        mode = "morning" if in_morning_transition else "evening"
        start_time, end_time = get_transition_times(mode, temp_transition_override, config)
        transition_duration = (
            datetime.combine(datetime.today(), end_time)
            - datetime.combine(datetime.today(), start_time)
        ).total_seconds()

        # Calculate total brightness change for the entire transition
        night_brightness = night_brightness_255
        day_brightness = day_brightness_255
        total_brightness_change = abs(day_brightness - night_brightness)

        if total_brightness_change > 0:
            # Determine the number of steps
            num_steps = total_brightness_change / MIN_BRIGHTNESS_CHANGE_FOR_UPDATE

            # Calculate the dynamic update interval
            dynamic_interval = max(MIN_UPDATE_INTERVAL, transition_duration / num_steps)

            # Check if enough time has passed since the last update
            last_update = last_update_times.get(light_entity_id)
            if last_update and (now - last_update).total_seconds() < dynamic_interval:
                _LOGGER.debug(f"[{light_entity_id}] Dynamic interval not met, skipping update.")
                return None

            # Set hardware transition
            hardware_transition = max(0, dynamic_interval - TRANSITION_SAFETY_BUFFER)

            # No hardware transition for small brightness changes
            if abs(current_brightness - circadian_brightness) <= MIN_BRIGHTNESS_CHANGE_FOR_UPDATE:
                transition = 0
            else:
                transition = hardware_transition

        # Store the update time
        last_update_times[light_entity_id] = now

    return {"brightness": circadian_brightness, "transition": transition}
