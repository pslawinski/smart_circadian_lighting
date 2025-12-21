from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING

from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_COLOR_TEMP_KELVIN
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.helpers.event import async_call_later

from . import circadian_logic
from .circadian_logic import _convert_percent_to_255

if TYPE_CHECKING:
    from .light import CircadianLight

_LOGGER = logging.getLogger(__name__)


async def start_test_transition(
    light: CircadianLight,
    mode: str,
    duration: int,
    hold_duration: int | None,
    include_color_temp: bool = False,
) -> None:
    """Start a test transition."""
    if light._unsub_tracker:
        light._unsub_tracker()
        light._unsub_tracker = None

    light._test_mode = True
    light.async_write_ha_state()

    night_brightness = _convert_percent_to_255(
        light._config["night_brightness"]
    )
    day_brightness = _convert_percent_to_255(light._config["day_brightness"])

    start_brightness, end_brightness = (
        (night_brightness, day_brightness)
        if mode == "morning"
        else (day_brightness, night_brightness)
    )

    service_data = {ATTR_BRIGHTNESS: start_brightness}

    if include_color_temp and light._color_temp_schedule:
        night_color_temp = light._config["night_color_temp_kelvin"]
        day_color_temp = light._config["day_color_temp_kelvin"]
        start_color_temp, end_color_temp = (
            (night_color_temp, day_color_temp)
            if mode == "morning"
            else (day_color_temp, night_color_temp)
        )
        service_data[ATTR_COLOR_TEMP_KELVIN] = start_color_temp

    # Set initial state
    await light.async_update_light(**service_data, transition=0)

    transition_service_data = {ATTR_BRIGHTNESS: end_brightness}
    if include_color_temp and light._color_temp_schedule:
        transition_service_data[ATTR_COLOR_TEMP_KELVIN] = end_color_temp

    # Start transition
    await light.async_update_light(**transition_service_data, transition=duration)

    if hold_duration:

        async def _cancel_test(now: datetime) -> None:
            await cancel_test_transition(light)

        light._test_mode_unsub = async_call_later(
            light._hass, duration + hold_duration, _cancel_test
        )


async def cancel_test_transition(light: CircadianLight) -> None:
    """Cancel a test transition."""
    light._test_mode = False
    if light._test_mode_unsub:
        light._test_mode_unsub()
        light._test_mode_unsub = None

    light._schedule_update()
    await light.async_force_update_circadian()
    light.async_write_ha_state()


async def set_temporary_transition(
    light: CircadianLight,
    mode: str,
    start_time: str | None,
    end_time: str | None,
    duration: int | None,
) -> None:
    """Set a temporary transition override."""
    light._temp_transition_override = {
        "mode": mode,
        "start_time": time.fromisoformat(start_time) if start_time else None,
        "end_time": time.fromisoformat(end_time) if end_time else None,
        "duration": duration,
    }
    # If a transition is currently active, we need to re-evaluate
    now = datetime.now()
    is_transitioning = circadian_logic.is_morning_transition(
        now, light._temp_transition_override, light._config
    ) or circadian_logic.is_evening_transition(
        now, light._temp_transition_override, light._config
    )
    if is_transitioning:
        await light.async_update_brightness(force_update=True)


async def end_current_transition(light: CircadianLight) -> None:
    """End the current transition."""
    now = datetime.now()
    is_morning = circadian_logic.is_morning_transition(
        now, light._temp_transition_override, light._config
    )
    is_evening = circadian_logic.is_evening_transition(
        now, light._temp_transition_override, light._config
    )
    if is_morning:
        target_brightness_pct = light._config["day_brightness"]
        light._brightness = _convert_percent_to_255(target_brightness_pct)
        await light.async_update_light(transition=1)
    elif is_evening:
        night_brightness_pct = (
            light._config["night_brightness"]
        )
        light._brightness = _convert_percent_to_255(night_brightness_pct)
        await light.async_update_light(transition=1)
    light._temp_transition_override = {}


async def async_run_test_cycle_all(lights: list[CircadianLight], duration: int) -> None:
    """Run a test cycle for all lights concurrently."""
    _LOGGER.info(f"Starting test cycle for all lights for {duration} seconds...")
    for light in lights:
        light._is_all_lights_test = True
    tasks = [async_run_test_cycle(light, duration) for light in lights]
    await asyncio.gather(*tasks)
    _LOGGER.info("All test cycles completed.")


async def async_run_test_cycle(light: CircadianLight, duration: int) -> None:
    """Run a test cycle of brightness changes using the component's real scheduler."""
    if light.is_testing:
        _LOGGER.warning(f"[{light._light_entity_id}] Test cycle for {light.name} is already running.")
        return

    _LOGGER.info(f"[{light._light_entity_id}] Starting test cycle for {light.name}")
    light.is_testing = True
    light._test_cancelled = False
    light.async_write_ha_state()

    # Ensure the component isn't in a manual override state to start
    if light._is_overridden:
        await light.async_clear_manual_override()

    # Force update to night brightness at start of test
    night_brightness_255 = _convert_percent_to_255(light._config["night_brightness"])
    await light.async_update_light(brightness=night_brightness_255, transition=0)

    try:
        # Morning simulation
        await _async_run_single_test_phase(light, "morning", duration)

        # Evening simulation
        if not light._test_cancelled:
            await _async_run_single_test_phase(light, "evening", duration)

    finally:
        _LOGGER.info(f"[{light._light_entity_id}] Restoring to current target brightness for {light.name}")
        # Explicitly cancel the last scheduled update to exit the test loop
        if light._unsub_tracker:
            light._unsub_tracker()
            light._unsub_tracker = None

        # Force update to current target circadian settings
        await light.async_force_update_circadian()

        light.is_testing = False
        light._is_all_lights_test = False
        light._temp_transition_override = {}
        light.async_write_ha_state()

        # Reschedule the main update loop to resume normal operation
        light._schedule_update()


async def _async_run_single_test_phase(
    light: CircadianLight, mode: str, duration: int
) -> None:
    """Run a single phase of the test cycle by leveraging the main scheduler."""
    if light._test_cancelled:
        _LOGGER.info(f"[{light._light_entity_id}] Skipping {mode} test phase due to cancellation.")
        return

    _LOGGER.info(f"[{light._light_entity_id}] Starting {mode} test phase for {light.name} for {duration} seconds...")

    # Set initial brightness before starting the transition
    if mode == "morning":
        # Start at night brightness
        start_brightness = _convert_percent_to_255(
            light._config["night_brightness"]
        )
    else:  # evening
        # Start at day brightness
        start_brightness = _convert_percent_to_255(light._config["day_brightness"])

    _LOGGER.debug(f"[{light._light_entity_id}] Setting initial brightness for {mode} test to {start_brightness}")
    await light.async_update_light(brightness=start_brightness, transition=0)

    # Poll to confirm brightness change, with a 30-second timeout
    confirmation_timeout = 30
    confirmed = False
    current_brightness = None
    for i in range(confirmation_timeout):
        await asyncio.sleep(1) # Wait 1 second before polling.

        _LOGGER.debug(f"[{light._light_entity_id}] Polling for brightness confirmation (Attempt {i+1}/{confirmation_timeout})...")

        # Check if the entity is online.
        if not light.available:
            _LOGGER.debug(
                f"[{light._light_entity_id}] Light {light.name} is unavailable. Continuing to poll..."
            )
            continue

        light_state = light._hass.states.get(light._light_entity_id)
        if light_state and light_state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            current_brightness = light_state.attributes.get(ATTR_BRIGHTNESS)
            _LOGGER.debug(
                f"[{light._light_entity_id}] Fetched state: "
                f"state='{light_state.state}', current_brightness={current_brightness}, "
                f"expected_brightness={start_brightness}"
            )
            # Check for brightness match with tolerance
            if current_brightness is not None and abs(current_brightness - start_brightness) <= 1:
                _LOGGER.debug(f"[{light._light_entity_id}] Initial brightness confirmed for {light.name} after {i+1} seconds.")
                confirmed = True
                break
        else:
            state_str = light_state.state if light_state else "None"
            _LOGGER.debug(
                f"[{light._light_entity_id}] Light state is unavailable or unknown during poll. State: {state_str}"
            )

    if not confirmed:
        _LOGGER.warning(
            f"[{light._light_entity_id}] Could not confirm initial brightness for {light.name} within {confirmation_timeout} seconds. The light may be unresponsive. Cancelling test."
        )
        await async_cancel_test_cycle(light)
        return

    # Set the last reported brightness to the confirmed value to avoid false override detection
    if current_brightness is not None:
        light._last_reported_brightness = current_brightness

    # Set a temporary transition that starts now and lasts for the given duration
    now = datetime.now()
    light._temp_transition_override = {
        "mode": mode,
        "start_time": (now - timedelta(seconds=5)).time(),
        "duration": duration,
    }

    # Cancel any existing scheduled update and start the test's update loop.
    # The scheduler will now run at the test interval (20s).
    if light._unsub_tracker:
        light._unsub_tracker()
        light._unsub_tracker = None
    light._schedule_update_and_run()

    # Wait for the duration of this test phase, checking for cancellation
    for _ in range(duration):
        if light._test_cancelled:
            _LOGGER.info(f"[{light._light_entity_id}] Test cycle for {light.name} was cancelled during {mode} phase.")
            return
        await asyncio.sleep(1)

    # Clear the temporary transition for the next phase
    light._temp_transition_override = {}


async def async_cancel_test_cycle_all(lights: list[CircadianLight]) -> None:
    """Cancel all running test cycles."""
    _LOGGER.info("Cancelling all running test cycles.")
    for light in lights:
        if light.is_testing:
            await async_cancel_test_cycle(light)


async def async_cancel_test_cycle(light: CircadianLight) -> None:
    """Cancel the running test cycle."""
    if not light.is_testing:
        _LOGGER.warning(f"[{light._light_entity_id}] No test cycle is currently running for {light.name}.")
        return

    _LOGGER.info(f"[{light._light_entity_id}] Cancelling test cycle for {light.name}.")
    light._test_cancelled = True
