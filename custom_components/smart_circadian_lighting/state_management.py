from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_COLOR_TEMP_KELVIN
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from .const import DOMAIN, SIGNAL_OVERRIDE_STATE_CHANGED

if TYPE_CHECKING:
    from .light import CircadianLight

_LOGGER = logging.getLogger(__name__)


def check_override_expiration(light: CircadianLight) -> bool:
    """Check if the current manual override has expired."""
    if not light._is_overridden or not light._override_timestamp:
        return False

    now = dt_util.now()
    morning_clear_time_str = light._config["morning_override_clear_time"]
    evening_clear_time_str = light._config["evening_override_clear_time"]

    try:
        morning_clear_time = time.fromisoformat(morning_clear_time_str)
        evening_clear_time = time.fromisoformat(evening_clear_time_str)
    except (ValueError, TypeError):
        _LOGGER.error("Invalid override clear time format. Please check your configuration.")
        return False  # Don't expire overrides if config is broken

    today_morning_clear = now.replace(
        hour=morning_clear_time.hour,
        minute=morning_clear_time.minute,
        second=0,
        microsecond=0,
    )
    today_evening_clear = now.replace(
        hour=evening_clear_time.hour,
        minute=evening_clear_time.minute,
        second=0,
        microsecond=0,
    )

    last_clear_time = None
    if now >= today_evening_clear:
        last_clear_time = today_evening_clear
    elif now >= today_morning_clear:
        last_clear_time = today_morning_clear
    else:
        # It's before the morning clear time, so the last clear was yesterday's evening
        last_clear_time = today_evening_clear - timedelta(days=1)

    if light._override_timestamp.astimezone(dt_util.UTC) < last_clear_time.astimezone(
        dt_util.UTC
    ):
        _LOGGER.debug(
            f"[{light._light_entity_id}] Override expired for {light.name}. Timestamp ({light._override_timestamp}) is before last clear time ({last_clear_time})"
        )
        return True

    return False


def _calculate_next_clear_time(light: CircadianLight) -> datetime:
    """Calculate the next time when the override should be cleared."""
    if hasattr(light, '_test_now'):
        now = light._test_now
    else:
        now = dt_util.now()
    morning_clear_time_str = light._config["morning_override_clear_time"]
    evening_clear_time_str = light._config["evening_override_clear_time"]

    try:
        morning_clear_time = time.fromisoformat(morning_clear_time_str)
        evening_clear_time = time.fromisoformat(evening_clear_time_str)
    except (ValueError, TypeError):
        _LOGGER.error("Invalid override clear time format. Using default 8:00 AM and 2:00 AM.")
        morning_clear_time = time(8, 0, 0)
        evening_clear_time = time(2, 0, 0)

    today_morning_clear = now.replace(
        hour=morning_clear_time.hour,
        minute=morning_clear_time.minute,
        second=0,
        microsecond=0,
    )
    today_evening_clear = now.replace(
        hour=evening_clear_time.hour,
        minute=evening_clear_time.minute,
        second=0,
        microsecond=0,
    )

    # Determine the next clear time
    if now < today_morning_clear:
        # Before morning clear time, next clear is morning
        next_clear_time = today_morning_clear
    elif now < today_evening_clear:
        # Between morning and evening clear, next clear is evening
        next_clear_time = today_evening_clear
    else:
        # After evening clear, next clear is tomorrow morning
        next_clear_time = today_morning_clear + timedelta(days=1)

    return next_clear_time


async def _async_clear_override_callback(light: CircadianLight) -> None:
    """Callback to clear an expired override."""
    if light._is_overridden:
        _LOGGER.info(f"[{light._light_entity_id}] Scheduled override expiration triggered for {light.name}")
        light._is_overridden = False
        light._expiration_callback_handle = None
        await async_save_override_state(light)
        # Set exact circadian targets
        await light._set_exact_circadian_targets()


def _create_clear_override_callback(light: CircadianLight):
    """Create a callback function that clears override for the specific light."""
    async def callback():
        await _async_clear_override_callback(light)
    return callback


async def async_save_override_state(light: CircadianLight) -> None:
    """Save the current override state to the store."""
    light._override_timestamp = dt_util.utcnow() if light._is_overridden else None
    saved_states = await light._store.async_load() or {}
    saved_states[light._light_entity_id] = {
        "is_overridden": light._is_overridden,
        "timestamp": light._override_timestamp.isoformat()
        if light._override_timestamp
        else None,
    }
    await light._store.async_save(saved_states)
    _LOGGER.debug(f"[{light._light_entity_id}] Saved override state for {light.name}: {light._is_overridden}")

    # Schedule or cancel expiration callback based on override state
    if light._is_overridden:
        # Cancel any existing callback
        if light._expiration_callback_handle:
            light._expiration_callback_handle()
            light._expiration_callback_handle = None

        # Schedule new callback for next clear time
        next_clear_time = _calculate_next_clear_time(light)
        now = light._test_now if hasattr(light, '_test_now') else dt_util.now()
        delay_seconds = (next_clear_time - now).total_seconds()
        if delay_seconds > 0:
            _LOGGER.debug(f"[{light._light_entity_id}] Scheduling override expiration for {light.name} in {delay_seconds} seconds (at {next_clear_time})")
            light._expiration_callback_handle = async_call_later(
                light.hass, delay_seconds, _create_clear_override_callback(light)
            )
        else:
            _LOGGER.warning(f"[{light._light_entity_id}] Calculated negative delay for override expiration, clearing immediately")
            await _async_clear_override_callback(light)
    else:
        # Clear any existing callback when override is removed
        if light._expiration_callback_handle:
            light._expiration_callback_handle()
            light._expiration_callback_handle = None

    async_dispatcher_send(light.hass, f"{SIGNAL_OVERRIDE_STATE_CHANGED}_{light.entity_id}")


async def async_load_override_state(light: CircadianLight) -> None:
    """Load saved override state and check for expiration."""
    saved_states = await light._store.async_load()
    if saved_states and light._light_entity_id in saved_states:
        state_data = saved_states[light._light_entity_id]
        light._is_overridden = state_data.get("is_overridden", False)
        timestamp_str = state_data.get("timestamp")
        if timestamp_str:
            light._override_timestamp = dt_util.parse_datetime(timestamp_str)

        _LOGGER.debug(
            f"[{light._light_entity_id}] Restored override state for {light.name}: overridden={light._is_overridden}, timestamp={light._override_timestamp}"
        )

        # Check if the loaded override has expired
        if light._is_overridden and check_override_expiration(light):
            _LOGGER.info(f"[{light._light_entity_id}] Restored override for {light.name} has expired. Clearing.")
            light._is_overridden = False
            light._override_timestamp = None
            await async_save_override_state(light)


async def async_clear_manual_override(light: CircadianLight) -> None:
    """Clear the manual override and force a brightness update."""
    _LOGGER.debug(f"[{light._light_entity_id}] Clearing manual override for {light.name}")
    if light._is_overridden:
        light._is_overridden = False
        # Cancel any scheduled expiration callback
        if light._expiration_callback_handle:
            light._expiration_callback_handle()
            light._expiration_callback_handle = None
        await async_save_override_state(light)
    await light.async_force_update_circadian()


async def _check_for_manual_override(
    light: CircadianLight,
    old_brightness: int | None,
    new_brightness: int | None,
    old_color_temp: int | None = None,
    new_color_temp: int | None = None,
    now: datetime = None
) -> None:
    """Check for a manual brightness or color temperature change that should trigger an override."""
    from . import circadian_logic  # Defer import to avoid circular dependency

    if not light._first_update_done:
        return

    # Check if manual overrides are enabled
    if not light._hass.data[DOMAIN][light._entry.entry_id].get("manual_overrides_enabled", True):
        return

    # Override detection is only active during transitions
    is_transition = circadian_logic.is_morning_transition(
        now, light._temp_transition_override, light._config
    ) or circadian_logic.is_evening_transition(
        now, light._temp_transition_override, light._config
    )
    if not is_transition:
        return

    is_morning = circadian_logic.is_morning_transition(
        now, light._temp_transition_override, light._config
    )

    brightness_override = False
    is_soft_override = False
    color_temp_override = False

    # Check brightness override
    if new_brightness is not None and old_brightness is not None and light._brightness is not None:
        brightness_diff = new_brightness - old_brightness
        max_error = light.max_quantization_error

        # An override is triggered if the user adjusts against the transition's direction
        # AND the new brightness level crosses the circadian setpoint by the threshold,
        # accounting for quantization error.
        if is_morning and brightness_diff < 0:  # Dimming during morning transition
            boundary = light._brightness - light._manual_override_threshold
            _LOGGER.debug(f"Morning brightness override check: new={new_brightness}, setpoint={light._brightness}, threshold={light._manual_override_threshold}, boundary={boundary}, max_error={max_error}, diff={brightness_diff}")
            if new_brightness < boundary + max_error:
                brightness_override = True
        elif not is_morning and brightness_diff > 0:  # Brightening during evening transition
            boundary = light._brightness + light._manual_override_threshold
            _LOGGER.debug(f"Evening brightness override check: new={new_brightness}, setpoint={light._brightness}, threshold={light._manual_override_threshold}, boundary={boundary}, max_error={max_error}, diff={brightness_diff}")
            if new_brightness >= boundary - max_error:
                brightness_override = True
        # Soft override detection: user adjusts in the same direction as the transition at transition start
        elif not brightness_override and new_brightness is not None and old_brightness is not None:
            # Check if we're at the transition start by calculating progress
            morning_start_time, morning_end_time = circadian_logic.get_transition_times(
                "morning", light._temp_transition_override, light._config
            )
            evening_start_time, evening_end_time = circadian_logic.get_transition_times(
                "evening", light._temp_transition_override, light._config
            )

            current_time = now.time()

            if is_morning:
                progress = circadian_logic.get_progress(current_time, morning_start_time, morning_end_time)
                # Soft override: brightening at transition start (adjusting in direction of transition)
                if brightness_diff > 0 and progress < 0.01:  # Within first 1% of transition duration
                    # Check if the adjustment is substantial (crosses threshold)
                    if new_brightness > old_brightness + light._manual_override_threshold:
                        brightness_override = True
                        is_soft_override = True
                        _LOGGER.debug(f"Morning soft override check: new={new_brightness}, old={old_brightness}, progress={progress}, diff={brightness_diff}, threshold={light._manual_override_threshold}")
            else:
                progress = circadian_logic.get_progress(current_time, evening_start_time, evening_end_time)
                # Soft override: dimming at transition start (adjusting in direction of transition)
                if brightness_diff < 0 and progress < 0.01:  # Within first 1% of transition duration
                    # Check if the adjustment is substantial (crosses threshold)
                    if new_brightness < old_brightness - light._manual_override_threshold:
                        brightness_override = True
                        is_soft_override = True
                        _LOGGER.debug(f"Evening soft override check: new={new_brightness}, old={old_brightness}, progress={progress}, diff={brightness_diff}, threshold={light._manual_override_threshold}")

        # Optimistic update filtering:
        # If the old_brightness is very close to our last-set target brightness,
        # it's likely this isn't a manual override, but a correction from
        # an optimistic state report from the light.
        # Skip this filtering for soft overrides, as they are intentional user adjustments.
        if brightness_override and not is_soft_override and abs(old_brightness - light._brightness) <= 5:
            brightness_override = False

    # Check color temperature override
    if (new_color_temp is not None and old_color_temp is not None and
        light._color_temp_kelvin is not None):
        color_temp_diff = abs(new_color_temp - old_color_temp)
        # Any significant color temperature change during transition is an override
        # since color temp transitions are gradual and user changes are intentional
        if color_temp_diff > light._color_temp_manual_override_threshold:
            color_temp_override = True

    if brightness_override:
        _LOGGER.info(
            f"[{light._light_entity_id}] Manual brightness override detected for {light.name}. "
            f"Brightness changed from {old_brightness} to {new_brightness} during {'morning' if is_morning else 'evening'} transition, "
            f"exceeding setpoint {light._brightness} by threshold."
        )
        light._is_overridden = True
        # Update the brightness to the user's manually set value for soft overrides
        if new_brightness is not None:
            light._brightness = new_brightness
        await async_save_override_state(light)
        light._event_throttle_time = now + timedelta(seconds=5)
    elif color_temp_override:
        _LOGGER.info(
            f"[{light._light_entity_id}] Manual color temperature override detected for {light.name}. "
            f"Color temperature changed from {old_color_temp}K to {new_color_temp}K during {'morning' if is_morning else 'evening'} transition, "
            f"exceeding threshold {light._color_temp_manual_override_threshold}K."
        )
        light._is_overridden = True
        await async_save_override_state(light)
        light._event_throttle_time = now + timedelta(seconds=5)


async def handle_entity_state_changed(light: CircadianLight, event: Any) -> None:
    """Handle state changes of the underlying entity."""
    if event is None or not hasattr(event, 'data') or event.data is None:
        return

    now = dt_util.utcnow()
    old_state = event.data.get("old_state")
    new_state = event.data.get("new_state")

    if not new_state:
        light._is_online = False
        light.async_write_ha_state()
        _LOGGER.info(f"[{light._light_entity_id}] {light.name} has become unavailable (entity removed).")
        return

    was_online = old_state is not None and old_state.state not in (
        STATE_UNAVAILABLE,
        STATE_UNKNOWN,
    )
    is_online = new_state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN)
    new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
    new_color_temp = new_state.attributes.get(ATTR_COLOR_TEMP_KELVIN)

    # Handle offline transition
    if was_online and not is_online:
        light._is_online = False
        # Save the last known brightness before going offline
        if old_state.attributes.get(ATTR_BRIGHTNESS) is not None:
            light._last_reported_brightness = old_state.attributes.get(ATTR_BRIGHTNESS)
        light.async_write_ha_state()
        _LOGGER.info(f"[{light._light_entity_id}] {light.name} has gone offline.")
        return

    # At this point, the light is online. Update its state.
    if not light._is_online:
        light._is_online = True
        light.async_write_ha_state()

    # Determine the brightness before the change
    if not was_online:  # came online
        _LOGGER.info(f"[{light._light_entity_id}] {light.name} has come back online.")
        old_brightness = light._last_reported_brightness
    else:  # was already online
        old_brightness = old_state.attributes.get(ATTR_BRIGHTNESS) if old_state else None

    # Determine the color temperature before the change
    if not was_online:  # came online
        old_color_temp = None  # Can't determine previous color temp when coming online
    else:  # was already online
        old_color_temp = old_state.attributes.get(ATTR_COLOR_TEMP_KELVIN) if old_state else None

    _LOGGER.debug(
        f"[{light._light_entity_id}] State change event: "
        f"old_brightness={old_brightness}, new_brightness={new_brightness}, "
        f"old_color_temp={old_color_temp}, new_color_temp={new_color_temp}"
    )

    # Hard query the current brightness from the device to ensure we have the most recent value
    # This prevents false negatives in override detection due to stale cached state
    if new_brightness is not None:
        queried_brightness = await light._get_current_brightness_with_refresh()
        if queried_brightness is not None and isinstance(queried_brightness, int):
            _LOGGER.debug(
                f"[{light._light_entity_id}] Refreshed brightness from {new_brightness} to {queried_brightness}"
            )
            new_brightness = queried_brightness
        else:
            _LOGGER.debug(
                f"[{light._light_entity_id}] Failed to refresh brightness, using event value: {new_brightness}"
            )

    # Update the last reported brightness for the next event
    if new_brightness is not None:
        light._last_reported_brightness = new_brightness

    # If an override is active and hasn't expired, do nothing.
    if light._is_overridden and not check_override_expiration(light):
        _LOGGER.debug(f"[{light._light_entity_id}] State change for {light.name} ignored: override is active.")
        return

    # Throttle event handling to avoid rapid firing
    if light._event_throttle_time and now < light._event_throttle_time:
        return

    if was_online:
        await _check_for_manual_override(light, old_brightness, new_brightness, old_color_temp, new_color_temp, now)
