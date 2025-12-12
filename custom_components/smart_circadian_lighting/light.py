import asyncio
import logging
from datetime import datetime, timedelta

from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP_KELVIN,
    ColorMode,
    LightEntity,
    LightEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_call_later, async_track_state_change_event
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from . import circadian_logic, state_management, testing
from .circadian_logic import _convert_percent_to_255
from .color_temp_logic import (
    _is_time_in_period,
    get_color_temp_schedule,
    get_ct_at_time,
    kelvin_to_mired,
)
from .const import (
    DOMAIN,
    LIGHT_UPDATE_TIMEOUT,
    MIN_COLOR_TEMP_CHANGE_FOR_UPDATE,
    MIN_UPDATE_INTERVAL,
    SIGNAL_CIRCADIAN_LIGHT_TESTING_STATE_CHANGED,
    STORAGE_KEY,
    STORAGE_VERSION,
    TRANSITION_UPDATE_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up the Smart Circadian Lighting light platform."""
    store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
    domain_data = hass.data[DOMAIN][entry.entry_id]
    config = domain_data["config"]
    lights = config.get("lights", [])

    circadian_lights = [
        CircadianLight(hass, light, config, entry, store) for light in lights
    ]

    # Store the list of circadian lights in the domain data for this entry,
    # so other platforms (like button) can access them.
    domain_data["circadian_lights"] = circadian_lights

    async_add_entities(circadian_lights, True)


class CircadianLight(LightEntity):
    """Representation of a virtual Circadian Light entity."""

    _attr_has_entity_name = True
    _attr_name = "Circadian"
    _attr_supported_features = LightEntityFeature.TRANSITION

    def __init__(
        self,
        hass: HomeAssistant,
        light_entity_id: str,
        config: dict[str, any],
        entry: ConfigEntry,
        store: Store
    ) -> None:
        """Initialize the Circadian Light entity.

        Args:
            hass: Home Assistant instance
            light_entity_id: Entity ID of the light to control
            config: Integration configuration
            entry: Config entry
            store: Storage instance for persistence
        """
        self._hass = hass
        self._light_entity_id = light_entity_id
        self._config = config
        self._entry = entry
        self._store = store
        light_state = self._hass.states.get(self._light_entity_id)

        self._is_on = True
        self._brightness = None
        self._color_temp_kelvin = None
        self._color_temp_mired = None
        self._attr_color_temp_kelvin = None
        self._attr_min_color_temp_kelvin = 1500  # Default min kelvin
        self._attr_max_color_temp_kelvin = 6500  # Default max kelvin
        self._unsub_tracker = None
        self._unsub_state_tracker = None
        self._test_mode = False
        self._is_testing = False
        self._is_all_lights_test = False
        self._test_mode_unsub = None
        self._temp_transition_override = {}
        self._test_cancelled = False
        self._last_circadian_update_time: datetime | None = None
        self._event_throttle_time: datetime | None = None

        self._color_temp_schedule = get_color_temp_schedule(self._hass, self._config)

        # Set color temperature range based on config
        if self._color_temp_schedule:
            night_temp = self._config.get("night_color_temp_kelvin", 1800)
            day_temp = self._config.get("day_color_temp_kelvin", 4800)
            self._attr_min_color_temp_kelvin = min(night_temp, day_temp)
            self._attr_max_color_temp_kelvin = max(night_temp, day_temp)
        else:
            self._attr_min_color_temp_kelvin = 1500
            self._attr_max_color_temp_kelvin = 6500

        # State tracking attributes
        self._is_overridden = False
        self._override_timestamp = None
        self._expiration_callback_handle = None  # Handle for scheduled expiration callback
        self._is_online = True  # Assume online initially
        self._first_update_done = False
        self._last_confirmed_brightness = None
        self._last_reported_brightness = None
        self._last_update_time = None
        self._last_set_brightness = None
        self._last_set_color_temp = None

        if light_state:
            self._last_reported_brightness = light_state.attributes.get(ATTR_BRIGHTNESS)

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self.unique_id)},
            name=light_state.name if light_state else light_entity_id,
        )

        # Convert brightness percentages to 0-255 scale
        self._day_brightness_255 = _convert_percent_to_255(self._config["day_brightness"])
        self._night_brightness_255 = _convert_percent_to_255(self._config["night_brightness"])

        self._manual_override_threshold = _convert_percent_to_255(self._config.get("manual_override_threshold", 5))
        self._color_temp_manual_override_threshold = self._config.get("color_temp_manual_override_threshold", 100)

    async def async_added_to_hass(self) -> None:
        """Register update listener and load state."""
        await state_management.async_load_override_state(self)

        # Check initial state of the light
        initial_state = self._hass.states.get(self._light_entity_id)
        if not initial_state or initial_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            self._is_online = False

        # If the light is not considered overridden, force an update
        if not self._is_overridden:
            _LOGGER.debug(f"[{self._light_entity_id}] Forcing initial circadian update for {self.name} as it is not overridden.")
            self._hass.async_create_task(self.async_force_update_circadian())

        # Listen for state changes of the underlying light
        self._unsub_state_tracker = async_track_state_change_event(
            self._hass, [self._light_entity_id], self._async_entity_state_changed
        )

        self.async_schedule_update_ha_state(True)  # Start the update loop
        self._schedule_update()

    async def async_will_remove_from_hass(self) -> None:
        """Unregister update listener."""
        if self._unsub_tracker:
            self._unsub_tracker()
            self._unsub_tracker = None
        if self._unsub_state_tracker:
            self._unsub_state_tracker()
            self._unsub_state_tracker = None
        if self._test_mode_unsub:
            self._test_mode_unsub()
            self._test_mode_unsub = None
        _LOGGER.debug(f"[{self._light_entity_id}] Successfully removed {self.name} and cleaned up listeners.")


    def _check_override_expiration(self) -> bool:
        """Check if the current manual override has expired."""
        return state_management.check_override_expiration(self)

    async def _async_save_override_state(self):
        """Save the current override state to the store."""
        await state_management.async_save_override_state(self)

    async def _set_exact_circadian_targets(self) -> None:
        """Set lights to exact current circadian targets when override clears.

        This ensures lights are immediately corrected to precise circadian values
        after manual intervention, accounting for device capabilities.
        """
        if not self._is_on or self._is_overridden:
            return

        # Calculate exact current targets (no transition offsets)
        target_brightness = circadian_logic.calculate_brightness(
            0, self._temp_transition_override, self._config,
            self._day_brightness_255, self._night_brightness_255, self._light_entity_id
        )

        now = dt_util.now()
        target_color_temp = get_ct_at_time(self._color_temp_schedule, now.time())

        # Update immediately with transition=0
        await self.async_update_light(brightness=target_brightness,
                                    color_temp=target_color_temp,
                                    transition=0,
                                    force_update=True)

    async def async_clear_manual_override(self) -> None:
        """Clear the manual override and set exact circadian targets."""
        await state_management.async_clear_manual_override(self)
        await self._set_exact_circadian_targets()

    @callback
    async def _async_entity_state_changed(self, event) -> None:
        """Handle state changes of the underlying entity."""
        new_state = event.data.get("new_state")
        if not new_state:
            return

        # Handle online/offline status
        if new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            if self._is_online:
                self._is_online = False
                _LOGGER.debug(f"[{self._light_entity_id}] Light has gone offline.")
                self.async_write_ha_state()
        else:
            if not self._is_online:
                self._is_online = True
                _LOGGER.debug(f"[{self._light_entity_id}] Light has come online.")
                # If the light was previously offline, force an update
                self._hass.async_create_task(self.async_force_update_circadian())
            else:
                # Check if light just turned on and matches last set values
                old_state = event.data.get("old_state")
                if old_state and old_state.state != STATE_ON and new_state.state == STATE_ON:
                    # Light just turned on
                    entity_registry = er.async_get(self._hass)
                    entity_entry = entity_registry.async_get(self._light_entity_id)
                    is_kasa_dimmer = entity_entry and entity_entry.platform == "kasa_smart_dim"

                    if is_kasa_dimmer and self._last_set_brightness is not None:
                        current_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
                        current_color_temp = new_state.attributes.get(ATTR_COLOR_TEMP_KELVIN)
                        if (current_brightness == self._last_set_brightness and
                            (self._last_set_color_temp is None or current_color_temp == self._last_set_color_temp)):
                            _LOGGER.debug(f"[{self._light_entity_id}] Light turned on with last set values, updating to current circadian.")
                            self._hass.async_create_task(self.async_force_update_circadian())

                # If the light is online, handle other state changes
                await state_management.handle_entity_state_changed(self, event)

                # Check if Z-Wave light just turned off and clear override
                if old_state and old_state.state == STATE_ON and new_state.state == STATE_OFF:
                    entity_registry = er.async_get(self._hass)
                    entity_entry = entity_registry.async_get(self._light_entity_id)
                    is_zwave_light = entity_entry and entity_entry.platform == "zwave_js"

                    if is_zwave_light and self._is_overridden:
                        _LOGGER.debug(f"[{self._light_entity_id}] Z-Wave light turned off, clearing manual override")
                        self._is_overridden = False
                        await state_management.async_save_override_state(self)

                # Check if light just turned on and needs circadian update
                if old_state and old_state.state != STATE_ON and new_state.state == STATE_ON:
                    # Light just turned on
                    should_update = False

                    entity_registry = er.async_get(self._hass)
                    entity_entry = entity_registry.async_get(self._light_entity_id)
                    is_kasa_dimmer = entity_entry and entity_entry.platform == "kasa_smart_dim"

                    if is_kasa_dimmer and self._last_set_brightness is not None:
                        current_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
                        if current_brightness == self._last_set_brightness:
                            should_update = True
                    elif self._color_temp_schedule:  # Color temp enabled
                        should_update = True

                    if should_update:
                        _LOGGER.debug(f"[{self._light_entity_id}] Light turned on, updating to current circadian values.")
                        self._hass.async_create_task(self.async_force_update_circadian())

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{DOMAIN}_{self._light_entity_id}"

    @property
    def is_on(self) -> bool:
        """Return true if the circadian functionality is enabled."""
        return self._is_on

    @property
    def brightness(self) -> int | None:
        """Return the current target brightness."""
        return self._brightness

    @property
    def color_temp(self) -> int | None:
        """Return the current color temperature in mireds."""
        return self._color_temp_mired

    @property
    def color_temp_kelvin(self) -> int | None:
        """Return the current color temperature in kelvin."""
        return self._color_temp_kelvin

    @property
    def available(self) -> bool:
        """Return true if the underlying light is available."""
        return self._is_online

    @property
    def test_mode(self) -> bool:
        """Return true if the light is in test mode."""
        return self._test_mode

    @property
    def is_testing(self) -> bool:
        """Return true if the light is in a test cycle."""
        return self._is_testing

    @is_testing.setter
    def is_testing(self, value: bool) -> None:
        """Set the testing state and dispatch a signal."""
        if self._is_testing != value:
            self._is_testing = value
            async_dispatcher_send(self._hass, SIGNAL_CIRCADIAN_LIGHT_TESTING_STATE_CHANGED, self.entity_id, self._is_testing)

    @property
    def circadian_mode(self) -> str:
        """Return the current circadian mode."""
        return circadian_logic.get_circadian_mode(dt_util.now(), self._temp_transition_override, self._config)

    @property
    def extra_state_attributes(self) -> dict:
        """Return the state attributes."""
        return {
            "light_entity_id": self._light_entity_id,
            "is_circadian": self._is_on,
            "is_overridden": self._is_overridden,
            "override_timestamp": self._override_timestamp.isoformat() if self._override_timestamp else None,
            "test_mode": self._test_mode,
            "is_testing": self._is_testing,
            "circadian_mode": self.circadian_mode,
            "last_confirmed_brightness": self._last_confirmed_brightness,
            "last_reported_brightness": self._last_reported_brightness,
            "day_brightness_255": self._day_brightness_255,
            "night_brightness_255": self._night_brightness_255,
            "color_temp_kelvin": self._color_temp_kelvin
        }

    @property
    def supported_color_modes(self) -> set[ColorMode]:
        """Return the set of supported color modes."""
        if self._color_temp_schedule:
            return {ColorMode.COLOR_TEMP}
        return {ColorMode.BRIGHTNESS}

    @property
    def color_mode(self) -> ColorMode:
        """Return the color mode of the light."""
        if self._color_temp_schedule:
            return ColorMode.COLOR_TEMP
        return ColorMode.BRIGHTNESS

    async def async_turn_on(self, **kwargs) -> None:
        """Enable circadian adjustments."""
        self._is_on = True
        if self._is_overridden:
            self._is_overridden = False
            await state_management.async_save_override_state(self)
            await self._set_exact_circadian_targets()
        self.async_write_ha_state()
        await self.async_update_brightness(force_update=True)

    async def async_turn_off(self, **kwargs) -> None:
        """Disable circadian adjustments."""
        self._is_on = False
        self.async_write_ha_state()

    def _schedule_update(self, now: datetime | None = None) -> None:
        """Schedule the next update."""
        if self._unsub_tracker:
            self._unsub_tracker()
            self._unsub_tracker = None

        if self._test_mode:
            return

        is_transition = self.circadian_mode in ["morning_transition", "evening_transition"]

        # Check if color temperature is in transition
        color_temp_in_transition = False
        if self._color_temp_schedule:
            now_time = dt_util.now().time()
            morning_start = self._color_temp_schedule.get("morning_start")
            morning_end = self._color_temp_schedule.get("morning_end")
            evening_start = self._color_temp_schedule.get("evening_start")
            evening_end = self._color_temp_schedule.get("evening_end")
            if morning_start and morning_end and _is_time_in_period(now_time, morning_start, morning_end):
                color_temp_in_transition = True
            elif evening_start and evening_end and _is_time_in_period(now_time, evening_start, evening_end):
                color_temp_in_transition = True

        # Use a shorter update interval during testing
        if self._is_all_lights_test:
            update_interval = TRANSITION_UPDATE_INTERVAL
        elif self._is_testing:
            update_interval = MIN_UPDATE_INTERVAL
        elif is_transition:
            update_interval = TRANSITION_UPDATE_INTERVAL
        elif color_temp_in_transition:
            update_interval = 300  # 5 minutes for color temperature transitions
        elif self._color_temp_schedule and self._is_on:
            # During daytime with color temp enabled, update every 5 minutes
            current_time = dt_util.now().time()
            sunrise = self._color_temp_schedule.get("sunrise")
            sunset = self._color_temp_schedule.get("sunset")
            if sunrise and sunset and sunrise <= current_time < sunset:
                update_interval = 300  # 5 minutes
            else:
                # Not in a transition, so calculate time until the next one starts
                update_interval = circadian_logic.get_seconds_until_next_transition(self._temp_transition_override, self._config, self._light_entity_id)
        else:
            # Not in a transition, so calculate time until the next one starts
            update_interval = circadian_logic.get_seconds_until_next_transition(self._temp_transition_override, self._config, self._light_entity_id)

        _LOGGER.debug(f"[{self._light_entity_id}] Scheduling next update for {self.name} in {update_interval} seconds.")
        self._unsub_tracker = async_call_later(
            self._hass, update_interval, self._schedule_update_and_run
        )

    @callback
    def _schedule_update_and_run(self, now: datetime | None = None) -> None:
        """Helper to schedule and run the update."""
        if self._is_testing and self._test_cancelled:
            return
        self._hass.async_create_task(self.async_update_brightness())
        self._schedule_update()

    async def async_update_brightness(self, now: datetime | None = None, force_update: bool = False) -> None:
        """Calculate and apply the target brightness."""
        now_dt = dt_util.now()
        if self._last_circadian_update_time and (now_dt - self._last_circadian_update_time) < timedelta(seconds=5):
            _LOGGER.debug(f"[{self._light_entity_id}] Skipping update: recent circadian update")
            return
        await self._async_calculate_and_apply_brightness(force_update=force_update)

    async def _schedule_verification(self) -> None:
        """Schedule transition completion verification with a small delay."""
        await asyncio.sleep(1)  # Wait 1 second for HA to process the update
        await self._verify_transition_completion()

    async def _verify_transition_completion(self) -> None:
        """Verify that transitions completed successfully and correct if needed.

        This method checks if the light reached its commanded brightness target.
        If not, and the difference doesn't indicate a manual override, it sends
        a correction command to ensure the light reaches the exact target.
        """
        if self._is_overridden:
            return  # Don't interfere with manual overrides

        light_state = self._hass.states.get(self._light_entity_id)
        if not light_state or light_state.state != STATE_ON:
            return  # Light is off or unavailable

        current_brightness = light_state.attributes.get(ATTR_BRIGHTNESS)
        if current_brightness is None or self._brightness is None:
            return

        # Check if light reached the exact target
        if current_brightness != self._brightness:
            # Check if this looks like a manual override
            should_correct = True

            if self._last_confirmed_brightness is not None:
                # Use existing override detection logic
                brightness_diff = abs(current_brightness - self._last_confirmed_brightness)
                if brightness_diff > self._manual_override_threshold:
                    # Significant difference from last confirmed value - likely manual override
                    should_correct = False
                    _LOGGER.debug(f"[{self._light_entity_id}] Brightness difference ({brightness_diff}) exceeds threshold ({self._manual_override_threshold}), not correcting")

            if should_correct:
                _LOGGER.info(f"[{self._light_entity_id}] Transition incomplete, correcting brightness from {current_brightness} to {self._brightness}")
                await self.async_update_light(brightness=self._brightness, transition=0)
            else:
                _LOGGER.debug(f"[{self._light_entity_id}] Not correcting brightness due to potential manual override")

    async def _async_calculate_and_apply_brightness(self, force_update: bool = False, transition_override: int | None = None) -> None:
        """Calculate and apply the target brightness."""
        now = dt_util.now()
        if not force_update and self._last_update_time and (now - self._last_update_time) < timedelta(seconds=MIN_UPDATE_INTERVAL):
            _LOGGER.debug(f"[{self._light_entity_id}] Skipping update for {self.name}: MIN_UPDATE_INTERVAL not passed.")
            return

        _LOGGER.debug(f"[{self._light_entity_id}] Running _async_calculate_and_apply_brightness for {self.name} (force_update: {force_update})")

        if not self._is_on:
            _LOGGER.debug(f"[{self._light_entity_id}] Skipping brightness update for {self.name}: circadian entity is disabled.")
            return

        if not self._is_online:
            _LOGGER.debug(f"[{self._light_entity_id}] Skipping brightness update for {self.name}: entity is offline.")
            return

        mode = self.circadian_mode
        is_currently_transition = mode in ["morning_transition", "evening_transition"]

        # Check for ahead overrides at transition start
        if is_currently_transition and not self._is_overridden:
            # Check if manual overrides are enabled
            if not self._hass.data[DOMAIN][self._entry.entry_id].get("manual_overrides_enabled", True):
                pass  # Skip override detection
            else:
                light_state = self._hass.states.get(self._light_entity_id)
                if light_state:
                    current_brightness = light_state.attributes.get(ATTR_BRIGHTNESS)
                    current_color_temp = light_state.attributes.get(ATTR_COLOR_TEMP_KELVIN)

                    should_override = False

                    if mode == "evening_transition":
                        # Evening transition: check if ahead (dimmer than day or warmer than day)
                        if current_brightness is not None and current_brightness < self._day_brightness_255 - self._manual_override_threshold:
                            should_override = True
                            _LOGGER.debug(f"[{self._light_entity_id}] Evening transition start: current brightness {current_brightness} < day brightness {self._day_brightness_255} - threshold {self._manual_override_threshold}, marking as overridden")
                        if (current_color_temp is not None and self._color_temp_schedule and
                            current_color_temp < self._config.get("day_color_temp_kelvin", 5000) - self._color_temp_manual_override_threshold):
                            should_override = True
                            _LOGGER.debug(f"[{self._light_entity_id}] Evening transition start: current color temp {current_color_temp}K < day color temp - threshold, marking as overridden")

                    elif mode == "morning_transition":
                        # Morning transition: check if ahead (brighter than night or cooler than night)
                        if current_brightness is not None and current_brightness > self._night_brightness_255 + self._manual_override_threshold:
                            should_override = True
                            _LOGGER.debug(f"[{self._light_entity_id}] Morning transition start: current brightness {current_brightness} > night brightness {self._night_brightness_255} + threshold {self._manual_override_threshold}, marking as overridden")
                        if (current_color_temp is not None and self._color_temp_schedule and
                            current_color_temp > self._config.get("night_color_temp_kelvin", 1800) + self._color_temp_manual_override_threshold):
                            should_override = True
                            _LOGGER.debug(f"[{self._light_entity_id}] Morning transition start: current color temp {current_color_temp}K > night color temp + threshold, marking as overridden")

                    if should_override:
                        self._is_overridden = True
                        self._override_timestamp = now
                        await state_management.async_save_override_state(self)
                        _LOGGER.info(f"[{self._light_entity_id}] Detected ahead adjustment at transition start, marked as overridden")

        if is_currently_transition:
            update_interval = MIN_UPDATE_INTERVAL if self._is_testing else TRANSITION_UPDATE_INTERVAL
            target_brightness_255 = circadian_logic.calculate_brightness(
                update_interval,
                self._temp_transition_override,
                self._config,
                self._day_brightness_255,
                self._night_brightness_255,
                self._light_entity_id
            )
            transition = update_interval
            _LOGGER.debug(f"[{self._light_entity_id}] In transition. Target brightness: {target_brightness_255}, update_interval: {update_interval}")
        else:
            target_brightness_255 = circadian_logic.calculate_brightness(
                0,
                self._temp_transition_override,
                self._config,
                self._day_brightness_255,
                self._night_brightness_255,
                self._light_entity_id
            )
            transition = 0
            _LOGGER.debug(f"[{self._light_entity_id}] Not in transition. Target brightness: {target_brightness_255}")

        self._color_temp_kelvin = get_ct_at_time(self._color_temp_schedule, now.time())
        sun_state = self._hass.states.get("sun.sun")
        sun_elevation = sun_state.attributes.get("elevation") if sun_state else None
        if self._color_temp_kelvin:
            self._color_temp_mired = kelvin_to_mired(self._color_temp_kelvin)
            self._attr_color_temp_kelvin = self._color_temp_kelvin
            _LOGGER.debug(f"[{self._light_entity_id}] Calculated color temperature: {self._color_temp_kelvin}K ({self._color_temp_mired} mired), sun elevation: {sun_elevation}°")
        else:
            self._color_temp_mired = None
            self._attr_color_temp_kelvin = None
            _LOGGER.debug(f"[{self._light_entity_id}] No color temperature schedule available, sun elevation: {sun_elevation}°")

        # Check for manual overrides during transitions
        if is_currently_transition and not self._is_overridden:
            # Check if manual overrides are enabled
            if self._hass.data[DOMAIN][self._entry.entry_id].get("manual_overrides_enabled", True):
                light_state = self._hass.states.get(self._light_entity_id)
                if light_state and light_state.state == STATE_ON:
                    current_brightness = light_state.attributes.get(ATTR_BRIGHTNESS)
                    if current_brightness is not None:
                        brightness_diff = abs(current_brightness - target_brightness_255)
                        if brightness_diff > self._manual_override_threshold:
                            self._is_overridden = True
                            self._override_timestamp = now
                            await state_management.async_save_override_state(self)
                            _LOGGER.debug(f"[{self._light_entity_id}] Detected manual override during transition: current {current_brightness}, target {target_brightness_255}, diff {brightness_diff}")

        # For Z-Wave lights, always set parameter 18 regardless of override state
        # This allows users to "return to schedule" by turning lights off/on
        entity_registry = er.async_get(self._hass)
        entity_entry = entity_registry.async_get(self._light_entity_id)
        is_zwave_light = entity_entry and entity_entry.platform == "zwave_js"

        if is_zwave_light and target_brightness_255 is not None:
            # Always set parameter 18 to current circadian target
            zwave_brightness = int(target_brightness_255 * 99 / 255)
            zwave_brightness = max(0, min(99, zwave_brightness))

            try:
                await asyncio.wait_for(
                    self._hass.services.async_call(
                        "zwave_js",
                        "set_config_parameter",
                        {
                            "device_id": entity_entry.device_id,
                            "parameter": 18,
                            "value": zwave_brightness,
                        },
                    ),
                    timeout=LIGHT_UPDATE_TIMEOUT,
                )
                _LOGGER.debug(f"[{self._light_entity_id}] Set Z-Wave parameter 18 to {zwave_brightness} (brightness: {target_brightness_255})")
            except TimeoutError:
                _LOGGER.warning(f"[{self._light_entity_id}] Timeout setting Z-Wave parameter 18.")
            except HomeAssistantError as e:
                _LOGGER.error(f"[{self._light_entity_id}] Error setting Z-Wave parameter 18: {e}")

        if self._is_overridden:
            light_state = self._hass.states.get(self._light_entity_id)
            if not light_state:
                _LOGGER.debug(f"[{self._light_entity_id}] Skipping: overridden but no light state.")
                return
            current_brightness = light_state.attributes.get(ATTR_BRIGHTNESS)

            # Only clear override for "ahead" adjustments that have been caught up to
            # "Behind" adjustments (dimmer than circadian) should persist until manually changed
            should_clear_override = False

            # Calculate what the circadian target was when override was detected
            original_circadian_target = circadian_logic.calculate_brightness(
                0, self._temp_transition_override, self._config,
                self._day_brightness_255, self._night_brightness_255, self._light_entity_id
            )

            if mode == "morning_transition":
                # In morning transition, clear override only if manual brightness was higher than circadian
                # AND circadian has now caught up to it
                if current_brightness > original_circadian_target and target_brightness_255 >= current_brightness:
                    should_clear_override = True
            elif mode == "evening_transition":
                # In evening transition, clear override only if manual brightness was lower than circadian
                # AND circadian has now caught up to it
                if current_brightness < original_circadian_target and target_brightness_255 <= current_brightness:
                    should_clear_override = True

            if should_clear_override:
                _LOGGER.info(f"[{self._light_entity_id}] Circadian brightness has caught up to manual override. Clearing override.")
                self._is_overridden = False
                await state_management.async_save_override_state(self)
                await self._set_exact_circadian_targets()
            else:
                _LOGGER.debug(f"[{self._light_entity_id}] Skipping: manually overridden. Target: {target_brightness_255}, Current: {current_brightness}")
                return

        if transition_override is not None:
            transition = transition_override
            _LOGGER.debug(f"[{self._light_entity_id}] Using transition override: {transition}")

        if self._brightness != target_brightness_255:
            _LOGGER.debug(f"[{self._light_entity_id}] Updating internal brightness from {self._brightness} to {target_brightness_255}")
            self._brightness = target_brightness_255
            self.async_write_ha_state()
        else:
            _LOGGER.debug(f"[{self._light_entity_id}] Internal brightness already {self._brightness}, no change.")

        if self.color_temp != self._color_temp_mired:
            _LOGGER.debug(f"[{self._light_entity_id}] Updating internal color temperature from {self.color_temp} mired to {self._color_temp_mired} mired ({self._color_temp_kelvin}K)")
            self.async_write_ha_state()

        self._last_update_time = now
        _LOGGER.debug(f"[{self._light_entity_id}] Proceeding with light update. Transition: {transition}")
        await self.async_update_light(transition=transition)

        # Schedule verification for final target updates (transition=0)
        # Small delay to let HA process the update before checking
        if transition == 0 and not self._is_overridden:
            self._hass.async_create_task(self._schedule_verification())

    async def async_force_update_circadian(self) -> None:
        """Force update the circadian settings (brightness and color temperature) of the light, overriding any manual adjustments."""
        _LOGGER.debug(f"[{self._light_entity_id}] Force updating brightness for {self.name}")

        if not self._is_online:
            _LOGGER.warning(f"[{self._light_entity_id}] Cannot force update {self.name}: light is offline.")
            return

        if not self._is_on:
            self._is_on = True

        if self._is_overridden:
            self._is_overridden = False
            await state_management.async_save_override_state(self)

        target_brightness = circadian_logic.calculate_brightness(
            0,
            self._temp_transition_override,
            self._config,
            self._day_brightness_255,
            self._night_brightness_255,
            self._light_entity_id
        )

        # Always calculate current color temperature
        now = dt_util.now()
        self._color_temp_kelvin = get_ct_at_time(self._color_temp_schedule, now.time())
        if self._color_temp_kelvin:
            self._color_temp_mired = kelvin_to_mired(self._color_temp_kelvin)
            self._attr_color_temp_kelvin = self._color_temp_kelvin

        # Always proceed with update for online lights
        # The _apply_light_state_checks will handle on/off logic appropriately
        self._brightness = target_brightness
        self.async_write_ha_state()
        await self.async_update_light(transition=0, force_update=True)

    async def async_config_updated(self) -> None:
        """Handle configuration updates by recalculating targets and rescheduling."""
        _LOGGER.debug(f"[{self._light_entity_id}] Configuration updated, recalculating targets for {self.name}")

        # Recalculate color temp schedule if color temp is enabled
        if self._config.get("color_temp_enabled"):
            self._color_temp_schedule = get_color_temp_schedule(self._hass, self._config)

        # Recalculate current targets
        now = dt_util.now()
        if self._color_temp_schedule:
            self._color_temp_kelvin = get_ct_at_time(self._color_temp_schedule, now.time())
            if self._color_temp_kelvin:
                self._color_temp_mired = kelvin_to_mired(self._color_temp_kelvin)
                self._attr_color_temp_kelvin = self._color_temp_kelvin
        else:
            self._color_temp_kelvin = None
            self._color_temp_mired = None
            self._attr_color_temp_kelvin = None

        # Recalculate brightness
        self._brightness = circadian_logic.calculate_brightness(
            0,
            self._temp_transition_override,
            self._config,
            self._day_brightness_255,
            self._night_brightness_255,
            self._light_entity_id
        )

        # Update internal state
        self.async_write_ha_state()

        # Reschedule updates based on new config
        self._schedule_update()

        _LOGGER.debug(f"[{self._light_entity_id}] Configuration update complete for {self.name}")

    async def async_update_light(
        self,
        transition: int | None = None,
        brightness: int | None = None,
        color_temp: int | None = None,
        force_update: bool = False
    ) -> None:
        """Update the underlying light entity with new brightness and color temperature settings.

        Args:
            transition: Transition duration in seconds
            brightness: Target brightness (0-255)
            color_temp: Target color temperature in kelvin
            force_update: Whether this is a force update that should bypass minimum change checks
        """
        await self._refresh_entity_state()
        service_data = self._prepare_service_data(brightness, color_temp)
        if not service_data:
            return

        light_state = self._hass.states.get(self._light_entity_id)
        if not light_state:
            _LOGGER.debug(f"[{self._light_entity_id}] No light state available, cannot determine if transition should be applied.")
            return

        self._apply_light_state_checks(service_data, light_state, transition, force_update)
        await self._execute_light_update(service_data, brightness)

    async def _refresh_entity_state(self) -> None:
        """Refresh the state of the underlying light entity."""
        _LOGGER.debug(f"[{self._light_entity_id}] Requesting entity update for {self.name}")
        try:
            await asyncio.wait_for(
                self._hass.services.async_call(
                    "homeassistant",
                    "update_entity",
                    {"entity_id": self._light_entity_id},
                    blocking=True,
                ),
                timeout=LIGHT_UPDATE_TIMEOUT,
            )
            _LOGGER.debug(f"[{self._light_entity_id}] Entity update completed for {self.name}")
        except TimeoutError:
            _LOGGER.warning(
                f"[{self._light_entity_id}] Timeout requesting entity update for {self.name}."
            )
        except HomeAssistantError as e:
            _LOGGER.error(f"[{self._light_entity_id}] Error requesting entity update for {self.name}: {e}")

    def _prepare_service_data(self, brightness: int | None, color_temp: int | None) -> dict[str, any] | None:
        """Prepare the service data for the light update.

        Args:
            brightness: Target brightness override
            color_temp: Target color temperature override in kelvin

        Returns:
            Service data dict or None if no update needed
        """
        target_brightness = brightness if brightness is not None else self._brightness
        if target_brightness is None:
            return None

        target_color_temp = color_temp if color_temp is not None else self._color_temp_kelvin

        service_data = {
            "entity_id": self._light_entity_id,
            ATTR_BRIGHTNESS: target_brightness,
        }

        if target_color_temp is not None:
            service_data[ATTR_COLOR_TEMP_KELVIN] = target_color_temp

        return service_data

    def _apply_light_state_checks(self, service_data: dict[str, any], light_state: State | None, transition: int | None, force_update: bool = False) -> None:
        """Apply checks based on current light state and modify service data accordingly.

        Args:
            service_data: The service call data to modify
            light_state: Current state of the light
            transition: Requested transition duration
            force_update: Whether this is a force update that should bypass minimum change checks
        """
        entity_registry = er.async_get(self._hass)
        entity_entry = entity_registry.async_get(self._light_entity_id)
        is_kasa_dimmer = entity_entry and entity_entry.platform == "kasa_smart_dim"
        is_zwave_light = entity_entry and entity_entry.platform == "zwave_js"

        # Do not send updates to lights that are off and don't support off-state control
        # Allow Kasa dimmers and Z-Wave lights when off (they can preload brightness)
        if not (is_kasa_dimmer or is_zwave_light) and light_state.state != STATE_ON:
            _LOGGER.debug(f"[{self._light_entity_id}] Skipping update because non-supported light is off.")
            service_data.clear()  # Signal to skip
            return

        supported_features = light_state.attributes.get("supported_features", 0)
        supports_transition = supported_features & LightEntityFeature.TRANSITION

        current_brightness = light_state.attributes.get(ATTR_BRIGHTNESS)
        target_brightness = service_data.get(ATTR_BRIGHTNESS)
        target_color_temp = service_data.get(ATTR_COLOR_TEMP_KELVIN)

        mode = self.circadian_mode
        if mode == "morning_transition" and current_brightness is not None and target_brightness < current_brightness:
            _LOGGER.debug(f"[{self._light_entity_id}] Skipping update: Morning transition, but target brightness ({target_brightness}) is less than current brightness ({current_brightness}).")
            service_data.clear()
            return
        if mode == "evening_transition" and current_brightness is not None and target_brightness > current_brightness:
            _LOGGER.debug(f"[{self._light_entity_id}] Skipping update: Evening transition, but target brightness ({target_brightness}) is greater than current brightness ({current_brightness}).")
            service_data.clear()
            return

        current_color_temp = light_state.attributes.get(ATTR_COLOR_TEMP_KELVIN)
        if not force_update and target_color_temp and current_color_temp and abs(target_color_temp - current_color_temp) < MIN_COLOR_TEMP_CHANGE_FOR_UPDATE:
            if ATTR_COLOR_TEMP_KELVIN in service_data:
                del service_data[ATTR_COLOR_TEMP_KELVIN]
            _LOGGER.debug(f"[{self._light_entity_id}] Color temperature change is too small, skipping update.")

        # Only add transition if the light supports it and the change is significant
        if supports_transition and transition is not None and transition > 0 and current_brightness is not None:
            if abs(target_brightness - current_brightness) > 3:  # Corresponds to ~1% brightness change
                service_data["transition"] = transition
            else:
                _LOGGER.debug(f"[{self._light_entity_id}] Brightness change is too small, skipping transition.")

    async def _execute_light_update(self, service_data: dict[str, any], brightness: int | None) -> None:
        """Execute the light update service call.

        Args:
            service_data: The prepared service data
            brightness: Original brightness parameter for tracking
        """
        if not service_data:
            return

        target_brightness = service_data.get(ATTR_BRIGHTNESS)
        target_color_temp = service_data.get(ATTR_COLOR_TEMP_KELVIN)

        # Check if this is a Z-Wave light that needs special handling
        entity_registry = er.async_get(self._hass)
        entity_entry = entity_registry.async_get(self._light_entity_id)
        is_zwave_light = entity_entry and entity_entry.platform == "zwave_js"

        light_state = self._hass.states.get(self._light_entity_id)
        is_light_on = light_state and light_state.state == STATE_ON

        # For Z-Wave lights when off, we've already set parameter 18 - no need to call light.turn_on
        if is_zwave_light and not is_light_on:
            _LOGGER.debug(f"[{self._light_entity_id}] Z-Wave light is off, parameter already set - skipping light.turn_on")
            if brightness is None:
                self._last_confirmed_brightness = self._brightness
            self._last_set_brightness = target_brightness
            if target_color_temp is not None:
                self._last_set_color_temp = target_color_temp
            self._first_update_done = True
            return

        sun_state = self._hass.states.get("sun.sun")
        sun_elevation = sun_state.attributes.get("elevation") if sun_state else None
        _LOGGER.debug(
            f"[{self._light_entity_id}] Calling light.turn_on service with data: {service_data}, sun elevation: {sun_elevation}°"
        )

        self._last_circadian_update_time = dt_util.now()
        try:
            await asyncio.wait_for(
                self._hass.services.async_call("light", "turn_on", service_data),
                timeout=LIGHT_UPDATE_TIMEOUT
            )
            if brightness is None:
                self._last_confirmed_brightness = self._brightness
            self._last_set_brightness = target_brightness
            if target_color_temp is not None:
                self._last_set_color_temp = target_color_temp
            self._first_update_done = True
        except TimeoutError:
            _LOGGER.warning(f"[{self._light_entity_id}] Timeout occurred while updating light {self._light_entity_id}.")
        except HomeAssistantError as e:
            _LOGGER.error(f"[{self._light_entity_id}] Error updating light {self._light_entity_id}: {e}")


    async def start_test_transition(
        self, mode: str, duration: int, hold_duration: int | None, include_color_temp: bool = False
    ) -> None:
        """Start a test transition."""
        await testing.start_test_transition(self, mode, duration, hold_duration, include_color_temp)

    async def cancel_test_transition(self) -> None:
        """Cancel a test transition."""
        await testing.cancel_test_transition(self)

    async def set_temporary_transition(
        self, mode: str, start_time: str | None, end_time: str | None, duration: int | None
    ) -> None:
        """Set a temporary transition override."""
        await testing.set_temporary_transition(self, mode, start_time, end_time, duration)

    async def end_current_transition(self) -> None:
        """End the current transition."""
        await testing.end_current_transition(self)

    async def async_run_test_cycle(self, duration: int) -> None:
        """Run a test cycle of brightness changes using the component's real scheduler."""
        await testing.async_run_test_cycle(self, duration)

    async def async_cancel_test_cycle(self) -> None:
        """Cancel the running test cycle."""
        await testing.async_cancel_test_cycle(self)
