
import datetime
import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import selector

from .const import (
    CONF_COLOR_CURVE_TYPE,
    CONF_COLOR_EVENING_END_FIXED_TIME,
    CONF_COLOR_EVENING_END_SUN_EVENT,
    CONF_COLOR_EVENING_END_SUN_OFFSET,
    CONF_COLOR_EVENING_END_TIME,
    CONF_COLOR_EVENING_END_TYPE,
    CONF_COLOR_EVENING_START_FIXED_TIME,
    CONF_COLOR_EVENING_START_SUN_EVENT,
    CONF_COLOR_EVENING_START_SUN_OFFSET,
    CONF_COLOR_EVENING_START_TIME,
    CONF_COLOR_EVENING_START_TYPE,
    CONF_COLOR_MORNING_END_FIXED_TIME,
    CONF_COLOR_MORNING_END_SUN_EVENT,
    CONF_COLOR_MORNING_END_SUN_OFFSET,
    CONF_COLOR_MORNING_END_TIME,
    CONF_COLOR_MORNING_END_TYPE,
    CONF_COLOR_MORNING_START_FIXED_TIME,
    CONF_COLOR_MORNING_START_SUN_EVENT,
    CONF_COLOR_MORNING_START_SUN_OFFSET,
    CONF_COLOR_MORNING_START_TIME,
    CONF_COLOR_MORNING_START_TYPE,
    CONF_COLOR_TEMP_ENABLED,
    CONF_COLOR_TEMP_MANUAL_OVERRIDE_THRESHOLD,
    CONF_COLOR_TIME_TYPE_FIXED,
    CONF_COLOR_TIME_TYPE_SUN,
    CONF_COLOR_TIME_TYPE_SYNC,
    CONF_DAY_BRIGHTNESS,
    CONF_EVENING_END_TIME,
    CONF_EVENING_OVERRIDE_CLEAR_TIME,
    CONF_EVENING_START_TIME,
    CONF_LIGHTS,
    CONF_MANUAL_OVERRIDE_THRESHOLD,
    CONF_MIDDAY_COLOR_TEMP_KELVIN,
    CONF_MORNING_END_TIME,
    CONF_MORNING_OVERRIDE_CLEAR_TIME,
    CONF_MORNING_START_TIME,
    CONF_NIGHT_BRIGHTNESS,
    CONF_NIGHT_COLOR_TEMP_KELVIN,
    CONF_SUNRISE_SUNSET_COLOR_TEMP_KELVIN,
    DEFAULT_COLOR_CURVE_TYPE,
    DEFAULT_COLOR_TEMP_ENABLED,
    DEFAULT_COLOR_TEMP_MANUAL_OVERRIDE_THRESHOLD,
    DEFAULT_DAY_BRIGHTNESS,
    DEFAULT_EVENING_END_TIME,
    DEFAULT_EVENING_OVERRIDE_CLEAR_TIME,
    DEFAULT_EVENING_START_TIME,
    DEFAULT_MANUAL_OVERRIDE_THRESHOLD,
    DEFAULT_MIDDAY_COLOR_TEMP_KELVIN,
    DEFAULT_MORNING_END_TIME,
    DEFAULT_MORNING_OVERRIDE_CLEAR_TIME,
    DEFAULT_MORNING_START_TIME,
    DEFAULT_NIGHT_BRIGHTNESS,
    DEFAULT_NIGHT_COLOR_TEMP_KELVIN,
    DEFAULT_SUNRISE_SUNSET_COLOR_TEMP_KELVIN,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


def _get_kasa_dimmer_entities(hass: HomeAssistant) -> list[str]:
    """Return a list of Kasa dimmer entity IDs from the kasa_smart_dim component."""
    ent_reg = er.async_get(hass)
    entity_ids = []
    for entity in ent_reg.entities.values():
        if entity.platform == "kasa_smart_dim" and entity.domain == "light":
            entity_ids.append(entity.entity_id)
    _LOGGER.debug(f"Found Kasa smart dim entities: {entity_ids}")
    return entity_ids


def get_main_schema(hass: HomeAssistant, config: dict[str, Any] | None = None) -> vol.Schema:
    """Generate the main schema for the config flow."""
    if config is None:
        config = {}

    return vol.Schema(
        {
            vol.Required(
                CONF_MORNING_START_TIME,
                default=config.get(CONF_MORNING_START_TIME, DEFAULT_MORNING_START_TIME),
                description="Time when morning brightness transition begins"
            ): selector.TimeSelector(),
            vol.Required(
                CONF_MORNING_END_TIME,
                default=config.get(CONF_MORNING_END_TIME, DEFAULT_MORNING_END_TIME),
                description="Time when morning brightness transition ends (daytime begins)"
            ): selector.TimeSelector(),
            vol.Required(
                CONF_EVENING_START_TIME,
                default=config.get(CONF_EVENING_START_TIME, DEFAULT_EVENING_START_TIME),
                description="Time when evening brightness transition begins (daytime ends)"
            ): selector.TimeSelector(),
            vol.Required(
                CONF_EVENING_END_TIME,
                default=config.get(CONF_EVENING_END_TIME, DEFAULT_EVENING_END_TIME),
                description="Time when evening brightness transition ends"
            ): selector.TimeSelector(),
            vol.Required(
                CONF_NIGHT_BRIGHTNESS,
                default=config.get(CONF_NIGHT_BRIGHTNESS, DEFAULT_NIGHT_BRIGHTNESS),
                description="Brightness level during nighttime hours (0-100%)"
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=100, mode="slider", step=1),
            ),
            vol.Required(
                CONF_DAY_BRIGHTNESS,
                default=config.get(CONF_DAY_BRIGHTNESS, DEFAULT_DAY_BRIGHTNESS),
                description="Brightness level during daytime hours (0-100%)"
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=100, mode="slider", step=1),
            ),
            vol.Required(
                CONF_MANUAL_OVERRIDE_THRESHOLD,
                default=config.get(CONF_MANUAL_OVERRIDE_THRESHOLD, DEFAULT_MANUAL_OVERRIDE_THRESHOLD),
                description="Minimum brightness change that triggers manual override detection"
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=2, max=20, mode="slider", step=1),
            ),
            vol.Optional(
                CONF_COLOR_TEMP_MANUAL_OVERRIDE_THRESHOLD,
                default=config.get(CONF_COLOR_TEMP_MANUAL_OVERRIDE_THRESHOLD, DEFAULT_COLOR_TEMP_MANUAL_OVERRIDE_THRESHOLD),
                description="Minimum color temperature change that triggers manual override detection"
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=50, max=500, mode="slider", step=10),
            ),
            vol.Required(
                CONF_LIGHTS,
                default=config.get(CONF_LIGHTS, []),
                description="Select the lights to control with circadian lighting"
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="light", multiple=True),
            ),
            vol.Optional(
                CONF_MORNING_OVERRIDE_CLEAR_TIME,
                default=config.get(CONF_MORNING_OVERRIDE_CLEAR_TIME, DEFAULT_MORNING_OVERRIDE_CLEAR_TIME),
                description="Time when morning manual overrides automatically expire"
            ): selector.TimeSelector(),
            vol.Optional(
                CONF_EVENING_OVERRIDE_CLEAR_TIME,
                default=config.get(CONF_EVENING_OVERRIDE_CLEAR_TIME, DEFAULT_EVENING_OVERRIDE_CLEAR_TIME),
                description="Time when evening manual overrides automatically expire"
            ): selector.TimeSelector(),
        }
    )


def _get_color_time_schema(
    config: dict[str, Any],
    type_key: str,
    fixed_time_key: str,
    sun_event_key: str,
    sun_offset_key: str,
) -> dict:
    """Generate the schema for a single color time setting."""
    # Determine the transition name for descriptions
    # For morning_end and evening_start, when sun sync, auto-use sunrise/sunset
    include_sun_event = not (
        ("morning_start" in type_key or "morning_end" in type_key or "evening_start" in type_key or "evening_end" in type_key) and
        config.get(type_key) == CONF_COLOR_TIME_TYPE_SUN
    )

    schema = {
        vol.Required(
            type_key,
            default=config.get(type_key, CONF_COLOR_TIME_TYPE_SYNC)
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    {"value": CONF_COLOR_TIME_TYPE_SYNC, "label": "Sync with brightness"},
                    {"value": CONF_COLOR_TIME_TYPE_FIXED, "label": "Fixed time"},
                    {"value": CONF_COLOR_TIME_TYPE_SUN, "label": "Sunrise/Sunset"},
                ],
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Optional(
            fixed_time_key,
            description="Fixed time"
        ): selector.TimeSelector(),
    }

    # Only show sun_event selector for morning_end and evening_start
    if include_sun_event and type_key in [CONF_COLOR_MORNING_END_TYPE, CONF_COLOR_EVENING_START_TYPE]:
        default_sun_event = "sunrise" if "morning" in type_key else "sunset"
        schema[vol.Optional(
            sun_event_key,
            default=config.get(sun_event_key, default_sun_event)
        )] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    {"value": "sunrise", "label": "Sunrise"},
                    {"value": "sunset", "label": "Sunset"},
                ],
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        )

    schema[vol.Optional(sun_offset_key, description="Offset from sun event")] = selector.DurationSelector()

    return schema


def get_color_schema(hass: HomeAssistant, config: dict[str, Any] | None = None) -> vol.Schema:
    """Generate the color schema for the config flow."""
    if config is None:
        config = {}

    schema = {
        vol.Required(
            CONF_COLOR_TEMP_ENABLED,
            default=config.get(CONF_COLOR_TEMP_ENABLED, DEFAULT_COLOR_TEMP_ENABLED),
            description="Enable color temperature adjustments"
        ): selector.BooleanSelector(),
        vol.Required(
            CONF_SUNRISE_SUNSET_COLOR_TEMP_KELVIN,
            default=config.get(CONF_SUNRISE_SUNSET_COLOR_TEMP_KELVIN, DEFAULT_SUNRISE_SUNSET_COLOR_TEMP_KELVIN),
            description="Color temperature at sunrise and sunset"
        ): selector.ColorTempSelector(
            selector.ColorTempSelectorConfig(
                unit="kelvin",
                min=1500,
                max=6500,
            )
        ),
        vol.Required(
            CONF_MIDDAY_COLOR_TEMP_KELVIN,
            default=config.get(CONF_MIDDAY_COLOR_TEMP_KELVIN, DEFAULT_MIDDAY_COLOR_TEMP_KELVIN),
            description="Color temperature during midday"
        ): selector.ColorTempSelector(
            selector.ColorTempSelectorConfig(
                unit="kelvin",
                min=1500,
                max=6500,
            )
        ),
        vol.Required(
            CONF_NIGHT_COLOR_TEMP_KELVIN,
            default=config.get(CONF_NIGHT_COLOR_TEMP_KELVIN, DEFAULT_NIGHT_COLOR_TEMP_KELVIN),
            description="Color temperature during nighttime"
        ): selector.ColorTempSelector(
            selector.ColorTempSelectorConfig(
                unit="kelvin",
                min=1500,
                max=6500,
            )
        ),
        vol.Required(
            CONF_COLOR_CURVE_TYPE,
            default=config.get(CONF_COLOR_CURVE_TYPE, DEFAULT_COLOR_CURVE_TYPE),
            description="Interpolation curve for transitions"
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    {"value": "linear", "label": "Linear"},
                    {"value": "cosine", "label": "Cosine (smooth)"},
                ],
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        ),
    }

    schema.update(
        _get_color_time_schema(
            config,
            CONF_COLOR_MORNING_START_TYPE,
            CONF_COLOR_MORNING_START_FIXED_TIME,
            CONF_COLOR_MORNING_START_SUN_EVENT,
            CONF_COLOR_MORNING_START_SUN_OFFSET,
        )
    )
    schema.update(
        _get_color_time_schema(
            config,
            CONF_COLOR_MORNING_END_TYPE,
            CONF_COLOR_MORNING_END_FIXED_TIME,
            CONF_COLOR_MORNING_END_SUN_EVENT,
            CONF_COLOR_MORNING_END_SUN_OFFSET,
        )
    )
    schema.update(
        _get_color_time_schema(
            config,
            CONF_COLOR_EVENING_START_TYPE,
            CONF_COLOR_EVENING_START_FIXED_TIME,
            CONF_COLOR_EVENING_START_SUN_EVENT,
            CONF_COLOR_EVENING_START_SUN_OFFSET,
        )
    )
    schema.update(
        _get_color_time_schema(
            config,
            CONF_COLOR_EVENING_END_TYPE,
            CONF_COLOR_EVENING_END_FIXED_TIME,
            CONF_COLOR_EVENING_END_SUN_EVENT,
            CONF_COLOR_EVENING_END_SUN_OFFSET,
        )
    )

    return vol.Schema(schema)


def _process_color_time_input(config: dict[str, Any], input_data: dict[str, Any], prefix: str) -> None:
    """Process the color time input from the user and update the config.

    Args:
        config: Configuration dict to update
        input_data: User input data
        prefix: Time period prefix (e.g., "morning_start")
    """
    type_key = f"color_{prefix}_type"
    time_str_key = f"color_{prefix}_time"
    fixed_time_key = f"color_{prefix}_fixed_time"
    sun_event_key = f"color_{prefix}_sun_event"
    sun_offset_key = f"color_{prefix}_sun_offset"

    time_type = input_data.get(type_key)
    if time_type:
        config[type_key] = time_type  # Save the type for proper form reloading
    # If no type in input, infer from time_str (for loading existing configs)
    elif config.get(time_str_key):
        time_str = config[time_str_key]
        if time_str == "sync":
            config[type_key] = CONF_COLOR_TIME_TYPE_SYNC
        elif "sunrise" in time_str or "sunset" in time_str:
            config[type_key] = CONF_COLOR_TIME_TYPE_SUN
        else:
            config[type_key] = CONF_COLOR_TIME_TYPE_FIXED

    if time_type == CONF_COLOR_TIME_TYPE_SYNC:
        config[time_str_key] = "sync"
    elif time_type == CONF_COLOR_TIME_TYPE_FIXED:
        config[time_str_key] = input_data.get(fixed_time_key)
    elif time_type == CONF_COLOR_TIME_TYPE_SUN:
        if prefix in ["morning_start", "morning_end"]:
            sun_event = "sunrise"
        elif prefix in ["evening_start", "evening_end"]:
            sun_event = "sunset"
        else:
            sun_event = input_data.get(sun_event_key)

        sun_offset = input_data.get(sun_offset_key)

        # Format the offset string correctly
        if sun_offset:
            # Home Assistant's DurationSelector returns a dict with `days`, `hours`,
            # `minutes`, `seconds`. A negative duration is indicated by a negative
            # value for one of these keys.
            total_seconds = sun_offset.get("seconds", 0) + \
                            sun_offset.get("minutes", 0) * 60 + \
                            sun_offset.get("hours", 0) * 3600 + \
                            sun_offset.get("days", 0) * 86400

            if total_seconds == 0:
                config[time_str_key] = sun_event
            else:
                is_negative = total_seconds < 0

                abs_seconds = abs(total_seconds)
                hours = abs_seconds // 3600
                minutes = (abs_seconds % 3600) // 60
                seconds = abs_seconds % 60

                offset_str = f"{hours:02}:{minutes:02}:{seconds:02}"

                if is_negative:
                    config[time_str_key] = f"{sun_event} - {offset_str}"
                else:
                    config[time_str_key] = f"{sun_event} + {offset_str}"
        else:
            config[time_str_key] = sun_event


class SmartCircadianLightingOptionsFlow(config_entries.OptionsFlow):
    """Handle an options flow for Smart Circadian Lighting."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.temp_config = {**config_entry.data, **config_entry.options}

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage the options."""
        try:
            if user_input is not None:
                _LOGGER.debug(f"Options init user input: {user_input}")
                self.temp_config.update(user_input)
                return await self.async_step_color()

            schema = get_main_schema(self.hass, self.temp_config)
            _LOGGER.debug("Options init schema created successfully")
            return self.async_show_form(
                step_id="init",
                data_schema=schema
            )
        except Exception as e:
            _LOGGER.error(f"Error in options async_step_init: {e}", exc_info=True)
            raise

    async def async_step_color(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the color temp options."""
        try:
            if user_input is not None:
                _LOGGER.debug(f"Options color user input: {user_input}")
                self.temp_config.update(user_input)

                _process_color_time_input(self.temp_config, user_input, "morning_start")
                _process_color_time_input(self.temp_config, user_input, "morning_end")
                _process_color_time_input(self.temp_config, user_input, "evening_start")
                _process_color_time_input(self.temp_config, user_input, "evening_end")

                # Clean up temporary keys
                keys_to_remove = [
                    CONF_COLOR_MORNING_START_TYPE, CONF_COLOR_MORNING_START_FIXED_TIME,
                    CONF_COLOR_MORNING_START_SUN_EVENT, CONF_COLOR_MORNING_START_SUN_OFFSET,
                    CONF_COLOR_MORNING_END_TYPE, CONF_COLOR_MORNING_END_FIXED_TIME,
                    CONF_COLOR_MORNING_END_SUN_EVENT, CONF_COLOR_MORNING_END_SUN_OFFSET,
                    CONF_COLOR_EVENING_START_TYPE, CONF_COLOR_EVENING_START_FIXED_TIME,
                    CONF_COLOR_EVENING_START_SUN_EVENT, CONF_COLOR_EVENING_START_SUN_OFFSET,
                    CONF_COLOR_EVENING_END_TYPE, CONF_COLOR_EVENING_END_FIXED_TIME,
                    CONF_COLOR_EVENING_END_SUN_EVENT, CONF_COLOR_EVENING_END_SUN_OFFSET,
                ]
                for key in keys_to_remove:
                    self.temp_config.pop(key, None)

                self.temp_config["morning_start_brightness"] = self.temp_config[CONF_NIGHT_BRIGHTNESS]
                self.temp_config["evening_end_brightness"] = self.temp_config[CONF_NIGHT_BRIGHTNESS]
                self.temp_config["morning_end_brightness"] = self.temp_config[CONF_DAY_BRIGHTNESS]
                self.temp_config["evening_start_brightness"] = self.temp_config[CONF_DAY_BRIGHTNESS]
                _LOGGER.debug(f"Options creating entry with config: {self.temp_config}")
                return self.async_create_entry(title="", data=self.temp_config)

            # Ensure defaults are set
            self.temp_config.setdefault("sunrise_sunset_color_temp_kelvin", DEFAULT_SUNRISE_SUNSET_COLOR_TEMP_KELVIN)
            self.temp_config.setdefault(CONF_MIDDAY_COLOR_TEMP_KELVIN, DEFAULT_MIDDAY_COLOR_TEMP_KELVIN)
            self.temp_config.setdefault(CONF_NIGHT_COLOR_TEMP_KELVIN, DEFAULT_NIGHT_COLOR_TEMP_KELVIN)
            self.temp_config.setdefault(CONF_COLOR_CURVE_TYPE, DEFAULT_COLOR_CURVE_TYPE)
            self.temp_config.setdefault(CONF_COLOR_MORNING_START_TIME, "sync")
            self.temp_config.setdefault(CONF_COLOR_MORNING_END_TIME, "06:00:00")
            self.temp_config.setdefault(CONF_COLOR_EVENING_START_TIME, "20:00:00")
            self.temp_config.setdefault(CONF_COLOR_EVENING_END_TIME, "sync")
            self.temp_config.setdefault(CONF_COLOR_MORNING_END_TYPE, CONF_COLOR_TIME_TYPE_FIXED)
            self.temp_config.setdefault(CONF_COLOR_EVENING_START_TYPE, CONF_COLOR_TIME_TYPE_FIXED)

            # Set defaults for sun events
            self.temp_config.setdefault(CONF_COLOR_MORNING_END_SUN_EVENT, "sunrise")
            self.temp_config.setdefault(CONF_COLOR_EVENING_START_SUN_EVENT, "sunset")

            # Set defaults for sun offsets
            self.temp_config.setdefault(CONF_COLOR_MORNING_START_SUN_OFFSET, {"hours": 0, "minutes": 0, "seconds": 0})
            self.temp_config.setdefault(CONF_COLOR_MORNING_END_SUN_OFFSET, {"hours": 0, "minutes": 0, "seconds": 0})
            self.temp_config.setdefault(CONF_COLOR_EVENING_START_SUN_OFFSET, {"hours": 0, "minutes": 0, "seconds": 0})
            self.temp_config.setdefault(CONF_COLOR_EVENING_END_SUN_OFFSET, {"hours": 0, "minutes": 0, "seconds": 0})

            # Set sun event defaults from saved time_str
            if self.temp_config.get(CONF_COLOR_MORNING_END_TIME) == "sunrise":
                self.temp_config[CONF_COLOR_MORNING_END_SUN_EVENT] = "sunrise"
            elif self.temp_config.get(CONF_COLOR_MORNING_END_TIME) == "sunset":
                self.temp_config[CONF_COLOR_MORNING_END_SUN_EVENT] = "sunset"

            if self.temp_config.get(CONF_COLOR_EVENING_START_TIME) == "sunrise":
                self.temp_config[CONF_COLOR_EVENING_START_SUN_EVENT] = "sunrise"
            elif self.temp_config.get(CONF_COLOR_EVENING_START_TIME) == "sunset":
                self.temp_config[CONF_COLOR_EVENING_START_SUN_EVENT] = "sunset"

            # Parse offset from time_str if present
            for prefix in ["morning_start", "morning_end", "evening_start", "evening_end"]:
                time_str_key = f"color_{prefix}_time"
                sun_offset_key = f"color_{prefix}_sun_offset"
                time_str = self.temp_config.get(time_str_key)
                if time_str and (" + " in time_str or " - " in time_str):
                    parts = time_str.split()
                    if len(parts) >= 3:
                        offset_str = parts[2]
                        try:
                            parsed_offset = datetime.timedelta(hours=int(offset_str.split(":")[0]), minutes=int(offset_str.split(":")[1]), seconds=int(offset_str.split(":")[2]))
                            total_seconds = int(parsed_offset.total_seconds())
                            abs_seconds = abs(total_seconds)
                            hours = abs_seconds // 3600
                            minutes = (abs_seconds % 3600) // 60
                            seconds = abs_seconds % 60
                            self.temp_config[sun_offset_key] = {"hours": hours, "minutes": minutes, "seconds": seconds}
                        except Exception as e:
                            _LOGGER.debug(f"Failed to parse offset from {time_str}: {e}")

            # Infer types from time_str for loading
            _process_color_time_input(self.temp_config, {}, "morning_start")
            _process_color_time_input(self.temp_config, {}, "morning_end")
            _process_color_time_input(self.temp_config, {}, "evening_start")
            _process_color_time_input(self.temp_config, {}, "evening_end")

            # Set fixed time defaults
            if self.temp_config.get(CONF_COLOR_MORNING_END_TYPE) == CONF_COLOR_TIME_TYPE_FIXED:
                self.temp_config[CONF_COLOR_MORNING_END_FIXED_TIME] = self.temp_config.get(CONF_COLOR_MORNING_END_TIME)

            if self.temp_config.get(CONF_COLOR_EVENING_START_TYPE) == CONF_COLOR_TIME_TYPE_FIXED:
                self.temp_config[CONF_COLOR_EVENING_START_FIXED_TIME] = self.temp_config.get(CONF_COLOR_EVENING_START_TIME)

            schema = get_color_schema(self.hass, self.temp_config)
            _LOGGER.debug("Options color schema created successfully")
            return self.async_show_form(
                step_id="color",
                data_schema=schema
            )
        except Exception as e:
            _LOGGER.error(f"Error in options async_step_color: {e}", exc_info=True)
            raise


class SmartCircadianLightingConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Smart Circadian Lighting."""

    VERSION = 1

    def __init__(self):
        """Initialize the config flow."""
        self.temp_config = {}

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> SmartCircadianLightingOptionsFlow:
        """Get the options flow for this handler."""
        return SmartCircadianLightingOptionsFlow(config_entry)

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial step."""
        try:
            errors: dict[str, str] = {}

            if user_input is not None:
                _LOGGER.debug(f"User input received: {user_input}")
                self.temp_config.update(user_input)
                return await self.async_step_color()

            # Pre-select all found dimmers by default on first run
            config_with_defaults = {CONF_LIGHTS: _get_kasa_dimmer_entities(self.hass)}
            _LOGGER.debug(f"Config with defaults: {config_with_defaults}")

            schema = get_main_schema(self.hass, config_with_defaults)
            _LOGGER.debug("Schema created successfully")
            return self.async_show_form(
                step_id="user",
                data_schema=schema,
                errors=errors if errors else None
            )
        except Exception as e:
            _LOGGER.error(f"Error in async_step_user: {e}", exc_info=True)
            raise

    async def async_step_color(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the color temp options."""
        try:
            errors: dict[str, str] = {}

            if user_input is not None:
                _LOGGER.debug(f"Color step user input: {user_input}")
                # Make sure lights is a list
                if not isinstance(self.temp_config.get(CONF_LIGHTS), list):
                    self.temp_config[CONF_LIGHTS] = [self.temp_config.get(CONF_LIGHTS)]

                if not self.temp_config.get(CONF_LIGHTS):
                    errors["base"] = "no_lights"
                else:
                    self.temp_config.update(user_input)

                    _process_color_time_input(self.temp_config, user_input, "morning_start")
                    _process_color_time_input(self.temp_config, user_input, "morning_end")
                    _process_color_time_input(self.temp_config, user_input, "evening_start")
                    _process_color_time_input(self.temp_config, user_input, "evening_end")

                    # Clean up temporary keys
                    keys_to_remove = [
                        CONF_COLOR_MORNING_START_TYPE, CONF_COLOR_MORNING_START_FIXED_TIME,
                        CONF_COLOR_MORNING_START_SUN_EVENT, CONF_COLOR_MORNING_START_SUN_OFFSET,
                        CONF_COLOR_MORNING_END_TYPE, CONF_COLOR_MORNING_END_FIXED_TIME,
                        CONF_COLOR_MORNING_END_SUN_EVENT, CONF_COLOR_MORNING_END_SUN_OFFSET,
                        CONF_COLOR_EVENING_START_TYPE, CONF_COLOR_EVENING_START_FIXED_TIME,
                        CONF_COLOR_EVENING_START_SUN_EVENT, CONF_COLOR_EVENING_START_SUN_OFFSET,
                        CONF_COLOR_EVENING_END_TYPE, CONF_COLOR_EVENING_END_FIXED_TIME,
                        CONF_COLOR_EVENING_END_SUN_EVENT, CONF_COLOR_EVENING_END_SUN_OFFSET,
                    ]
                    for key in keys_to_remove:
                        self.temp_config.pop(key, None)

                    self.temp_config["morning_start_brightness"] = self.temp_config[CONF_NIGHT_BRIGHTNESS]
                    self.temp_config["evening_end_brightness"] = self.temp_config[CONF_NIGHT_BRIGHTNESS]
                    self.temp_config["morning_end_brightness"] = self.temp_config[CONF_DAY_BRIGHTNESS]
                    self.temp_config["evening_start_brightness"] = self.temp_config[CONF_DAY_BRIGHTNESS]
                    _LOGGER.debug(f"Creating entry with config: {self.temp_config}")
                    return self.async_create_entry(title="Smart Circadian Lighting", data=self.temp_config)

            schema = get_color_schema(self.hass, self.temp_config)
            _LOGGER.debug("Color schema created successfully")
            return self.async_show_form(
                step_id="color",
                data_schema=schema,
                errors=errors if errors else None
            )
        except Exception as e:
            _LOGGER.error(f"Error in async_step_color: {e}", exc_info=True)
            raise
