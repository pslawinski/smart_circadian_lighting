import logging
import hashlib
from datetime import time, datetime, timedelta

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    DOMAIN,
    CONF_COLOR_TEMP_ENABLED
)
from . import circadian_logic
from .circadian_logic import _convert_percent_to_255
from .color_temp_logic import get_color_temp_schedule, get_ct_at_time, _is_time_in_period

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up the Smart Circadian Lighting sensor platform."""
    domain_data = hass.data[DOMAIN][entry.entry_id]
    circadian_lights = domain_data.get("circadian_lights", [])
    config = domain_data["config"]

    sensors = [CircadianModeSensor(hass, light, entry) for light in circadian_lights if light is not None]

    # Add a single sensor for the brightness graph
    sensors.append(CircadianBrightnessGraphSensor(hass, entry, config))
    if config.get(CONF_COLOR_TEMP_ENABLED):
        sensors.append(CircadianColorTempGraphSensor(hass, entry, config))

    async_add_entities(sensors)


class CircadianModeSensor(SensorEntity):
    """Representation of a sensor that shows the current circadian mode for a light."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:theme-light-dark"

    def __init__(self, hass: HomeAssistant, light: object, entry: ConfigEntry) -> None:
        """Initialize the sensor."""
        self._hass = hass
        self._light = light
        self._entry = entry
        self._state = None
        self._unsub_tracker = None
        self._attr_name = f"{light.name} Circadian Mode"

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{self._light.unique_id}_circadian_mode"

    @property
    def state(self) -> str:
        """Return the state of the sensor."""
        return self._state

    async def async_added_to_hass(self) -> None:
        """Register update listener."""
        self._unsub_tracker = async_track_time_interval(
            self._hass, self.async_update_state, timedelta(seconds=60)
        )

    async def async_will_remove_from_hass(self) -> None:
        """Unregister update listener."""
        if self._unsub_tracker:
            self._unsub_tracker()
            self._unsub_tracker = None

    async def async_update_state(self, now=None):
        """Update the sensor state."""
        self._state = self._light.circadian_mode
        self.async_write_ha_state()

    async def async_update(self) -> None:
        """Request an update of the sensor."""
        await self.async_update_state()


class CircadianBrightnessGraphSensor(SensorEntity):
    """Representation of a sensor that holds the daily brightness graph for the integration."""

    _attr_icon = "mdi:chart-line"
    _attr_name = "Circadian Brightness Graph"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, config: dict) -> None:
        """Initialize the sensor."""
        self._hass = hass
        self._entry = entry
        self._config = config
        self._attr_extra_state_attributes = {}
        self._unsub_tracker = None
        self.entity_id = f"sensor.circadian_brightness_graph_{self._entry.entry_id}"
        # Compute hash of relevant config parts for brightness calculation
        relevant_config = {
            "morning_start_time": config.get("morning_start_time"),
            "morning_end_time": config.get("morning_end_time"),
            "evening_start_time": config.get("evening_start_time"),
            "evening_end_time": config.get("evening_end_time"),
            "day_brightness": config.get("day_brightness"),
            "night_brightness": config.get("night_brightness"),
        }
        self._config_hash = hashlib.md5(str(sorted(relevant_config.items())).encode()).hexdigest()
        # Generate initial graph
        self._attr_extra_state_attributes["daily_brightness_graph"] = self._generate_brightness_graph()

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{DOMAIN}_{self._entry.entry_id}_circadian_brightness_graph"

    @property
    def state(self) -> str:
        """Return the state of the sensor (the current generic circadian mode)."""
        return circadian_logic.get_circadian_mode(datetime.now(), {}, self._config)

    def _generate_brightness_graph(self) -> list[tuple[str, int]]:
        """Generate a list of [timestamp, brightness] pairs for a 24-hour period."""
        brightness_points = set()
        today = datetime.now().date()
        start_of_day = datetime.combine(today, time.min)

        morning_start, morning_end = circadian_logic.get_transition_times(
            "morning", {}, self._config
        )
        evening_start, evening_end = circadian_logic.get_transition_times(
            "evening", {}, self._config
        )

        key_times = {
            start_of_day,
            datetime.combine(today, morning_start),
            datetime.combine(today, morning_end),
            datetime.combine(today, evening_start),
            datetime.combine(today, evening_end),
            start_of_day + timedelta(days=1) - timedelta(seconds=1),
        }

        current_time = datetime.combine(today, morning_start)
        while current_time < datetime.combine(today, morning_end):
            key_times.add(current_time)
            current_time += timedelta(minutes=1)

        # Add points during daytime (morning_end to evening_start) every 30 minutes
        current_time = datetime.combine(today, morning_end)
        while current_time < datetime.combine(today, evening_start):
            key_times.add(current_time)
            current_time += timedelta(minutes=30)

        current_time = datetime.combine(today, evening_start)
        while current_time < datetime.combine(today, evening_end):
            key_times.add(current_time)
            current_time += timedelta(minutes=1)

        day_brightness_255 = _convert_percent_to_255(self._config["day_brightness"])
        night_brightness_255 = _convert_percent_to_255(self._config["night_brightness"])

        for dt in sorted(list(key_times)):
            brightness = circadian_logic.calculate_brightness_for_time(
                dt,
                {},
                self._config,
                day_brightness_255,
                night_brightness_255,
                None,  # No specific light entity
                debug_enable=False,
            )
            brightness_points.add((dt.isoformat(), brightness))

        return sorted(list(brightness_points))

    async def async_added_to_hass(self) -> None:
        """Register update listener."""
        self._unsub_tracker = async_track_time_interval(
            self._hass, self.async_update, timedelta(hours=1)
        )

    async def async_will_remove_from_hass(self) -> None:
        """Unregister update listener."""
        if self._unsub_tracker:
            self._unsub_tracker()
            self._unsub_tracker = None

    async def async_update(self, now=None) -> None:
        """Request an update of the sensor."""
        # Check if config has changed
        relevant_config = {
            "morning_start_time": self._config.get("morning_start_time"),
            "morning_end_time": self._config.get("morning_end_time"),
            "evening_start_time": self._config.get("evening_start_time"),
            "evening_end_time": self._config.get("evening_end_time"),
            "day_brightness": self._config.get("day_brightness"),
            "night_brightness": self._config.get("night_brightness"),
        }
        new_hash = hashlib.md5(str(sorted(relevant_config.items())).encode()).hexdigest()
        if new_hash != self._config_hash:
            self._config_hash = new_hash
            self._attr_extra_state_attributes[
                "daily_brightness_graph"
            ] = self._generate_brightness_graph()
        self.async_write_ha_state()

class CircadianColorTempGraphSensor(SensorEntity):
    """Representation of a sensor that holds the daily color temperature graph for the integration."""

    _attr_icon = "mdi:chart-line"
    _attr_name = "Circadian Color Temp Graph"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, config: dict) -> None:
        """Initialize the sensor."""
        self._hass = hass
        self._entry = entry
        self._config = config
        self._attr_extra_state_attributes = {}
        self._unsub_tracker = None
        self.entity_id = f"sensor.circadian_color_temp_graph_{self._entry.entry_id}"
        self._color_temp_schedule = None
        # Compute hash of relevant config parts for color temp calculation
        relevant_config = {
            "color_temp_enabled": config.get("color_temp_enabled", False),
            "sunrise_sunset_color_temp_kelvin": config.get("sunrise_sunset_color_temp_kelvin", 2700),
            "midday_color_temp_kelvin": config.get("midday_color_temp_kelvin", 5000),
            "night_color_temp_kelvin": config.get("night_color_temp_kelvin", 1800),
            "color_curve_type": config.get("color_curve_type", "cosine"),
            "color_morning_start_time": config.get("color_morning_start_time", "sync"),
            "color_morning_end_time": config.get("color_morning_end_time", "06:00:00"),
            "color_evening_start_time": config.get("color_evening_start_time", "20:00:00"),
            "color_evening_end_time": config.get("color_evening_end_time", "sync"),
        }
        self._config_hash = hashlib.md5(str(sorted(relevant_config.items())).encode()).hexdigest()

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{DOMAIN}_{self._entry.entry_id}_circadian_color_temp_graph"

    @property
    def state(self) -> str:
        """Return the state of the sensor (the current color temperature)."""
        if not self._color_temp_schedule:
            _LOGGER.debug(f"[{self._entry.entry_id}] No color temp schedule, state unknown")
            return None
        kelvin = get_ct_at_time(self._color_temp_schedule, datetime.now().time())
        sun_state = self._hass.states.get("sun.sun")
        sun_elevation = sun_state.attributes.get("elevation") if sun_state else None
        if kelvin is not None:
            _LOGGER.debug(f"[{self._entry.entry_id}] Current color temp: {kelvin}K, sun elevation: {sun_elevation}°")
            return str(kelvin)
        else:
            _LOGGER.debug(f"[{self._entry.entry_id}] No color temp for current time, sun elevation: {sun_elevation}°")
            return None

    def _get_update_interval(self, current_time):
        """Get the update interval in seconds based on current time and schedule."""
        # Check if in color temperature transition
        morning_start = self._color_temp_schedule.get("morning_start")
        morning_end = self._color_temp_schedule.get("morning_end")
        evening_start = self._color_temp_schedule.get("evening_start")
        evening_end = self._color_temp_schedule.get("evening_end")

        if (morning_start and morning_end and _is_time_in_period(current_time, morning_start, morning_end)) or \
           (evening_start and evening_end and _is_time_in_period(current_time, evening_start, evening_end)):
            return 300  # 5 minutes during color temp transitions

        # During daytime, update every 5 minutes
        sunrise = self._color_temp_schedule.get("sunrise")
        sunset = self._color_temp_schedule.get("sunset")
        if sunrise and sunset and sunrise <= current_time < sunset:
            return 300  # 5 minutes

        # Otherwise, 1 hour
        return 3600

    def _generate_color_temp_graph(self) -> list[tuple[str, int]]:
        """Generate a list of [timestamp, kelvin] pairs for a 24-hour period at actual update times."""
        _LOGGER.debug(f"[{self._entry.entry_id}] Generating color temp graph using scheduler simulation")
        if not self._color_temp_schedule:
            _LOGGER.debug(f"[{self._entry.entry_id}] No color temp schedule available for graph")
            return []

        color_temp_points = []
        current_time = time(0, 0, 0)  # Start at midnight
        end_time = time(23, 59, 59)

        while current_time < end_time:
            # Get current temperature
            kelvin = get_ct_at_time(self._color_temp_schedule, current_time)
            if kelvin is not None:
                timestamp = f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}"
                color_temp_points.append((f"{datetime.now().date()}T{timestamp}", kelvin))

            # Get update interval and advance time
            interval = self._get_update_interval(current_time)
            current_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
            new_seconds = current_seconds + interval

            # Handle day wraparound
            if new_seconds >= 86400:  # 24 hours in seconds
                break

            new_hour = new_seconds // 3600
            new_minute = (new_seconds % 3600) // 60
            new_second = new_seconds % 60
            current_time = time(new_hour, new_minute, new_second)

        _LOGGER.debug(f"[{self._entry.entry_id}] Generated {len(color_temp_points)} color temp points at update times")
        return color_temp_points

    async def async_added_to_hass(self) -> None:
        """Register update listener."""
        await self.async_update()
        self._unsub_tracker = async_track_time_interval(
            self._hass, self.async_update, timedelta(hours=1)
        )

    async def async_will_remove_from_hass(self) -> None:
        """Unregister update listener."""
        if self._unsub_tracker:
            self._unsub_tracker()
            self._unsub_tracker = None

    async def async_update(self, now=None) -> None:
        """Request an update of the sensor."""
        _LOGGER.debug(f"[{self._entry.entry_id}] Updating color temp graph sensor")
        # Check if config has changed
        relevant_config = {
            "color_temp_enabled": self._config.get("color_temp_enabled"),
            "sunrise_sunset_color_temp_kelvin": self._config.get("sunrise_sunset_color_temp_kelvin"),
            "midday_color_temp_kelvin": self._config.get("midday_color_temp_kelvin"),
            "night_color_temp_kelvin": self._config.get("night_color_temp_kelvin"),
            "color_curve_type": self._config.get("color_curve_type"),
            "color_morning_start_time": self._config.get("color_morning_start_time"),
            "color_morning_end_time": self._config.get("color_morning_end_time"),
            "color_evening_start_time": self._config.get("color_evening_start_time"),
            "color_evening_end_time": self._config.get("color_evening_end_time"),
        }
        new_hash = hashlib.md5(str(sorted(relevant_config.items())).encode()).hexdigest()
        if new_hash != self._config_hash or self._color_temp_schedule is None:
            _LOGGER.debug(f"[{self._entry.entry_id}] Config changed or no schedule, regenerating schedule and graph")
            self._config_hash = new_hash
            self._color_temp_schedule = get_color_temp_schedule(self._hass, self._config)
            if self._color_temp_schedule:
                self._attr_extra_state_attributes[
                    "daily_color_temp_graph"
                ] = self._generate_color_temp_graph()
            else:
                _LOGGER.warning(f"[{self._entry.entry_id}] Failed to generate color temp schedule")
                self._attr_extra_state_attributes["daily_color_temp_graph"] = []
        else:
            _LOGGER.debug(f"[{self._entry.entry_id}] Config unchanged and schedule exists, skipping regeneration")
        self.async_write_ha_state()
