import logging

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, SIGNAL_OVERRIDE_STATE_CHANGED

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up the Smart Circadian Lighting binary sensor platform."""
    domain_data = hass.data[DOMAIN][entry.entry_id]
    circadian_lights = domain_data.get("circadian_lights", [])

    sensors = [
        CircadianOverrideSensor(light) for light in circadian_lights if light is not None
    ]
    if sensors:
        async_add_entities(sensors)


class CircadianOverrideSensor(BinarySensorEntity):
    """Representation of a sensor that shows if a circadian light is overridden."""

    _attr_has_entity_name = True

    def __init__(self, light) -> None:
        """Initialize the sensor."""
        self._light = light
        self._attr_name = "Manual Override"
        self._attr_unique_id = f"{light.unique_id}_override_sensor"
        self._attr_device_info = light.device_info

    @property
    def is_on(self) -> bool:
        """Return the state of the sensor."""
        return self._light._is_overridden

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_OVERRIDE_STATE_CHANGED}_{self._light.entity_id}",
                self.async_write_ha_state
            )
        )
