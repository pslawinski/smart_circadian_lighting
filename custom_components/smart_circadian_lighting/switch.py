"""Switch platform for Smart Circadian Lighting."""

from __future__ import annotations

import logging

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the switch platform."""
    domain_data = hass.data[DOMAIN][config_entry.entry_id]

    # Add global switch
    async_add_entities([DisableManualOverridesSwitch(config_entry, domain_data)])


class DisableManualOverridesSwitch(SwitchEntity):
    """Representation of a switch to disable manual overrides."""

    _attr_has_entity_name = True
    _attr_name = "Disable Manual Overrides"

    def __init__(self, config_entry: ConfigEntry, domain_data: dict) -> None:
        """Initialize the switch."""
        self._config_entry = config_entry
        self._domain_data = domain_data
        self._attr_unique_id = f"{config_entry.entry_id}_disable_manual_overrides"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            name=config_entry.title,
        )

    @property
    def is_on(self) -> bool:
        """Return true if manual overrides are disabled."""
        return not self._domain_data.get("manual_overrides_enabled", True)

    async def async_turn_on(self, **kwargs) -> None:
        """Disable manual overrides."""
        self._domain_data["manual_overrides_enabled"] = False
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs) -> None:
        """Enable manual overrides."""
        self._domain_data["manual_overrides_enabled"] = True
        self.async_write_ha_state()
