"""Button platform for Smart Circadian Lighting."""
from __future__ import annotations

import logging
from typing import cast

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import testing
from .const import DEBUG_FEATURES, DOMAIN, SIGNAL_CIRCADIAN_LIGHT_TESTING_STATE_CHANGED
from .light import CircadianLight

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the button platform."""
    domain_data = hass.data[DOMAIN][config_entry.entry_id]
    circadian_lights = cast(list[CircadianLight], domain_data.get("circadian_lights", []))

    if not circadian_lights:
        _LOGGER.warning("No circadian lights found to create buttons for.")

    buttons = []
    # Add per-entity buttons
    for light in circadian_lights:
        if light is None:
            continue
        buttons.append(ClearOverrideButton(light))
        buttons.append(ForceUpdateButton(light))
        if DEBUG_FEATURES:
            buttons.append(RunTestCycleButton(light))
            buttons.append(CancelTestCycleButton(light))

    # Add global buttons
    valid_lights = [light for light in circadian_lights if light is not None]
    buttons.append(GlobalClearOverrideButton(config_entry, valid_lights))
    buttons.append(GlobalForceUpdateButton(config_entry, valid_lights))
    if DEBUG_FEATURES:
        buttons.append(GlobalRunTestCycleButton(config_entry, valid_lights))
        buttons.append(GlobalCancelTestCycleButton(config_entry, valid_lights))

    async_add_entities(buttons)

class ClearOverrideButton(ButtonEntity):
    """Representation of a button to clear manual override for a single entity."""
    _attr_has_entity_name = True

    def __init__(self, circadian_light: CircadianLight) -> None:
        """Initialize the button."""
        self._circadian_light = circadian_light
        self._attr_name = "Clear Manual Override"
        self._attr_unique_id = f"{self._circadian_light.unique_id}_clear_override"

        # Derive underlying light entity_id from the circadian light's unique_id
        light_entity_id = self._circadian_light.unique_id.replace(f"{DOMAIN}_", "")
        light_state = self._circadian_light.hass.states.get(light_entity_id)
        device_name = light_state.name if light_state else light_entity_id

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._circadian_light.unique_id)},
            name=device_name,
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        _LOGGER.info(f"Clear manual override button pressed for {self._circadian_light.name}")
        await self._circadian_light.async_clear_manual_override()


class ForceUpdateButton(ButtonEntity):
    """Representation of a button to force an update for a single entity."""
    _attr_has_entity_name = True

    def __init__(self, circadian_light: CircadianLight) -> None:
        """Initialize the button."""
        self._circadian_light = circadian_light
        self._attr_name = "Force Circadian Update"
        self._attr_unique_id = f"{self._circadian_light.unique_id}_force_update"

        # Derive underlying light entity_id from the circadian light's unique_id
        light_entity_id = self._circadian_light.unique_id.replace(f"{DOMAIN}_", "")
        light_state = self._circadian_light.hass.states.get(light_entity_id)
        device_name = light_state.name if light_state else light_entity_id

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._circadian_light.unique_id)},
            name=device_name,
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        _LOGGER.info(f"Force circadian update button pressed for {self._circadian_light.name}")
        await self._circadian_light.async_force_update_circadian()


class RunTestCycleButton(ButtonEntity):
    """Representation of a button to run a test cycle for a single entity."""
    _attr_has_entity_name = True

    def __init__(self, circadian_light: CircadianLight) -> None:
        """Initialize the button."""
        self._circadian_light = circadian_light
        self._attr_name = "Run Test Cycle"
        self._attr_unique_id = f"{self._circadian_light.unique_id}_run_test_cycle"

        # Derive underlying light entity_id from the circadian light's unique_id
        light_entity_id = self._circadian_light.unique_id.replace(f"{DOMAIN}_", "")
        light_state = self._circadian_light.hass.states.get(light_entity_id)
        device_name = light_state.name if light_state else light_entity_id

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._circadian_light.unique_id)},
            name=device_name,
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        _LOGGER.info(f"Run test cycle button pressed for {self._circadian_light.name}")
        await self._circadian_light.async_run_test_cycle(60)

class CancelTestCycleButton(ButtonEntity):
    """Representation of a button to cancel a test cycle for a single entity."""
    _attr_has_entity_name = True

    def __init__(self, circadian_light: CircadianLight) -> None:
        """Initialize the button."""
        self._circadian_light = circadian_light
        self._attr_name = "Cancel Test Cycle"
        self._attr_unique_id = f"{self._circadian_light.unique_id}_cancel_test_cycle"
        self._attr_available = self._circadian_light.is_testing

        # Derive underlying light entity_id from the circadian light's unique_id
        light_entity_id = self._circadian_light.unique_id.replace(f"{DOMAIN}_", "")
        light_state = self._circadian_light.hass.states.get(light_entity_id)
        device_name = light_state.name if light_state else light_entity_id

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._circadian_light.unique_id)},
            name=device_name,
        )

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_CIRCADIAN_LIGHT_TESTING_STATE_CHANGED,
                self._update_availability,
            )
        )

    @callback
    def _update_availability(self, entity_id: str, is_testing: bool) -> None:
        """Update the button's availability if the signal is for this light."""
        if entity_id == self._circadian_light.entity_id:
            self._attr_available = is_testing
            self.async_write_ha_state()

    async def async_press(self) -> None:
        """Handle the button press."""
        _LOGGER.info(f"Cancel test cycle button pressed for {self._circadian_light.name}")
        await self._circadian_light.async_cancel_test_cycle()

class GlobalClearOverrideButton(ButtonEntity):
    """Representation of a button to clear manual override for all entities."""
    _attr_has_entity_name = True

    def __init__(self, config_entry: ConfigEntry, circadian_lights: list[CircadianLight]) -> None:
        """Initialize the global button."""
        self._circadian_lights = circadian_lights
        self._attr_name = "Clear Manual Override All"
        self._attr_unique_id = f"{config_entry.entry_id}_clear_override_all"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            name=config_entry.title,
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        _LOGGER.info("Global clear manual override button pressed")
        for light in self._circadian_lights:
            await light.async_clear_manual_override()


class GlobalForceUpdateButton(ButtonEntity):
    """Representation of a button to force an update for all entities."""
    _attr_has_entity_name = True

    def __init__(self, config_entry: ConfigEntry, circadian_lights: list[CircadianLight]) -> None:
        """Initialize the global button."""
        self._circadian_lights = circadian_lights
        self._attr_name = "Force Circadian Update All"
        self._attr_unique_id = f"{config_entry.entry_id}_force_update_all"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            name=config_entry.title,
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        _LOGGER.info("Global force update button pressed")
        for light in self._circadian_lights:
            await light.async_force_update_circadian()


class GlobalRunTestCycleButton(ButtonEntity):
    """Representation of a button to run a test cycle for all entities."""
    _attr_has_entity_name = True

    def __init__(self, config_entry: ConfigEntry, circadian_lights: list[CircadianLight]) -> None:
        """Initialize the global button."""
        self._circadian_lights = circadian_lights
        self._attr_name = "Run Test Cycle All"
        self._attr_unique_id = f"{config_entry.entry_id}_run_test_cycle_all"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            name=config_entry.title,
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        _LOGGER.info("Global run test cycle button pressed")
        await testing.async_run_test_cycle_all(self._circadian_lights, 300)

class GlobalCancelTestCycleButton(ButtonEntity):
    """Representation of a button to cancel a test cycle for all entities."""
    _attr_has_entity_name = True

    def __init__(self, config_entry: ConfigEntry, circadian_lights: list[CircadianLight]) -> None:
        """Initialize the global button."""
        self._circadian_lights = circadian_lights
        self._attr_name = "Cancel Test Cycle All"
        self._attr_unique_id = f"{config_entry.entry_id}_cancel_test_cycle_all"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            name=config_entry.title,
        )
        self._attr_available = any(light.is_testing for light in self._circadian_lights)

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_CIRCADIAN_LIGHT_TESTING_STATE_CHANGED,
                self._update_availability,
            )
        )

    @callback
    def _update_availability(self, entity_id: str, is_testing: bool) -> None:
        """Update the button's availability."""
        self._attr_available = any(light.is_testing for light in self._circadian_lights)
        self.async_write_ha_state()

    async def async_press(self) -> None:
        """Handle the button press."""
        _LOGGER.info("Global cancel test cycle button pressed")
        for light in self._circadian_lights:
            if light.is_testing:
                await light.async_cancel_test_cycle()


