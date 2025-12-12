"""The Smart Circadian Lighting integration."""
import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.service import async_register_admin_service

from .const import DOMAIN

LIGHT_PLATFORM = ["light"]
DEPENDENT_PLATFORMS = ["button", "binary_sensor", "sensor", "switch"]
_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Smart Circadian Lighting component."""

    async def _async_force_circadian_update_service_handler(
        service_call: ServiceCall,
    ) -> None:
        """Handle the force_circadian_update service by iterating over all entries."""
        _LOGGER.debug("Force circadian update service called for domain %s", DOMAIN)
        domain_data = hass.data.get(DOMAIN)
        if not domain_data:
            _LOGGER.warning("No data found for domain %s. Cannot force update.", DOMAIN)
            return

        update_tasks = []
        for entry_id, entry_data in domain_data.items():
            circadian_lights = entry_data.get("circadian_lights", [])
            _LOGGER.debug("Found %d lights for entry %s", len(circadian_lights), entry_id)
            for light in circadian_lights:
                update_tasks.append(light.async_force_update_circadian())

        if not update_tasks:
            _LOGGER.warning("No circadian lights found to force update.")
            return

        await asyncio.gather(*update_tasks)

    # Register the service if it's not already registered.
    if not hass.services.has_service(DOMAIN, "force_circadian_update"):
        async_register_admin_service(
            hass,
            DOMAIN,
            "force_circadian_update",
            _async_force_circadian_update_service_handler,
        )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Smart Circadian Lighting from a config entry."""
    entry.async_on_unload(entry.add_update_listener(update_listener))

    # Initialize the structure for this entry
    domain_data = hass.data.setdefault(DOMAIN, {})
    # Merge data and options for complete config
    config = {**entry.data, **entry.options}
    # Ensure color temp defaults are set if enabled
    if config.get("color_temp_enabled"):
        from .const import (
            DEFAULT_COLOR_CURVE_TYPE,
            DEFAULT_MIDDAY_COLOR_TEMP_KELVIN,
            DEFAULT_NIGHT_COLOR_TEMP_KELVIN,
            DEFAULT_SUNRISE_SUNSET_COLOR_TEMP_KELVIN,
        )
        config.setdefault("sunrise_sunset_color_temp_kelvin", DEFAULT_SUNRISE_SUNSET_COLOR_TEMP_KELVIN)
        config.setdefault("midday_color_temp_kelvin", DEFAULT_MIDDAY_COLOR_TEMP_KELVIN)
        config.setdefault("night_color_temp_kelvin", DEFAULT_NIGHT_COLOR_TEMP_KELVIN)
        config.setdefault("color_curve_type", DEFAULT_COLOR_CURVE_TYPE)
        config.setdefault("color_morning_start_time", "sync")
        config.setdefault("color_morning_end_time", "sunrise")
        config.setdefault("color_evening_start_time", "sunset")
        config.setdefault("color_evening_end_time", "sync")
    domain_data[entry.entry_id] = {
        "config": config,
        "circadian_lights": [],
        "manual_overrides_enabled": True,
    }

    # Set up the light platform first and wait for it to finish
    await hass.config_entries.async_forward_entry_setups(entry, LIGHT_PLATFORM)

    # Now set up the dependent platforms
    await hass.config_entries.async_forward_entry_setups(entry, DEPENDENT_PLATFORMS)

    if not hass.services.has_service(DOMAIN, "start_test_transition"):
        _LOGGER.info("Registering services")

        async def _async_start_test_transition_service_handler(
            service_call: ServiceCall,
        ) -> None:
            """Handle the start_test_transition service."""
            entity_id = service_call.data.get("entity_id")
            mode = service_call.data.get("mode")
            duration = service_call.data.get("duration")
            hold_duration = service_call.data.get("hold_duration")
            include_color_temp = service_call.data.get("include_color_temp", False)
            for light in hass.data[DOMAIN][entry.entry_id]["circadian_lights"]:
                if light.entity_id == entity_id:
                    await light.start_test_transition(
                        mode, duration, hold_duration, include_color_temp
                    )
                    break

        async def _async_cancel_test_transition_service_handler(
            service_call: ServiceCall,
        ) -> None:
            """Handle the cancel_test_transition service."""
            entity_id = service_call.data.get("entity_id")
            for light in hass.data[DOMAIN][entry.entry_id]["circadian_lights"]:
                if light.entity_id == entity_id:
                    await light.cancel_test_transition()
                    break

        async def _async_set_temporary_transition_service_handler(
            service_call: ServiceCall,
        ) -> None:
            """Handle the set_temporary_transition service."""
            entity_id = service_call.data.get("entity_id")
            mode = service_call.data.get("mode")
            start_time = service_call.data.get("start_time")
            end_time = service_call.data.get("end_time")
            duration = service_call.data.get("duration")
            for light in hass.data[DOMAIN][entry.entry_id]["circadian_lights"]:
                if light.entity_id == entity_id:
                    await light.set_temporary_transition(
                        mode, start_time, end_time, duration
                    )
                    break

        async def _async_end_current_transition_service_handler(
            service_call: ServiceCall,
        ) -> None:
            """Handle the end_current_transition service."""
            entity_id = service_call.data.get("entity_id")
            for light in hass.data[DOMAIN][entry.entry_id]["circadian_lights"]:
                if light.entity_id == entity_id:
                    await light.end_current_transition()
                    break

        hass.services.async_register(
            DOMAIN,
            "start_test_transition",
            _async_start_test_transition_service_handler,
        )
        hass.services.async_register(
            DOMAIN,
            "cancel_test_transition",
            _async_cancel_test_transition_service_handler,
        )
        hass.services.async_register(
            DOMAIN,
            "set_temporary_transition",
            _async_set_temporary_transition_service_handler,
        )
        hass.services.async_register(
            DOMAIN,
            "end_current_transition",
            _async_end_current_transition_service_handler,
        )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    all_platforms = LIGHT_PLATFORM + DEPENDENT_PLATFORMS
    unload_ok = await hass.config_entries.async_unload_platforms(entry, all_platforms)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)

    return unload_ok


async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)
