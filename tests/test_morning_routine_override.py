import asyncio
import logging
from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from freezegun import freeze_time
from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.const import STATE_ON
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_circadian_lighting import DOMAIN
from custom_components.smart_circadian_lighting.light import CircadianLight

_LOGGER = logging.getLogger(__name__)

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock()
    hass.data = {"entity_registry": MagicMock()}
    
    tasks = []
    def async_create_task(coro, name=None):
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task
        
    hass.async_create_task = MagicMock(side_effect=async_create_task)
    hass._tasks = tasks
    return hass

@pytest.fixture
def config_entry():
    return MockConfigEntry(
        domain=DOMAIN,
        data={
            "lights": ["light.test_bulb"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "05:15:00",
            "morning_end_time": "06:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:30:00",
            "manual_override_threshold": 5,
            "morning_override_clear_time": "08:00:00",
            "evening_override_clear_time": "02:00:00",
            "color_temp_enabled": False,
        },
        entry_id="test_entry",
    )

@pytest.mark.asyncio
async def test_zwave_morning_routine_override_repro(mock_hass, config_entry):
    # Setup Z-Wave entity registry entry
    mock_registry = mock_hass.data["entity_registry"]
    mock_entry = MagicMock()
    mock_entry.platform = "zwave_js"
    mock_registry.async_get.return_value = mock_entry

    store = MagicMock()
    store.async_load = AsyncMock(return_value={})
    store.async_save = AsyncMock()
    
    config = config_entry.data.copy()
    mock_hass.data[DOMAIN] = {config_entry.entry_id: {"config": config, "manual_overrides_enabled": True}}
    
    with freeze_time("2024-01-01 05:15:00"):
        light = CircadianLight(mock_hass, "light.test_bulb", config, config_entry, store)
        light.hass = mock_hass
        light.entity_id = "light.circadian_test_bulb"
        light._first_update_done = True
        
        # Initial state: OFF
        old_state = MagicMock(state="off", attributes={})
        mock_hass.states.get.return_value = old_state
        await light.async_added_to_hass()
        await light.async_update_brightness(force_update=True)
        
        # User script commands Z-Wave light to 100%
        # However, sometimes Z-Wave sends a state change for just turning ON first,
        # then another with the brightness.
        on_only_state = MagicMock(state=STATE_ON, attributes={})
        on_only_event = MagicMock(data={"new_state": on_only_state, "old_state": old_state})
        
        # User script finally sets 100%
        full_on_state = MagicMock(state=STATE_ON, attributes={ATTR_BRIGHTNESS: 255})
        full_on_event = MagicMock(data={"new_state": full_on_state, "old_state": on_only_state})
        
        # Track calls to update the light
        light.async_update_light = AsyncMock()
        
        # Simulate the sequence
        with patch.object(light, "_get_current_brightness_with_refresh", AsyncMock(side_effect=[None, 255, 255, 255])):
            # 1. Light turns ON (brightness not yet reported/refreshed)
            mock_hass.states.get.return_value = on_only_state
            print("\nDEBUG: Sending ON-only event")
            await light._async_entity_state_changed(on_only_event)
            print(f"DEBUG: After ON-only event: overridden={light._is_overridden}")
            
            # 2. Next update cycle (BEFORE user script brightness reached HA)
            # This is where the "slamming" might happen
            print("\nDEBUG: Running update cycle before 255 reaches HA")
            mock_hass.states.get.return_value = on_only_state # Light is ON but brightness is unknown or default
            await light.async_update_brightness(force_update=True)
            
            # 3. User script brightness reaches HA
            mock_hass.states.get.return_value = full_on_state
            print("\nDEBUG: Sending full ON event (255)")
            await light._async_entity_state_changed(full_on_event)
            print(f"DEBUG: After full ON event: overridden={light._is_overridden}, is_soft={light._is_soft_override}")
            print(f"DEBUG: After update cycle: overridden={light._is_overridden}, brightness={light._brightness}")
        
        # Assertions
        assert light._is_overridden is True, "Override was not detected"
        
        for call in light.async_update_light.call_args_list:
            brightness = call.kwargs.get("brightness")
            if brightness is not None and brightness < 250:
                 pytest.fail(f"Z-Wave light was forced to {brightness} despite override!")

        # Cleanup
        await light.async_will_remove_from_hass()
        if mock_hass._tasks:
            for task in mock_hass._tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*mock_hass._tasks, return_exceptions=True)

    print("\nTest passed: Z-Wave override respected.")
