"""Tests for button.py - integration tests using pytest-homeassistant-custom-component."""

import pytest
from unittest.mock import patch

from homeassistant.components.button import DOMAIN as BUTTON_DOMAIN

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_circadian_lighting.const import DOMAIN
import custom_components.smart_circadian_lighting.light

from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_global_force_update_button_press(mock_config_entry):
    """Test global force update button press calls async_force_update_circadian on all lights."""
    hass, entry = await mock_config_entry
    # Update entry data to match test expectations
    entry.async_update_entry = AsyncMock()
    await entry.async_update_entry(hass, {
        "lights": ["light.test_light1", "light.test_light2"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "06:00:00",
        "morning_end_time": "07:00:00",
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
        "color_temp_enabled": False,
    })

    with patch('custom_components.smart_circadian_lighting.light.CircadianLight.async_force_update_circadian') as mock_force_update, \
         patch('homeassistant.core.ServiceRegistry.async_call', new_callable=AsyncMock) as mock_service_call:

        # Press the global force update button
        button_entity_id = f"button.{entry.entry_id}_force_update_all"
        await hass.services.async_call(
            BUTTON_DOMAIN,
            "press",
            {"entity_id": button_entity_id},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify called on lights
        assert mock_force_update.call_count >= 2

@pytest.mark.asyncio
async def test_global_clear_override_button_press(mock_config_entry):
    """Test global clear override button press."""
    hass, entry = await mock_config_entry
    # Add async_update_entry method to avoid deprecation warning
    entry.async_update_entry = AsyncMock()
    # Update entry data to match test expectations
    await entry.async_update_entry(hass, {
        "lights": ["light.test_light"],
        "day_brightness": 100,
        "night_brightness": 10,
        "morning_start_time": "06:00:00",
        "morning_end_time": "07:00:00",
        "evening_start_time": "20:00:00",
        "evening_end_time": "21:00:00",
        "color_temp_enabled": False,
    })

    with patch('homeassistant.core.ServiceRegistry.async_call', new_callable=AsyncMock) as mock_service_call:

        button_entity_id = f"button.{entry.entry_id}_clear_override_all"
        await hass.services.async_call(
            BUTTON_DOMAIN,
            "press",
            {"entity_id": button_entity_id},
            blocking=True,
        )
        await hass.async_block_till_done()
