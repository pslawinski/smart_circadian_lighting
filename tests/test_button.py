"""Tests for button.py - integration tests using pytest-homeassistant-custom-component."""

import asyncio
import pytest
from unittest.mock import patch

from homeassistant.components.button import DOMAIN as BUTTON_DOMAIN

from homeassistant.const import STATE_ON

from homeassistant.core import HomeAssistant, ServiceRegistry

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_circadian_lighting.const import DOMAIN
import custom_components.smart_circadian_lighting.light

from unittest.mock import MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_global_force_update_button_press():
    """Test global force update button press calls async_force_update_circadian on all lights."""
    hass = MagicMock()
    hass.config_entries = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.async_block_till_done = AsyncMock()

    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="test_multiple_lights",
        data={
            "lights": ["light.test_light1", "light.test_light2"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
            "color_temp_enabled": False,
        },
    )

    with patch('custom_components.smart_circadian_lighting.light.CircadianLight.async_force_update_circadian') as mock_force_update:
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
async def test_global_clear_override_button_press():
    """Test global clear override button press."""
    hass = MagicMock()
    hass.config_entries = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.async_block_till_done = AsyncMock()

    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="test_single_light",
        data={
            "lights": ["light.test_light"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
            "color_temp_enabled": False,
        },
    )

    button_entity_id = f"button.{entry.entry_id}_clear_override_all"
    await hass.services.async_call(
        BUTTON_DOMAIN,
        "press",
        {"entity_id": button_entity_id},
        blocking=True,
    )
    await hass.async_block_till_done()
