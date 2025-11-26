"""Tests for button.py - integration tests using pytest-homeassistant-custom-component."""

import pytest

from homeassistant.components.button import DOMAIN as BUTTON_DOMAIN

from homeassistant.const import STATE_ON

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_circadian_lighting.const import DOMAIN

@pytest.mark.asyncio
async def test_global_force_update_button_press(hass):
    """Test global force update button press calls async_force_update_circadian on all lights."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="test_entry",
        data={
            "lights": ["light.test_light1", "light.test_light2"],
            "day_brightness": 100,
            "night_brightness": 10,
            "morning_start_time": "06:00:00",
            "morning_end_time": "07:00:00",
            "evening_start_time": "20:00:00",
            "evening_end_time": "21:00:00",
        },
    )
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Patch the light async_force_update_circadian
    with patch("custom_components.smart_circadian_lighting.light.CircadianLight.async_force_update_circadian") as mock_force_update:
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
async def test_global_clear_override_button_press(hass):
    """Test global clear override button press."""
    # Similar setup
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="test_entry",
        data={
            "lights": ["light.test_light1"],
            "day_brightness": 100,
            "night_brightness": 10,
        },
    )
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    button_entity_id = f"button.{entry.entry_id}_clear_override_all"
    await hass.services.async_call(
        BUTTON_DOMAIN,
        "press",
        {"entity_id": button_entity_id},
        blocking=True,
    )
    await hass.async_block_till_done()
