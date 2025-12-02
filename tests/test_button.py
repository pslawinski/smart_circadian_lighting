"""Tests for button.py - integration tests using pytest-homeassistant-custom-component."""

import pytest
from unittest.mock import patch, MagicMock

from homeassistant.components.button import DOMAIN as BUTTON_DOMAIN

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_circadian_lighting.const import DOMAIN
import custom_components.smart_circadian_lighting.light

from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_global_force_update_button_press(mock_config_entry):
    """Test global force update button press calls async_force_update_circadian on all lights."""
    hass, entry = mock_config_entry

    # Create mock lights
    light1 = MagicMock()
    light1.async_force_update_circadian = AsyncMock()
    light2 = MagicMock()
    light2.async_force_update_circadian = AsyncMock()

    # Set up component data structure so button setup works
    hass.data = {DOMAIN: {entry.entry_id: {"circadian_lights": [light1, light2]}}}

    # Manually create the global force update button
    from custom_components.smart_circadian_lighting.button import GlobalForceUpdateButton
    button = GlobalForceUpdateButton(entry, [light1, light2])
    button_entity_id = f"button.{entry.entry_id}_force_update_all"

    # Add button to hass
    hass.states.async_set(button_entity_id, "on")

    # Press the button
    await button.async_press()

    # Verify called on both lights
    light1.async_force_update_circadian.assert_called_once()
    light2.async_force_update_circadian.assert_called_once()

@pytest.mark.asyncio
async def test_global_clear_override_button_press(mock_config_entry):
    """Test global clear override button press."""
    hass, entry = mock_config_entry

    # Create mock light
    light = MagicMock()
    light.async_clear_manual_override = AsyncMock()

    # Set up component data structure
    hass.data = {DOMAIN: {entry.entry_id: {"circadian_lights": [light]}}}

    # Manually create the global clear override button
    from custom_components.smart_circadian_lighting.button import GlobalClearOverrideButton
    button = GlobalClearOverrideButton(entry, [light])
    button_entity_id = f"button.{entry.entry_id}_clear_override_all"

    # Add button to hass
    hass.states.async_set(button_entity_id, "on")

    # Press the button
    await button.async_press()

    # Verify called on light
    light.async_clear_manual_override.assert_called_once()
