"""Pytest configuration for mocking Home Assistant modules."""

import sys
from unittest.mock import MagicMock

# Mock all Home Assistant modules before any test collection
mock_modules = [
    'homeassistant',
    'homeassistant.config_entries',
    'homeassistant.core',
    'homeassistant.helpers.sun',
    'homeassistant.helpers.entity',
    'homeassistant.helpers.entity_platform',
    'homeassistant.helpers.event',
    'homeassistant.helpers.storage',
    'homeassistant.helpers.dispatcher',
    'homeassistant.helpers',
    'homeassistant.helpers.entity_registry',
    'homeassistant.helpers.selector',
    'homeassistant.helpers.service',
    'homeassistant.components.light',
    'homeassistant.components.sensor',
    'homeassistant.const',
    'homeassistant.data_entry_flow',
    'homeassistant.exceptions',
    'homeassistant.util',
    'homeassistant.util.dt',
# Removed smart_circadian_lighting modules from mocking to allow real imports
]

for module in mock_modules:
    sys.modules[module] = MagicMock()
