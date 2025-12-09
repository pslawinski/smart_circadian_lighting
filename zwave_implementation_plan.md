# Z-Wave JS Light Support Implementation Plan

## Overview
Add support for Z-Wave JS lights (including Zooz ZEN77 dimmers) to the Smart Circadian Lighting component. Zooz ZEN77 uses parameter 18 for custom brightness preloading, which is standard for Z-Wave dimmers.

## Key Requirements
- When Z-Wave light is **off**: Set parameter 18 to preload brightness for next turn-on
- When Z-Wave light is **on**: Set both parameter 18 (preload) AND current brightness via light.turn_on
- During transitions: Handle ramping appropriately based on on/off state
- Update config flow to pre-select Z-Wave dimmers
- Comprehensive testing using official Z-Wave JS mock driver

## Implementation Steps

### 1. Update Config Flow (`config_flow.py`)
- Modify `_get_kasa_dimmer_entities()` to also return Z-Wave dimmer entities (platform == "zwave_js")
- Rename function to `_get_supported_dimmer_entities()` for clarity
- Pre-select both Kasa and Z-Wave dimmers in initial config flow

### 2. Enhance Light Handling (`light.py`)
- Add Z-Wave detection: `entity_entry.platform == "zwave_js"`
- **When off**: Call `zwave_js.set_config_parameter` with parameter 18 = target_brightness (scaled 0-99)
- **When on**: Call both `zwave_js.set_config_parameter` (parameter 18) AND `light.turn_on` (current brightness + transition)
- Handle transitions: if off set parameter 18, if on use light.turn_on with transition time
- Update state change handling for Z-Wave lights turning on with preloaded brightness

### 3. Service Compatibility
- Verify `force_circadian_update` works with Z-Wave dual-setting logic
- Ensure all existing services support Z-Wave lights

### 4. Comprehensive Testing (`test_light.py`)
**Testing Infrastructure:**
- Use `MockZwaveJsServer` fixture from `homeassistant.components.zwave_js.tests.conftest`
- Set up real Z-Wave dimmer devices with Multilevel Switch and Configuration command classes
- Drive all state changes via real `ValueNotification` events
- Use Syrupy for snapshot testing

**Test Cases:**
- `test_zwave_off_parameter_setting`: Verify parameter 18 setting via ValueNotifications when light off
- `test_zwave_on_dual_setting`: Verify both parameter 18 and brightness updates when light on
- `test_zwave_transitions_ramping`: Test multilevel switch CC ramping via ValueNotification sequences
- `test_zwave_node_lifecycle`: Test node states (not ready → ready, asleep → awake, dead)
- `test_zwave_force_refresh`: Test force_refresh service via real refresh commands
- `test_zwave_services_all`: Test all services via `hass.services.async_call()` with ValueNotification verification
- `test_zwave_reinterview_firmware`: Test re-interview completion and firmware update events
- `test_zwave_snapshots`: Syrupy snapshots for entity states, diagnostics, and attributes
- `test_regression_full_suite`: Ensure all existing tests pass

**Testing Best Practices:**
- No manual mocking of Z-Wave services - use official mock driver only
- All assertions via public APIs (`async_wait_for_entity_state`, entity registry)
- Real service calls verified through resulting ValueNotifications
- Comprehensive node lifecycle and event testing

## Checklist
- [x] Research Zooz ZEN77 and confirm generic Z-Wave dimmer
- [x] Analyze current Kasa dimmer special handling
- [x] Update config flow for Z-Wave pre-selection
- [x] Implement Z-Wave detection in light.py
- [x] Implement parameter 18 setting when off
- [x] Implement dual setting (parameter + brightness) when on
- [x] Handle transitions properly for Z-Wave lights
- [x] Verify all services work with Z-Wave lights
- [x] Create test_light.py with MockZwaveJsServer
- [ ] Set up Z-Wave dimmer devices in mock driver
- [ ] Add Syrupy snapshot tests
- [ ] Test parameter setting via ValueNotifications
- [ ] Test dual setting via ValueNotifications
- [ ] Test transitions with ramping notifications
- [ ] Test node lifecycle states
- [ ] Test force_refresh via real commands
- [ ] Test re-interview and firmware events
- [ ] Test all services with event verification
- [x] Run full test suite - no regressions

## Testing Notes
- Run test suite after each major change
- Use `pytest --snapshot-update` to update Syrupy snapshots when expected
- Focus on integration testing with real HA services and mock driver events
- Never mock Z-Wave internals - let mock driver handle all Z-Wave protocol simulation