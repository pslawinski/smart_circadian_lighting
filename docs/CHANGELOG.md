# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-21

### Added
- **Initial release** of Smart Circadian Lighting integration
- **Natural circadian rhythm simulation** with automatic brightness and color temperature adjustments
- **Flexible scheduling** with sunrise/sunset-based or fixed time transitions
- **Smart manual override detection** that respects user preferences
- **Device-aware behavior** with special handling for Kasa dimmers and standard lights
- **Comprehensive color temperature support** with customizable day/night temperatures
- **Smooth transitions** between lighting states with configurable duration
- **Post-transition verification** to ensure lights reach exact target values
- **Extensive test suite** with 129+ automated tests
- **HACS compatibility** for easy installation through Home Assistant Community Store

### Features
- **Brightness Control**: Automatic adjustment between day (100%) and night (10%) brightness levels
- **Color Temperature**: Support for 1500K-6500K range with smooth transitions
- **Transition Timing**: Configurable morning/evening transition periods
- **Override Management**: Intelligent detection and automatic expiration of manual adjustments
- **Light Compatibility**: Works with any Home Assistant light entity
- **Kasa Integration**: Enhanced support for Kasa smart dimmers with offline brightness control

### Technical
- **Robust state management** with persistence across restarts
- **Performance optimized** with entity registry caching and rate-limited logging
- **Error resilient** with retry logic and graceful degradation
- **Type-safe** with comprehensive type hints
- **Well-tested** with high test coverage and integration tests

### Documentation
- **Complete README** with installation and configuration instructions
- **Contributing guidelines** for developers
- **Development setup** with modern Python tooling (ruff, black, mypy)
- **License information** (Creative Commons Attribution-NonCommercial 4.0)

---

## Development Notes

This is the initial release of Smart Circadian Lighting. The integration has been thoroughly tested and is production-ready.

### Known Limitations
- Color temperature transitions require compatible light bulbs
- Some device-specific transition behaviors may vary
- Advanced scheduling features may require sun integration setup

### Future Plans
- Progressive configuration UI improvements
- Additional light brand integrations
- Enhanced automation and scene support
- Performance optimizations for large installations

---

**Full Changelog**: [View on GitHub](https://github.com/pslawinski/smart_circadian_lighting/compare/v0.1.0...HEAD)