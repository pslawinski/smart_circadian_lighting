# Smart Circadian Lighting

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)

A sophisticated Home Assistant integration that automatically adjusts your smart lights based on natural circadian rhythms, promoting better sleep and well-being.

## Features

- **Natural Light Simulation**: Automatically adjusts brightness and color temperature throughout the day to mimic natural sunlight patterns
- **Flexible Scheduling**: Supports sunrise/sunset-based timing or fixed time schedules
- **Smart Transitions**: Smooth brightness and color temperature transitions between day and night
- **Manual Override Detection**: Automatically detects when users manually adjust lights and respects their preferences
- **Device-Aware**: Special handling for different light types (Kasa dimmers, standard lights)
- **Comprehensive Testing**: 129+ automated tests ensure reliability

## Installation

### HACS (Recommended)

1. Ensure [HACS](https://hacs.xyz/) is installed
2. Add this repository as a custom repository in HACS:
   - URL: `https://github.com/pslawinski/smart_circadian_lighting`
   - Category: Integration
3. Search for "Smart Circadian Lighting" and install
4. Restart Home Assistant

### Manual Installation

1. Download the latest release from [GitHub Releases](https://github.com/pslawinski/smart_circadian_lighting/releases)
2. Extract the `custom_components/smart_circadian_lighting/` folder to your Home Assistant `custom_components` directory
3. Restart Home Assistant

## Configuration

After installation, add the integration through the Home Assistant UI:

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "Smart Circadian Lighting"
3. Select your lights and configure the schedule

### Configuration Options

- **Transition Times**: Set morning/evening transition start and end times
- **Brightness Levels**: Configure day and night brightness (0-100%)
- **Color Temperature**: Enable color temperature adjustments with customizable temperatures
- **Manual Override**: Set sensitivity for detecting manual light adjustments
- **Time Sources**: Choose between fixed times or sunrise/sunset-based scheduling

## How It Works

The integration creates virtual light entities that control your physical lights according to natural circadian rhythms:

- **Morning Transition**: Gradually increases brightness and adjusts color temperature from cool nighttime to warm daytime
- **Daytime**: Maintains bright, cool white light
- **Evening Transition**: Gradually decreases brightness and shifts to warmer color temperatures
- **Nighttime**: Low brightness with warm color temperature

### Manual Override Handling

The integration intelligently detects when users manually adjust lights and temporarily suspends automatic control to respect user preferences. Overrides automatically expire based on your configured schedule.

## Requirements

- Home Assistant 2024.1.0 or later
- Smart lights that support brightness control
- Optional: Color temperature support for enhanced circadian simulation

## Contributing

Contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](LICENSE) file for details.

## Support

- [GitHub Issues](https://github.com/pslawinski/smart_circadian_lighting/issues) for bug reports and feature requests
- [Discussions](https://github.com/pslawinski/smart_circadian_lighting/discussions) for questions and general discussion

## Credits

Created by [pslawinski](https://github.com/pslawinski)