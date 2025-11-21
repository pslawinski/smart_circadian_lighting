"""Constants for the Smart Circadian Lighting integration."""

DOMAIN = "smart_circadian_lighting"

# The minimum time in seconds between updates sent to a light.
MIN_UPDATE_INTERVAL = 20

# A buffer in seconds to subtract from the calculated hardware transition time.
# This ensures the hardware transition finishes before the next calculated update,
# preventing visual jumps.
TRANSITION_SAFETY_BUFFER = 0.5

# The minimum change in brightness (on a 0-255 scale) required to trigger an
# update. This prevents sending updates for tiny, imperceptible changes.
MIN_BRIGHTNESS_CHANGE_FOR_UPDATE = 3

# The minimum change in kelvin required to trigger an update.
MIN_COLOR_TEMP_CHANGE_FOR_UPDATE = 10

# Default transition update interval in seconds
TRANSITION_UPDATE_INTERVAL = 60

# Configuration keys
CONF_LIGHTS = "lights"
CONF_MORNING_START_TIME = "morning_start_time"
CONF_MORNING_END_TIME = "morning_end_time"
CONF_EVENING_START_TIME = "evening_start_time"
CONF_EVENING_END_TIME = "evening_end_time"
CONF_NIGHT_BRIGHTNESS = "night_brightness"
CONF_DAY_BRIGHTNESS = "day_brightness"
CONF_MANUAL_OVERRIDE_THRESHOLD = "manual_override_threshold"
CONF_COLOR_TEMP_MANUAL_OVERRIDE_THRESHOLD = "color_temp_manual_override_threshold"
CONF_MORNING_OVERRIDE_CLEAR_TIME = "morning_override_clear_time"
CONF_EVENING_OVERRIDE_CLEAR_TIME = "evening_override_clear_time"
CONF_COLOR_TEMP_ENABLED = "color_temp_enabled"
CONF_NIGHT_COLOR_TEMP_KELVIN = "night_color_temp_kelvin"
CONF_MIDDAY_COLOR_TEMP_KELVIN = "midday_color_temp_kelvin"
CONF_SUNRISE_SUNSET_COLOR_TEMP_KELVIN = "sunrise_sunset_color_temp_kelvin"
CONF_COLOR_CURVE_TYPE = "color_curve_type"
CONF_COLOR_MORNING_START_TIME = "color_morning_start_time"
CONF_COLOR_MORNING_END_TIME = "color_morning_end_time"
CONF_COLOR_EVENING_START_TIME = "color_evening_start_time"
CONF_COLOR_EVENING_END_TIME = "color_evening_end_time"

# New keys for color time UI
CONF_COLOR_TIME_TYPE_SYNC = "sync"
CONF_COLOR_TIME_TYPE_FIXED = "fixed"
CONF_COLOR_TIME_TYPE_SUN = "sun"

CONF_COLOR_MORNING_START_TYPE = "color_morning_start_type"
CONF_COLOR_MORNING_START_FIXED_TIME = "color_morning_start_fixed_time"
CONF_COLOR_MORNING_START_SUN_EVENT = "color_morning_start_sun_event"
CONF_COLOR_MORNING_START_SUN_OFFSET = "color_morning_start_sun_offset"

CONF_COLOR_MORNING_END_TYPE = "color_morning_end_type"
CONF_COLOR_MORNING_END_FIXED_TIME = "color_morning_end_fixed_time"
CONF_COLOR_MORNING_END_SUN_EVENT = "color_morning_end_sun_event"
CONF_COLOR_MORNING_END_SUN_OFFSET = "color_morning_end_sun_offset"

CONF_COLOR_EVENING_START_TYPE = "color_evening_start_type"
CONF_COLOR_EVENING_START_FIXED_TIME = "color_evening_start_fixed_time"
CONF_COLOR_EVENING_START_SUN_EVENT = "color_evening_start_sun_event"
CONF_COLOR_EVENING_START_SUN_OFFSET = "color_evening_start_sun_offset"

CONF_COLOR_EVENING_END_TYPE = "color_evening_end_type"
CONF_COLOR_EVENING_END_FIXED_TIME = "color_evening_end_fixed_time"
CONF_COLOR_EVENING_END_SUN_EVENT = "color_evening_end_sun_event"
CONF_COLOR_EVENING_END_SUN_OFFSET = "color_evening_end_sun_offset"


# Default values
DEFAULT_NIGHT_BRIGHTNESS = 10
DEFAULT_DAY_BRIGHTNESS = 100
DEFAULT_MANUAL_OVERRIDE_THRESHOLD = 5
DEFAULT_COLOR_TEMP_MANUAL_OVERRIDE_THRESHOLD = 100  # Kelvin
DEFAULT_MORNING_START_TIME = "05:15:00"
DEFAULT_MORNING_END_TIME = "06:00:00"
DEFAULT_EVENING_START_TIME = "20:00:00"
DEFAULT_EVENING_END_TIME = "21:30:00"
DEFAULT_MORNING_OVERRIDE_CLEAR_TIME = "08:00:00"
DEFAULT_EVENING_OVERRIDE_CLEAR_TIME = "02:00:00"
DEFAULT_COLOR_TEMP_ENABLED = False
DEFAULT_NIGHT_COLOR_TEMP_KELVIN = 1800
DEFAULT_MIDDAY_COLOR_TEMP_KELVIN = 5000
DEFAULT_SUNRISE_SUNSET_COLOR_TEMP_KELVIN = 2700
DEFAULT_COLOR_CURVE_TYPE = "cosine"
DEFAULT_COLOR_TIME_OPTION = CONF_COLOR_TIME_TYPE_SYNC

# Storage version for manual overrides
STORAGE_VERSION = 1

# Storage key for manual overrides
STORAGE_KEY = f"{DOMAIN}_manual_overrides"

# Timeout in seconds for light update requests
LIGHT_UPDATE_TIMEOUT = 10

# Signals
SIGNAL_CIRCADIAN_LIGHT_TESTING_STATE_CHANGED = "smart_circadian_lighting.testing_state_changed"
SIGNAL_OVERRIDE_STATE_CHANGED = "smart_circadian_lighting.override_state_changed"

# Debugging
DEBUG_FEATURES = True