# Manual Overrides in Smart Circadian Lighting

A "manual override" occurs when a user's manual brightness adjustment conflicts with an active circadian transition. When a light is overridden, the component stops sending automatic brightness updates until the override is cleared.

This document covers how an override is triggered, its behavior, how it is cleared, and how the system handles lights that go offline.

---

## 1. Triggering an Override

An override is triggered only if **all** of the following conditions are met:

1.  **Active Transition:** The override detection is only active during the morning or evening brightness transitions. Manual changes at other times do not trigger an override.
2.  **Adjustment Against Transition:** The user must adjust the brightness in the opposite direction of the current transition.
    *   **Morning Transition** (increasing brightness): User **dims** the light.
    *   **Evening Transition** (decreasing brightness): User **brightens** the light.
3.  **Crosses Setpoint Threshold:** The new brightness level must be beyond the calculated circadian setpoint by at least the configured `manual_override_threshold`.
4.  **Light must be ON:** Override detection is only performed on lights that are physically on (`STATE_ON`). Changes to a light's state while it is off (e.g., "preloading" brightness for a future turn-on) are ignored to prevent false overrides.
5.  **Does Not Match Last Commanded State:** If the reported state (brightness/color temp) matches the value last commanded by the system (within the quantization threshold), it is **never** considered a manual override. This prevents smart bulbs that restore their previous state on power-on from being incorrectly flagged.

For example, if the current circadian brightness is 150, and a user dims the light to 130, an override will only be triggered if `130 < (150 - manual_override_threshold)`. This allows for small adjustments without disabling the circadian rhythm, as long as the change doesn't significantly deviate from the setpoint.

Adjusting brightness in the *same* direction as the transition does not trigger a "hard" manual override *during the transition* for most lights. However, there is special handling for Z-Wave lights and "soft adjustments" at the start of a transition.

### Z-Wave Specific In-Direction Overrides (Soft Overrides)

For Z-Wave lights, any substantial manual adjustment (exceeding the `manual_override_threshold`) made **in the direction** of an active transition will trigger a "soft override."

- **Captured Value:** The system captures the manually set brightness.
- **Z-Wave Parameter 18:** The system sets parameter 18 to your **manual brightness value** and stops updating it. This "pins" the switch's local preloaded brightness to your manual setting.
- **Persistence:** Soft overrides **persist** through power cycles. When the light is turned off and back on, the system skips the immediate circadian update, allowing the light to resume at the manual level preloaded in parameter 18.
- **Catch-up:** The override remains active until the background circadian target "catches up" to your manual value, at which point normal control resumes.

### Against-Direction Overrides (Hard Overrides)

When a manual adjustment is made **against** the transition direction (e.g., dimming in the morning):

- **Z-Wave Parameter 18:** Unlike soft overrides, the system **continues to update** parameter 18 with the background circadian target.
- **Toggling:** Because parameter 18 tracks the circadian rhythm, turning the light off and then back on will cause it to "return to schedule" at the correct circadian brightness, effectively clearing the hard override.

### Soft Overrides (Transition Start)

If the light has been adjusted in the direction of the transition *before the transition starts* (i.e., at transition start time), this is detected as a "soft override": the system acknowledges the user's pre-adjustment and holds the light at that brightness level until the circadian target "catches up," then resumes normal circadian control.

For example:
- **Evening transition scenario:** If a user dims the light to 25% before the evening transition starts (at which time the system plans to begin decreasing brightness further), the system detects this pre-adjustment and applies a soft override. The light remains at 25% as the evening transition progresses, until the circadian calculation naturally reaches or passes 25%, at which point the override is cleared and normal control resumes.
- **Morning transition scenario:** If a user brightens the light to 80% before the morning transition starts (at which time the system plans to begin increasing brightness further), the system detects this pre-adjustment and applies a soft override. The light remains at 80% as the morning transition progresses, until the circadian calculation naturally reaches or passes 80%, at which point the override is cleared and normal control resumes.

---

## 2. Behavior During an Override

-   **Automatic Updates Stop:** The component stops sending brightness adjustments to the light. It will remain at the user-set brightness.
-   **Background Monitoring:** The component continues calculating the correct circadian brightness. If the calculated brightness "catches up" to the manual level (e.g., the morning transition brightens to match the dimmed level), the override is automatically cleared.

---

## 3. Clearing an Override

### Automatic Clearing
1.  **Time-Based Expiration:** Overrides are cleared at configured times of day.
    *   `morning_override_clear_time`: Clears any override from the previous evening/night.
    *   `evening_override_clear_time`: Clears any override from the day.
2.  **Circadian "Catch-Up":** The override is cleared if the calculated circadian brightness meets or exceeds the manually set level.

### Manual Clearing
1.  **"Clear Manual Override" Button:** Immediately clears the override and syncs the light to the correct circadian brightness.
2.  **"Force Update" Button:** Same as the "Clear" button.
3.  **Toggling the Circadian Entity:** Turning the main `CircadianLight` entity off and on again clears the override.

---

## 4. Handling Transitions, Offline Lights, and Quantization Error

The system must account for brightness value conversions between different device scales and Home Assistant's internal scale, which can introduce minor rounding or truncation errors (quantization errors).

### Brightness Scales and Maximum Quantization Error

Home Assistant uses an internal 0-255 brightness scale. Different hardware uses different scales:
*   **Home Assistant (Internal):** 0-255 (256 steps)
*   **Kasa Dimmers:** 1-100% (100 steps)
*   **Z-Wave Dimmers:** 1-99% (99 steps)

The maximum potential quantization error must be determined to avoid falsely flagging an override due to scaling inaccuracies. The system should dynamically calculate the acceptable error threshold at runtime based on the specific light's native scale.

### Preventing False Overrides During Hardware Transitions

When the system uses a light's built-in hardware transition (e.g., via the `transition:` parameter in Home Assistant), the light changes brightness gradually on the device side.

To prevent false overrides during normal progression:
- No override is triggered as long as each reported brightness is between the transition start brightness and the target brightness (inclusive), accounting for quantization error.

To detect manual intervention during the transition:
- Track the highest (for morning transition) or lowest (for evening transition) reported brightness seen so far during the current hardware transition.
- If a new reported brightness moves against the transition direction relative to this tracked extreme value by more than the `manual_override_threshold`, and the new value also deviates from the current circadian setpoint in the against-transition direction by at least `manual_override_threshold`, trigger an override.

This detects cases where the light progresses partway through the transition but is then manually adjusted backward significantly.

General override conditions (active transition, against direction, setpoint deviation) still apply to any reported value outside the expected start-to-target range.

### Offline Scenarios

When a light goes offline during a transition period, the override logic is applied when it reconnects, considering the potential quantization error and the principles above.

*   **Scenario 1: Brightness Matches Last Command (within error threshold)**
    *   **Behavior:** When the light comes back online, its reported brightness is within the calculated error threshold of the last brightness value commanded by the system.
    *   **Outcome:** The override status remains unchanged.

*   **Scenario 2: Adjusted In Direction of Transition (Soft Adjustment)**
    *   **Behavior:** The light reports a brightness different from the last command (beyond the error threshold), but the adjustment is in the *same direction* as the current transition.
    *   **Outcome:** The override status remains unchanged. This is treated as a "soft" adjustment; it will *not* trigger an override, even if the light is significantly behind the current circadian setpoint, as long as it is moving in the correct direction.

*   **Scenario 3: Adjusted Against Direction of Transition (Override Triggered)**
    *   **Behavior:** The light reports a brightness different from the last command *against* the direction of the current transition, by more than the acceptable error threshold.
    *   **Outcome:** The system checks the new brightness against the `manual_override_threshold`. If the new level significantly deviates from the current circadian calculation (crossing the setpoint threshold), a "hard" manual override is triggered automatically upon reconnection.