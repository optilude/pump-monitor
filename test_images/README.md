# Test Images

Test images are automatically discovered by the test suite using filename conventions.

## Filename Convention

```
test_<pump_state>_<temperature>C_<descriptor>.jpg
```

**The descriptor is REQUIRED** - this allows you to capture multiple images at the same pump state and temperature on different days or with different setups.

### Components:
- **pump_state**: `on` or `off`
- **temperature**: The actual temperature shown on the gauge (integer or decimal)
- **descriptor**: **REQUIRED** - Date, time, setup iteration, lighting condition, or any combination

### Examples:

```
# Multiple captures at same conditions on different days
test_on_40C_20231115.jpg         → Pump ON, 40°C, Nov 15, 2023
test_on_40C_20231116.jpg         → Pump ON, 40°C, Nov 16, 2023
test_on_40C_20231120.jpg         → Pump ON, 40°C, Nov 20, 2023

# Multiple captures same day, different times
test_off_42C_20231115_am.jpg     → Pump OFF, 42°C, morning
test_off_42C_20231115_pm.jpg     → Pump OFF, 42°C, evening

# Different lighting conditions
test_off_38C_ambient.jpg         → Room light ON
test_off_38C_dark.jpg            → Room light OFF
test_off_38C_20231115_dark.jpg   → Specific date + lighting

# Setup iterations
test_on_40C_setup1.jpg           → LED configuration 1
test_on_40C_setup2.jpg           → LED configuration 2  
test_on_40C_led_left.jpg         → LED positioned left
test_on_40C_led_close.jpg        → LED closer to gauge

# Combinations
test_on_55C_20231115_setup2.jpg  → Date + setup iteration
test_off_60C_20231116_ambient.jpg → Date + lighting
```

### Recommended Naming Schemes:

**For daily monitoring:**
```
test_<state>_<temp>C_YYYYMMDD.jpg
test_<state>_<temp>C_YYYYMMDD_HHMM.jpg
```

**For setup iterations:**
```
test_<state>_<temp>C_setup1.jpg
test_<state>_<temp>C_setup2.jpg
test_<state>_<temp>C_led_<position>.jpg
```

**For lighting variations:**
```
test_<state>_<temp>C_ambient.jpg
test_<state>_<temp>C_dark.jpg
test_<state>_<temp>C_<date>_ambient.jpg
```

## Tolerance

All tests use a fixed **2°C tolerance**. If a test fails with more than 2°C error, this indicates:
1. Physical setup needs adjustment (camera angle, IR LED position)
2. Gauge needs calibration (use `gauge_calibration.json`)
3. Needle detection algorithm needs improvement

## Workflow for Iterating on Setup

1. **Capture image** with current setup
2. **Name appropriately** (e.g., `test_on_45C_setup1.jpg`)
3. **Run tests**: `pytest tests/test_cv.py -v`
4. **Review failures**: Test output shows actual vs expected temperature
5. **Adjust**:
   - Move IR LEDs to reduce glare or improve contrast
   - Adjust camera angle
   - Use `calibrate_gauge.py` to create calibration file
6. **Recapture** with new setup (e.g., `test_on_45C_setup2.jpg`)
7. **Compare** results

## Current Test Images

Run `python analyze_test_images.py` to see current status of all test images.

Example images (yours may differ):
- `test_on_40C_baseline.jpg` - Initial capture, pump on
- `test_off_40C_baseline.jpg` - Initial capture, pump off
- `test_off_42C_20231115.jpg` - Captured Nov 15
- `test_off_42C_20231116.jpg` - Captured Nov 16 (compare with previous day)
- `test_off_42C_ambient.jpg` - Room light on
- `test_on_50C_setup1.jpg` - LED setup iteration 1
- `test_on_50C_setup2.jpg` - LED setup iteration 2 (compare improvements)

## Calibration

To calibrate the gauge using two known temperature points:

```bash
python pump_monitor.py calibrate path/to/lowtemp.jpg 20 path/to/hightemp.jpg 60
```

This creates `gauge_calibration.json` which the system will automatically use.

## Tips for Good Test Images

1. **IR LED Positioning**: Balance illumination vs glare on gauge face
2. **Pump State**: Capture both ON and OFF states at same temperature if possible
3. **Temperature Range**: Capture across full expected range (20-80°C)
4. **Consistency**: Use same camera position for all captures
5. **Focus**: Ensure gauge and needle are in focus
6. **Descriptors**: Use descriptive tags to track different setup configurations
