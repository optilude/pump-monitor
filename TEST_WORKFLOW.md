# Pump Monitor - Test & Calibration Workflow

This document describes the iterative workflow for improving the temperature gauge reading accuracy through physical setup adjustments, calibration, and algorithm improvements.

## Goal

Achieve **≤2°C accuracy** for all temperature readings across the full operating range.

## Quick Start

### 1. Analyze Current Status
```bash
python analyze_test_images.py
```

This shows which test images pass/fail and provides suggestions.

### 2. Run Full Test Suite
```bash
pytest tests/test_cv.py -v
```

Detailed test output shows exactly where failures occur.

### 3. Iterate on Setup

When tests fail, try these improvements in order:

#### A. Adjust IR LED Position
- Move LEDs to reduce glare on gauge face
- Balance illumination vs. reflection
- Ensure needle has good contrast against gauge face
- Recapture test images with new LED position

#### B. Calibrate the Gauge
```bash
python calibrate_gauge.py
```

Follow interactive prompts to create `gauge_calibration.json` using two known temperature points.

#### C. Improve Camera Position
- Ensure gauge is centered in frame
- Check focus - needle should be sharp
- Minimize camera angle (face-on is best)

## Detailed Workflow

### Capturing Test Images

1. **Setup**: Position camera and IR LEDs
2. **Capture**: Take photo when gauge shows known temperature
3. **Name**: Use convention `test_<on|off>_<temp>C_<descriptor>.jpg`
   ```
   # Date-based (recommended for daily monitoring)
   test_on_40C_20231115.jpg          # Nov 15, 2023
   test_on_40C_20231116.jpg          # Nov 16, 2023
   test_off_42C_20231115_am.jpg      # Morning capture
   test_off_42C_20231115_pm.jpg      # Evening capture
   
   # Setup iterations
   test_on_40C_setup1.jpg            # First LED configuration
   test_on_40C_setup2.jpg            # Second LED configuration
   test_on_55C_led_left.jpg          # LED positioned left
   
   # Lighting conditions
   test_off_38C_ambient.jpg          # Room light on
   test_off_38C_dark.jpg             # Room light off
   test_off_38C_20231115_dark.jpg    # Date + lighting
   ```
4. **Test**: Run `pytest tests/test_cv.py -v` or `python analyze_test_images.py`
5. **Review**: Check errors and decide on adjustments

**Note**: The descriptor is **required** - this allows multiple captures at the same temperature/pump state.

### Calibration Process

Use `calibrate_gauge.py` when you have two images at known temperatures:

**Interactive mode:**
```bash
python calibrate_gauge.py
```

**Direct mode:**
```bash
python calibrate_gauge.py test_on_30C.jpg 30 test_on_60C.jpg 60
```

This creates `gauge_calibration.json` which overrides the default 0-180° mapping.

### Understanding Test Failures

Test output shows specific errors:

```
FAILED test_read_gauge_end_to_end[off_42C]
AssertionError: test_off_42C.jpg: Temperature reading off by 8.5°C 
(expected 42.0°C, got 33.5°C). 
Consider: adjusting IR LED position, recalibrating gauge, or improving needle detection.
```

Common causes:
- **Large errors (>5°C)**: Wrong needle end detected, or wrong color detected (glare/reflection)
- **Consistent offset (e.g., always +3°C)**: Calibration issue - run `calibrate_gauge.py`
- **Inconsistent errors**: Setup issue - IR LED glare or poor contrast

### Tracking Setup Iterations

Use descriptive filenames to track different configurations:

```
test_on_40C_baseline.jpg          # Initial setup
test_on_40C_led_left.jpg          # Moved LED to left
test_on_40C_led_closer.jpg        # Moved LED closer
test_on_40C_noambient.jpg         # Tested in complete darkness
```

Or track over time:

```
test_on_40C_20231115.jpg          # Day 1
test_on_40C_20231116.jpg          # Day 2
test_on_40C_20231120.jpg          # Day 5 (after recalibration)
```

Compare results across iterations to find optimal setup and monitor consistency over time.

## Current Status

Run `python analyze_test_images.py` to see current passing/failing images.

**Example output:**
```
✅ test_on_40C.jpg (0.0°C error)
✅ test_off_42C_ambient.jpg (1.6°C error)
❌ test_off_40C.jpg (5.0°C error) ← needs attention
❌ test_off_42C.jpg (8.5°C error) ← needs attention

Passing: 2/4
Failing: 2/4
```

## Files Reference

- `tests/test_cv.py` - Auto-discovers and tests all images in `test_images/`
- `test_images/README.md` - Filename conventions and tips
- `calibrate_gauge.py` - Interactive calibration tool
- `analyze_test_images.py` - Quick status check
- `gauge_calibration.json` - Active calibration (created by calibration tool)
- `settings.json` - System configuration including default gauge parameters

## Needle Detection Details

The system tries to detect needles using these colors in order:
1. **black** - Primary needle color, works best with IR LED illumination
2. **red** - Alternative for red-tipped needles
3. **white** - Can detect gauge markings or reflections (less reliable)

When multiple colors detect different angles, the first successful detection is used. This can cause issues if white reflections are detected instead of the actual needle.

## Tips for Success

1. **Temperature Range**: Capture test images across the full range (20-80°C)
2. **Both States**: Capture pump ON and OFF at same temperatures to compare
3. **Consistency**: Use same camera/LED position for all captures in a test set
4. **Descriptive Names**: Use descriptors to track setup variations
5. **Calibration**: Do calibration with widely spaced temperatures (e.g., 30°C and 70°C)
6. **Iterative**: Small adjustments + recapture + retest = fastest path to accuracy

## Expected Performance

With good setup and calibration:
- Pump ON (with IR LEDs): ✅ 0-2°C error
- Pump OFF (with IR LEDs): ✅ 0-2°C error
- Ambient light only: ✅ 0-2°C error (if sufficient contrast)
- Poor lighting/glare: ❌ >5°C error (needs setup improvement)

## Next Steps When Failing

1. Check `analyze_test_images.py` output
2. Identify which images fail
3. Look for patterns:
   - All pump OFF? → LED illumination needed
   - Consistent offset? → Run calibration
   - Random errors? → Reduce glare, improve contrast
4. Make ONE change at a time
5. Recapture and retest
6. Repeat until all tests pass

## Questions?

Review the code comments in:
- `pump_monitor.py` - Core detection algorithms
- `tests/test_cv.py` - Test logic and auto-discovery
- `test_images/README.md` - Image naming details
