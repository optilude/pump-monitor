# Gauge Calibration Guide

## When to Recalibrate

You need to recalibrate the gauge if you:
- Move the camera position
- Change the IR LED placement
- Replace the camera
- Notice consistent temperature reading errors across all test images

## Quick Start

The easiest way to calibrate uses two existing test images:

```bash
python calibrate_gauge.py test_images/image1.jpg 31 test_images/image2.jpg 41
```

Where:
- `image1.jpg` shows the gauge at 31°C
- `image2.jpg` shows the gauge at 41°C

The script will:
1. Detect needle angles in both images
2. Handle the 180° ambiguity automatically
3. Calculate the full 0-80°C calibration range
4. Save to `gauge_calibration.json`

## Detailed Calibration Steps

### Step 1: Capture Two Test Images

Capture images at two known temperatures, ideally:
- **8-12°C apart** for good accuracy
- Both with **IR-only lighting** (room dark, pump LEDs visible)
- **Same camera/LED setup** you'll use in production

Example temperatures: 31°C and 41°C, or 35°C and 45°C

### Step 2: Run Calibration

```bash
python calibrate_gauge.py test_images/test_off_31C_dark.jpg 31 \\
                           test_images/test_on_41C_dark.jpg 41
```

### Step 3: Verify

```bash
./analyze_test_images.py
```

All images should show <2°C error. If not, recalibrate with different images or improve your physical setup.

## How It Works

The gauge rotates through an arc (typically 240-260°) to show the full temperature range. Given two known points, the script:

1. Detects the needle angle at each temperature (considering 180° ambiguity)
2. Calculates degrees per °C: `(angle2 - angle1) / (temp2 - temp1)`
3. Extrapolates backward to find 0°C position
4. Extrapolates forward to find 80°C position

**Example:**
- At 31°C, needle is at 67°
- At 41°C, needle is at 100°
- Change: 33° over 10°C = 3.3°/°C
- Extrapolate to 0°C: 67° - (31 × 3.3°) = -35° = 325°
- Extrapolate to 80°C: 67° + (49 × 3.3°) = 229°

## Command Line Options

```bash
python calibrate_gauge.py IMAGE1 TEMP1 IMAGE2 TEMP2 [OPTIONS]
```

**Required:**
- `IMAGE1`: Path to first test image
- `TEMP1`: Temperature shown in first image (°C)
- `IMAGE2`: Path to second test image  
- `TEMP2`: Temperature shown in second image (°C)

**Optional:**
- `--min-temp N`: Minimum gauge temperature (default: 0)
- `--max-temp N`: Maximum gauge temperature (default: 80)
- `--show-all`: Show all candidate calibrations (for debugging)

## Examples

**Basic calibration:**
```bash
python calibrate_gauge.py test_images/test_off_31C_dark.jpg 31 \\
                           test_images/test_on_41C_dark.jpg 41
```

**Different temperature range (0-100°C gauge):**
```bash
python calibrate_gauge.py image1.jpg 35 image2.jpg 45 --min-temp 0 --max-temp 100
```

**Debug mode (see all candidates):**
```bash
python calibrate_gauge.py image1.jpg 31 image2.jpg 41 --show-all
```

## Best Practices

### Image Selection

✅ **Good:**
- Dark room, IR LEDs only
- Clear needle visibility
- 8-12°C temperature difference
- Both images from similar pump state

❌ **Avoid:**
- Ambient lighting creating glare
- Extreme temperatures (too close to 0° or 80°)
- Very small temperature difference (<5°C)
- Blurry or out-of-focus images

### Temperature Spacing

- **Minimum:** 5°C apart (script enforces this)
- **Recommended:** 8-12°C apart
- **Good combinations:** 30-40°C, 31-41°C, 35-45°C
- **Avoid:** Endpoints (0-10°C or 70-80°C) as extrapolation error increases

## Understanding the Output

When you run calibration, you'll see:

```
Processing image 1: test_images/test_off_31C_20251119_dark.jpg (31.0°C)
  Gauge detected: center=(1002, 722), radius=109
  Raw needle angle: 247.27°
  Angles: 247.27° (direct) or 67.27° (flipped)

Processing image 2: test_images/test_on_41C_20251119_dark.jpg (41.0°C)
  Gauge detected: center=(1000, 718), radius=93
  Raw needle angle: 100.20°
  Angles: 100.20° (direct) or 280.20° (flipped)

Resolving 180° ambiguity and calculating calibration...

Best calibration (score: 250):
  Using angles: 67.27° @ 31.0°C, 100.20° @ 41.0°C
  Calculated span: 263.5°
  Degrees per °C: 3.293°/°C

============================================================
✅ Calibration saved to gauge_calibration.json
============================================================
  Zero angle (0°C):  325.1758°
  Max angle (80°C):   228.6454°
  Angular span:      263.5°
  Degrees per °C:    3.293°/°C
============================================================
```

The script automatically:
1. Detects both the direct and 180° flipped angles
2. Tries all 4 combinations
3. Scores them based on reasonableness (typical span, typical degrees/°C, typical zero position)
4. Picks the best one

## Current Calibration

Your current calibration values (in `gauge_calibration.json`):
```json
{
  "zero_angle": 330.1222,
  "max_angle": 213.6778,
  "min_temp": 0,
  "max_temp": 80
}
```

This means:
- The needle points at ~330° when the gauge shows 0°C
- The needle points at ~214° when the gauge shows 80°C
- The needle rotates clockwise through ~244° to cover the full 0-80°C range

## Troubleshooting

**"ERROR: Temperatures should be at least 5°C apart"**
- Use images with greater temperature difference (8-12°C recommended)
- Example: 31°C and 41°C instead of 39°C and 41°C

**"Could not detect gauge circle"**
- Improve lighting (ensure IR LEDs are working)
- Clean the gauge face
- Check camera focus
- Ensure full gauge is visible in frame

**"Could not detect needle"**
- Add/adjust IR LED diffusion (tissue paper works well)
- Reduce glare on gauge face
- Ensure needle has good contrast against background
- Check that needle isn't at an angle where it blends with tick marks

**Calibration produces wrong values (all readings off)**
- Try `--show-all` to see all candidates
- The scoring might have picked the wrong ambiguity resolution
- Check that your input temperatures are correct
- Try different image pairs with better needle visibility

**Some test images pass, others fail**
- This is likely a detection issue, not calibration
- Improve physical setup (lighting, diffusion, stability)
- Recapture failing images with better conditions
- The calibration is based on averages, so some variation is normal

**Want to understand the scoring (debug mode)**
```bash
python calibrate_gauge.py image1.jpg 31 image2.jpg 41 --show-all
```

This shows all 4 angle combinations and their scores so you can verify the correct one was chosen.

## Tips for Best Results

1. **Use IR-only images** (dark room, no ambient light)
2. **Ensure good diffusion** on IR LEDs to avoid glare
3. **Pick temperatures in the middle range** (30-50°C) rather than extremes
4. **Space temperatures 8-12°C apart** for optimal accuracy
5. **Verify with multiple test images** after calibration
6. **Recalibrate if you move anything** in the physical setup
