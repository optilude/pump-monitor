# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Raspberry Pi Zero 2 W pump monitoring system using computer vision to:
- Detect pump status via green LED detection (HSV color space)
- Read temperature from analog gauge needle using radial darkness detection
- Publish data to Home Assistant via MQTT
- 100% local processing (no cloud/API costs)

**Hardware**: Pi Zero 2 W, Camera Module 3 NoIR, 2× IR LEDs (GPIO 26, 27)

## Architecture

### Core Algorithm: Radial Darkness Detection
The temperature reading uses an advanced CV algorithm (`detect_needle()` in [pump_monitor.py:184-254](pump_monitor.py#L184-L254)):
- CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes lighting
- Bilateral filtering reduces noise while preserving edges
- Samples 360 radial lines from gauge center, measuring darkness along each
- Handles 180° needle ambiguity by scoring both orientations against calibration arc
- Achieves 82.6% accuracy (19/23 test images within 2°C)

### Smart Scheduling
Adaptive monitoring intervals ([should_check_temperature():1194-1229](pump_monitor.py#L1194-L1229)):
- LED status checked every 5 minutes
- Temperature checked every 5 min when pump ON, every 30 min when pump OFF
- First cycle always includes temperature reading
- Reduces processing and storage while maintaining responsiveness

### Configuration System
Multi-layered settings ([apply_pump_settings():652-659](pump_monitor.py#L652-L659)):
- Default settings in `DEFAULT_SETTINGS` dict
- Override via `settings.json` (merged recursively)
- Gauge calibration in separate `gauge_calibration.json`
- All paths resolved relative to config file directory

### Reliability Features
Confidence-gated publishing ([publish_to_mqtt():1041-1188](pump_monitor.py#L1041-L1188)):
- Temperature only published when confidence is "high"
- Pump status requires minimum LED confidence (10%)
- Publishes `temperature_published` topic indicating if temp was published this cycle
- Publishes `reading_failed` topic indicating LED or temperature detection failures
- Publishes `available` topic (online unless detection failed) for Home Assistant availability
- Publishes `detection_quality` JSON with diagnostic metrics
- Low-confidence readings logged but not sent to HA

## Common Development Commands

### Testing
```bash
# Run full test suite
pytest tests/test_cv.py -v

# Quick status of all test images
python analyze_test_images.py

# Test on single image
python pump_monitor.py test-image <image_path>
```

### Calibration
```bash
# Interactive calibration from two known temperatures (8-12°C apart recommended)
python calibrate_gauge.py test_images/test_off_31C_dark.jpg 31 \
                          test_images/test_on_41C_dark.jpg 41

# After calibration, verify accuracy
python analyze_test_images.py
```

### Running
```bash
# Run monitoring loop
python pump_monitor.py monitor

# With custom config
python pump_monitor.py monitor --config /path/to/settings.json

# On Raspberry Pi, runs as systemd service:
sudo systemctl status pump-monitor.service
sudo systemctl restart pump-monitor.service
sudo journalctl -u pump-monitor.service -f
```

## Test Image Convention

Test images auto-discovered from `test_images/` using filename format:
```
test_<pump_state>_<temp>C_<descriptor>.jpg
```

Examples:
- `test_on_40C_20231115.jpg` - Pump on, 40°C, captured Nov 15, 2023
- `test_off_42C_ambient.jpg` - Pump off, 42°C, ambient lighting
- `test_on_55C_setup2.jpg` - Pump on, 55°C, LED setup iteration 2

The descriptor is **required** and allows multiple captures at same conditions for:
- Tracking setup iterations (setup1, setup2, led_left)
- Dating captures (20231115, 20231116_am)
- Lighting conditions (ambient, dark, ir_only)

**Fixed tolerance**: All tests use 2°C tolerance. Images exceeding this indicate:
- Physical setup issues (camera position, IR LED placement, glare)
- Calibration error (run `calibrate_gauge.py`)
- Algorithm limitation (requires code improvement)

## File Structure

### Main Components
- `pump_monitor.py` - Core monitoring system (CLI with subcommands: monitor, test-image, calibrate)
- `settings.json` - Runtime configuration (MQTT, GPIO, timing, detection parameters)
- `gauge_calibration.json` - Angle-to-temperature mapping (created by calibration)
- `state.json` - Persistent runtime state (last check times, pump status, LED region)

### Utilities
- `calibrate_gauge.py` - Interactive calibration tool (handles 180° ambiguity resolution)
- `analyze_test_images.py` - Quick validation of all test images
- `capture_test_image.py` - Capture test images with IR illumination

### Tests
- `tests/test_cv.py` - Auto-discovers and validates test images
- `tests/test_gauge.py` - Unit tests for gauge detection functions
- `tests/test_config.py` - Configuration merging and path resolution tests

### Documentation
- `README.txt` - Complete hardware setup and deployment guide
- `TEST_WORKFLOW.md` - Iterative testing and calibration workflow
- `CALIBRATION.md` - Detailed calibration instructions
- `RELIABILITY_IMPROVEMENTS.md` - Confidence-gating implementation notes

## Key Implementation Details

### Needle Detection Ambiguity Resolution
The needle appears as a thin line that could be detected from either end (180° apart). The system:
1. Detects the raw angle using radial darkness analysis
2. Considers both the raw angle and (angle + 180°) % 360
3. Scores each against calibration arc and reasonable temperature range
4. Prefers angles within the calibrated arc (strongest signal)
5. See [read_gauge():382-485](pump_monitor.py#L382-L485) for implementation

### Configuration Merging
Settings are merged recursively:
- `DEFAULT_SETTINGS` provides base configuration
- `settings.json` overrides specific values
- Nested dicts are merged (not replaced)
- See [merge_settings():564-593](pump_monitor.py#L564-L593)

### Image Cleanup
Automatic retention management:
- Images older than `image_retention_hours` (default: 4) auto-deleted
- Uses embedded timestamp from filename (`pump_YYYYMMDD_HHMMSS*.jpg`)
- Falls back to file mtime if timestamp parsing fails
- See [cleanup_old_images():889-910](pump_monitor.py#L889-L910)

### MQTT Publishing Pattern
```python
# Build list of publish calls
publish_calls = []
publish_calls.append(client.publish(topic, payload, retain=True))

# Wait for all publishes to complete
for info in publish_calls:
    info.wait_for_publish()

# Then disconnect
client.disconnect()
```

## Development Environment

This code runs on both:
- **Raspberry Pi** (production): Full functionality including GPIO, picamera2, MQTT
- **Development machine** (testing): CV algorithms and test suite work without Pi-specific modules

Conditional imports handle missing dependencies gracefully:
```python
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None
```

Runtime checks raise helpful errors when Pi-specific features are used on dev machines.

## Calibration Theory

The gauge rotates through an arc (typically 240-260°) for the full 0-80°C range. Given two known temperature/angle pairs:
1. Calculate degrees per °C: `(angle2 - angle1) / (temp2 - temp1)`
2. Extrapolate to find 0°C position: `angle1 - (temp1 × deg_per_c)`
3. Extrapolate to find 80°C position: `angle1 + ((80 - temp1) × deg_per_c)`
4. Handle 180° needle ambiguity by scoring all 4 combinations and selecting best fit

For best results:
- Use images 8-12°C apart (avoids extrapolation error)
- Use dark/IR-only images (better needle contrast)
- Choose mid-range temperatures (30-50°C rather than endpoints)

## Testing Philosophy

**Test-driven setup improvement**: Instead of tweaking code to match bad images:
1. Capture test images at known temperatures
2. Run `pytest tests/test_cv.py -v` to identify failures
3. Improve physical setup (LED position, camera angle, diffusion)
4. Recapture failing images
5. Re-run tests until all pass with 2°C tolerance
6. Only then consider algorithm improvements

This ensures the system works reliably in production conditions, not just for cherry-picked test cases.

## Common Tasks

### Adding new test image
1. Capture image when gauge shows known temperature
2. Name using convention: `test_<on|off>_<temp>C_<descriptor>.jpg`
3. Place in `test_images/` directory
4. Run `pytest tests/test_cv.py -v` - test automatically discovered

### Improving accuracy
1. Check current status: `python analyze_test_images.py`
2. If consistent offset: recalibrate with `calibrate_gauge.py`
3. If random errors: improve physical setup (reduce glare, adjust IR LEDs)
4. If algorithm limitation: modify detection in `detect_needle()` or `read_gauge()`

### Changing monitoring intervals
Edit `settings.json`:
```json
{
  "timing": {
    "led_check_interval_seconds": 300,
    "temp_check_interval_pump_on": 300,
    "temp_check_interval_pump_off": 1800
  }
}
```
Then restart: `sudo systemctl restart pump-monitor.service`

### Debugging detection issues
```bash
# Test with debug output
python pump_monitor.py test-image test_images/test_on_40C.jpg

# Check logs on Pi
tail -f /home/pi/pump-monitor/pump_monitor.log

# View detection quality metrics in Home Assistant
# Topic: home/pump/detection_quality (JSON)
```
