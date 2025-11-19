"""Regression tests for the computer-vision portions of pump_monitor.

These tests rely on OpenCV and captured images with IR illumination.
They deliberately avoid any GPIO, Picamera2, or MQTT functionality so they can
run on development hardware.

Test images are automatically discovered from the test_images/ directory using
this filename convention:
    test_<pump_state>_<temp>C_<descriptor>.jpg

The descriptor is REQUIRED and allows multiple captures at the same conditions:
    test_on_40C_20231115.jpg       → pump on, 40°C, captured Nov 15, 2023
    test_off_42C_20231115_am.jpg   → pump off, 42°C, Nov 15 morning
    test_off_42C_20231116.jpg      → pump off, 42°C, Nov 16 (different day)
    test_on_40C_setup1.jpg         → pump on, 40°C, LED setup iteration 1
    test_on_40C_setup2.jpg         → pump on, 40°C, LED setup iteration 2
    test_off_38C_ambient.jpg       → pump off, 38°C, room light on
    test_off_38C_dark.jpg          → pump off, 38°C, room light off

The descriptor can be:
- Date/time: 20231115, 20231115_am, 20231115_1430
- Setup iteration: setup1, setup2, led_left, led_close
- Lighting: ambient, dark, room_light
- Any combination: 20231115_ambient, setup2_dark

All tests use a fixed 2°C tolerance. Images that fail indicate issues with:
- Physical setup (camera position, IR LED placement, glare)
- Needle detection algorithm
- Calibration (adjust via gauge_calibration.json)

To add new test cases, simply drop appropriately named images into test_images/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_DIR = PROJECT_ROOT / "test_images"

# Fixed tolerance for all tests - if images don't pass with 2°C, the system needs improvement
TOLERANCE_C = 2.0


def _load_calibration() -> dict[str, float]:
    """Load calibration from gauge_calibration.json, with fallback to defaults."""
    from pump_monitor import load_calibration
    
    calibration = load_calibration()
    if calibration:
        return calibration
    
    # Fallback to defaults if no calibration file exists
    return {
        "zero_angle": 0.0,
        "max_angle": 180.0,
        "min_temp": 0.0,
        "max_temp": 80.0,
    }


# Load actual calibration once at module level
CALIBRATION = _load_calibration()


@dataclass(frozen=True)
class CvTestCase:
    name: str
    filename: str
    pump_on: bool
    expected_temperature_c: float
    descriptor: str = ""
    
    @property
    def tolerance_c(self) -> float:
        """All tests use fixed 2°C tolerance."""
        return TOLERANCE_C
    
    @property
    def min_led_confidence(self) -> float:
        """Minimum LED confidence - only relevant when pump is on."""
        return 20.0 if self.pump_on else 0.0
    
    def calibration_dict(self) -> dict[str, float]:
        """Use actual calibration from gauge_calibration.json."""
        return CALIBRATION.copy()


def _parse_test_image_filename(filename: str) -> Optional[CvTestCase]:
    """Parse a test image filename into a CvTestCase.
    
    Expected format: test_<state>_<temp>C_<descriptor>.jpg
    The descriptor is required to support multiple captures at same conditions.
    
    Examples:
        test_on_40C_20231115.jpg → pump on, 40°C, descriptor='20231115'
        test_off_42C_setup1.jpg → pump off, 42°C, descriptor='setup1'
        test_on_55C_20231115_am.jpg → pump on, 55°C, descriptor='20231115_am'
    """
    # Match: test_<on|off>_<number>C_<descriptor>.jpg (descriptor required)
    pattern = r'^test_(on|off)_(\d+(?:\.\d+)?)C_([^.]+)\.jpg$'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    state, temp_str, descriptor = match.groups()
    pump_on = (state == "on")
    temp = float(temp_str)
    
    # Create a readable name for test output
    name = f"{state}_{int(temp)}C_{descriptor}"
    
    return CvTestCase(
        name=name,
        filename=filename,
        pump_on=pump_on,
        expected_temperature_c=temp,
        descriptor=descriptor,
    )


def _discover_test_cases() -> list[CvTestCase]:
    """Automatically discover test cases from test_images/ directory."""
    if not TEST_IMAGE_DIR.exists():
        return []
    
    cases = []
    for image_path in sorted(TEST_IMAGE_DIR.glob("test_*.jpg")):
        case = _parse_test_image_filename(image_path.name)
        if case:
            cases.append(case)
    
    return cases


# Automatically discover all test cases from filenames
TEST_CASES = _discover_test_cases()

# If no test cases found, provide a helpful message
if not TEST_CASES:
    pytest.skip(
        f"No test images found in {TEST_IMAGE_DIR}. "
        "Add images using format: test_<on|off>_<temp>C_<descriptor>.jpg "
        "(descriptor is required, e.g., date/time or setup iteration)",
        allow_module_level=True
    )


@pytest.fixture(params=TEST_CASES, ids=lambda case: case.name)
def cv_case(request) -> tuple[CvTestCase, np.ndarray]:
    case: CvTestCase = request.param
    image_path = TEST_IMAGE_DIR / case.filename
    frame = cv2.imread(str(image_path))
    assert frame is not None, f"Test image missing: {image_path}"
    return case, frame


@pytest.fixture
def gauge_result(cv_case) -> tuple[CvTestCase, np.ndarray, dict]:
    from pump_monitor import read_gauge

    case, frame = cv_case
    result = read_gauge(frame, calibration=case.calibration_dict(), debug=False)
    return case, frame, result


def test_detect_pump_leds(cv_case) -> None:
    from pump_monitor import detect_pump_leds

    case, frame = cv_case
    pump_on, confidence, region = detect_pump_leds(frame, region=None)

    assert pump_on is case.pump_on, f"Unexpected pump state for case {case.name}"
    if case.pump_on:
        assert confidence >= case.min_led_confidence, "LED confidence unexpectedly low"
        assert region, "LED detection should return a region when pump is on"

    if region:
        x, y, w, h = region
        height, width = frame.shape[:2]
        assert 0 <= x < width
        assert 0 <= y < height
        assert x + w <= width
        assert y + h <= height


def test_detect_gauge_components(cv_case) -> None:
    from pump_monitor import crop_to_gauge, detect_gauge_circle, detect_needle

    case, frame = cv_case

    gauge = detect_gauge_circle(frame)
    assert gauge is not None, f"Gauge circle should be detected for {case.name}"

    cx, cy, radius = gauge
    assert radius > 30, "Gauge radius unexpectedly small"

    cropped, new_center = crop_to_gauge(frame, (cx, cy), radius)
    
    # Detect needle (uses multi-strategy edge detection)
    angle = detect_needle(cropped, new_center, radius)
    
    assert angle is not None, f"Needle angle should be detected for {case.name}"


def test_read_gauge_end_to_end(gauge_result) -> None:
    case, _frame, result = gauge_result

    assert result["gauge_detected"], f"Gauge should be detected in {case.filename}"
    assert result["needle_detected"], f"Needle should be detected in {case.filename}"
    assert result["confidence"] == "high", f"Confidence should be high for {case.filename}"
    
    # Core test: temperature reading must be within 2°C tolerance
    detected_temp = result["temperature_c"]
    expected_temp = case.expected_temperature_c
    error = abs(detected_temp - expected_temp)
    
    assert detected_temp == pytest.approx(expected_temp, abs=case.tolerance_c), (
        f"{case.filename}: Temperature reading off by {error:.1f}°C "
        f"(expected {expected_temp}°C, got {detected_temp}°C). "
        f"Consider: adjusting IR LED position, recalibrating gauge, or improving needle detection."
    )


def test_image_interpretation_matches_expectations(cv_case, gauge_result) -> None:
    from pump_monitor import detect_pump_leds

    case, frame = cv_case
    _, _, result = gauge_result

    # Verify pump LED detection matches expected state
    pump_on, led_confidence, _ = detect_pump_leds(frame, region=None)
    assert pump_on is case.pump_on, (
        f"{case.filename}: Expected pump {'ON' if case.pump_on else 'OFF'}, "
        f"detected pump {'ON' if pump_on else 'OFF'}"
    )

    if case.pump_on:
        assert led_confidence >= case.min_led_confidence, (
            f"{case.filename}: LED confidence too low ({led_confidence:.1f})"
        )

    # Verify temperature reading
    detected_temp = result["temperature_c"]
    expected_temp = case.expected_temperature_c
    error = abs(detected_temp - expected_temp)
    
    assert detected_temp == pytest.approx(expected_temp, abs=case.tolerance_c), (
        f"{case.filename}: Temperature off by {error:.1f}°C "
        f"(expected {expected_temp}°C, got {detected_temp}°C)"
    )
    
    # Check that the temperature value appears in diagnostic notes
    temp_str = str(detected_temp)
    assert temp_str in result["notes"], (
        f"{case.filename}: Diagnostic notes should include temperature ({temp_str})"
    )