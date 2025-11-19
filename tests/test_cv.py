"""Regression tests for the computer-vision portions of pump_monitor.

These tests rely on OpenCV and a captured dark-scene image with IR illumination.
They deliberately avoid any GPIO, Picamera2, or MQTT functionality so they can
run on development hardware. The structure is prepared for adding more captured
scenarios with varying pump states and temperatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_DIR = PROJECT_ROOT / "test_images"

DEFAULT_CALIBRATION = {
    "zero_angle": 0.0,
    "max_angle": 180.0,
    "min_temp": 0.0,
    "max_temp": 80.0,
}


@dataclass(frozen=True)
class CvTestCase:
    name: str
    filename: str
    pump_on: bool
    expected_temperature_c: Optional[float]
    tolerance_c: float = 5.0
    expected_confidence: str = "high"
    expect_needle: bool = True
    expect_direct_needle: bool = True
    expected_angle_deg: Optional[float] = None
    calibration: Optional[dict[str, float]] = None
    min_led_confidence: float = 20.0

    def calibration_dict(self) -> dict[str, float]:
        base = DEFAULT_CALIBRATION.copy()
        if self.calibration:
            base.update(self.calibration)
        return base


TEST_CASES = [
    CvTestCase(
        name="dark_on_40c",
        filename="test_dark_pump_on_40C.jpg",
        pump_on=True,
        expected_temperature_c=40.0,
        expected_angle_deg=90.0,
        min_led_confidence=20.0,
    ),
    CvTestCase(
        name="dark_off_40c",
        filename="test_dark_pump_off_40C.jpg",
        pump_on=False,
        expected_temperature_c=40.0,
        tolerance_c=6.0,
        expected_angle_deg=90.0,
        min_led_confidence=0.0,
        expect_direct_needle=False,
    ),
    CvTestCase(
        name="dark_off_42C",
        filename="test_dark_pump_off_42C.jpg",
        pump_on=False,
        expected_temperature_c=42.0,
        tolerance_c=6.0,
        expected_angle_deg=95.0,
        min_led_confidence=0.0,
        expect_direct_needle=False,
    ),
    CvTestCase(
        name="light_off_42C",
        filename="test_light_pump_off_42C.jpg",
        pump_on=False,
        expected_temperature_c=42.0,
        tolerance_c=6.0,
        expected_angle_deg=95.0,
        min_led_confidence=0.0,
        expect_direct_needle=False,
    ),
    # Additional cases can be appended here as new captures arrive.
]


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
    if not case.expect_needle or not case.expect_direct_needle:
        pytest.skip("Needle detection not expected for this scenario")

    gauge = detect_gauge_circle(frame)
    assert gauge is not None, f"Gauge circle should be detected for {case.name}"

    cx, cy, radius = gauge
    assert radius > 30, "Gauge radius unexpectedly small"

    cropped, new_center = crop_to_gauge(frame, (cx, cy), radius)
    angle = detect_needle(cropped, new_center, radius, needle_color="black")
    assert angle is not None, "Needle angle should be detected"

    if case.expected_angle_deg is not None:
        assert angle == pytest.approx(case.expected_angle_deg, abs=20)


def test_read_gauge_end_to_end(gauge_result) -> None:
    case, _frame, result = gauge_result

    assert result["gauge_detected"], "Gauge should be detected in the test frame"

    if case.expect_needle:
        assert result["needle_detected"], "Needle should be detected in the test frame"
    else:
        assert not result["needle_detected"], "Needle detection was not expected"

    if case.expected_confidence is not None:
        assert result["confidence"] == case.expected_confidence

    if case.expected_temperature_c is not None:
        assert result["temperature_c"] == pytest.approx(case.expected_temperature_c, abs=case.tolerance_c)
    else:
        assert result["temperature_c"] is None

    if case.expected_angle_deg is not None and result.get("angle") is not None:
        assert result["angle"] == pytest.approx(case.expected_angle_deg, abs=20)


def test_image_interpretation_matches_expectations(cv_case, gauge_result) -> None:
    from pump_monitor import detect_pump_leds

    case, frame = cv_case
    _, _, result = gauge_result

    pump_on, led_confidence, _ = detect_pump_leds(frame, region=None)
    assert pump_on is case.pump_on

    if case.pump_on:
        assert led_confidence >= case.min_led_confidence

    if case.expected_temperature_c is not None:
        assert result["temperature_c"] == pytest.approx(case.expected_temperature_c, abs=case.tolerance_c)
        expected_string = f"{round(result['temperature_c']):.0f}"
        assert expected_string in result["notes"], "Diagnostic notes should include the reported temperature"
    else:
        assert result["temperature_c"] is None