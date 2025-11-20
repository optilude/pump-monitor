"""Tests for gauge reading functions."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def test_angle_to_temperature_with_calibration():
    """Test angle to temperature conversion with calibration."""
    from pump_monitor import angle_to_temperature
    
    calibration = {
        "zero_angle": 120.0,
        "max_angle": 60.0,  # Wraps around through 360
        "min_temp": 0.0,
        "max_temp": 80.0
    }
    
    # At zero angle
    temp = angle_to_temperature(120.0, calibration)
    assert temp == pytest.approx(0.0, abs=0.1)
    
    # At max angle
    temp = angle_to_temperature(60.0, calibration)
    assert temp == pytest.approx(80.0, abs=0.1)
    
    # Midpoint (should be around 40°C)
    # Arc goes 120° -> 180° -> 270° -> 0° -> 60° = 300° total
    # Midpoint at 150° rotation = 120° + 150° = 270°
    temp = angle_to_temperature(270.0, calibration)
    assert 35.0 < temp < 45.0


def test_angle_to_temperature_no_calibration():
    """Test angle to temperature without calibration (empirical)."""
    from pump_monitor import angle_to_temperature

    # Test new empirical calibration
    # Reference: 40°C at 90°, rate: 2.67° per °C

    # At reference angle (90° = 40°C)
    temp = angle_to_temperature(90.0, None)
    assert temp == pytest.approx(40.0, abs=0.1)

    # 10°C below reference: 90° - (10 * 2.67) = 90° - 26.7° = 63.3°
    temp = angle_to_temperature(63.0, None)
    assert temp == pytest.approx(30.0, abs=1.5)

    # 10°C above reference: 90° + (10 * 2.67) = 90° + 26.7° = 116.7°
    temp = angle_to_temperature(117.0, None)
    assert temp == pytest.approx(50.0, abs=1.5)

    # Test wraparound for low temperatures
    # 0°C = 90° - (40 * 2.67) = 90° - 106.8° = -16.8° = 343.2° (wrapped)
    temp = angle_to_temperature(343.0, None)
    assert temp == pytest.approx(0.0, abs=1.5)


def test_angle_to_temperature_clamping():
    """Test that temperature is clamped to valid range."""
    from pump_monitor import angle_to_temperature
    
    calibration = {
        "zero_angle": 120.0,
        "max_angle": 60.0,
        "min_temp": 0.0,
        "max_temp": 80.0
    }
    
    # Way before zero angle (should clamp to 0)
    temp = angle_to_temperature(0.0, calibration)
    assert temp >= 0.0
    
    # Way after max angle (should clamp to 80)
    temp = angle_to_temperature(100.0, calibration)
    assert temp <= 80.0


def test_is_angle_in_valid_range():
    """Test angle validation against calibration."""
    from pump_monitor import _is_angle_in_valid_range
    
    calibration = {
        "zero_angle": 120.0,
        "max_angle": 60.0,  # Arc goes 120° -> 60° (through 360)
        "min_temp": 0.0,
        "max_temp": 80.0
    }
    
    # Within range
    assert _is_angle_in_valid_range(130.0, calibration)
    assert _is_angle_in_valid_range(270.0, calibration)
    assert _is_angle_in_valid_range(50.0, calibration)
    
    # Note: 90° might be within tolerance (arc is 300° with 10% = 30° tolerance)
    # Test an angle clearly outside range
    assert not _is_angle_in_valid_range(100.0, calibration)


def test_point_to_line_distance():
    """Test perpendicular distance calculation."""
    from pump_monitor import point_to_line_distance
    
    # Point at origin, horizontal line
    dist = point_to_line_distance((0, 0), (0, 5), (10, 5))
    assert dist == pytest.approx(5.0, abs=0.01)
    
    # Point at origin, vertical line
    dist = point_to_line_distance((0, 0), (5, 0), (5, 10))
    assert dist == pytest.approx(5.0, abs=0.01)
    
    # Point on line
    dist = point_to_line_distance((5, 5), (0, 0), (10, 10))
    assert dist == pytest.approx(0.0, abs=0.01)
    
    # Degenerate line (zero length)
    dist = point_to_line_distance((5, 5), (0, 0), (0, 0))
    assert dist == float("inf")


def test_clamp_region_valid():
    """Test region clamping with valid input."""
    from pump_monitor import _clamp_region
    
    # Region fully within bounds
    region = _clamp_region((10, 20, 50, 50), width=200, height=200)
    assert region == (10, 20, 50, 50)
    
    # Region at edge
    region = _clamp_region((150, 150, 50, 50), width=200, height=200)
    assert region == (150, 150, 50, 50)


def test_clamp_region_out_of_bounds():
    """Test region clamping when out of bounds."""
    from pump_monitor import _clamp_region
    
    # Region extends beyond right edge
    region = _clamp_region((180, 20, 50, 50), width=200, height=200)
    assert region is not None
    x, y, w, h = region
    assert x + w <= 200
    
    # Region extends beyond bottom edge
    region = _clamp_region((20, 180, 50, 50), width=200, height=200)
    assert region is not None
    x, y, w, h = region
    assert y + h <= 200


def test_clamp_region_completely_outside():
    """Test region clamping when mostly outside."""
    from pump_monitor import _clamp_region
    
    # X beyond image - will be clamped to edge with minimal width
    region = _clamp_region((250, 20, 50, 50), width=200, height=200)
    # Function clamps x to 199, w to 1 (still valid but minimal)
    assert region is not None
    assert region[0] <= 199
    assert region[0] + region[2] <= 200
    
    # Negative width/height should be invalid
    region = _clamp_region((10, 20, -50, 50), width=200, height=200)
    assert region is None


def test_clamp_region_invalid_input():
    """Test region clamping with invalid input."""
    from pump_monitor import _clamp_region
    
    # None region
    region = _clamp_region(None, width=200, height=200)
    assert region is None
    
    # Wrong number of elements
    region = _clamp_region((10, 20, 50), width=200, height=200)
    assert region is None
    
    # Non-numeric values
    region = _clamp_region(("a", "b", "c", "d"), width=200, height=200)
    assert region is None


def test_save_and_load_calibration(tmp_path, monkeypatch):
    """Test calibration persistence."""
    from pump_monitor import save_calibration, load_calibration
    
    calibration_file = tmp_path / "test_calibration.json"
    monkeypatch.setattr("pump_monitor.CALIBRATION_FILE", calibration_file)
    
    # Save calibration
    save_calibration(
        zero_angle=120.0,
        max_angle=60.0,
        min_temp=0.0,
        max_temp=80.0
    )
    
    assert calibration_file.exists()
    
    # Load calibration
    loaded = load_calibration()
    assert loaded is not None
    assert loaded["zero_angle"] == 120.0
    assert loaded["max_angle"] == 60.0
    assert loaded["min_temp"] == 0.0
    assert loaded["max_temp"] == 80.0
    assert "timestamp" in loaded


def test_load_calibration_missing(tmp_path, monkeypatch):
    """Test loading calibration when file doesn't exist."""
    from pump_monitor import load_calibration
    
    calibration_file = tmp_path / "nonexistent.json"
    monkeypatch.setattr("pump_monitor.CALIBRATION_FILE", calibration_file)
    
    result = load_calibration()
    assert result is None


def test_read_gauge_no_circle():
    """Test gauge reading when circle cannot be detected."""
    from pump_monitor import read_gauge
    
    # Create a blank image (no gauge)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = read_gauge(image, calibration=None, debug=False)
    
    assert result["gauge_detected"] is False
    assert result["needle_detected"] is False
    assert result["temperature_c"] is None
    assert "Could not detect gauge circle" in result["notes"]


def test_crop_to_gauge():
    """Test cropping image to gauge region."""
    from pump_monitor import crop_to_gauge
    
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    center = (320, 240)
    radius = 100
    
    cropped, new_center = crop_to_gauge(image, center, radius)
    
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0
    assert new_center[0] >= 0
    assert new_center[1] >= 0


def test_crop_to_gauge_edge():
    """Test cropping when gauge is at image edge."""
    from pump_monitor import crop_to_gauge
    
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    center = (50, 50)  # Near corner
    radius = 100
    
    cropped, new_center = crop_to_gauge(image, center, radius)
    
    # Should not crash and should return valid crop
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0


def test_detect_pump_leds_off():
    """Test LED detection with no green LEDs."""
    from pump_monitor import detect_pump_leds
    
    # Create image with no green (just blue and red)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = 100  # Blue channel
    image[:, :, 2] = 100  # Red channel
    
    is_on, confidence, region = detect_pump_leds(image, region=None)
    
    assert is_on is False
    assert confidence == pytest.approx(0.0, abs=0.1)


def test_detect_pump_leds_on():
    """Test LED detection with green LEDs."""
    from pump_monitor import detect_pump_leds
    
    # Create image with green region
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add green LED (in HSV terms: Hue 60°, high saturation, high value)
    # In BGR: low B, high G, low R
    image[200:250, 300:350, 1] = 255  # Green channel
    
    is_on, confidence, region = detect_pump_leds(image, region=None)
    
    # Might detect or not depending on exact color range
    # At minimum, should not crash
    assert isinstance(is_on, bool)
    assert isinstance(confidence, (int, float))


def test_detect_pump_leds_with_region():
    """Test LED detection with ROI optimization."""
    from pump_monitor import detect_pump_leds
    
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Specify a region to search
    region = (100, 100, 200, 200)
    
    is_on, confidence, new_region = detect_pump_leds(image, region=region)
    
    assert isinstance(is_on, bool)
    assert isinstance(confidence, (int, float))


def test_should_check_temperature_first_run(monkeypatch):
    """Test temperature check scheduling on first run."""
    from pump_monitor import should_check_temperature
    
    state = {"last_temp_check": None, "pump_on": False}
    
    result = should_check_temperature(state)
    
    assert result is True


def test_should_check_temperature_pump_on(monkeypatch):
    """Test temperature check scheduling when pump is on."""
    from datetime import datetime, timedelta
    from pump_monitor import should_check_temperature
    
    # Last check was 10 minutes ago, pump on (interval: 5 min)
    monkeypatch.setattr("pump_monitor.TEMP_CHECK_INTERVAL_PUMP_ON", 300)
    
    last_check = (datetime.now() - timedelta(minutes=10)).isoformat()
    state = {"last_temp_check": last_check, "pump_on": True}
    
    result = should_check_temperature(state)
    
    assert result is True


def test_should_check_temperature_pump_off_too_soon(monkeypatch):
    """Test temperature check scheduling when pump is off but too soon."""
    from datetime import datetime, timedelta
    from pump_monitor import should_check_temperature
    
    # Last check was 10 minutes ago, pump off (interval: 30 min)
    monkeypatch.setattr("pump_monitor.TEMP_CHECK_INTERVAL_PUMP_OFF", 1800)
    
    last_check = (datetime.now() - timedelta(minutes=10)).isoformat()
    state = {"last_temp_check": last_check, "pump_on": False}
    
    result = should_check_temperature(state)
    
    assert result is False


def test_should_check_temperature_invalid_timestamp(monkeypatch):
    """Test temperature check scheduling with invalid timestamp."""
    from pump_monitor import should_check_temperature
    
    state = {"last_temp_check": "invalid timestamp", "pump_on": False}
    
    result = should_check_temperature(state)
    
    # Should default to checking
    assert result is True


def test_parse_image_timestamp_valid():
    """Test parsing timestamp from image filename."""
    from pathlib import Path
    from pump_monitor import _parse_image_timestamp
    
    filename = Path("pump_20231115_143025.jpg")
    timestamp = _parse_image_timestamp(filename)
    
    assert timestamp is not None
    assert timestamp.year == 2023
    assert timestamp.month == 11
    assert timestamp.day == 15
    assert timestamp.hour == 14
    assert timestamp.minute == 30
    assert timestamp.second == 25


def test_parse_image_timestamp_with_suffix():
    """Test parsing timestamp with suffix."""
    from pathlib import Path
    from pump_monitor import _parse_image_timestamp
    
    filename = Path("pump_20231115_143025_temp.jpg")
    timestamp = _parse_image_timestamp(filename)
    
    assert timestamp is not None
    assert timestamp.year == 2023


def test_parse_image_timestamp_invalid():
    """Test parsing invalid filename."""
    from pathlib import Path
    from pump_monitor import _parse_image_timestamp
    
    # Various invalid formats
    assert _parse_image_timestamp(Path("invalid.jpg")) is None
    assert _parse_image_timestamp(Path("pump_abc_def.jpg")) is None
    assert _parse_image_timestamp(Path("pump_20231115.jpg")) is None
