#!/usr/bin/env python3
"""
Temperature Gauge Needle Angle Detection System
Reads analog circular temperature gauge by detecting needle angle
No AI/API required - pure computer vision
"""

import argparse
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

DEFAULT_CONFIG_PATH = Path(__file__).with_name("settings.json")

DEFAULT_SETTINGS = {
    "gauge": {
        "min_temp": 0,
        "max_temp": 80,
        "arc_degrees": 270,
        "zero_angle": 225,
        "min_radius": 30,
        "max_radius": 200,
        "needle_color_range": {
            "black": [[0, 0, 0], [180, 255, 80]],
            "red": [[0, 100, 100], [10, 255, 255]],
            "white": [[0, 0, 200], [180, 30, 255]]
        },
        "calibration_file": "gauge_calibration.json"
    }
}

# Runtime configuration values populated from settings
GAUGE_MIN_TEMP = DEFAULT_SETTINGS["gauge"]["min_temp"]
GAUGE_MAX_TEMP = DEFAULT_SETTINGS["gauge"]["max_temp"]
GAUGE_ARC_DEGREES = DEFAULT_SETTINGS["gauge"]["arc_degrees"]
GAUGE_ZERO_ANGLE = DEFAULT_SETTINGS["gauge"]["zero_angle"]
GAUGE_MIN_RADIUS = DEFAULT_SETTINGS["gauge"]["min_radius"]
GAUGE_MAX_RADIUS = DEFAULT_SETTINGS["gauge"]["max_radius"]
NEEDLE_COLOR_RANGE = {}
CALIBRATION_FILE = Path(DEFAULT_SETTINGS["gauge"]["calibration_file"])
CURRENT_CONFIG_PATH = None


def _resolve_path(value, base_dir):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def apply_settings(settings, base_dir):
    """Merge gauge settings from config into module globals."""

    global GAUGE_MIN_TEMP
    global GAUGE_MAX_TEMP
    global GAUGE_ARC_DEGREES
    global GAUGE_ZERO_ANGLE
    global GAUGE_MIN_RADIUS
    global GAUGE_MAX_RADIUS
    global NEEDLE_COLOR_RANGE
    global CALIBRATION_FILE

    gauge_settings = DEFAULT_SETTINGS["gauge"].copy()
    incoming = settings.get("gauge", {})

    for key, value in incoming.items():
        gauge_settings[key] = value

    GAUGE_MIN_TEMP = gauge_settings["min_temp"]
    GAUGE_MAX_TEMP = gauge_settings["max_temp"]
    GAUGE_ARC_DEGREES = gauge_settings["arc_degrees"]
    GAUGE_ZERO_ANGLE = gauge_settings["zero_angle"]
    GAUGE_MIN_RADIUS = gauge_settings["min_radius"]
    GAUGE_MAX_RADIUS = gauge_settings["max_radius"]

    needle_colors = {}
    for name, bounds in gauge_settings.get("needle_color_range", {}).items():
        lower, upper = bounds
        needle_colors[name] = (
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8)
        )
    NEEDLE_COLOR_RANGE = needle_colors

    CALIBRATION_FILE = _resolve_path(gauge_settings["calibration_file"], base_dir)


def _normalize_config_path(candidate):
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def configure_from_file(config_path=None):
    """Load settings JSON and apply them."""

    global CURRENT_CONFIG_PATH

    if config_path is None:
        path = _normalize_config_path(DEFAULT_CONFIG_PATH)
    else:
        path = _normalize_config_path(config_path)

    with open(path, "r", encoding="utf-8") as handle:
        settings = json.load(handle)

    apply_settings(settings, path.parent)
    CURRENT_CONFIG_PATH = path
    return settings


# Ensure defaults are applied at import
apply_settings(DEFAULT_SETTINGS, DEFAULT_CONFIG_PATH.parent)

try:
    configure_from_file(DEFAULT_CONFIG_PATH)
except FileNotFoundError:
    CURRENT_CONFIG_PATH = None

# ============================================================================
# GAUGE DETECTION
# ============================================================================

def detect_gauge_circle(image):
    """
    Detect the circular gauge in the image
    Returns: (center_x, center_y, radius) or None
    """
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=GAUGE_MIN_RADIUS,
        maxRadius=GAUGE_MAX_RADIUS
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Return the most prominent circle (first detected)
        x, y, r = circles[0]
        return (x, y, r)
    
    return None

def crop_to_gauge(image, center, radius):
    """
    Crop image to gauge region with some padding
    Returns: cropped image, new center coordinates
    """
    x, y, r = center[0], center[1], radius
    padding = int(r * 0.3)
    
    # Calculate crop boundaries
    x1 = max(0, x - r - padding)
    y1 = max(0, y - r - padding)
    x2 = min(image.shape[1], x + r + padding)
    y2 = min(image.shape[0], y + r + padding)
    
    # Crop
    cropped = image[y1:y2, x1:x2]
    
    # New center in cropped image
    new_center = (x - x1, y - y1)
    
    return cropped, new_center

# ============================================================================
# NEEDLE DETECTION
# ============================================================================

def detect_needle(image, center, radius, needle_color='black'):
    """
    Detect needle in gauge using edge detection and line finding
    Returns: angle in degrees (0° = right, counterclockwise) or None
    """
    
    if needle_color not in NEEDLE_COLOR_RANGE:
        return None

    # Create mask for gauge area (exclude outer edges and center)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, int(radius * 0.9), 255, -1)
    cv2.circle(mask, center, int(radius * 0.2), 0, -1)  # Exclude center
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Try to detect needle by color
    lower, upper = NEEDLE_COLOR_RANGE[needle_color]
    color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    # Combine with gauge mask
    needle_mask = cv2.bitwise_and(color_mask, mask)
    
    # If color detection fails, try edge detection
    if cv2.countNonZero(needle_mask) < 10:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        needle_mask = cv2.bitwise_and(edges, mask)
    
    # Find lines using Hough Line Transform
    lines = cv2.HoughLinesP(
        needle_mask,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=int(radius * 0.3),
        maxLineGap=10
    )
    
    if lines is None:
        return None
    
    # Find the line that passes closest to center
    best_line = None
    min_distance = float('inf')
    
    line_segments = np.array(lines, dtype=np.int32).reshape(-1, 4)

    for x1, y1, x2, y2 in line_segments:
        # Calculate distance from line to center
        distance = point_to_line_distance(center, (x1, y1), (x2, y2))
        
        if distance < min_distance:
            min_distance = distance
            best_line = (x1, y1, x2, y2)
    
    if best_line is None:
        return None
    
    x1, y1, x2, y2 = best_line
    
    # Calculate angle from center
    # Use the point furthest from center as needle tip
    dist1 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
    dist2 = np.sqrt((x2 - center[0])**2 + (y2 - center[1])**2)
    
    if dist1 > dist2:
        tip_x, tip_y = x1, y1
    else:
        tip_x, tip_y = x2, y2
    
    # Calculate angle (0° = right, counterclockwise)
    angle = np.arctan2(tip_y - center[1], tip_x - center[0])
    angle_degrees = np.degrees(angle)
    
    # Normalize to 0-360
    if angle_degrees < 0:
        angle_degrees += 360
    
    return angle_degrees

def point_to_line_distance(point, line_start, line_end):
    """Calculate perpendicular distance from point to line segment"""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate distance
    num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    
    if den == 0:
        return float('inf')
    
    return num / den

# ============================================================================
# ANGLE TO TEMPERATURE CONVERSION
# ============================================================================

def angle_to_temperature(angle, calibration=None):
    """
    Convert needle angle to temperature
    Uses calibration data if available, otherwise uses default mapping
    """
    
    if calibration:
        # Use calibrated mapping
        zero_angle = calibration['zero_angle']
        max_angle = calibration['max_angle']
        min_temp = calibration['min_temp']
        max_temp = calibration['max_temp']
        
        # Normalize angle relative to zero
        relative_angle = angle - zero_angle
        if relative_angle < 0:
            relative_angle += 360
        
        arc_span = max_angle - zero_angle
        if arc_span < 0:
            arc_span += 360
        
        # Calculate temperature
        temp_range = max_temp - min_temp
        temperature = min_temp + (relative_angle / arc_span) * temp_range
        
    else:
        # Use default mapping (0-80°C over 270°, starting at 225°)
        relative_angle = angle - GAUGE_ZERO_ANGLE
        if relative_angle < 0:
            relative_angle += 360
        
        # Map to temperature
        temperature = (relative_angle / GAUGE_ARC_DEGREES) * GAUGE_MAX_TEMP
    
    # Clamp to valid range
    temperature = max(GAUGE_MIN_TEMP, min(GAUGE_MAX_TEMP, temperature))
    
    return round(temperature, 1)

# ============================================================================
# CALIBRATION
# ============================================================================

def save_calibration(zero_angle, max_angle, min_temp, max_temp):
    """Save calibration data"""
    calibration = {
        'zero_angle': zero_angle,
        'max_angle': max_angle,
        'min_temp': min_temp,
        'max_temp': max_temp,
        'timestamp': datetime.now().isoformat()
    }
    
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"Calibration saved: {calibration}")

def load_calibration():
    """Load calibration data"""
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    return None

def calibrate_gauge(image1, temp1, image2, temp2):
    """
    Calibrate gauge using two known temperature readings
    image1: Image at known temperature temp1
    image2: Image at known temperature temp2
    """
    
    print(f"Calibrating with {temp1}°C and {temp2}°C...")
    
    # Detect needle angles for both images
    result1 = read_gauge(image1, calibration=None, debug=True)
    result2 = read_gauge(image2, calibration=None, debug=True)
    
    if result1['angle'] is None or result2['angle'] is None:
        print("ERROR: Could not detect needle in one or both calibration images")
        return False
    
    angle1 = result1['angle']
    angle2 = result2['angle']
    
    print(f"Detected angles: {angle1}° at {temp1}°C, {angle2}° at {temp2}°C")
    
    # Determine which is zero and which is max
    if temp1 < temp2:
        zero_angle, max_angle = angle1, angle2
        min_temp, max_temp = temp1, temp2
    else:
        zero_angle, max_angle = angle2, angle1
        min_temp, max_temp = temp2, temp1
    
    # Save calibration
    save_calibration(zero_angle, max_angle, min_temp, max_temp)
    
    return True

# ============================================================================
# MAIN READING FUNCTION
# ============================================================================

def read_gauge(image, calibration=None, debug=False):
    """
    Read temperature from gauge image
    Returns: dict with temperature, angle, confidence, etc.
    """
    
    result = {
        'temperature_c': None,
        'angle': None,
        'confidence': 'low',
        'gauge_detected': False,
        'needle_detected': False,
        'notes': ''
    }
    
    # Detect gauge circle
    gauge = detect_gauge_circle(image)
    
    if gauge is None:
        result['notes'] = 'Could not detect gauge circle'
        return result
    
    result['gauge_detected'] = True
    center_x, center_y, radius = gauge
    
    if debug:
        print(f"Gauge detected: center=({center_x}, {center_y}), radius={radius}")
    
    # Crop to gauge region
    cropped, new_center = crop_to_gauge(image, (center_x, center_y), radius)
    
    # Try different needle colors
    angle = None
    detected_color = None
    color_candidates = ['black', 'red', 'white']
    for extra_color in NEEDLE_COLOR_RANGE.keys():
        if extra_color not in color_candidates:
            color_candidates.append(extra_color)

    for needle_color in color_candidates:
        if needle_color not in NEEDLE_COLOR_RANGE:
            continue
        angle = detect_needle(cropped, new_center, radius, needle_color)
        if angle is not None:
            detected_color = needle_color
            if debug:
                print(f"Needle detected ({needle_color}): angle={angle:.1f}°")
            break
    
    if angle is None:
        result['notes'] = 'Could not detect needle'
        return result
    
    result['needle_detected'] = True
    result['angle'] = angle
    
    # Convert angle to temperature
    temperature = angle_to_temperature(angle, calibration)
    result['temperature_c'] = temperature
    
    # Determine confidence
    if gauge and angle is not None:
        result['confidence'] = 'high'
    else:
        result['confidence'] = 'medium'
    
    result['notes'] = f'Needle at {angle:.1f}° = {temperature}°C'
    
    if debug:
        print(f"Result: {result}")
    
    return result

# ============================================================================
# TESTING / STANDALONE USAGE
# ============================================================================

def test_on_image(image_path):
    """Test gauge reading on a single image"""
    
    print(f"\nTesting gauge reading on: {image_path}")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    
    # Load calibration if available
    calibration = load_calibration()
    if calibration:
        print(f"Using calibration: {calibration}")
    else:
        print("No calibration found, using default mapping")
    
    # Read gauge
    result = read_gauge(image, calibration=calibration, debug=True)
    
    print("\nFinal Result:")
    print(f"  Temperature: {result['temperature_c']}°C")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Notes: {result['notes']}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Read or calibrate the temperature gauge using computer vision."
    )
    parser.add_argument(
        "--config",
        help="Path to settings JSON file overriding defaults"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser("test", help="Run gauge reading on an image")
    test_parser.add_argument("image_path", help="Path to the gauge image to analyse")

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate the gauge using two reference images"
    )
    calibrate_parser.add_argument("image1", help="Image path at temperature temp1")
    calibrate_parser.add_argument("temp1", type=float, help="Known temperature for image1")
    calibrate_parser.add_argument("image2", help="Image path at temperature temp2")
    calibrate_parser.add_argument("temp2", type=float, help="Known temperature for image2")

    args = parser.parse_args()

    if args.config:
        try:
            configure_from_file(args.config)
        except FileNotFoundError:
            parser.error(f"Settings file not found: {args.config}")
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid settings file '{args.config}': {exc}")

    if args.command == "test":
        test_on_image(args.image_path)
    elif args.command == "calibrate":
        image1 = cv2.imread(str(args.image1))
        image2 = cv2.imread(str(args.image2))

        if image1 is None or image2 is None:
            parser.error("Could not load one or both calibration images")

        success = calibrate_gauge(image1, args.temp1, image2, args.temp2)
        if success:
            print("\nCalibration successful!")
        else:
            print("\nCalibration failed!")


if __name__ == "__main__":
    main()