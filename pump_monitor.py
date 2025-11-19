#!/usr/bin/env python3
"""
Pump Monitor System for Raspberry Pi Zero 2 W - Final Version
Monitors pump status (green LED) and temperature (gauge needle angle)
Uses 100% local computer vision - no API costs
Smart scheduling: checks temp frequently when pump ON, less when OFF
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - expected on non-Pi dev systems
    Picamera2 = None

try:
    import RPi.GPIO as GPIO
except ImportError:  # pragma: no cover - expected on non-Pi dev systems
    GPIO = None

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_CONFIG_PATH = Path(__file__).with_name("settings.json")

DEFAULT_SETTINGS = {
    "mqtt": {
        "broker": "192.168.1.100",
        "port": 1883,
        "username": None,
        "password": None,
        "topic_prefix": "home/pump"
    },
    "led_detection": {
        "color_lower": [35, 100, 100],
        "color_upper": [85, 255, 255],
        "min_area": 50
    },
    "gpio": {
        "ir_led_pins": [26, 27]
    },
    "timing": {
        "led_check_interval_seconds": 300,
        "temp_check_interval_pump_on": 300,
        "temp_check_interval_pump_off": 1800,
        "image_retention_hours": 4
    },
    "storage": {
        "image_dir": "images",
        "log_file": "pump_monitor.log",
        "state_file": "state.json"
    },
    "gauge": {
        "min_temp": 0,
        "max_temp": 80,
        "arc_degrees": 270,
        "zero_angle": 225,
        "min_radius": 80,
        "max_radius": 130,
        "needle_color_range": {
            "black": [[0, 0, 0], [180, 255, 80]],
            "red": [[0, 100, 100], [10, 255, 255]],
            "white": [[0, 0, 200], [180, 30, 255]]
        },
        "calibration_file": "gauge_calibration.json"
    }
}
CONFIG: dict = {}
CONFIG_DIR = DEFAULT_CONFIG_PATH.parent
CURRENT_CONFIG_PATH = None


# ==========================================================================
# GAUGE SETTINGS
# ==========================================================================

GAUGE_MIN_TEMP = DEFAULT_SETTINGS["gauge"]["min_temp"]
GAUGE_MAX_TEMP = DEFAULT_SETTINGS["gauge"]["max_temp"]
GAUGE_ARC_DEGREES = DEFAULT_SETTINGS["gauge"]["arc_degrees"]
GAUGE_ZERO_ANGLE = DEFAULT_SETTINGS["gauge"]["zero_angle"]
GAUGE_MIN_RADIUS = DEFAULT_SETTINGS["gauge"]["min_radius"]
GAUGE_MAX_RADIUS = DEFAULT_SETTINGS["gauge"]["max_radius"]
NEEDLE_COLOR_RANGE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
CALIBRATION_FILE = Path(DEFAULT_SETTINGS["gauge"]["calibration_file"]).resolve()


def apply_gauge_settings(settings, base_dir):
    """Merge gauge settings from config into module globals."""

    global GAUGE_MIN_TEMP
    global GAUGE_MAX_TEMP
    global GAUGE_ARC_DEGREES
    global GAUGE_ZERO_ANGLE
    global GAUGE_MIN_RADIUS
    global GAUGE_MAX_RADIUS
    global NEEDLE_COLOR_RANGE
    global CALIBRATION_FILE

    defaults = DEFAULT_SETTINGS["gauge"].copy()
    overrides = settings.get("gauge", {}) if settings else {}

    gauge_settings = {**defaults, **overrides}

    GAUGE_MIN_TEMP = gauge_settings["min_temp"]
    GAUGE_MAX_TEMP = gauge_settings["max_temp"]
    GAUGE_ARC_DEGREES = gauge_settings["arc_degrees"]
    GAUGE_ZERO_ANGLE = gauge_settings["zero_angle"]
    GAUGE_MIN_RADIUS = gauge_settings["min_radius"]
    GAUGE_MAX_RADIUS = gauge_settings["max_radius"]

    needle_colors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, bounds in gauge_settings.get("needle_color_range", {}).items():
        lower, upper = bounds
        needle_colors[name] = (
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8),
        )
    NEEDLE_COLOR_RANGE = needle_colors

    CALIBRATION_FILE = _resolve_path(gauge_settings["calibration_file"], base_dir)


# ==========================================================================
# GAUGE DETECTION UTILITIES
# ==========================================================================


def detect_gauge_circle(image):
    """Detect the circular gauge and return (center_x, center_y, radius)."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use improved parameters from pump_vision.py for more reliable detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=60,  # Increased for stricter edge detection
        param2=40,  # Increased to avoid false positives
        minRadius=GAUGE_MIN_RADIUS,
        maxRadius=GAUGE_MAX_RADIUS,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]
        return (x, y, r)

    return None


def crop_to_gauge(image, center, radius):
    """Crop image to the gauge region and return (cropped_image, new_center)."""

    x, y = center
    padding = int(radius * 0.3)

    x1 = max(0, x - radius - padding)
    y1 = max(0, y - radius - padding)
    x2 = min(image.shape[1], x + radius + padding)
    y2 = min(image.shape[0], y + radius + padding)

    cropped = image[y1:y2, x1:x2]
    new_center = (x - x1, y - y1)

    return cropped, new_center


def detect_needle(image, center, radius):
    """Detect the needle angle in degrees, or return None if not found.
    
    Uses improved multi-strategy edge detection for better reliability across
    different lighting conditions. Color-based detection removed as edge-based
    approach is more reliable.
    """
    cx, cy = center
    
    # Create annular mask (ring) for the gauge area
    # Exclude center (needle pivot) and outer edge
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(radius * 0.85), 255, -1)
    cv2.circle(mask, (cx, cy), int(radius * 0.15), 0, -1)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try multiple edge detection strategies for robustness
    edges_list = [
        cv2.Canny(gray, 50, 150),   # Standard
        cv2.Canny(gray, 30, 100),   # More sensitive
        cv2.Canny(gray, 70, 200),   # Less sensitive (cleaner)
    ]
    
    best_angle = None
    best_score = -1
    
    for edges in edges_list:
        # Apply mask to focus only on gauge area
        edges_masked = cv2.bitwise_and(edges, mask)
        
        # Detect line segments with relaxed parameters
        lines = cv2.HoughLinesP(
            edges_masked,
            rho=1,
            theta=np.pi / 180,
            threshold=10,  # Lower threshold for better detection
            minLineLength=int(radius * 0.25),  # Shorter minimum
            maxLineGap=10
        )
        
        if lines is None:
            continue
        
        # Find the line that passes closest to the center
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate perpendicular distance from center to line
            distance = point_to_line_distance((cx, cy), (x1, y1), (x2, y2))
            
            # Check that line is reasonably long
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Score based on: close to center + reasonably long
            if line_length > radius * 0.3:
                score = line_length / (distance + 1)  # Higher score = better
                
                if score > best_score:
                    best_score = score
                    
                    # Determine which endpoint is farther from center (needle tip)
                    dist1 = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
                    dist2 = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
                    
                    if dist1 > dist2:
                        tip_x, tip_y = x1, y1
                    else:
                        tip_x, tip_y = x2, y2
                    
                    # Calculate angle from center to tip
                    angle_rad = np.arctan2(tip_y - cy, tip_x - cx)
                    angle_deg = np.degrees(angle_rad)
                    
                    # Convert to 0-360 range
                    if angle_deg < 0:
                        angle_deg += 360
                    
                    best_angle = angle_deg
    
    return best_angle


def point_to_line_distance(point, line_start, line_end):
    """Return perpendicular distance from point to a line segment."""

    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = np.hypot(y2 - y1, x2 - x1)

    if den == 0:
        return float("inf")

    return num / den


def angle_to_temperature(angle, calibration=None):
    """Convert a needle angle to temperature using calibration if available.
    
    Uses improved empirical calibration from pump_vision.py when no calibration
    file exists, providing accurate readings for the standard gauge setup.
    """

    if calibration:
        zero_angle = calibration["zero_angle"]
        max_angle = calibration["max_angle"]
        min_temp = calibration["min_temp"]
        max_temp = calibration["max_temp"]

        relative_angle = angle - zero_angle
        if relative_angle < 0:
            relative_angle += 360

        arc_span = max_angle - zero_angle
        if arc_span < 0:
            arc_span += 360

        temp_range = max_temp - min_temp
        if arc_span <= 0 or temp_range <= 0:
            return round(min_temp, 1)

        temperature = min_temp + (relative_angle / arc_span) * temp_range
    else:
        # Use empirical calibration from pump_vision.py analysis
        # The needle rotates clockwise starting from ~126° (0°C),
        # going through 180°, 270°, 0°/360°, ending at ~67° (80°C)
        # This represents approximately 3.77° per degree Celsius
        EMPIRICAL_ZERO_ANGLE = 126.0
        EMPIRICAL_DEGREES_PER_C = 3.77
        
        # Calculate how many degrees we've rotated from zero position
        degrees_from_zero = angle - EMPIRICAL_ZERO_ANGLE
        
        # Handle wraparound: if we've gone past 360° back to small angles,
        # we need to add 360 to get the total rotation
        if degrees_from_zero < -180:  # We've wrapped around
            degrees_from_zero += 360
        
        # Convert rotation to temperature
        temperature = degrees_from_zero / EMPIRICAL_DEGREES_PER_C

    temperature = max(GAUGE_MIN_TEMP, min(GAUGE_MAX_TEMP, temperature))
    return round(temperature, 1)


def _is_angle_in_valid_range(angle, calibration=None):
    """Check if an angle is within a reasonable range for the gauge calibration."""
    if calibration:
        zero_angle = calibration["zero_angle"]
        max_angle = calibration["max_angle"]
    else:
        zero_angle = GAUGE_ZERO_ANGLE
        max_angle = zero_angle + GAUGE_ARC_DEGREES
        if max_angle >= 360:
            max_angle -= 360

    # Calculate if angle is within the arc from zero to max
    relative_angle = angle - zero_angle
    if relative_angle < 0:
        relative_angle += 360

    arc_span = max_angle - zero_angle
    if arc_span < 0:
        arc_span += 360

    # Allow some tolerance beyond the arc (e.g., 10% extra on each end)
    tolerance = arc_span * 0.1
    return -tolerance <= relative_angle <= arc_span + tolerance


def _normalize_needle_angle(angle, calibration=None):
    """Normalize a needle angle, flipping by 180° if it appears to be detecting the wrong end."""
    if _is_angle_in_valid_range(angle, calibration):
        return angle

    # Try flipping by 180°
    flipped_angle = (angle + 180) % 360
    if _is_angle_in_valid_range(flipped_angle, calibration):
        return flipped_angle

    # If neither works, return original angle
    return angle



def save_calibration(zero_angle, max_angle, min_temp, max_temp):
    """Persist calibration data to disk."""

    calibration = {
        "zero_angle": zero_angle,
        "max_angle": max_angle,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "timestamp": datetime.now().isoformat(),
    }

    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)

    print(f"Calibration saved: {calibration}")


def load_calibration():
    """Load calibration data if present."""

    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def read_gauge(image, calibration=None, debug=False):
    """Read gauge temperature from an image, returning a result dict."""

    result = {
        "temperature_c": None,
        "angle": None,
        "confidence": "low",
        "gauge_detected": False,
        "needle_detected": False,
        "notes": "",
    }

    gauge = detect_gauge_circle(image)
    if gauge is None:
        result["notes"] = "Could not detect gauge circle"
        return result

    result["gauge_detected"] = True
    center_x, center_y, radius = gauge

    if debug:
        print(f"Gauge detected: center=({center_x}, {center_y}), radius={radius}")

    cropped, new_center = crop_to_gauge(image, (center_x, center_y), radius)

    # Detect needle using edge-based detection
    angle = detect_needle(cropped, new_center, radius)
    
    if angle is None:
        result["notes"] = "Could not detect needle"
        return result
    
    if debug:
        print(f"Needle detected: angle={angle:.1f}°")
    
    # Try both the raw angle and flipped version to find best temperature match
    best_angle = None
    best_temp = None
    best_score = float('inf')
    
    for angle_to_try in [angle, (angle + 180) % 360]:
            # Check if this angle is in the valid calibration range
            is_in_range = _is_angle_in_valid_range(angle_to_try, calibration)
            
            # Calculate temperature
            temp = angle_to_temperature(angle_to_try, calibration)
            
            # Score based on temperature reasonableness
            in_temp_range = GAUGE_MIN_TEMP <= temp <= GAUGE_MAX_TEMP
            
            # Lower score is better
            score = 0
            
            if not in_temp_range:
                # Heavily penalize out-of-range temps
                score += 1000
                score += abs(temp - 40.0)  # How far outside
            else:
                # STRONG preference for angles within the valid calibration arc
                # This is the most reliable signal for choosing the correct angle
                if not is_in_range:
                    score += 100  # Large penalty for being outside calibrated arc
                else:
                    # Reward being in the calibrated range
                    score += 0.5
                
                # Secondary factor: prefer reasonable temperature ranges
                # But don't let this override the calibration range check
                if temp < 15:
                    score += (15 - temp) * 0.5
                elif temp > 65:
                    score += (temp - 65) * 0.5
            
            was_flipped = (abs(angle_to_try - angle) > 1.0)
            flip_str = "(flipped)" if was_flipped else ""
            
            if debug:
                in_arc_str = "in-arc" if is_in_range else "out-arc"
                print(f"  {angle:.1f}° → {angle_to_try:.1f}° {flip_str} [{in_arc_str}] → {temp:.1f}°C (score: {score:.1f})")
            
            if score < best_score:
                best_angle = angle_to_try
                best_temp = temp
                best_score = score
    
    # Use selected angle and temperature
    if best_angle is None:
        best_angle = angle
        best_temp = angle_to_temperature(best_angle, calibration)

    result["needle_detected"] = True
    
    result["angle"] = angle

    temperature = best_temp
    result["temperature_c"] = temperature

    result["confidence"] = "high" if angle is not None else "medium"
    result["notes"] = f"Needle at {angle:.1f}° = {temperature}°C"

    if debug:
        print(f"Result: {result}")

    return result


def calibrate_gauge(image1, temp1, image2, temp2):
    """Calibrate gauge using two known temperature readings."""

    print(f"Calibrating with {temp1}°C and {temp2}°C...")

    result1 = read_gauge(image1, calibration=None, debug=True)
    result2 = read_gauge(image2, calibration=None, debug=True)

    if result1["angle"] is None or result2["angle"] is None:
        print("ERROR: Could not detect needle in one or both calibration images")
        return False

    angle1 = result1["angle"]
    angle2 = result2["angle"]

    print(f"Detected angles: {angle1}° at {temp1}°C, {angle2}° at {temp2}°C")

    if temp1 < temp2:
        zero_angle, max_angle = angle1, angle2
        min_temp, max_temp = temp1, temp2
    else:
        zero_angle, max_angle = angle2, angle1
        min_temp, max_temp = temp2, temp1

    save_calibration(zero_angle, max_angle, min_temp, max_temp)
    return True


def test_on_image(image_path):
    """Run the gauge reader on a single image."""

    print(f"\nTesting gauge reading on: {image_path}")
    print("=" * 60)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None

    calibration = load_calibration()
    if calibration:
        print(f"Using calibration: {calibration}")
    else:
        print("No calibration found, using default mapping")

    result = read_gauge(image, calibration=calibration, debug=True)

    print("\nFinal Result:")
    print(f"  Temperature: {result['temperature_c']}°C")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Notes: {result['notes']}")

    return result


def _normalize_config_path(candidate):
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def merge_settings(defaults, overrides):
    result = {}
    keys = set(defaults.keys()) | set(overrides.keys())
    for key in keys:
        default_value = defaults.get(key)
        override_value = overrides.get(key)

        if isinstance(default_value, dict):
            override_dict = override_value if isinstance(override_value, dict) else {}
            result[key] = merge_settings(default_value, override_dict)
        elif override_value is not None:
            result[key] = override_value
        else:
            result[key] = default_value

    return result


def _apply_config_dict(settings, base_dir):
    global MQTT_BROKER
    global MQTT_PORT
    global MQTT_USER
    global MQTT_PASS
    global MQTT_TOPIC_PREFIX
    global LED_COLOR_LOWER
    global LED_COLOR_UPPER
    global LED_MIN_AREA
    global IR_LED_PINS
    global LED_CHECK_INTERVAL_SECONDS
    global TEMP_CHECK_INTERVAL_PUMP_ON
    global TEMP_CHECK_INTERVAL_PUMP_OFF
    global IMAGE_RETENTION_HOURS
    global IMAGE_DIR
    global LOG_FILE
    global STATE_FILE

    mqtt_config = settings["mqtt"]
    led_config = settings["led_detection"]
    gpio_config = settings["gpio"]
    timing_config = settings["timing"]
    storage_config = settings["storage"]

    MQTT_BROKER = mqtt_config["broker"]
    MQTT_PORT = int(mqtt_config["port"])
    MQTT_USER = mqtt_config.get("username")
    MQTT_PASS = mqtt_config.get("password")
    MQTT_TOPIC_PREFIX = mqtt_config["topic_prefix"]

    LED_COLOR_LOWER = np.array(led_config["color_lower"], dtype=np.uint8)
    LED_COLOR_UPPER = np.array(led_config["color_upper"], dtype=np.uint8)
    LED_MIN_AREA = int(led_config["min_area"])

    IR_LED_PINS = [int(pin) for pin in gpio_config.get("ir_led_pins", [])]

    LED_CHECK_INTERVAL_SECONDS = int(timing_config["led_check_interval_seconds"])
    TEMP_CHECK_INTERVAL_PUMP_ON = int(timing_config["temp_check_interval_pump_on"])
    TEMP_CHECK_INTERVAL_PUMP_OFF = int(timing_config["temp_check_interval_pump_off"])
    IMAGE_RETENTION_HOURS = int(timing_config["image_retention_hours"])

    IMAGE_DIR = _resolve_path(storage_config["image_dir"], base_dir)
    LOG_FILE = _resolve_path(storage_config["log_file"], base_dir)
    STATE_FILE = _resolve_path(storage_config["state_file"], base_dir)

    apply_gauge_settings(settings, base_dir)


def apply_pump_settings(settings, base_dir):
    global CONFIG
    global CONFIG_DIR

    CONFIG_DIR = base_dir
    CONFIG = merge_settings(DEFAULT_SETTINGS, settings)
    _apply_config_dict(CONFIG, base_dir)
    return CONFIG


def _resolve_path(value, base_dir):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def configure_from_file(config_path=None):
    global CURRENT_CONFIG_PATH

    if config_path is None:
        path = _normalize_config_path(DEFAULT_CONFIG_PATH)
    else:
        path = _normalize_config_path(config_path)

    with open(path, "r", encoding="utf-8") as handle:
        overrides = json.load(handle)

    apply_pump_settings(overrides, path.parent)
    CURRENT_CONFIG_PATH = path
    return CONFIG


def prepare_storage():
    if IMAGE_DIR:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_FILE:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


# Apply defaults at import so dependent functions have usable values
apply_pump_settings({}, DEFAULT_CONFIG_PATH.parent)

try:
    configure_from_file(DEFAULT_CONFIG_PATH)
except FileNotFoundError:
    CURRENT_CONFIG_PATH = None

# ============================================================================
# SYSTEM SETUP
# ============================================================================

def log(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    
    if LOG_FILE is None:
        return

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def load_state():
    """Load persistent state"""
    default_state = {
        "last_temp_check": None,
        "last_temperature": None,
        "pump_on": False,
        "led_region": None
    }
    
    if STATE_FILE is None:
        return default_state

    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            log("Loaded previous state")
            # Validate loaded state has expected keys
            for key in default_state:
                if key not in state:
                    log(f"Warning: Missing key '{key}' in state file, using default")
                    state[key] = default_state[key]
            return state
        except (json.JSONDecodeError, ValueError) as e:
            log(f"State file corrupted: {e}, using defaults")
        except Exception as e:
            log(f"Error loading state: {e}, using defaults")
    
    return default_state

def save_state(state):
    """Save persistent state"""
    try:
        if STATE_FILE is None:
            return
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log(f"Error saving state: {e}")

# ============================================================================
# GPIO CONTROL
# ============================================================================


def _require_gpio():
    if GPIO is None:
        raise RuntimeError(
            "RPi.GPIO module not available; monitoring requires Raspberry Pi hardware."
        )


def setup_gpio():
    """Initialize GPIO pins for IR LEDs"""
    _require_gpio()
    assert GPIO is not None
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    for pin in IR_LED_PINS:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    
    log(f"GPIO initialized: IR LEDs on pins {IR_LED_PINS}")

def ir_leds_on():
    """Turn on IR LEDs"""
    _require_gpio()
    assert GPIO is not None
    for pin in IR_LED_PINS:
        GPIO.output(pin, GPIO.HIGH)

def ir_leds_off():
    """Turn off IR LEDs"""
    _require_gpio()
    assert GPIO is not None
    for pin in IR_LED_PINS:
        GPIO.output(pin, GPIO.LOW)

# ============================================================================
# CAMERA CONTROL
# ============================================================================


def _require_camera():
    if Picamera2 is None:
        raise RuntimeError(
            "picamera2 library not available; monitoring requires Raspberry Pi camera stack."
        )


def setup_camera():
    """Initialize and configure the camera"""
    _require_camera()
    assert Picamera2 is not None
    camera = Picamera2()
    
    config = camera.create_still_configuration(
        main={"size": (1920, 1080)},
        buffer_count=2
    )
    camera.configure(config)
    
    camera.start()
    time.sleep(2)
    
    log("Camera initialized: 1920x1080")
    return camera

def capture_image_with_ir(camera):
    """Capture image with IR illumination"""
    ir_leds_on()
    try:
        time.sleep(0.5)  # Let IR stabilize
        image_array = camera.capture_array()
    finally:
        ir_leds_off()

    # Convert from RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    return image_bgr

def save_image(image, timestamp, suffix=""):
    """Save image to disk with timestamp"""
    filename = IMAGE_DIR / f"pump_{timestamp.strftime('%Y%m%d_%H%M%S')}{suffix}.jpg"
    cv2.imwrite(str(filename), image)
    log(f"Image saved: {filename.name}")
    return filename


def _parse_image_timestamp(image_file: Path):
    """Attempt to parse the timestamp embedded in a pump_YYYYMMDD_HHMMSS*.jpg name."""
    parts = image_file.stem.split('_')
    if len(parts) < 3:
        return None

    date_part = parts[1]
    time_part = parts[2]
    time_digits = ''.join(ch for ch in time_part if ch.isdigit())
    if len(date_part) != 8 or len(time_digits) < 6:
        return None

    try:
        return datetime.strptime(f"{date_part}_{time_digits[:6]}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None

def cleanup_old_images():
    """Remove images older than retention period"""
    if IMAGE_DIR is None or not IMAGE_DIR.exists():
        return

    cutoff_time = datetime.now() - timedelta(hours=IMAGE_RETENTION_HOURS)

    deleted_count = 0
    for image_file in IMAGE_DIR.glob("pump_*.jpg"):
        try:
            file_time = _parse_image_timestamp(image_file)
            if file_time is None:
                file_time = datetime.fromtimestamp(image_file.stat().st_mtime)

            if file_time < cutoff_time:
                image_file.unlink()
                deleted_count += 1
        except Exception as e:
            log(f"Error checking file {image_file.name}: {e}")

    if deleted_count > 0:
        log(f"Cleaned up {deleted_count} old images")

# ============================================================================
# GREEN LED DETECTION (Pump Status)
# ============================================================================


def _clamp_region(region, width, height):
    """Ensure a persisted ROI stays inside the image bounds."""
    if not region:
        return None

    try:
        x, y, w, h = (int(value) for value in region)
    except (TypeError, ValueError):
        return None

    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return None

    width = max(1, width)
    height = max(1, height)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    max_w = width - x
    max_h = height - y
    if max_w <= 0 or max_h <= 0:
        return None

    w = max(1, min(w, max_w))
    h = max(1, min(h, max_h))
    return (x, y, w, h)


def detect_pump_leds(image, region=None):
    """
    Detect green LEDs on pump face (handles multiple LEDs)
    Returns: (is_on: bool, confidence: float, led_positions: list)
    
    Uses improved LED detection algorithm from pump_vision.py for better
    reliability in both ambient and dark (IR-only) lighting conditions.
    """
    height, width = image.shape[:2]
    sanitized_region = _clamp_region(region, width, height)

    if sanitized_region:
        x, y, w, h = sanitized_region
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            roi = image
            offset = (0, 0)
            sanitized_region = None
        else:
            offset = (x, y)
    else:
        roi = image
        offset = (0, 0)

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Create mask for green colors using configured bounds
    green_mask = cv2.inRange(hsv, LED_COLOR_LOWER, LED_COLOR_UPPER)
    
    # Find contours (potential LEDs)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area and extract LED information
    led_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= LED_MIN_AREA:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + offset[0]
                cy = int(M["m01"] / M["m00"]) + offset[1]
                led_regions.append({
                    'center': (cx, cy),
                    'area': area,
                    'y_position': cy
                })
    
    # Sort by area (largest first) for consistent selection
    led_regions.sort(key=lambda x: x['area'], reverse=True)
    
    # Determine pump state: ON if we found at least one LED
    num_leds = len(led_regions)
    is_on = num_leds >= 1
    
    # Calculate confidence based on LED characteristics
    if led_regions:
        largest_led = led_regions[0]
        green_pixels = cv2.countNonZero(green_mask)
        total_pixels = max(1, roi.shape[0] * roi.shape[1])
        
        # Confidence increases with:
        # - Larger LED area
        # - More green pixels overall
        # - Multiple LEDs detected (typical for this pump)
        confidence = min(100.0, max(0.0, 
                       largest_led['area'] / 10.0 + 
                       (green_pixels / total_pixels * 100.0) * 2.0 + 
                       num_leds * 10.0))
    else:
        confidence = 0.0
    
    log(f"LED: {'ON' if is_on else 'OFF'} ({num_leds} LED(s) detected, "
        f"conf: {confidence:.1f}%)")
    
    # Calculate region for future optimization
    detected_region = sanitized_region
    if led_regions and sanitized_region is None:
        # Calculate bounding box around all LEDs with padding
        all_x = [led['center'][0] for led in led_regions]
        all_y = [led['center'][1] for led in led_regions]
        x_min, x_max = min(all_x) - 50, max(all_x) + 50
        y_min, y_max = min(all_y) - 50, max(all_y) + 50
        new_region = _clamp_region(
            (x_min, y_min, x_max - x_min, y_max - y_min), width, height
        )
        if new_region:
            detected_region = new_region
    
    return is_on, confidence, detected_region

# ============================================================================
# MQTT / HOME ASSISTANT
# ============================================================================

def publish_to_mqtt(pump_on, temperature_c, led_confidence, temp_confidence, notes):
    """Publish pump status to Home Assistant via MQTT"""
    client = None
    loop_running = False
    connected = False

    try:
        client = mqtt.Client(client_id="pump_monitor", protocol=mqtt.MQTTv311)

        if MQTT_USER and MQTT_PASS:
            client.username_pw_set(MQTT_USER, MQTT_PASS)

        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        connected = True

        client.loop_start()
        loop_running = True
        time.sleep(0.1)  # Brief pause to ensure loop is running

        timestamp = datetime.now().isoformat()
        publish_calls = []

        publish_calls.append(
            client.publish(
                f"{MQTT_TOPIC_PREFIX}/status",
                "on" if pump_on else "off",
                retain=True,
            )
        )

        if temperature_c is not None:
            # Validate temperature is within reasonable range
            if GAUGE_MIN_TEMP <= temperature_c <= GAUGE_MAX_TEMP:
                publish_calls.append(
                    client.publish(
                        f"{MQTT_TOPIC_PREFIX}/temperature",
                        str(temperature_c),
                        retain=True,
                    )
                )
            else:
                log(f"Warning: Temperature {temperature_c}°C outside valid range [{GAUGE_MIN_TEMP}, {GAUGE_MAX_TEMP}]")

        publish_calls.append(
            client.publish(
                f"{MQTT_TOPIC_PREFIX}/last_check",
                timestamp,
                retain=True,
            )
        )
        publish_calls.append(
            client.publish(
                f"{MQTT_TOPIC_PREFIX}/led_confidence",
                f"{led_confidence:.0f}%",
                retain=True,
            )
        )

        if temp_confidence:
            publish_calls.append(
                client.publish(
                    f"{MQTT_TOPIC_PREFIX}/temp_confidence",
                    temp_confidence,
                    retain=True,
                )
            )

        if notes:
            publish_calls.append(
                client.publish(
                    f"{MQTT_TOPIC_PREFIX}/notes",
                    str(notes),
                    retain=True,
                )
            )

        for info in publish_calls:
            try:
                info.wait_for_publish()
            except Exception as publish_error:
                log(f"MQTT publish wait error: {publish_error}")

        client.disconnect()
        connected = False

        log(f"Published to MQTT: pump={'on' if pump_on else 'off'}, temp={temperature_c}°C")

    except Exception as e:
        log(f"MQTT error: {e}")
    finally:
        if connected and client is not None:
            try:
                client.disconnect()
            except Exception as disconnect_error:
                log(f"MQTT disconnect error: {disconnect_error}")
        if loop_running and client is not None:
            try:
                client.loop_stop()
            except Exception as loop_error:
                log(f"MQTT loop stop error: {loop_error}")

# ============================================================================
# SMART SCHEDULING
# ============================================================================

def should_check_temperature(state):
    """Determine if temperature should be checked this cycle"""
    
    last_check = state.get('last_temp_check')
    pump_on = state.get('pump_on', False)
    
    # First run - always check
    if last_check is None:
        log("First temperature check")
        return True
    
    # Calculate time since last check
    try:
        last_check_time = datetime.fromisoformat(last_check)
    except (TypeError, ValueError):
        log("Could not parse last check time, checking temperature")
        return True

    elapsed_seconds = (datetime.now() - last_check_time).total_seconds()
    
    # Determine interval based on pump status
    if pump_on:
        interval = TEMP_CHECK_INTERVAL_PUMP_ON
        reason = "pump ON"
    else:
        interval = TEMP_CHECK_INTERVAL_PUMP_OFF
        reason = "pump OFF"
    
    # Check if enough time has elapsed
    if elapsed_seconds >= interval:
        log(f"Temp check due: {elapsed_seconds:.0f}s elapsed ({reason}, interval: {interval}s)")
        return True
    else:
        remaining = interval - elapsed_seconds
        log(f"Skip temp check: {remaining:.0f}s remaining ({reason})")
        return False

# ============================================================================
# MAIN MONITORING LOOP
# ============================================================================

def run_monitoring_cycle(camera, state, calibration):
    """Execute one monitoring cycle"""
    
    timestamp = datetime.now()
    log("=" * 60)
    log(f"Starting cycle")
    
    try:
        # Capture image with IR
        log("Capturing image...")
        image = capture_image_with_ir(camera)
        
        # Always detect LED status (local CV - free and fast)
        log("Detecting LED status...")
        previous_region = state.get('led_region')
        led_on, led_confidence, detected_led_region = detect_pump_leds(
            image,
            previous_region,
        )

        if detected_led_region:
            if detected_led_region != previous_region:
                state['led_region'] = detected_led_region
                log(f"LED region updated: {detected_led_region}")
        elif previous_region:
            state['led_region'] = None
            log("LED region cleared")
        
        # Update pump status in state
        state['pump_on'] = led_on
        
        # Determine if we should check temperature
        check_temp = should_check_temperature(state)
        
        # Use last known temperature if not checking
        temperature = state.get('last_temperature')
        temp_confidence = None
        temp_notes = ""
        
        if check_temp:
            # Read temperature with CV needle detection
            log("Reading temperature gauge (needle angle detection)...")
            
            result = read_gauge(image, calibration=calibration, debug=False)
            
            temperature = result.get('temperature_c')
            temp_confidence = result.get('confidence', 'low')
            temp_notes = result.get('notes', '')
            
            # Update state
            state['last_temp_check'] = datetime.now().isoformat()
            if temperature is not None:
                state['last_temperature'] = temperature
            
            # Save image when temperature is checked
            save_image(image, timestamp, suffix="_temp")
        else:
            # Save LED check image occasionally (every 15 minutes)
            if timestamp.minute % 15 == 0:
                save_image(image, timestamp, suffix="_led")
        
        # Publish to Home Assistant
        log("Publishing to Home Assistant...")
        publish_to_mqtt(
            pump_on=led_on,
            temperature_c=temperature,
            led_confidence=led_confidence,
            temp_confidence=temp_confidence,
            notes=temp_notes
        )
        
        # Save state
        save_state(state)
        
        # Cleanup old images
        cleanup_old_images()
        
        log("Cycle complete")
        
    except KeyboardInterrupt:
        raise  # Re-raise to allow graceful shutdown
    except Exception as e:
        log(f"Cycle error: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        
        # Try to publish error state, but don't fail if MQTT is the problem
        try:
            publish_to_mqtt(
                pump_on=state.get('pump_on', False),  # Use last known state
                temperature_c=state.get('last_temperature'),
                led_confidence=0,
                temp_confidence='low',
                notes=f'Error: {str(e)}'
            )
        except Exception as mqtt_err:
            log(f"Could not publish error state: {mqtt_err}")


def run_monitor():
    """Run the continuous monitoring loop."""

    prepare_storage()

    log("=" * 60)
    log("PUMP MONITOR SYSTEM - FINAL VERSION")
    log("=" * 60)
    if CURRENT_CONFIG_PATH:
        log(f"Configuration file: {CURRENT_CONFIG_PATH}")
    else:
        log("Configuration: built-in defaults")
    log(f"MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
    log(f"LED check interval: {LED_CHECK_INTERVAL_SECONDS}s ({LED_CHECK_INTERVAL_SECONDS/60} min)")
    log(f"Temp check when pump ON: {TEMP_CHECK_INTERVAL_PUMP_ON}s ({TEMP_CHECK_INTERVAL_PUMP_ON/60} min)")
    log(f"Temp check when pump OFF: {TEMP_CHECK_INTERVAL_PUMP_OFF}s ({TEMP_CHECK_INTERVAL_PUMP_OFF/60} min)")
    log(f"Image retention: {IMAGE_RETENTION_HOURS} hours")
    log(f"Method: Local CV (Green LED + Needle Angle Detection)")
    log(f"Monthly cost: £0")
    log("")

    state = load_state()
    calibration = load_calibration()

    if calibration:
        log(f"Loaded calibration: {calibration}")
    else:
        log("WARNING: No calibration found - using default mapping")
        log("Recommend running: python3 pump_monitor.py calibrate <image1> <temp1> <image2> <temp2>")

    log("")

    camera = None
    try:
        setup_gpio()
        camera = setup_camera()

        log("System ready. Starting monitoring loop...")
        log("Press Ctrl+C to stop")
        log("")

        while True:
            run_monitoring_cycle(camera, state, calibration)

            log(f"Waiting {LED_CHECK_INTERVAL_SECONDS}s until next check...")
            log("")
            time.sleep(LED_CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        log("\nShutdown requested by user")
    except Exception as exc:
        log(f"Fatal error: {exc}")
        import traceback
        log(traceback.format_exc())
    finally:
        log("Cleaning up...")
        if camera is not None:
            camera.stop()
        if GPIO is not None:
            try:
                ir_leds_off()
            except RuntimeError:
                pass
            GPIO.cleanup()
        log("Pump monitor stopped")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pump monitor and gauge utilities for the embedded camera system."
    )
    parser.add_argument(
        "--config",
        help="Path to settings JSON file overriding defaults"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "monitor",
        help="Run the continuous monitoring loop"
    )

    test_parser = subparsers.add_parser(
        "test-image",
        help="Read temperature from a still image"
    )
    test_parser.add_argument("image_path", type=Path, help="Path to gauge image")

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate the gauge using two reference images"
    )
    calibrate_parser.add_argument("image1", type=Path, help="Image captured at temp1")
    calibrate_parser.add_argument("temp1", type=float, help="Known temperature for image1")
    calibrate_parser.add_argument("image2", type=Path, help="Image captured at temp2")
    calibrate_parser.add_argument("temp2", type=float, help="Known temperature for image2")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        try:
            configure_from_file(args.config)
        except FileNotFoundError:
            parser.error(f"Settings file not found: {args.config}")
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid settings file '{args.config}': {exc}")

    if args.command == "monitor":
        run_monitor()
    elif args.command == "test-image":
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
    else:  # pragma: no cover - defensive programming
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()