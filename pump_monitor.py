#!/usr/bin/env python3
"""
Pump Monitor System for Raspberry Pi Zero 2 W - Final Version
Monitors pump status (green LED) and temperature (gauge needle angle)
Uses 100% local computer vision - no API costs
Smart scheduling: checks temp frequently when pump ON, less when OFF
"""

import argparse
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt
import cv2
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Import gauge reader
sys.path.append(str(Path(__file__).parent))
try:
    from gauge_reader import read_gauge, load_calibration, apply_settings as apply_gauge_settings
except ImportError:
    print("ERROR: gauge_reader.py not found in same directory!")
    print("Make sure gauge_reader.py is in ~/pump-monitor/")
    sys.exit(1)

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
CONFIG: dict = {}
CONFIG_DIR = DEFAULT_CONFIG_PATH.parent
CURRENT_CONFIG_PATH = None


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
            return state
        except Exception as e:
            log(f"Error loading state: {e}")
    
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

def setup_gpio():
    """Initialize GPIO pins for IR LEDs"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    for pin in IR_LED_PINS:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    
    log(f"GPIO initialized: IR LEDs on pins {IR_LED_PINS}")

def ir_leds_on():
    """Turn on IR LEDs"""
    for pin in IR_LED_PINS:
        GPIO.output(pin, GPIO.HIGH)

def ir_leds_off():
    """Turn off IR LEDs"""
    for pin in IR_LED_PINS:
        GPIO.output(pin, GPIO.LOW)

# ============================================================================
# CAMERA CONTROL
# ============================================================================

def setup_camera():
    """Initialize and configure the camera"""
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
    time.sleep(0.5)  # Let IR stabilize
    
    image_array = camera.capture_array()
    
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

def cleanup_old_images():
    """Remove images older than retention period"""
    if IMAGE_DIR is None:
        return

    cutoff_time = datetime.now() - timedelta(hours=IMAGE_RETENTION_HOURS)
    
    deleted_count = 0
    for image_file in IMAGE_DIR.glob("pump_*.jpg"):
        try:
            parts = image_file.stem.split('_')
            if len(parts) >= 2:
                timestamp_str = parts[1]
                # Extract just the date part
                date_str = timestamp_str[:8]  # YYYYMMDD
                file_time = datetime.strptime(date_str, '%Y%m%d')
                
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

def detect_pump_leds(image, region=None):
    """
    Detect green LEDs on pump face (handles multiple LEDs)
    Returns: (is_on: bool, confidence: float, led_positions: list)
    """
    
    if region:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        offset = (x, y)
    else:
        roi = image
        offset = (0, 0)
    
    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Detect bright green (neon green LEDs)
    mask = cv2.inRange(hsv, LED_COLOR_LOWER, LED_COLOR_UPPER)
    
    # Find all green contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size (LEDs are small but distinct)
    led_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > LED_MIN_AREA:  # Large enough to be an LED
            # Get position
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + offset[0]
                cy = int(M["m01"] / M["m00"]) + offset[1]
                led_contours.append({
                    'center': (cx, cy),
                    'area': area,
                    'y_position': cy  # For finding topmost
                })
    
    if not led_contours:
        return False, 0, []
    
    # Sort by Y position (topmost = lowest Y value)
    led_contours.sort(key=lambda x: x['y_position'])
    
    # The topmost LED is the status indicator
    status_led = led_contours[0]
    
    # Pump is ON if we found at least one bright green LED
    is_on = True
    
    # Confidence based on number of LEDs found and brightness
    green_pixels = cv2.countNonZero(mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100
    
    # Higher confidence if multiple LEDs detected (as expected)
    num_leds = len(led_contours)
    confidence = min(100, status_led['area'] / 10 + green_percentage * 2 + num_leds * 10)
    
    log(f"LED: {'ON' if is_on else 'OFF'} ({num_leds} LED(s) detected, "
        f"conf: {confidence:.1f}%, status LED at {status_led['center']})")
    
    # Return detected region for future optimization
    if led_contours and not region:
        # Calculate bounding box around all LEDs
        all_x = [led['center'][0] for led in led_contours]
        all_y = [led['center'][1] for led in led_contours]
        x_min, x_max = min(all_x) - 50, max(all_x) + 50
        y_min, y_max = min(all_y) - 50, max(all_y) + 50
        detected_region = (x_min, y_min, x_max - x_min, y_max - y_min)
    else:
        detected_region = region
    
    return is_on, confidence, detected_region

# ============================================================================
# MQTT / HOME ASSISTANT
# ============================================================================

def publish_to_mqtt(pump_on, temperature_c, led_confidence, temp_confidence, notes):
    """Publish pump status to Home Assistant via MQTT"""
    
    try:
        client = mqtt.Client(client_id="pump_monitor", protocol=mqtt.MQTTv311)
        
        if MQTT_USER and MQTT_PASS:
            client.username_pw_set(MQTT_USER, MQTT_PASS)
        
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        
        timestamp = datetime.now().isoformat()
        
        # Publish pump status
        client.publish(f"{MQTT_TOPIC_PREFIX}/status", 
                      "on" if pump_on else "off", retain=True)
        
        # Publish temperature
        if temperature_c is not None:
            client.publish(f"{MQTT_TOPIC_PREFIX}/temperature", 
                          str(temperature_c), retain=True)
        
        # Publish metadata
        client.publish(f"{MQTT_TOPIC_PREFIX}/last_check", timestamp, retain=True)
        client.publish(f"{MQTT_TOPIC_PREFIX}/led_confidence", 
                      f"{led_confidence:.0f}%", retain=True)
        
        if temp_confidence:
            client.publish(f"{MQTT_TOPIC_PREFIX}/temp_confidence", 
                          temp_confidence, retain=True)
        
        if notes:
            client.publish(f"{MQTT_TOPIC_PREFIX}/notes", notes, retain=True)
        
        client.disconnect()
        
        log(f"Published to MQTT: pump={'on' if pump_on else 'off'}, temp={temperature_c}°C")
        
    except Exception as e:
        log(f"MQTT error: {e}")

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
        elapsed_seconds = (datetime.now() - last_check_time).total_seconds()
    except:
        log("Could not parse last check time, checking temperature")
        return True
    
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
        led_on, led_confidence, detected_led_region = detect_pump_leds(
            image, 
            state.get('led_region')
        )
        
        # Update LED region if detected and not already set
        if detected_led_region and not state.get('led_region'):
            state['led_region'] = detected_led_region
            log(f"LED region saved: {detected_led_region}")
        
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
        
    except Exception as e:
        log(f"Cycle error: {e}")
        # Publish error state
        publish_to_mqtt(
            pump_on=False,
            temperature_c=None,
            led_confidence=0,
            temp_confidence='low',
            notes=f'Error: {str(e)}'
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor the pump status and temperature using local computer vision."
    )
    parser.add_argument(
        "--config",
        help="Path to settings JSON file overriding defaults"
    )
    return parser.parse_args()

def main():
    """Main entry point"""

    args = parse_args()

    if args.config:
        try:
            configure_from_file(args.config)
        except FileNotFoundError:
            print(f"ERROR: Settings file not found: {args.config}")
            sys.exit(1)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid settings file '{args.config}': {exc}")
            sys.exit(1)

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
    
    # Load state and calibration
    state = load_state()
    calibration = load_calibration()
    
    if calibration:
        log(f"Loaded calibration: {calibration}")
    else:
        log("WARNING: No calibration found - using default mapping")
        log("Recommend running: python3 gauge_reader.py calibrate ...")
    
    log("")
    
    # Setup hardware
    setup_gpio()
    camera = setup_camera()
    
    log("System ready. Starting monitoring loop...")
    log("Press Ctrl+C to stop")
    log("")
    
    try:
        while True:
            run_monitoring_cycle(camera, state, calibration)
            
            log(f"Waiting {LED_CHECK_INTERVAL_SECONDS}s until next check...")
            log("")
            time.sleep(LED_CHECK_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        log("\nShutdown requested by user")
    except Exception as e:
        log(f"Fatal error: {e}")
        import traceback
        log(traceback.format_exc())
    finally:
        log("Cleaning up...")
        camera.stop()
        ir_leds_off()
        GPIO.cleanup()
        log("Pump monitor stopped")

if __name__ == "__main__":
    main()