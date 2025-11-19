#!/usr/bin/env python3
"""
Quick analysis tool for test images.

Shows detected angles, temperatures, and errors for all test images.
Helps identify which images need recapture or which setup changes to try.

Usage:
    python analyze_test_images.py
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from pump_monitor import read_gauge, load_calibration

# Import test utilities
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from test_cv import _discover_test_cases


def main():
    # Load actual calibration from file
    calibration = load_calibration()
    if calibration:
        print(f"Using calibration: zero={calibration['zero_angle']:.1f}°, max={calibration['max_angle']:.1f}°")
    else:
        print("No calibration found, using defaults")
        calibration = None
    
    print("=" * 80)
    print("Test Image Analysis")
    print("=" * 80)
    print()
    
    test_cases = _discover_test_cases()
    
    if not test_cases:
        print("No test images found in test_images/")
        print("Add images using format: test_<on|off>_<temp>C.jpg")
        return
    
    print(f"Found {len(test_cases)} test image(s)\n")
    
    passing = []
    failing = []
    
    for case in test_cases:
        image_path = Path("test_images") / case.filename
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"❌ {case.filename}: Could not load image")
            continue
        
        result = read_gauge(image, calibration=calibration, debug=False)
        
        detected_temp = result.get("temperature_c")
        angle = result.get("angle")
        expected_temp = case.expected_temperature_c
        
        if detected_temp is None:
            print(f"❌ {case.filename}: No temperature detected")
            failing.append((case, None, None))
            continue
        
        error = abs(detected_temp - expected_temp)
        status = "✅" if error <= 2.0 else "❌"
        
        print(f"{status} {case.filename}")
        print(f"   Pump: {'ON' if case.pump_on else 'OFF'}")
        print(f"   Expected: {expected_temp}°C")
        print(f"   Detected: {detected_temp}°C (angle: {angle:.1f}°)")
        print(f"   Error: {error:.1f}°C")
        
        if error <= 2.0:
            passing.append(case)
        else:
            failing.append((case, detected_temp, error))
            print(f"   ⚠️  Exceeds 2°C tolerance")
        
        print()
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Passing: {len(passing)}/{len(test_cases)}")
    print(f"Failing: {len(failing)}/{len(test_cases)}")
    print()
    
    if failing:
        print("Images needing attention:")
        for item in failing:
            case = item[0]
            detected = item[1]
            error = item[2]
            if detected is not None:
                print(f"  • {case.filename}: {error:.1f}°C error")
            else:
                print(f"  • {case.filename}: Detection failed")
        print()
        print("Suggestions:")
        print("  1. Adjust IR LED position to improve needle contrast")
        print("  2. Check for glare on gauge face")
        print("  3. Recapture images with better lighting setup")
        print("  4. Run calibration: python calibrate_gauge.py")
    else:
        print("✅ All test images pass with 2°C tolerance!")
        print("Your setup is well-calibrated.")
    
    print()


if __name__ == "__main__":
    main()
