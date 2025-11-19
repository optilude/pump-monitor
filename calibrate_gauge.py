#!/usr/bin/env python3
"""
Gauge Calibration Tool

This script calculates gauge calibration from test images at two known temperatures.
It detects the needle angles and extrapolates to find the zero and max positions.

Usage:
    python calibrate_gauge.py test_images/test_off_31C_20251119_dark.jpg 31 \\
                               test_images/test_on_41C_20251119_dark.jpg 41

This will:
1. Detect needle angles from both images
2. Calculate the angular change per degree Celsius
3. Extrapolate to find 0°C and 80°C positions
4. Save calibration to gauge_calibration.json

For best results:
- Use images that are 8-12°C apart
- Use dark (IR-only) images for better needle detection
- Ensure both images show the pump in the same state if possible
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pump_monitor import (
    detect_gauge_circle,
    crop_to_gauge,
    detect_needle,
    CALIBRATION_FILE,
    load_calibration,
)


def load_image_from_file(filepath):
    """Load an image from a file."""
    image = cv2.imread(str(filepath))
    if image is None:
        print(f"ERROR: Could not load image from {filepath}")
        sys.exit(1)
    return image


def detect_angle_from_image(image, image_name=""):
    """Detect the needle angle from an image, considering 180° ambiguity."""
    # Detect gauge circle
    gauge = detect_gauge_circle(image)
    if gauge is None:
        print(f"ERROR: Could not detect gauge circle in {image_name}")
        print("Make sure the gauge is clearly visible in the image")
        sys.exit(1)
    
    cx, cy, radius = gauge
    print(f"  Gauge detected: center=({cx}, {cy}), radius={radius}")
    
    # Crop to gauge region
    cropped, new_center = crop_to_gauge(image, (cx, cy), radius)
    
    # Detect needle
    angle = detect_needle(cropped, new_center, radius)
    if angle is None:
        print(f"ERROR: Could not detect needle in {image_name}")
        print("Make sure the needle is clearly visible and gauge is well-lit")
        sys.exit(1)
    
    print(f"  Raw needle angle: {angle:.2f}°")
    
    # Return both the angle and its 180° flip for ambiguity resolution
    flipped = (angle + 180) % 360
    return angle, flipped


def angular_distance(angle1, angle2):
    """Calculate the shortest angular distance between two angles."""
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def calculate_calibration_from_two_points(temp1, angle1, temp2, angle2, min_temp=0, max_temp=80):
    """
    Calculate zero and max angles from two known temperature/angle pairs.
    
    Returns (zero_angle, max_angle) or raises ValueError if invalid.
    """
    if abs(temp1 - temp2) < 5:
        raise ValueError("Temperatures should be at least 5°C apart for accurate calibration")
    
    # Calculate degrees per Celsius
    temp_diff = temp2 - temp1
    
    # Determine angular direction (clockwise or counter-clockwise)
    # Try both directions and pick the one that makes sense
    angle_diff_cw = (angle2 - angle1) % 360
    angle_diff_ccw = (angle1 - angle2) % 360
    
    # Typically gauges rotate ~200-300° for the full range
    # Pick the direction that gives a reasonable span
    if angle_diff_cw < angle_diff_ccw:
        angle_diff = angle_diff_cw
        if angle_diff > 180:  # Wrap around
            angle_diff = -(360 - angle_diff)
    else:
        angle_diff = -angle_diff_ccw
        if angle_diff < -180:  # Wrap around
            angle_diff = 360 + angle_diff
    
    degrees_per_c = angle_diff / temp_diff
    
    # Extrapolate to find zero angle (min_temp)
    zero_angle = (angle1 - degrees_per_c * (temp1 - min_temp)) % 360
    
    # Extrapolate to find max angle (max_temp)
    max_angle = (angle1 + degrees_per_c * (max_temp - temp1)) % 360
    
    # Calculate total span
    if max_angle < zero_angle:
        span = (360 - zero_angle) + max_angle
    else:
        span = max_angle - zero_angle
    
    return zero_angle, max_angle, span, degrees_per_c


def resolve_angle_ambiguity(temp1, angles1, temp2, angles2, min_temp, max_temp):
    """
    Resolve 180° ambiguity by trying all combinations and picking the most sensible one.
    
    angles1 and angles2 are tuples of (direct, flipped) for each temperature.
    Returns the best (angle1, angle2) pair.
    """
    candidates = []
    
    # Try all 4 combinations
    for a1 in angles1:
        for a2 in angles2:
            try:
                zero, max_ang, span, deg_per_c = calculate_calibration_from_two_points(
                    temp1, a1, temp2, a2, min_temp, max_temp
                )
                
                # Score based on reasonableness
                score = 0
                
                # Prefer spans between 200-280° (typical for gauges)
                if 220 <= span <= 260:
                    score += 100
                elif 200 <= span <= 280:
                    score += 75
                elif 180 <= span <= 300:
                    score += 50
                else:
                    score += 0  # Too small or too large
                
                # Prefer degrees_per_c in reasonable range (2.5-3.5°/°C for typical gauge)
                abs_deg = abs(deg_per_c)
                if 2.8 <= abs_deg <= 3.3:
                    score += 100
                elif 2.5 <= abs_deg <= 3.5:
                    score += 75
                elif 2.0 <= abs_deg <= 4.0:
                    score += 50
                elif 1.5 <= abs_deg <= 5.0:
                    score += 25
                
                # Prefer zero angle in the 300-360° or 0-60° range (typical positioning)
                if 300 <= zero <= 360 or 0 <= zero <= 60:
                    score += 50
                elif 240 <= zero < 300 or 60 < zero <= 120:
                    score += 25
                
                # Prefer positive rotation (clockwise)
                if deg_per_c > 0:
                    score += 25
                
                candidates.append({
                    'angle1': a1,
                    'angle2': a2,
                    'zero': zero,
                    'max': max_ang,
                    'span': span,
                    'deg_per_c': deg_per_c,
                    'score': score
                })
            except ValueError:
                continue
    
    if not candidates:
        raise ValueError("Could not find valid calibration from any angle combination")
    
    # Sort by score (highest first)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return candidates


def save_calibration_values(zero_angle, max_angle, min_temp=0, max_temp=80):
    """Save calibration to file."""
    calibration = {
        "zero_angle": round(zero_angle, 4),
        "max_angle": round(max_angle, 4),
        "min_temp": min_temp,
        "max_temp": max_temp,
    }
    
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(calibration, indent=2, fp=f)
    
    print(f"\n{'='*60}")
    print(f"✅ Calibration saved to {CALIBRATION_FILE}")
    print(f"{'='*60}")
    print(f"  Zero angle ({min_temp}°C):  {zero_angle:.4f}°")
    print(f"  Max angle ({max_temp}°C):   {max_angle:.4f}°")
    
    # Calculate span
    if max_angle < zero_angle:
        span = (360 - zero_angle) + max_angle
    else:
        span = max_angle - zero_angle
    print(f"  Angular span:      {span:.1f}°")
    print(f"  Degrees per °C:    {span/max_temp:.3f}°/°C")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate gauge from two test images at known temperatures",
        epilog="""
Examples:
  python calibrate_gauge.py test_images/test_off_31C_dark.jpg 31 \\
                             test_images/test_on_41C_dark.jpg 41
  
  python calibrate_gauge.py image1.jpg 35 image2.jpg 45 --min-temp 0 --max-temp 80
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "image1",
        type=str,
        help="First test image file path"
    )
    parser.add_argument(
        "temp1",
        type=float,
        help="Temperature shown in first image (°C)"
    )
    parser.add_argument(
        "image2",
        type=str,
        help="Second test image file path"
    )
    parser.add_argument(
        "temp2",
        type=float,
        help="Temperature shown in second image (°C)"
    )
    
    parser.add_argument(
        "--min-temp",
        type=float,
        default=0,
        help="Minimum temperature of gauge range (default: 0)"
    )
    parser.add_argument(
        "--max-temp",
        type=float,
        default=80,
        help="Maximum temperature of gauge range (default: 80)"
    )
    
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all candidate calibrations (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if abs(args.temp1 - args.temp2) < 5:
        print("ERROR: Temperatures should be at least 5°C apart for accurate calibration")
        print(f"You provided: {args.temp1}°C and {args.temp2}°C (difference: {abs(args.temp1 - args.temp2):.1f}°C)")
        sys.exit(1)
    
    # Load existing calibration if available
    current_cal = load_calibration()
    if current_cal:
        print("Current calibration:")
        print(f"  Zero angle: {current_cal['zero_angle']:.4f}°")
        print(f"  Max angle: {current_cal['max_angle']:.4f}°")
        print()
    
    print("="*60)
    print("GAUGE CALIBRATION FROM TWO TEST IMAGES")
    print("="*60)
    print()
    
    # Load and process first image
    print(f"Processing image 1: {args.image1} ({args.temp1}°C)")
    image1 = load_image_from_file(args.image1)
    angles1 = detect_angle_from_image(image1, args.image1)
    print(f"  Angles: {angles1[0]:.2f}° (direct) or {angles1[1]:.2f}° (flipped)")
    print()
    
    # Load and process second image
    print(f"Processing image 2: {args.image2} ({args.temp2}°C)")
    image2 = load_image_from_file(args.image2)
    angles2 = detect_angle_from_image(image2, args.image2)
    print(f"  Angles: {angles2[0]:.2f}° (direct) or {angles2[1]:.2f}° (flipped)")
    print()
    
    # Resolve ambiguity and calculate calibration
    print("Resolving 180° ambiguity and calculating calibration...")
    print()
    
    try:
        candidates = resolve_angle_ambiguity(
            args.temp1, angles1,
            args.temp2, angles2,
            args.min_temp, args.max_temp
        )
        
        if args.show_all:
            print("All candidate calibrations (sorted by score):")
            print()
            for i, cand in enumerate(candidates, 1):
                print(f"Candidate {i} (score: {cand['score']}):")
                print(f"  {args.temp1}°C angle: {cand['angle1']:.2f}°")
                print(f"  {args.temp2}°C angle: {cand['angle2']:.2f}°")
                print(f"  Zero angle: {cand['zero']:.2f}°")
                print(f"  Max angle: {cand['max']:.2f}°")
                print(f"  Span: {cand['span']:.1f}°")
                print(f"  Degrees/°C: {cand['deg_per_c']:.3f}")
                print()
        
        best = candidates[0]
        
        print(f"Best calibration (score: {best['score']}):")
        print(f"  Using angles: {best['angle1']:.2f}° @ {args.temp1}°C, {best['angle2']:.2f}° @ {args.temp2}°C")
        print(f"  Calculated span: {best['span']:.1f}°")
        print(f"  Degrees per °C: {best['deg_per_c']:.3f}°/°C")
        
        # Save the calibration
        save_calibration_values(
            best['zero'],
            best['max'],
            args.min_temp,
            args.max_temp
        )
        
        print()
        print("✅ Calibration complete!")
        print()
        print("Next steps:")
        print("  1. Run: ./analyze_test_images.py")
        print("  2. Verify all test images pass with <2°C error")
        print("  3. If errors are too large, recalibrate with different images")
        
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
