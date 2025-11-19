#!/usr/bin/env python3
"""
Exploratory script to analyze test images and understand LED/gauge detection requirements
"""

import cv2
import numpy as np
import os

def analyze_image(filepath):
    """Analyze a single test image for green LEDs and gauge characteristics"""
    img = cv2.imread(filepath)
    if img is None:
        print(f"Failed to load {filepath}")
        return
    
    filename = os.path.basename(filepath)
    print(f"\n{'='*80}")
    print(f"Analyzing: {filename}")
    print(f"Shape: {img.shape}")
    
    # Extract expected values from filename
    parts = filename.split('_')
    expected_state = parts[1]  # 'on' or 'off'
    expected_temp = parts[2]   # e.g., '31C'
    lighting = parts[4].split('.')[0]  # 'ambient' or 'dark'
    print(f"Expected: Pump={expected_state}, Temp={expected_temp}, Lighting={lighting}")
    
    # 1. Analyze green LED detection
    print("\n--- GREEN LED DETECTION ---")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Try multiple green ranges
    green_ranges = [
        ("Narrow", [35, 100, 100], [85, 255, 255]),
        ("Wider", [30, 50, 50], [90, 255, 255]),
        ("Bright only", [35, 100, 150], [85, 255, 255])
    ]
    
    for name, lower, upper in green_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = [c for c in contours if cv2.contourArea(c) > 50]
        print(f"  {name} range: {len(significant)} bright green regions (area > 50)")
    
    # 2. Analyze circular gauge detection
    print("\n--- CIRCULAR GAUGE DETECTION ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Try different Hough circle parameters
    param_sets = [
        ("Default", 50, 30, 30, 200),
        ("Relaxed", 40, 25, 30, 200),
        ("Strict", 60, 40, 30, 200)
    ]
    
    for name, p1, p2, minR, maxR in param_sets:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=p1,
            param2=p2,
            minRadius=minR,
            maxRadius=maxR
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"  {name}: Found {len(circles)} circles")
            for i, (x, y, r) in enumerate(circles[:3]):  # Show first 3
                print(f"    Circle {i}: center=({x},{y}), radius={r}")
        else:
            print(f"  {name}: No circles found")
    
    # 3. Analyze needle detection area
    print("\n--- NEEDLE REGION ANALYSIS ---")
    # Use first detected circle if available
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=50, param2=30, minRadius=30, maxRadius=200
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        cx, cy, radius = circles[0]
        
        # Create gauge mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(radius * 0.9), 255, -1)
        cv2.circle(mask, (cx, cy), int(radius * 0.2), 0, -1)  # Remove center
        
        # Analyze dark/black regions (potential needle)
        hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask)
        v_channel = hsv_masked[:, :, 2]  # Value channel
        
        dark_mask = cv2.inRange(v_channel, np.array([0]), np.array([80]))
        dark_pixels = cv2.countNonZero(dark_mask)
        
        print(f"  Gauge center: ({cx}, {cy}), radius: {radius}")
        print(f"  Dark pixels in annular region: {dark_pixels}")
        
        # Try edge detection
        edges = cv2.Canny(gray, 50, 150)
        edges_in_gauge = cv2.bitwise_and(edges, mask)
        edge_pixels = cv2.countNonZero(edges_in_gauge)
        print(f"  Edge pixels in gauge region: {edge_pixels}")
        
        # Try Hough lines
        lines = cv2.HoughLinesP(
            edges_in_gauge,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=int(radius * 0.3),
            maxLineGap=10
        )
        
        if lines is not None:
            print(f"  Detected {len(lines)} line segments")
        else:
            print(f"  No line segments detected")
    else:
        print("  Cannot analyze - no gauge circle detected")

def main():
    test_dir = 'test_images'
    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
    
    for filename in image_files:
        filepath = os.path.join(test_dir, filename)
        analyze_image(filepath)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
