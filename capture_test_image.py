#!/usr/bin/env python3
"""
Capture and catalog test image from Raspberry Pi camera.

This script:
1. Triggers the camera on the Pi to capture an image with IR LEDs
2. Downloads the image
3. Shows it to the user
4. Prompts for pump state, temperature, and descriptor
5. Saves it with the proper naming convention in test_images/

Usage:
    python capture_test_image.py
"""

import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
import shutil

# Configuration
SSH_HOST = "optilude@picam.local"
REMOTE_SCRIPT = "~/test_camera_ir.py"
REMOTE_IMAGE = "test_with_ir.jpg"
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"{description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed")
        print(f"Command: {cmd}")
        print(f"Error: {e.stderr}")
        return False


def main():
    print("=" * 70)
    print("Test Image Capture from Raspberry Pi")
    print("=" * 70)
    print()
    
    # Step 1: Execute camera script on Pi
    print("Step 1: Capturing image on Raspberry Pi...")
    cmd = f"ssh {SSH_HOST} '{REMOTE_SCRIPT}'"
    if not run_command(cmd, "Remote capture"):
        sys.exit(1)
    
    print("✓ Image captured on Pi")
    print()
    
    # Step 2: Download image to temp directory
    print("Step 2: Downloading image...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_with_ir.jpg"
        
        cmd = f"scp {SSH_HOST}:{REMOTE_IMAGE} {tmp_path}"
        if not run_command(cmd, "Download"):
            sys.exit(1)
        
        if not tmp_path.exists():
            print("ERROR: Downloaded file not found")
            sys.exit(1)
        
        print(f"✓ Image downloaded to {tmp_path}")
        print()
        
        # Step 3: Open image for user to view
        print("Step 3: Opening image...")
        cmd = f"open {tmp_path}"
        subprocess.run(cmd, shell=True)
        print("✓ Image opened (check your screen)")
        print()
        
        # Step 4: Prompt user for details
        print("Step 4: Image details")
        print("-" * 70)
        
        # Pump state
        while True:
            pump_state = input("Pump state (on/off): ").strip().lower()
            if pump_state in ['on', 'off']:
                break
            print("  Please enter 'on' or 'off'")
        
        # Temperature
        while True:
            try:
                temp_input = input("Temperature reading (°C): ").strip()
                temperature = float(temp_input)
                if 0 <= temperature <= 100:
                    break
                print("  Temperature should be between 0 and 100°C")
            except ValueError:
                print("  Please enter a valid number")
        
        # Descriptor
        print("\nDescriptor examples:")
        print("  - Date: 20231115, 20231115_am, 20231115_1430")
        print("  - Setup: baseline, setup1, led_left, led_close")
        print("  - Lighting: ambient, dark, morning")
        print("  - Combination: 20231115_setup2, setup1_dark")
        
        while True:
            descriptor = input("\nDescriptor: ").strip()
            if descriptor and ' ' not in descriptor:
                break
            if not descriptor:
                print("  Descriptor is required")
            else:
                print("  Descriptor cannot contain spaces (use underscores)")
        
        print()
        
        # Step 5: Generate filename and move
        print("Step 5: Saving image...")
        
        # Format temperature as integer if it's a whole number, otherwise with decimals
        if temperature == int(temperature):
            temp_str = str(int(temperature))
        else:
            temp_str = str(temperature).replace('.', '_')
        
        filename = f"test_{pump_state}_{temp_str}C_{descriptor}.jpg"
        dest_path = TEST_IMAGES_DIR / filename
        
        # Check if file already exists
        if dest_path.exists():
            overwrite = input(f"\n⚠️  File {filename} already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("\nCancelled. Image not saved.")
                return
        
        # Create test_images directory if it doesn't exist
        TEST_IMAGES_DIR.mkdir(exist_ok=True)
        
        # Move file
        shutil.copy2(tmp_path, dest_path)
        
        print(f"✓ Image saved as: {filename}")
        print(f"  Location: {dest_path}")
        print()
        
    # Step 6: Summary and next steps
    print("=" * 70)
    print("Capture Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print(f"  1. Run tests:    pytest tests/test_cv.py::test_read_gauge_end_to_end[{pump_state}_{temp_str}C_{descriptor}] -v")
    print(f"  2. Quick check:  python analyze_test_images.py")
    print(f"  3. View all:     pytest tests/test_cv.py -v")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
