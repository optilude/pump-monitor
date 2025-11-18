# Complete Pump Monitor Setup Guide - Final Version

## Hardware Configuration

- **Raspberry Pi Zero 2 W** with fresh Pi OS installation
- **Camera Module 3 NoIR** with Pi Zero camera cable
- **2× IR LEDs (940nm)** connected to:
  - LED 1: GPIO 17 (Pin 11) via 220Ω resistor → Ground (Pin 9)
  - LED 2: GPIO 27 (Pin 13) via 220Ω resistor → Ground (Pin 14)
- **Power supply** (5V, 2.5A micro-USB)

## Part 1: System Configuration

### 1.1 Update System

```bash
# SSH into your Pi
ssh pi@pumpmonitor.local

# Update system packages
sudo apt update && sudo apt upgrade -y
```

### 1.2 Enable Camera

```bash
sudo raspi-config
```

Navigate:
- **Interface Options** → **Camera** → **Enable**
- Select **Finish**
- **Reboot now** → Yes

Wait 30 seconds, then reconnect:
```bash
ssh pi@pumpmonitor.local
```

### 1.3 Install Required Packages

```bash
# Install system packages
sudo apt install -y python3-pip python3-picamera2 python3-opencv git

# Install Python libraries
pip3 install paho-mqtt opencv-python numpy --break-system-packages
```

## Part 2: Hardware Testing

### 2.1 Test Camera

```bash
# Test camera detection
libcamera-hello --list-cameras
```

**Expected output:**
```
Available cameras
-----------------
0 : imx708 [4608x2592] (/base/soc/i2c0mux/i2c@1/imx708@1a)
```

**Capture test image:**
```bash
libcamera-jpeg -o ~/test_camera.jpg --width 1920 --height 1080
```

**Download to your Mac to verify:**
```bash
# On your Mac
scp pi@pumpmonitor.local:~/test_camera.jpg ~/Desktop/
```

Open and verify image quality.

### 2.2 Test IR LEDs

Create test script:
```bash
nano ~/test_leds.py
```

Paste:
```python
#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

IR_LED_PINS = [17, 27]

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in IR_LED_PINS:
    GPIO.setup(pin, GPIO.OUT)

print("Testing IR LEDs - view through phone camera!")
print("LED 1 (GPIO 17) blinking...")
for i in range(3):
    GPIO.output(17, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(17, GPIO.LOW)
    time.sleep(0.5)

print("LED 2 (GPIO 27) blinking...")
for i in range(3):
    GPIO.output(27, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(27, GPIO.LOW)
    time.sleep(0.5)

print("Both LEDs on for 3 seconds...")
GPIO.output(17, GPIO.HIGH)
GPIO.output(27, GPIO.HIGH)
time.sleep(3)

GPIO.output(17, GPIO.LOW)
GPIO.output(27, GPIO.LOW)
GPIO.cleanup()
print("Test complete!")
```

Run test:
```bash
chmod +x ~/test_leds.py
python3 ~/test_leds.py
```

**Point your phone camera at the LEDs** - you should see them glow purple/white.

### 2.3 Test Camera with IR LEDs

```bash
nano ~/test_camera_ir.py
```

Paste:
```python
#!/usr/bin/env python3
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time

IR_LED_PINS = [17, 27]

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in IR_LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Setup camera
camera = Picamera2()
config = camera.create_still_configuration(main={"size": (1920, 1080)})
camera.configure(config)
camera.start()
time.sleep(2)

print("Capturing WITHOUT IR...")
camera.capture_file("test_no_ir.jpg")

print("Capturing WITH IR...")
GPIO.output(17, GPIO.HIGH)
GPIO.output(27, GPIO.HIGH)
time.sleep(0.5)
camera.capture_file("test_with_ir.jpg")
GPIO.output(17, GPIO.LOW)
GPIO.output(27, GPIO.LOW)

print("Done! Images saved.")
camera.stop()
GPIO.cleanup()
```

Run:
```bash
chmod +x ~/test_camera_ir.py
python3 ~/test_camera_ir.py
```

Download both images:
```bash
# On Mac
scp pi@pumpmonitor.local:~/test_*.jpg ~/Desktop/
```

Compare - the IR version should be brighter in darkness.

## Part 3: Install Monitoring Software

### 3.1 Create Project Directory

```bash
mkdir -p ~/pump-monitor/images
cd ~/pump-monitor
```

### 3.2 Install Gauge Reader

Create `gauge_reader.py`:
```bash
nano ~/pump-monitor/gauge_reader.py
```

**Copy the entire `gauge_reader.py` script from the artifact** and paste it into nano.

Save: `Ctrl+X`, `Y`, `Enter`

Make executable:
```bash
chmod +x ~/pump-monitor/gauge_reader.py
```

### 3.3 Install Main Monitor Script

Create `pump_monitor.py`:
```bash
nano ~/pump-monitor/pump_monitor.py
```

**Copy the entire `pump_monitor.py` script from the artifact** and paste it into nano.

Save: `Ctrl+X`, `Y`, `Enter`

Make executable:
```bash
chmod +x ~/pump-monitor/pump_monitor.py
```

### 3.4 Configure MQTT Settings

Edit configuration:
```bash
nano ~/pump-monitor/pump_monitor.py
```

Find and update these lines (near top of file):
```python
# MQTT Configuration
MQTT_BROKER = "192.168.1.100"  # YOUR NAS IP ADDRESS
MQTT_PORT = 1883
MQTT_USER = None  # Your MQTT username or None
MQTT_PASS = None  # Your MQTT password or None
```

Save: `Ctrl+X`, `Y`, `Enter`

## Part 4: Mount and Position Camera

### 4.1 Physical Setup

1. **Mount Pi + Camera** near pump using case with mounting holes
2. **Position camera** to see BOTH:
   - The 3 green LEDs on pump face
   - The temperature gauge (needle and numbers visible)
3. **Distance:** 0.5-2 meters optimal
4. **Angle:** Straight-on view of gauge face (minimize glare)

### 4.2 Capture Positioning Test Image

```bash
cd ~/pump-monitor

# Turn on IR LEDs and capture
python3 -c "
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
for pin in [17, 27]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)

camera = Picamera2()
camera.configure(camera.create_still_configuration(main={'size': (1920, 1080)}))
camera.start()
time.sleep(2)
camera.capture_file('positioning_test.jpg')
camera.stop()

for pin in [17, 27]:
    GPIO.output(pin, GPIO.LOW)
GPIO.cleanup()
print('Image captured: positioning_test.jpg')
"
```

Download and inspect:
```bash
# On Mac
scp pi@pumpmonitor.local:~/pump-monitor/positioning_test.jpg ~/Desktop/
```

**Verify:**
- ✅ Can clearly see 3 green LEDs on pump
- ✅ Can clearly see temperature gauge
- ✅ Gauge needle is visible
- ✅ Gauge numbers are readable
- ✅ No excessive glare on gauge face

**Adjust camera position if needed**, then take another test image.

## Part 5: Calibration

### 5.1 Test Green LED Detection

```bash
cd ~/pump-monitor

python3 -c "
import cv2
import numpy as np

img = cv2.imread('positioning_test.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Detect green
lower = np.array([35, 100, 100])
upper = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

cv2.imwrite('green_detection.jpg', mask)
green_pixels = cv2.countNonZero(mask)
print(f'Green pixels detected: {green_pixels}')
print('Saved green_detection.jpg - white areas show detected green')
"
```

Download `green_detection.jpg`:
```bash
# On Mac
scp pi@pumpmonitor.local:~/pump-monitor/green_detection.jpg ~/Desktop/
```

**Check:** The 3 green LEDs should appear as white spots on black background.

**If LEDs not detected:**
```bash
nano ~/pump-monitor/pump_monitor.py
```

Adjust:
```python
LED_COLOR_LOWER = np.array([30, 80, 80])   # Wider range
LED_COLOR_UPPER = np.array([90, 255, 255])
LED_MIN_AREA = 20  # Lower if LEDs are very small
```

### 5.2 Test Gauge Detection

```bash
python3 ~/pump-monitor/gauge_reader.py test positioning_test.jpg
```

**Expected output:**
```
Testing gauge reading on: positioning_test.jpg
============================================================
Gauge detected: center=(x, y), radius=r
Needle detected (black): angle=xxx.x°
Result: temperature=XX.X°C

Final Result:
  Temperature: XX.X°C
  Confidence: high/medium
  Notes: Needle at xxx.x° = XX.X°C
```

**If gauge not detected:**
```bash
nano ~/pump-monitor/gauge_reader.py
```

Adjust:
```python
GAUGE_MIN_RADIUS = 20   # Lower if gauge is very small
GAUGE_MAX_RADIUS = 300  # Higher if gauge is large
```

**If needle not detected:**

Try adjusting needle color detection - check what color your needle is (black/red/white).

### 5.3 Calibrate Gauge for Accuracy

**Capture two images at known temperatures:**

**Image 1 - When pump is cold (e.g., 20°C):**
```bash
# Wait until gauge shows approximately 20°C
# Then capture:
python3 ~/test_camera_ir.py
mv test_with_ir.jpg gauge_20c.jpg
```

**Image 2 - When pump is hot (e.g., 60°C):**
```bash
# Run pump until gauge shows approximately 60°C
# Then capture:
python3 ~/test_camera_ir.py
mv test_with_ir.jpg gauge_60c.jpg
```

**Run calibration:**
```bash
cd ~/pump-monitor
python3 gauge_reader.py calibrate gauge_20c.jpg 20 gauge_60c.jpg 60
```

**Expected output:**
```
Calibrating with 20°C and 60°C...
Gauge detected: center=(x, y), radius=r
Needle detected: angle=xxx°
Detected angles: xxx° at 20°C, yyy° at 60°C
Calibration saved: {...}
Calibration successful!
```

This creates `gauge_calibration.json` for precise readings.

**Note:** Use actual temperatures shown on gauge, not examples. Can use any two temperatures at least 30°C apart.

## Part 6: Test Full System

### 6.1 Manual Test Run

```bash
cd ~/pump-monitor
python3 pump_monitor.py
```

**Watch output - should see:**
```
============================================================
PUMP MONITOR - CV NEEDLE DETECTION VERSION
============================================================
LED check: 300s (5.0 min)
Temp when ON: 300s (5.0 min)
Temp when OFF: 1800s (30.0 min)
Method: Local CV (LED + Needle Angle Detection)
Monthly cost: £0
Loaded calibration: {...}
GPIO initialized: IR LEDs on pins [17, 27]
Camera initialized: 1920x1080
System ready
============================================================
Starting cycle
Capturing image...
Detecting LED (local CV)...
LED: ON/OFF (X LEDs detected, conf: XX%, status LED at (x, y))
First temperature check
Reading temperature gauge (CV needle detection)...
Gauge detected: center=(x, y), radius=r
Needle detected: angle=xxx°
Result: XX.X°C
Publishing to HA...
Published: pump=on/off, temp=XX.X°C
Cycle complete
Waiting 300s...
```

**Let it run for 10-15 minutes (2-3 cycles).**

Press `Ctrl+C` to stop.

### 6.2 Check Generated Files

```bash
# Check images
ls -lh ~/pump-monitor/images/

# Check logs
tail -50 ~/pump-monitor/pump_monitor.log

# Check state
cat ~/pump-monitor/state.json
```

### 6.3 Verify Detection Accuracy

**Download a recent image:**
```bash
# On Mac
scp pi@pumpmonitor.local:~/pump-monitor/images/pump_*_temp.jpg ~/Desktop/
```

**Check:**
- Are the green LEDs visible?
- Is the gauge needle clearly visible?
- Does the detected temperature match the gauge reading?

**If temperature readings are off by more than ±5°C:**
- Recalibrate with more accurate known temperatures
- Ensure gauge is clearly visible without glare
- Check needle detection is working (`Needle detected` in logs)

## Part 7: Home Assistant Configuration

### 7.1 Ensure MQTT Integration

In Home Assistant:
1. Go to **Settings** → **Devices & Services**
2. Check if **MQTT** integration is installed
3. If not: **Add Integration** → search **MQTT** → configure broker

### 7.2 Add Sensor Configuration

Edit `configuration.yaml`:

```yaml
mqtt:
  sensor:
    - name: "Pump Status"
      state_topic: "home/pump/status"
      icon: mdi:pump
      
    - name: "Pump Temperature"
      state_topic: "home/pump/temperature"
      unit_of_measurement: "°C"
      device_class: temperature
      state_class: measurement
      icon: mdi:thermometer
      
    - name: "Pump Last Check"
      state_topic: "home/pump/last_check"
      device_class: timestamp
      icon: mdi:clock-check
      
    - name: "Pump LED Confidence"
      state_topic: "home/pump/led_confidence"
      icon: mdi:check-circle
      
    - name: "Pump Temperature Confidence"
      state_topic: "home/pump/temp_confidence"
      icon: mdi:thermometer-check
      
    - name: "Pump Notes"
      state_topic: "home/pump/notes"
      icon: mdi:note-text
```

### 7.3 Restart Home Assistant

In Home Assistant:
1. **Settings** → **System** → **Restart**
2. Click **Check Configuration** (wait for validation)
3. Click **Restart**

Wait 1-2 minutes for restart.

### 7.4 Verify Sensors

After restart:
1. **Settings** → **Devices & Services** → **MQTT**
2. Look for your pump sensors
3. Or: **Developer Tools** → **States** → search "pump"

You should see:
- `sensor.pump_status`
- `sensor.pump_temperature`
- `sensor.pump_last_check`
- `sensor.pump_led_confidence`
- `sensor.pump_temperature_confidence`
- `sensor.pump_notes`

### 7.5 Add Dashboard Card

Go to your dashboard, click **Edit**, then **Add Card**:

```yaml
type: vertical-stack
cards:
  - type: entity
    entity: sensor.pump_status
    name: Pump Status
    icon: mdi:pump
    
  - type: gauge
    entity: sensor.pump_temperature
    min: 0
    max: 80
    name: Pump Temperature
    needle: true
    severity:
      green: 0
      yellow: 50
      orange: 60
      red: 70
      
  - type: entities
    entities:
      - entity: sensor.pump_last_check
        name: Last Check
      - entity: sensor.pump_led_confidence
        name: LED Confidence
      - entity: sensor.pump_temperature_confidence
        name: Temperature Confidence
      - entity: sensor.pump_notes
        name: Notes
```

Save the dashboard.

### 7.6 Test MQTT Publishing

With `pump_monitor.py` running on Pi, check Home Assistant dashboard.

Sensors should update every 5 minutes.

## Part 8: Run as Automatic Service

### 8.1 Create Systemd Service

```bash
sudo nano /etc/systemd/system/pump-monitor.service
```

Paste:
```ini
[Unit]
Description=Pump Monitor Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pump-monitor
ExecStart=/usr/bin/python3 /home/pi/pump-monitor/pump_monitor.py
Restart=always
RestartSec=10
StandardOutput=append:/home/pi/pump-monitor/service.log
StandardError=append:/home/pi/pump-monitor/service.log

[Install]
WantedBy=multi-user.target
```

Save: `Ctrl+X`, `Y`, `Enter`

### 8.2 Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable pump-monitor.service

# Start service now
sudo systemctl start pump-monitor.service

# Check status
sudo systemctl status pump-monitor.service
```

Should show: `Active: active (running)` in green.

### 8.3 Monitor Service

```bash
# View live logs
sudo journalctl -u pump-monitor.service -f

# View recent logs
sudo journalctl -u pump-monitor.service -n 50

# View application log
tail -f ~/pump-monitor/pump_monitor.log
```

### 8.4 Service Management Commands

```bash
# Stop service
sudo systemctl stop pump-monitor.service

# Restart service
sudo systemctl restart pump-monitor.service

# Disable auto-start
sudo systemctl disable pump-monitor.service

# Check status
sudo systemctl status pump-monitor.service
```

## Part 9: Monitoring and Maintenance

### 9.1 Check System Status

```bash
# Service status
sudo systemctl status pump-monitor.service

# Recent activity
tail -30 ~/pump-monitor/pump_monitor.log

# Recent detections
grep "LED:\|Result:" ~/pump-monitor/pump_monitor.log | tail -20

# Disk space
df -h
du -sh ~/pump-monitor/images/
```

### 9.2 Adjust Check Intervals

If you want to change how often things are checked:

```bash
nano ~/pump-monitor/pump_monitor.py
```

Modify:
```python
LED_CHECK_INTERVAL_SECONDS = 300   # Check LED every 5 min
TEMP_CHECK_INTERVAL_PUMP_ON = 300  # Check temp every 5 min when ON
TEMP_CHECK_INTERVAL_PUMP_OFF = 1800 # Check temp every 30 min when OFF
```

Then restart:
```bash
sudo systemctl restart pump-monitor.service
```

### 9.3 Recalibrate Gauge

If readings become inaccurate:

```bash
# Stop service
sudo systemctl stop pump-monitor.service

# Capture new calibration images
cd ~/pump-monitor
python3 ~/test_camera_ir.py
mv test_with_ir.jpg gauge_new1.jpg
# (wait for different temperature)
python3 ~/test_camera_ir.py
mv test_with_ir.jpg gauge_new2.jpg

# Recalibrate
python3 gauge_reader.py calibrate gauge_new1.jpg 25 gauge_new2.jpg 55

# Restart service
sudo systemctl start pump-monitor.service
```

### 9.4 Backup Configuration

```bash
# On your Mac
scp pi@pumpmonitor.local:~/pump-monitor/*.py ~/backup/
scp pi@pumpmonitor.local:~/pump-monitor/*.json ~/backup/
```

## Part 10: Troubleshooting

### Camera Issues

**Camera not detected:**
```bash
libcamera-hello --list-cameras
# If no camera, check cable connection and reboot
```

**Poor image quality:**
- Check camera focus (may need adjustment)
- Ensure adequate IR LED illumination
- Check for lens obstruction

### LED Detection Issues

**LEDs not detected:**
```bash
# Test green detection
grep "LED:" ~/pump-monitor/pump_monitor.log | tail -10
```

If confidence low, adjust color range in `pump_monitor.py`.

### Gauge Reading Issues

**Gauge not found:**
```bash
# Test manually
cd ~/pump-monitor
python3 ~/test_camera_ir.py
python3 gauge_reader.py test test_with_ir.jpg
```

Check radius parameters in `gauge_reader.py`.

**Temperature readings incorrect:**
- Recalibrate with known temperatures
- Check gauge face is clearly visible
- Verify needle color detection

### MQTT Issues

**Sensors not updating in HA:**
```bash
# Test MQTT connection
sudo apt install mosquitto-clients
mosquitto_pub -h YOUR_NAS_IP -t test/topic -m "hello"
```

Check:
- MQTT broker running on NAS
- Correct IP address in configuration
- Firewall not blocking port 1883

### Service Issues

**Service not running:**
```bash
sudo systemctl status pump-monitor.service
sudo journalctl -u pump-monitor.service -n 100
```

Common causes:
- Python errors (check logs)
- Missing dependencies
- Permission issues

## System Specifications

### Performance
- **LED detection:** ~0.1-0.2 seconds
- **Gauge detection:** ~0.5-1.0 seconds
- **Processing per cycle:** ~1-2 seconds
- **CPU usage:** <5% average
- **RAM usage:** ~150-200MB

### Storage
- **Images:** ~2MB each
- **Retention:** 4 hours (auto-cleanup)
- **Total storage:** ~100-200MB

### Cost
- **Hardware:** £60-80 (one-time)
- **Electricity:** ~1W = negligible
- **API/Cloud:** £0/month
- **Total monthly cost: £0** ✅

## Quick Reference

### Start/Stop Commands
```bash
sudo systemctl start pump-monitor.service   # Start
sudo systemctl stop pump-monitor.service    # Stop
sudo systemctl restart pump-monitor.service # Restart
sudo systemctl status pump-monitor.service  # Status
```

### View Logs
```bash
tail -f ~/pump-monitor/pump_monitor.log           # Application log
sudo journalctl -u pump-monitor.service -f        # Service log
```

### Test Components
```bash
python3 ~/test_leds.py                            # Test IR LEDs
python3 ~/test_camera_ir.py                       # Test camera
python3 ~/pump-monitor/gauge_reader.py test <img> # Test gauge reading
```

## Success Criteria

✅ Camera captures clear images of pump and gauge  
✅ IR LEDs illuminate scene in darkness  
✅ Green LEDs detected reliably (confidence >80%)  
✅ Gauge needle detected and temperature read (confidence high/medium)  
✅ Temperature readings within ±2-3°C of actual  
✅ Data publishes to Home Assistant via MQTT  
✅ Service runs automatically on boot  
✅ System operates continuously without intervention

## Support

If you encounter issues:

1. **Check logs** for error messages
2. **Test components individually** (camera, LEDs, gauge detection)
3. **Verify network connectivity** (MQTT broker accessible)
4. **Review calibration** if readings are inaccurate

System is now complete and monitoring your pump!