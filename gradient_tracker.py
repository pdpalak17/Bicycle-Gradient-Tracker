#!/usr/bin/env python3.11
import sys
sys.path.insert(1, "/home/disha/realsense_setup/librealsense/build/wrappers/python")

import pyrealsense2 as rs
import numpy as np
import cv2
import math
import os
import csv
from datetime import datetime

# Suppress annoying Qt font warnings from modern opencv-python on Linux/ARM
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts=false"

def calculate_gradient_from_image(image):
    """
    Attempts to calculate the uphill/downhill gradient from the raw image
    by detecting the horizon line using OpenCV.
    """
    height, width = image.shape
    
    # We focus on the center region to minimize fisheye distortion effects on the edges
    roi_top = int(height * 0.2)
    roi_bottom = int(height * 0.8)
    roi_left = int(width * 0.2)
    roi_right = int(width * 0.8)
    
    roi = image[roi_top:roi_bottom, roi_left:roi_right]
    
    # Edge detection
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Line detection using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    horizon_y = None
    
    if lines is not None:
        # Filter for horizontal-ish lines (representing the horizon or road boundaries)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle in degrees
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # Look for roughly horizontal lines (-25 to 25 degrees)
            if -25 < angle < 25: 
                horizontal_lines.append(line[0])
                
        if horizontal_lines:
            # Pick the longest horizontal line as our likely horizon candidate
            longest_line = None
            max_length = 0
            for (x1, y1, x2, y2) in horizontal_lines:
                length = math.hypot(x2 - x1, y2 - y1)
                if length > max_length:
                    max_length = length
                    longest_line = (x1, y1, x2, y2)
            
            if longest_line:
                # Get the average Y position of the horizon line in the full image
                horizon_y = (longest_line[1] + longest_line[3]) / 2.0 + roi_top
                
    # Calculate pitch and gradient based on pixel displacement from center
    center_y = height / 2.0
    
    pitch_deg = 0.0
    gradient_percent = 0.0
    
    if horizon_y is not None:
        # T265 Fisheye vertical FOV is roughly 163 degrees. 
        # We do a linear approximation: degrees per pixel (for center region)
        deg_per_pixel = 163.0 / height
        
        # If horizon is above center, camera is pointing down (pitch negative for downhill)
        # If horizon is below center, camera is pointing up (pitch positive for uphill)
        pixel_shift = center_y - horizon_y 
        pitch_deg = pixel_shift * deg_per_pixel
        
        # Gradient = tan(pitch) * 100
        gradient_percent = math.tan(math.radians(pitch_deg)) * 100.0
        
    return pitch_deg, gradient_percent, horizon_y

def calculate_pedal_force(pitch_deg, total_mass=97.0, r_wheel=0.34, y_crank=0.20):
    """
    Calculate the pedal force required to climb a hill at a given pitch angle.
    Formula: F_pedal = mg sinθ * R_wheel / Y_crank
    """
    # If the pitch is negative (downhill), gravity helps, so required pedal force to overcome gravity is 0.
    if pitch_deg <= 0:
        return 0.0
        
    g = 9.81  # m/s^2
    theta_rad = math.radians(pitch_deg)
    
    # Calculate force due to gravity pulling down the slope
    f_gravity = total_mass * g * math.sin(theta_rad)
    
    # Calculate required pedal force from torque balance
    f_pedal = f_gravity * r_wheel / y_crank
    
    return f_pedal

def main():
    # Setup RealSense Pipeline
    pipe = rs.pipeline()
    cfg = rs.config()
    
    # Only enable the Fisheye stream (No pose stream used here!)
    # T265 Native Resolution for Fisheye is 848x800 at 30 fps. Both must be enabled.
    cfg.enable_stream(rs.stream.fisheye, 1, 848, 800, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.fisheye, 2, 848, 800, rs.format.y8, 30)

    print("Starting T265 pipeline... Please wait")
    try:
        pipe.start(cfg)
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        print("\nMake sure the T265 is plugged in and udev rules are installed.")
        return

    print("\nCamera started! Tracking gradient using pure OpenCV from images...")
    print("Press 'q' in the image window to stop, or Ctrl+C in terminal.\n")

    # CSV Logging Setup
    csv_filename = f"gradient_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Pitch_deg', 'Gradient_percent', 'Direction', 'Pedal_Force_N'])

    # EMA Smoothing Variables
    alpha = 0.15 # 15% new value, 85% old value. Lower = smoother but slower to update.
    smoothed_pitch = 0.0
    first_frame = True

    try:
        while True:
            # Wait for frames
            frames = pipe.wait_for_frames()
            fisheye_frame = frames.get_fisheye_frame(1)
            
            if fisheye_frame:
                # Convert to numpy array (grayscale)
                frame_data = np.asanyarray(fisheye_frame.get_data())
                
                # Calculate the gradient fully from the image data using OpenCV
                pitch_deg, gradient_percent, horizon_y = calculate_gradient_from_image(frame_data)
                
                # Calculate required pedal force
                pedal_force_n = calculate_pedal_force(pitch_deg)
                
                # Determine direction string
                if horizon_y is None:
                    direction = "NO HORIZON DETECTED"
                    # If no horizon, gradually ease pitch to 0
                    smoothed_pitch = smoothed_pitch * (1 - alpha)
                    display_pitch = smoothed_pitch
                    display_gradient = 0.0
                else:
                    # Apply Exponential Moving Average (EMA) smoothing for stability
                    if first_frame:
                        smoothed_pitch = pitch_deg
                        first_frame = False
                    else:
                        smoothed_pitch = (alpha * pitch_deg) + ((1 - alpha) * smoothed_pitch)
                    
                    display_pitch = smoothed_pitch
                    display_gradient = math.tan(math.radians(display_pitch)) * 100.0
                    
                direction = "UPHILL" if display_pitch > 2 else "DOWNHILL" if display_pitch < -2 else "FLAT"
                
                # Calculate required pedal force using smoothed pitch
                pedal_force_n = calculate_pedal_force(display_pitch)

                # Log data to CSV
                csv_writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    round(display_pitch, 2),
                    round(display_gradient, 2),
                    direction,
                    round(pedal_force_n, 2)
                ])

                # -----------------------------
                # Display on Captured Image
                # -----------------------------
                # Convert Y8 to BGR for drawing colored overlays
                image_bgr = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
                
                # Annotate the image
                text1 = f"CV Pitch: {display_pitch:+.2f} deg"
                text2 = f"CV Gradient: {display_gradient:+.2f}%"
                text3 = f"Direction: {direction}"
                text4 = f"Pedal Force: {pedal_force_n:.1f} N"
                
                # Draw a dark semi-transparent background for text visibility
                overlay = image_bgr.copy()
                cv2.rectangle(overlay, (10, 15), (320, 160), (0, 0, 0), -1)
                image_bgr = cv2.addWeighted(overlay, 0.6, image_bgr, 0.4, 0)
                
                cv2.putText(image_bgr, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image_bgr, text2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(image_bgr, text3, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(image_bgr, text4, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # Draw a fixed optical center crosshair
                h, w = image_bgr.shape[:2]
                center_x, center_y = w // 2, h // 2
                cv2.drawMarker(image_bgr, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
                
                # Draw the detected horizon line in red
                if horizon_y is not None:
                    hy = int(horizon_y)
                    cv2.line(image_bgr, (0, hy), (w, hy), (0, 0, 255), 3)

                # Show the image window
                cv2.imshow("CV Bicycle Gradient Tracker", image_bgr)
                
                # Dynamic terminal output
                print(f"\r{text1}   |   {text2}   |   {text3}   |   {text4}          ", end="", flush=True)
                
            # OpenCV waitKey to render window and handle 'q' key quitting
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n'q' pressed. Stopping...")
                break
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'csv_file' in locals():
            csv_file.close()
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

]
