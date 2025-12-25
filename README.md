              Real-Time Object Tracking
This repository contains my first computer vision project focused on real-time object tracking.
The project implements a color-based tracking system that detects an object, tracks its motion, and visualizes movement-related information directly on the video stream.
The main goal of this project was to understand core tracking logic, reduce noise, and produce stable and readable motion data.

               Features & Techniques
This project covers the following tracking and motion analysis techniques:
Color-Based Object Detection
Contour Detection
Bounding Box & Orientation
Centroid Tracking
Position & Area Smoothing
Speed Calculation
Direction Detection and Stabilization
Distance Change Analysis
Motion State Classification
Jump & Noise Filtering
Motion Prediction
Trajectory Visualization

         How It Works (High Level)
The system detects a colored object using HSV color space.
Contours are extracted and filtered by area.
Object position is smoothed to reduce jitter.
Speed, direction, and motion state are calculated frame by frame.
Sudden jumps are filtered to improve stability.
When the object is lost, the system resets tracking safely.
All tracking data is visualized in real time on a HUD panel.
Object Lost Detection
Real-Time Visual Overlay

This project represents my first step into object tracking.
The focus was not performance optimization, but understanding tracking fundamentals clearly and correctly.
