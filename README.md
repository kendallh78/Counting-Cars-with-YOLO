Overview

This project is designed to monitor road traffic and count objects (humans, cars, and bikes) crossing pedestrian lines in a given video. The solution leverages machine vision algorithms and pre-trained models to achieve accurate detection and counting. The final results are displayed on-screen in real time, and a detailed report of the methodology and outcomes is included.

Objectives
- Read and process the provided video file (TrafficVideo.mp4) frame by frame.
- Segment pedestrian lines using computer vision techniques.
- Detect and classify objects (humans, cars, and bikes) using a pre-trained model.
- Implement a custom protocol to accurately count crossing actions while handling edge cases.
- Display the counts of each object type in real-time on the video.

Features
- Pedestrian Line Segmentation: Identifies and segments pedestrian lines in the video.
- Object Detection: Uses YOLO or TensorFlow pre-trained models for human, car, and bike identification.
- Accurate Counting: Tracks objects crossing pedestrian lines and ensures each crossing is counted only once.
- Real-Time Visualization: Displays the counts dynamically in the top-left corner of the video.
- Robust Protocols: Handles occlusions and re-entry of objects to avoid duplicate counts.
