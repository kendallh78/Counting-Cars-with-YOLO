# Overview

This project was created for my computer vision course, designed to monitor road traffic and count objects (humans, cars, and bikes) crossing pedestrian lines in a provided video. The code leverages machine vision algorithms and pre-trained models to achieve accurate detection and counting. The final results are displayed on-screen in real time, and a detailed report of the methodology and outcomes is included. I've only included a portion of the video due to file constraints, but attached a screenshot of the counting mechanism working.

# Objectives
- Read and process the provided video file frame by frame
- Segment pedestrian lines using computer vision techniques
- Detect and classify objects (e.g. humans, cars, and bikes) using a pre-trained model
- Implement a custom protocol to accurately count crossing actions while handling edge cases
- Display the counts of each object type in real-time on the video

# Included Features
- Pedestrian Line Segmentation: Identifies and segments pedestrian lines in the video
- Object Detection: Uses YOLOv5 model for human, car, and bike identification
- Accurate Counting: Tracks objects crossing pedestrian lines and ensures each crossing is counted only once
- Real-Time Visualization: Displays the counts dynamically in the top-left corner of the video
- Robust Protocols: Handles occlusions and re-entry of objects to avoid duplicate counts
