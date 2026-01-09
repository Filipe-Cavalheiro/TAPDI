# Motion-Based Interactive Game Using Computer Vision and GPU Acceleration

This repository contains a real-time motion-controlled game developed for the Computer Vision course of the Master’s Degree in Electrical and Computer Engineering (2025/2026). The game allows users to interact using body movements captured by a standard webcam, without requiring any additional sensors or controllers.

The system combines deep learning–based face detection, GPU-accelerated template matching using OpenCL, and pose estimation with OpenPose. Face detection is used to initialize and stabilize tracking, while template matching keeps the user centered in the frame in real time. A body region of interest is extracted based on the detected face, and pose estimation is applied to detect hand keypoints.

These hand keypoints are used to control a “Simon Says” style game in which the player must place their hands over specific colored regions according to on-screen instructions. The game includes multiple states, collision detection based on hand position, and increasing difficulty as the score grows.

Most computationally intensive image processing tasks are accelerated on the GPU using custom OpenCL kernels, while the high-level logic and game flow are implemented in Python.

This project demonstrates the practical integration of computer vision algorithms, GPU programming, and interactive application design in a real-time environment.
