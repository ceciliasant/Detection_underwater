# Detection in underwater videos

Underwater Detection using OpenCV.

## Overview

This project is designed to detect underwater objects, particularly green algae and fishes, in videos using OpenCV. The system processes video frames, applies color-based segmentation, filters noise, and tracks detected objects.

## Features

- Color-based segmentation: Uses HSV color space for detection.

- Morphological filtering: Reduces noise and enhances detection accuracy.

- Contour detection: Identifies objects across frames.

- Detection with and without tracking.

- Customizable parameters: Allows adjusting HSV limits interactively.

## Project Structure

```
📂 code
├── algae_final.py         # Main script for algae detection
├── color.py               # HSV range selection tool
├── without_tracking.py    # Main version for tracking fishes
├── with_tracking.py       # Advanced version with tracking
├── limits.json            # Saved HSV limits for algae detection
├── peixe.mp4              # Sample video for testing
├── utils/
│   ├── frame_processing.py # Utility functions for frame processing
```

## Requirements

- Python 3.x

- OpenCV

- NumPy


## Usage

1. Extract HSV Ranges

      To determine the correct HSV limits for algae detection, run:
      
      ```
      python color.py
      ```
      
      Use trackbars to adjust HSV values.
      
      Press l to save the current HSV range.
      
      Press w to save all collected ranges to limits.json.
      
      Press q to exit.

2. Run Algae Detection

      Once HSV limits are set, run:
      
      ```
      python algae_final.py
      ```
      
      This script processes the video and applies color segmentation and contour detection across frames for algae detection.

3. Run Fishes Detection without Tracking

      ```
      python with_tracking.py
      ```
      
      This script detects fishes based on motion detection, without explicit tracking, by analysing contours through frames based on their spatial proximity.
    
4.  Run Fishes Detection with Tracking

      ```
      python with_tracking.py
      ```

      This script detects fishes using both contour based detection and optical flow tracking.
      
**Output**

The processed video frames with detected objects will be displayed.

Press q to exit playback.

## Contributors

Artur Almeida

Cecília Santos
