# 🦁 Animal Detection Using Thermal Vision 🐾

This Python script utilizes Ultralytics YOLO (You Only Look Once) model to detect animals in a thermal video stream. It draws bounding boxes around detected animal regions, tracks their movement, and activates brakes if an animal is detected crossing a certain threshold.

## 📋 Requirements

- **Python 3.x**
- **Ultralytics YOLO**: You need to download the YOLO model file (`yolov8l.pt`) from the Ultralytics repository.
- **opencv-python**: OpenCV library (`pip install opencv-python`)
- **numpy**: NumPy library (`pip install numpy`)

## 🔧 Usage

1. **Download the YOLO model file (`yolov8l.pt`) from the Ultralytics repository and place it in the same directory as the script.**

2. **Install the required Python libraries using the provided commands:**

    ```bash
    pip install opencv-python numpy
    ```

3. **Run the script and specify the path to the thermal video file (`video_path`).    here animal.mp4**

    ```bash
    YOLOFINAL.py
    ```

4. **Press 'q' to exit the video stream.**

## 🛠️ Functionality

- **Animal Detection**: The script uses YOLO to detect animals in the thermal video stream and draws bounding boxes around them.
- **Movement Tracking**: It tracks the movement of detected animals and calculates the direction of their movement.
- **Brake Activation**: If an animal is detected moving downwards and close to the center line, the script activates brakes to prevent accidents.

- **NOTE**: test.mp4 is the final video from camera and animal.mp4 is the original one.
