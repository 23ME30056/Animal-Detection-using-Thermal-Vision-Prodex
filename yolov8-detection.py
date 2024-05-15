
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8l.pt')

# Defining classes of common animals found near highways
common_animals = ["cat", "dog", "horse", "sheep", "cow", "deer", "rabbit", "fox", "tiger", "lion"]

# Load class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video file
video_path = "animal2.mp4"  # Path to your video file
cap = cv2.VideoCapture(video_path)

# OpenCV window properties
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Prepare to write to video file
video=cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'MJPG'),25,(596,336))

# Store the previous positions of bounding box centers
previous_centers = {}

while True:
    # Read frame from the video
    ret, frame = cap.read()
    result = model.predict(frame)[0]
    if not ret:
        break  # Break the loop if there are no more frames
    
    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), True, crop=False)

    # Set input

    # Initialize lists for bounding boxes, confidence scores, and class IDs
    boxes = []
    class_ids = []

    for box in result.boxes:
        xywh = [int(box.xywh[0][0].item()), int(box.xywh[0][1]), int(box.xywh[0][2].item()), int(box.xywh[0][3].item())]
        boxes.append(xywh)
        class_ids.append(box.cls[0].item())


    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a thermal colormap
    thermal_frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Draw bounding boxes, labels, and center lines on the thermal frame
    font = cv2.FONT_HERSHEY_PLAIN
    height, width, _ = frame.shape
    center_line_y = int(height * 0.6)

    cv2.line(thermal_frame, (0, center_line_y-15), (width, center_line_y-15), (0, 0, 255), 2)  # Draw the horizontal center line

    # 15: 'cat',
    # 16: 'dog',
    # 17: 'horse',
    # 18: 'sheep',
    # 19: 'cow',
    # 20: 'elephant',
    # 21: 'bear',
    # 22: 'zebra',
    # 23: 'giraffe

    for i in range(len(boxes)):
        if class_ids[i] > 23 or class_ids[i] < 15: continue
        x, y, w, h = boxes[i]
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.rectangle(thermal_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
        cv2.putText(thermal_frame, "Animal", (x, y - 10), font, 1, (255, 255, 255), 2)
        cv2.circle(thermal_frame, (x, y), 5, (255, 0, 0), -1)  # Draw the center of the bounding box

        # Track the movement of the bounding box center
        if i not in previous_centers:
            previous_centers[i] = []

        previous_centers[i].append((center_x, center_y))

        if len(previous_centers[i]) > 2:
            # Calculate the movement direction
            prev_center_x, prev_center_y = previous_centers[i][-2]
            direction_y = center_y - prev_center_y
            
            if direction_y > 0 and center_y > center_line_y:  # Moving downwards and close to the center line
                cv2.putText(thermal_frame, "Brakes Activated !!", (50, 50), font, 3, (255, 0, 0), 3) 
                    # blue 

    # Display frame
    cv2.imshow('Object Detection', thermal_frame)
    video.write(thermal_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
