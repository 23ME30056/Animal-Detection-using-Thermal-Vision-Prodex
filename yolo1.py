
import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

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

# Store the previous positions of bounding box centers
previous_centers = {}

while True:
    # Read frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break  # Break the loop if there are no more frames
    
    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), True, crop=False)

    # Set input
    net.setInput(blob)

    # Get output layers
    output_layers = net.getUnconnectedOutLayersNames()

    # Forward pass
    outputs = net.forward(output_layers)

    # Initialize lists for bounding boxes, confidence scores, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop through outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and classes[class_id] in common_animals:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a thermal colormap
    thermal_frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Draw bounding boxes, labels, and center lines on the thermal frame
    font = cv2.FONT_HERSHEY_PLAIN
    height, width, _ = frame.shape
    center_line_y = int(height * 0.6)

    cv2.line(thermal_frame, (0, center_line_y), (width, center_line_y), (0, 0, 255), 2)  # Draw the horizontal center line

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(thermal_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(thermal_frame, "Animal", (x, y - 10), font, 1, (255, 255, 255), 2)
            cv2.circle(thermal_frame, (center_x, center_y), 5, (255, 0, 0), -1)  # Draw the center of the bounding box

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

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
