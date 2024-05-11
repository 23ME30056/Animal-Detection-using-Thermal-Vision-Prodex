
import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture from default camera
cap = cv2.VideoCapture(0)  # 0 for default camera, change it if needed

# OpenCV window properties for fullscreen
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    
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
            if confidence > 0.5:
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

    # Draw bounding boxes and labels
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, (255, 255, 255), 2)

    # Display frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()