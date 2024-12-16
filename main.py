import cv2
import torch
import numpy as np
from yolov5 import YOLOv5
from collections import defaultdict

# Load YOLOv5 model
model_path = '/Users/kendallhaddigan/Downloads/CS549/week9assignment/pythonProject1/yolov5s.pt'
model = YOLOv5(model_path)

# Load video
video_path = '/Users/kendallhaddigan/Downloads/TrafficVideo.mp4'
video = cv2.VideoCapture(video_path)

# Define object labels and count
labels = {'person': 0, 'bicycle': 0, 'car': 0}

# Define tracking
tracked_objects = {}
next_id = 0
object_crossings = defaultdict(lambda: {'counted': False, 'last_position': None})

def iou(box1, box2, threshold=0.7):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou_value = intersection / float(area1 + area2 - intersection)

    return iou_value

def detect_crossing(obj):
    box = obj['box']
    last_position = obj['last_position']
    if last_position is None:
        return False
    y_center = (box[1] + box[3]) / 2
    last_y_center = (last_position[1] + last_position[3]) / 2
    if (last_y_center < frame.shape[0] // 2 and y_center >= frame.shape[0] // 2) or (last_y_center >= frame.shape[0] // 2 and y_center < frame.shape[0] // 2):
        return True
    return False

frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to RGB for YOLOv5
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model.predict(frame_rgb)

    # Process detections
    detections = results.pandas().xyxy[0]
    current_objects = {}

    for _, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']
        if confidence > 0.4 and label in labels:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Check if this object matches any previously tracked object
            matched = False
            for obj_id, obj in tracked_objects.items():
                if iou([x1, y1, x2, y2], obj['box'], threshold=0.9) and label == obj['label']:
                    current_objects[obj_id] = {'box': [x1, y1, x2, y2], 'label': label, 'last_position': obj['box']}
                    matched = True
                    if detect_crossing(current_objects[obj_id]):
                        if not object_crossings[obj_id]['counted']:
                            labels[label] += 1
                            object_crossings[obj_id]['counted'] = True
                    break
            # If not matched, create a new tracked object
            if not matched:
                current_objects[next_id] = {'box': [x1, y1, x2, y2], 'label': label, 'last_position': None}
                next_id += 1

            # Draw boxes and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Update tracked objects
    tracked_objects = current_objects

    # Display counts
    cv2.putText(frame, f"Person: {labels['person']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Bicycles: {labels['bicycle']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Cars: {labels['car']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Traffic Video', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

video.release()
cv2.destroyAllWindows()
