from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Define the video path
video_path = r"C:\Users\Admin\Downloads\vehicle-counting.mp4"

# Open the video capture
cap = cv2.VideoCapture(video_path)

# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create a window for displaying results
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Tracking", frame_width, frame_height)

# Create a LineZone for counting objects crossing the line
LINE_START = sv.Point(50, 1500)
LINE_END = sv.Point(3840-50, 1500)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

# Create a LineZoneAnnotator for annotating the line
line_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=3, text_scale=2)

# Create a BoxAnnotator for annotating bounding boxes
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=3, text_scale=2)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        detections = sv.Detections.from_ultralytics(results[0])

        annotated_frame = results[0].orig_img

        if results[0].boxes.id is not None:
                detections.tracker_id = results[0].boxes.id.cpu().numpy().astype(int)

        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id
            in detections
        ]

        # Trigger the LineZone to count objects crossing the line
        line_counter.trigger(detections=detections)

        # Annotate the frame with the line and bounding boxes
        line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)
        box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
