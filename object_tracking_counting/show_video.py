import cv2

video_path = r"C:\\Users\\Admin\Downloads\\vehicle-counting.mp4"
cap = cv2.VideoCapture(video_path)

# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create a window with the same size as the video frame
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Tracking", frame_width, frame_height)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("YOLOv8 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()