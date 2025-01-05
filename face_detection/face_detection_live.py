import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import cv2
# import os

from ultralytics import YOLO
print("Libraries imported successfully!")

model = YOLO("face_detection.pt")
print("Model loaded successfully!")


# Trying on live video feed
print("Starting camera...")
cap = cv2.VideoCapture(0)
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(source=frame, stream=True,
                        device="cuda", iou=0.8, conf=0.3)

        # Visualize the results on the frame
        annotated_frame = next(results).plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
