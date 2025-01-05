import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

from ultralytics import YOLO
print("Libraries imported successfully!")

model = YOLO("face_detection.pt")
print("Model loaded successfully!")

cwd = os.getcwd()
imgtestpath = os.path.join(cwd, "test")
savepath = os.path.join(cwd, "output")
test_images = os.listdir(imgtestpath)


def infer_images():
    n = len(test_images)
    for idx, img_name in enumerate(test_images):
        print(f"Processing image {idx + 1}/{n} - {img_name}")

        # Create a new figure for each image pair
        fig = plt.figure(figsize=(12, 6))  # Adjust the size for better quality

        # Full path to the test image
        test_image_path = os.path.join(imgtestpath, img_name)

        # Read and display the actual image
        ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, position 1
        image = cv2.imread(test_image_path)
        image_rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        ax1.imshow(image_rgb)
        ax1.axis('off')
        ax1.set_title("Actual Image", fontsize=15)

        # Predict using the YOLO model
        res = model.predict(test_image_path, iou=0.8, conf=0.3, device='cuda')
        res_plotted = res[0].plot()  # Get the annotated image from YOLO

        # Display image with predictions
        ax2 = fig.add_subplot(1, 2, 2)  # 1 row, 2 columns, position 2
        ax2.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        ax2.axis('off')
        ax2.set_title("Image with Predictions", fontsize=15)

        # Save the figure as an image file
        output_filename = os.path.join(savepath, img_name)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300,
                    bbox_inches="tight")  # High resolution
        print(f"Saved image pair to {output_filename}")

        # Close the figure to free memory
        plt.close(fig)


def infer_video(file_name):
    video_path = os.path.join(cwd, "test", file_name)
    output_path = os.path.join(cwd, "output", file_name)

    print(f"Processing video: {file_name}...")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (frame_width, frame_height))

    # Run inference on the video
    results = model(source=video_path, stream=True)

    # Iterate over the results generator
    for result in results:
        # Process each frame
        annotated_frame = result.plot()  # Annotated frame
        # Write the annotated frame to the output video
        out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


infer_video("vid.mp4")
