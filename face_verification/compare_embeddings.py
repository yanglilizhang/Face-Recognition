from typing import Union
from torchvision.transforms import functional as F
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
import os
import cv2
import pickle
from PIL import Image
from ultralytics import YOLO
print("YOLO imported successfully!")
print("Torch imported successfully!")

device = "cuda" if torch.cuda.is_available() else "cpu"
cwd = os.getcwd()

infer_dir = os.path.join(cwd, 'data', 'infer_imgs')
# print(infer_dir)
embd_dir = os.path.join(cwd, 'data', 'encodings', 'encodings.pkl')
# print(embd_dir)

yolo = YOLO('face_detection.pt')
print("YOLO loaded successfully!")

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Resnet loaded successfully!")


def load_encodings(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def get_closest(vector, database):
    names = list(database.keys())
    embeddings = np.array(list(database.values()))
    distances = np.linalg.norm(embeddings - vector, axis=1)
    sorted_idx = np.argsort(distances)
    if distances[sorted_idx[0]] > 0.6:
        name = "Unknown"
    else:
        min_idx = sorted_idx[0]
        name = names[min_idx]
    return name


def process_frame(img_pil: Union[Image.Image, np.ndarray], db):
    model_result = {}
    results = yolo.predict(img_pil, iou=0.8, conf=0.3,
                           imgsz=640, device=device)
    img_tensor = F.to_tensor(img_pil).unsqueeze(0)
    for box in results[0].boxes.xyxy:
        box = box.int()
        top, left, height, width = box[1], box[0], box[3]-box[1], box[2]-box[0]
        box = box.cpu().numpy()
        cropped = F.crop(img_tensor, top, left, height, width)
        cropped = F.resize(cropped, size=(160, 160))
        embedding = resnet(cropped.to(device))
        embedding = embedding.cpu().detach().numpy().squeeze()
        name = get_closest(embedding, db)
        model_result |= {
            name: {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}}
    return model_result

# Convert RGB to BGR (OpenCV uses BGR format)


def image_resize(image: np.ndarray, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def annotate_image(img_pil: Union[Image.Image, np.ndarray], annotations):
    if isinstance(img_pil, Image.Image):
        image_np = np.array(img_pil)
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image = img_pil  # Already in BGR format when in numpy array

    # Get the image dimensions
    img_height, img_width = image.shape[:2]

    for name, coords in annotations.items():
        # Scale coordinates based on image size
        x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']

        # Calculate dynamic properties
        # Thickness of the rectangle
        rect_thickness = max(2, int(0.005 * img_width))
        # Font size proportional to image width
        font_scale = max(0.5, 0.002 * img_width)
        font_thickness = max(1, int(0.005 * img_width)
                             )  # Thickness of the text

        # Draw the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (141, 33, 173), rect_thickness)

        # Determine the text size
        text_size = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_width, text_height = text_size

        # Calculate the position of the text background and text
        text_x = x1
        text_y = y1 - 10  # 10 pixels above the rectangle
        background_x1 = text_x
        # Ensure it doesn't go out of bounds
        background_y1 = max(0, text_y - text_height - 5)
        background_x2 = text_x + text_width + 10
        background_y2 = text_y

        # Draw the black rectangle for the text background
        cv2.rectangle(image, (background_x1, background_y1),
                      (background_x2, background_y2), (0, 0, 0), -1)

        # Put the name in white on top of the black background
        cv2.putText(image, name, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness)

    return image


database = load_encodings(embd_dir)
img_name = "a.jpg"
img_path = os.path.join(infer_dir, img_name)

print(f'Processing {img_name}...')
output = process_frame(Image.open(img_path), database)
img = annotate_image(Image.open(img_path), output)

cv2.imshow("Output", img)
cv2.waitKey(0)
