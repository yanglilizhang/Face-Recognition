from torchvision.transforms import functional as F
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
import os
import pickle
from PIL import Image
import shutil
from ultralytics import YOLO
print("YOLO imported successfully!")
print("Torch imported successfully!")

device = "cuda" if torch.cuda.is_available() else "cpu"
cwd = os.getcwd()

img_dir = os.path.join(cwd, 'data', 'input_imgs')
# print(img_dir)
temp_dir = os.path.join(cwd, 'data', 'temp')
# print(temp_dir)
embd_dir = os.path.join(cwd, 'data', 'encodings', 'encodings.pkl')
# print(embd_dir)

model = YOLO('face_detection.pt')
print("YOLO loaded successfully!")

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Resnet loaded successfully!")

# Iterate through new images to create embeddings
# these images then gets moved to temp folder
data = {}
imgs = os.listdir(img_dir)
for img_name in imgs:
    img_path = os.path.join(img_dir, img_name)
    img_pil = Image.open(img_path)
    print(f'Processing {img_name}...')
    results = model.predict(img_pil, iou=0.8, conf=0.3,
                            imgsz=640, device=device)

    img_tensor = F.to_tensor(img_pil).unsqueeze(0)
    for box in results[0].boxes.xyxy:
        box = box.int()
        top, left, height, width = box[1], box[0], box[3]-box[1], box[2]-box[0]
        cropped = F.crop(img_tensor, top, left, height, width)
        cropped = F.resize(cropped, size=(160, 160))
        embedding = resnet(cropped.to(device))
        embedding = embedding.cpu().detach().numpy().squeeze()

        name = img_name.split('.')[0]
        data |= {name: embedding}
        print(f'{name} embedding added to data')

    shutil.move(img_path, os.path.join(temp_dir, img_name))


def update_encoding(new_data, file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            data.update(new_data)
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
    else:
        with open(file_path, "wb") as file:
            pickle.dump(new_data, file)


update_encoding(data, embd_dir)
