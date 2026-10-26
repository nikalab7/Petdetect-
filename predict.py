"""CLI demo alternative: python predict.py photo.jpg"""
import argparse

import torch
from PIL import Image

from core import load_checkpoint, predict_image

p = argparse.ArgumentParser()
p.add_argument('image')
a = p.parse_args()
model, meta = load_checkpoint('artifacts/best.pt', torch.device('cpu'))
with Image.open(a.image) as image:
    for name, score in predict_image(model, image, meta['image_size'], torch.device('cpu'), meta['class_names']):
        print(f'{name}: {score:.2%}')
