from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-base-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-base-hf").to("cuda:0").eval()

def depth(image, resize=True):
  inputs = image_processor(images=image, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)
  
