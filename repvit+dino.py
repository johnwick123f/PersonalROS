## SHOULD RESTART RUNTIME BEFORE EXECUTING CODE
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from repvit_sam import sam_model_registry, SamPredictor
from torchvision.ops import box_convert
grounding_model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")## LOADS GROUNDING MODEL
sam_checkpoint = "../weights/repvit_sam.pt"
sam = sam_model_registry["repvit"](checkpoint=sam_checkpoint).to("cuda:0").eval()## LOADS SEGMENT ANYTHING MODEL
predictor = SamPredictor(sam)
### DOWN IS FUNCTION FOR SHOWING MASK
def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
### DOWN IS FUNCTION FOR SHOWING BOX
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
  
## DOWN IS FUNCTION FOR SEGMENTATION
def segment(boxes):
  input_box = np.array(boxes)
  masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
  )
  return masks

### BELOW IS FUNCTION FOR 0 SHOT OBJECT DETECTION
def grounding(text, image):
  image_source, image = load_image(image)

  boxes, logits, phrases = predict(
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=0.35,
    text_threshold=0.35
  )
  return boxes, logits, phrases, image_source

### BELOW IS FUNCTION FOR THE WHOLE PIPELINE
### TEXT SHOULD BE A LIST AND IMAGE SHOULD BE PATH
def sam_dino(text, image):
  boxes, logits, phrases, image_source = grounding(text, image)
  masks = []
  cimage = cv2.imread(image)
  cimage = cv2.cvtColor(cimage, cv2.COLOR_BGR2RGB)
  source_h, source_w, _ = cimage.shape
  predictor.set_image(cimage)
  boxes3 = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
  xyxy = box_convert(boxes=boxes3, in_fmt="cxcywh", out_fmt="xyxy").numpy()
  for box in xyxy:
    box = np.array(box)
    mask = segment(box)### MIGHT HAVE TO CHANGE!!!!
    masks.append(mask)
  return masks, xyxy, logits, phrases, image_source



## FEW COMMENTS
# - takes 0.6 seconds for grounding dino with a few objects
# - rep vit takes 0.1 seconds so nice
# - not exactly great for hard objects
# - Biggest open grounding model t is the best quality
  
