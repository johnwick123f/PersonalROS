import cv2
import numpy as np
### BELOW FUNCTION IS FOR EXTRACTING MASK FROM ANOTHER IMAGE
def extract_mask(mask_img, surface_normal_img):
  mask = cv2.imread(mask_img, cv2.IMREAD_GRAYSCALE)
  image = cv2.imread(surface_normal_img)

# Convert mask to binary
  _, binary_mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)

# Invert the binary mask
  binary_mask = cv2.bitwise_not(binary_mask)

# Apply mask to the image
  extracted_object = cv2.bitwise_and(image, image, mask=binary_mask)
  return extracted_object

### BELOW FUNCTION IS FOR FINDING CENTROID FROM A MASK
def find_centroid(mask):
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

## BELOW FUNCTION IS FOR FINDING DIFFERENT COLORS FROM SURFACE NORMALS - HELPS FIND TOP, BOTTOM, LEFT SIDE, RIGHT SIDE!
def extract_and_find_centroid(image, color):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define thresholds for pink, blue, and green
    if color == 'pink':
        lower = np.array([140, 50, 50])
        upper = np.array([180, 255, 255])
    elif color == 'blue':
        lower = np.array([90, 40, 50])
        upper = np.array([120, 255, 255])
    elif color == 'green':
        lower = np.array([35, 50, 50])
        upper = np.array([90, 255, 255])
    # Create a mask for the specified color
    mask = cv2.inRange(hsv, lower, upper)
    return mask
### IMPORTANT DETAILS
#- ITS STILL IN PROGRESSA LOT

