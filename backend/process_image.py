import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to apply color or pattern to the segmented wall
def apply_pattern_or_color(image, mask, pattern_image_path=None, color_code=None):
    mask_3d = np.stack([mask] * 3, axis=-1)

    if pattern_image_path and os.path.exists(pattern_image_path):  # Apply pattern only if pattern image is provided and exists
        pattern_image = cv2.imread(pattern_image_path)
        if pattern_image is None:
            logger.error("Pattern image could not be loaded.")
            return image
        
        pattern_image = cv2.resize(pattern_image, (image.shape[1], image.shape[0]))
        image = np.where(mask_3d, pattern_image, image)
    elif color_code:  # Apply color only if color code is provided
        try:
            # Convert hex color code to RGB tuple, handling the '#' prefix
            color_code = color_code.lstrip('#')
            if len(color_code) != 6:
                raise ValueError("Invalid hex color code length.")
            
            # Convert the hex color code to an RGB tuple
            color = tuple(int(color_code[i:i+2], 16) for i in (0, 2, 4))
            # Convert RGB to BGR for OpenCV
            color_bgr = (color[2], color[1], color[0])
            logger.info(f"Applying color: {color_bgr} from hex code: {color_code}")

        except ValueError as e:
            logger.error(f"Invalid color code: {color_code}. Error: {str(e)}")
            return image
        
        color_image = np.full_like(image, color_bgr)
        image = np.where(mask_3d, color_image, image)
    else:
        logger.warning("No pattern image or color code provided, returning original image.")
        
    return image

# Main function to run the inference
def run_inference(room_image_path, pattern_image_path=None, color_code=None):
    try:
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h"
        CHECKPOINT_PATH = "models/deeplabv3/sam_vit_h.pth"

        logger.info("Loading SAM model...")
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)

        logger.info("Reading input image...")
        image = cv2.imread(room_image_path)
        if image is None:
            logger.error("Room image could not be loaded.")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logger.info("Generating masks...")
        mask_generator = SamAutomaticMaskGenerator(sam)
        result = mask_generator.generate(image_rgb)
        
        largest_mask = None
        max_area = 0
        for item in result:
            area = item['area']
            if area > max_area:
                max_area = area
                largest_mask = item['segmentation']
        
        if largest_mask is not None:
            logger.info("Applying pattern or color...")
            processed_image = apply_pattern_or_color(image, largest_mask, pattern_image_path, color_code)
        else:
            logger.warning("No mask detected, returning original image.")
            processed_image = image

        processed_image_path = 'processed_image.jpg'
        cv2.imwrite(processed_image_path, processed_image)
        return processed_image_path
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None
