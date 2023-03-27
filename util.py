import numpy as np
import cv2
import matplotlib.pyplot as plt


scale_subregion = float(3) / 4
scale_mask = float(1)/(scale_subregion*4)

def bbox_to_rect(bbox, color):
    '''
        bbox : (xmin, ymin, xmax, ymax)
        color: blue, red, etc
    '''
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]),      
        width=bbox[2]-bbox[0],      
        height=bbox[3]-bbox[1],     
        fill=False,             
        edgecolor=color,
        linewidth=2
    )

def calculate_iou(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j)/float(i))
    return iou

def genBoxFromAnnotation(boxs, imgShape):
    '''
        bbox : (xmin, ymin, xmax, ymax)
        imgShape : (640, 640)
    '''
    length_annotation = boxs.shape[0]
    masks = np.zeros([imgShape[0], imgShape[1], length_annotation])
    for i in range(0, length_annotation):
        masks[int(boxs[i, 0]):int(boxs[i, 2]), int(boxs[i, 1]):int(boxs[i, 3]), i] = 1
    return masks

def findMaxBox(gt_masks, region_mask):
    _, _, n = gt_masks.shape
    max_iou = 0.0
    for k in range(n):
        gt_mask = gt_masks[:,:,k]
        iou = calculate_iou(region_mask, gt_mask)
        if max_iou < iou:
            max_iou = iou
    return max_iou

def get_crop_image_and_mask(original_shape, offset, region_image, size_mask, action):
    """
    Args:
        original_shape: shape of original image (H x W)
        offset: the current image's left-top coordinate base on the original image
        region_image: the image to be cropped
        size_mask: the size of region_image
        action: the action choose by agent. can be 1,2,3,4,5.
        
    Returns:
        offset: the cropped image's left-top coordinate base on original image
        region_image: the cropped image
        size_mask: the size of the cropped image
        region_mask: the masked image which mask cropped region and has same size with original image
    
    """
    region_mask = np.zeros(original_shape) # mask at original image 
    size_mask = (int(size_mask[0] * scale_subregion), int(size_mask[1] * scale_subregion)) # the size of croped image

    if action == 0:
        offset_aux = (0, 0)
    elif action == 1:
        offset_aux = (0, int(size_mask[1] * scale_mask))
        offset = (offset[0], offset[1] + int(size_mask[1] * scale_mask))
    elif action == 2:
        offset_aux = (int(size_mask[0] * scale_mask), 0)
        offset = (offset[0] + int(size_mask[0] * scale_mask), offset[1])
    elif action == 3:
        offset_aux = (int(size_mask[0] * scale_mask), 
                      int(size_mask[1] * scale_mask))
        offset = (offset[0] + int(size_mask[0] * scale_mask),
                  offset[1] + int(size_mask[1] * scale_mask))
    elif action == 4:
        offset_aux = (int(size_mask[0] * scale_mask / 2),
                      int(size_mask[0] * scale_mask / 2))
        offset = (offset[0] + int(size_mask[0] * scale_mask / 2),
                  offset[1] + int(size_mask[0] * scale_mask / 2))
        
    region_image = region_image[:, :, offset_aux[0]:offset_aux[0] + size_mask[0],
                   offset_aux[1]:offset_aux[1] + size_mask[1]]
    region_mask[offset[0]:offset[0] + size_mask[0], offset[1]:offset[1] + size_mask[1]] = 1

    return offset, region_image, size_mask, region_mask