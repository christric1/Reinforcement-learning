from dataset import yoloDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from utils import bbox_to_rect
import torch

from reinforcement import *
from utils import *
from dataset import *
from nets.backbone import Backbone


# TestDataset & TestLoader
testDataset = yoloDataset("Pascal2007/VOCtest_06-Nov-2007/VOCdevkit", [256, 256], dataset_property="aeroplane_train")
testDataloader = DataLoader(testDataset, batch_size=1, shuffle=True)

# Parameter
model_path = 'runs\train\exp\model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
steps = 10

# Create model
obs_dim, action_dim = 8*8*1024 + 6*4, 6
backbone = Backbone(transition_channels=32, block_channels=32, n=4, phi='l', pretrained=True).to(device)
dqn = torch.load(model_path)
for param in backbone.parameters():
    param.requires_grad = False


#---------------------------------------#
#   Start training
#---------------------------------------#
for i, data in enumerate(testDataloader):
    img, target = data
    labels, boxs = target["labels"], target["boxes"]
    imgShape = img.shape[2], img.shape[3]

    # the iou part
    region_mask = np.ones(imgShape)
    gt_masks = genBoxFromAnnotation(boxs[0], imgShape)

    # choose the max bouding box
    iou = findMaxBox(gt_masks, region_mask)

    region_image = img
    history_vector = torch.zeros((4, 6), device=device)
    state = get_state(region_image, history_vector, backbone, device)
    done = False

    for step in range(steps):
        if iou > 0.5:
            action = 5
        else:
            qval = dqn(state)
            _, predicted = torch.max(qval.data, 1)
            action = predicted[0]

        if action == 5:
            next_state = None
            done = True
        else:
            offset, region_image, size_mask, region_mask = get_crop_image_and_mask(imgShape, offset,
                                                            region_image, size_mask, action)
            # Get next state
            history_vector = update_history_vector(history_vector, action).to(device)
            next_state = get_state(region_image, history_vector, backbone, device)
            
            # find the max bounding box in the region image
            new_iou = findMaxBox(gt_masks, region_mask)
            iou = new_iou

        # Move to the next state
        state = next_state

        if done:
            break
    
    # Get box
    nonzero = torch.nonzero(region_mask)
    top_left = nonzero[0]
    bottom_right = nonzero[-1]
    xmin, ymin = top_left.tolist()
    xmax, ymax = bottom_right.tolist()

    # Draw
    img = img.squeeze(dim=0).numpy()
    img = np.transpose(img, (1, 2, 0))
    fig = plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect((xmin, ymin, xmax, ymax), 'blue'))
    fig.axes.add_patch(bbox_to_rect(boxs[0], 'green'))
    plt.show()

    input("按 Enter 鍵繼續...")