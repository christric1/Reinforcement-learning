import argparse
import yaml
from tqdm import tqdm

from reinforcement import *
from util import *
from dataset import *
from nets.backbone import Backbone

import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/yolov7_backbone_weights.pth', help='initial weights path')
    parser.add_argument('--hyp', type=str, default='data/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--phi', type=str, default='l', help='type of yolov7')
    parser.add_argument('--img-size', type=int, default=[640, 640], help='image sizes')

    opt = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dqn model & freeze backbone weight
    obs_dim, action_dim = 20*20*1024 + 6*4, 6
    backbone = Backbone(transition_channels=32, block_channels=32, n=4, phi='l', pretrained=True).to(device)
    agent = DQNAgent(obs_dim, action_dim) 
    for param in backbone.parameters():
        param.requires_grad = False

    # Trainloader & Testloader
    trainDataset = yoloDataset("Pascal2007", opt.img_size, dataset_property="aeroplane_train")
    valDataset = yoloDataset("Pascal2007", opt.img_size, dataset_property="aeroplane_val")
    trainDataloader = DataLoader(trainDataset, batch_size=1, shuffle=True)
    valDataloader = DataLoader(valDataset, batch_size=1, shuffle=True)

    update_cnt = 0
    target_update = 10

    #---------------------------------------#
    #   Start training
    #---------------------------------------#
    for epoch in range(opt.epochs):
        pbar = tqdm(enumerate(trainDataloader), total=len(trainDataloader))
        for i, data in pbar:
            '''
                labels = [nums of the img labels]
                boxs = [nums of the img labels, 4]  4->(xmin, ymin, xmax, ymax)
                imgShape = (640, 640)
            '''
            img, target = data
            labels, boxs = target["labels"].squeeze(dim=0), target["boxes"].squeeze(dim=0)
            image = img.squeeze(dim=0)
            imgShape = image.shape[1], image.shape[2]

            # the iou part
            region_mask = np.ones(imgShape)
            gt_masks = genBoxFromAnnotation(boxs, imgShape)

            # choose the max bouding box
            iou = findMaxBox(gt_masks, region_mask)
            
            # the initial part
            region_image = image
            size_mask = imgShape
            offset = (0, 0)
            history_vector = torch.zeros((4, 6))
            state = get_state(region_image, history_vector, backbone)
            done = False

            for step in range(opt.steps):
                # Select action, the author force terminal action if case actual IoU is higher than 0.5
                if iou > 0.5:
                    action = 6
                else:
                    action = agent.select_action(state)

                # Perform the action and observe new state
                if action == 6:
                    next_state = None
                    reward = get_reward_trigger(iou)
                    done = True
                else:
                    offset, region_image, size_mask, region_mask = get_crop_image_and_mask(imgShape, offset,
                                                                    region_image, size_mask, action)
                    # update history vector and get next state
                    history_vector = update_history_vector(history_vector, action)
                    next_state = get_state(region_image, history_vector, backbone)
                    
                    # find the max bounding box in the region image
                    new_iou = findMaxBox(gt_masks, region_mask)
                    reward = get_reward_movement(iou, new_iou)
                    iou = new_iou
                    
                # Store the transition in memory
                agent.memory.push(state, action-1, next_state, reward)
                
                # Move to the next state
                state = next_state
                
                # Perform one step of the optimization (on the target network)
                loss = agent.update_model()
                update_cnt += 1

                # Print
                pbar.set_description((f"Epoch [{epoch+1}/{opt.epochs}]"))
                pbar.set_postfix(loss = loss.item())

                if update_cnt % target_update == 0:
                    agent.target_hard_update()

                if done:
                    break

            # end batch -------------------------------------------------------------
        # end epoch ---------------------------------------------------------

        # Validation


    print("End Training\n")