import argparse
from pathlib import Path
from tqdm import tqdm

from reinforcement import *
from utils import *
from dataset import *
from nets.backbone import Backbone

import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/yolov7_backbone_weights.pth', help='initial weights path')
    parser.add_argument('--hyp', type=str, default='data/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--dataset-path', type=str, default='Pascal2007/VOCtrainval_06-Nov-2007/VOCdevkit')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--phi', type=str, default='l', help='type of yolov7')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--img-size', type=int, default=[256, 256], help='image sizes')
    
    opt = parser.parse_args()

    # Result directary
    save_dir = increment_path(Path(opt.project) / opt.name)  # increment run

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # Create dqn model & freeze backbone weight
    obs_dim, action_dim = 8*8*1024 + 6*4, 6
    backbone = Backbone(transition_channels=32, block_channels=32, n=4, phi='l', pretrained=True).to(device)
    agent = DQNAgent(obs_dim, action_dim, opt.batch_size, device)
    for param in backbone.parameters():
        param.requires_grad = False

    # Trainloader & Testloader
    trainDataset = yoloDataset(opt.dataset_path, opt.img_size, dataset_property="aeroplane_train")
    valDataset = yoloDataset(opt.dataset_path, opt.img_size, dataset_property="aeroplane_val")
    trainDataloader = DataLoader(trainDataset, batch_size=1, shuffle=True)
    valDataloader = DataLoader(valDataset, batch_size=1, shuffle=True)

    update_cnt = 0
    target_update = 10

    #---------------------------------------#
    #   Start training
    #---------------------------------------#
    for epoch in range(opt.epochs):
        pbar = tqdm(enumerate(trainDataloader), total=len(trainDataloader))
        agent.train()
        for i, data in pbar:
            '''
                labels = [nums of the img labels]
                boxs = [nums of the img labels, 4]  4->(xmin, ymin, xmax, ymax)
                imgShape = (640, 640)
            '''
            img, target = data
            labels, boxs = target["labels"], target["boxes"]
            imgShape = img.shape[2], img.shape[3]

            # the iou part
            region_mask = np.ones(imgShape)
            gt_masks = genBoxFromAnnotation(boxs[0], imgShape)

            # choose the max bouding box
            iou = findMaxBox(gt_masks, region_mask)
            
            # the initial part
            region_image = img
            size_mask = imgShape
            offset = (0, 0)
            history_vector = torch.zeros((4, 6), device=device)
            state = get_state(region_image, history_vector, backbone, device)
            done = False

            for step in range(opt.steps):
                '''
                    Select action, the author force terminal action if case actual IoU is higher than 0.5
                    action = 0, 1, 2, 3, 4, 5
                    0 -> top left corner
                    1 -> bottom left corner
                    2 -> top right corner
                    3 -> bottom right corner
                    4 -> center
                    5 -> break
                '''
                if iou > 0.5:
                    action = 5
                else:
                    action = agent.select_action(state)

                # Perform the action and observe new state
                if action == 5:
                    next_state = None
                    reward = get_reward_trigger(iou)
                    done = True
                else:
                    offset, region_image, size_mask, region_mask = get_crop_image_and_mask(imgShape, offset,
                                                                    region_image, size_mask, action)
                    # update history vector and get next state
                    history_vector = update_history_vector(history_vector, action).to(device)
                    next_state = get_state(region_image, history_vector, backbone, device)
                    
                    # find the max bounding box in the region image
                    new_iou = findMaxBox(gt_masks, region_mask)
                    reward = get_reward_movement(iou, new_iou)
                    iou = new_iou
                    
                # Store the transition in memory
                agent.memory.push(state, action, next_state, reward)
                
                # Move to the next state
                state = next_state
                
                # Perform one step of the optimization (on the target network)
                loss = agent.update_model()
                update_cnt += 1

                # Print
                pbar.set_description((f"Epoch [{epoch+1}/{opt.epochs}]"))
                pbar.set_postfix(loss=loss)

                if update_cnt % target_update == 0:
                    agent.target_hard_update()

                if done:
                    break

            # end batch -------------------------------------------------------------
        # end epoch ---------------------------------------------------------

        # Validation
        agent.eval()
        pbar = tqdm(enumerate(valDataloader), total=len(trainDataloader))
        for i, data in pbar:
            img, target = data
            labels, boxs = target["labels"], target["boxes"]
            imgShape = img.shape[2], img.shape[3]

            # the iou part
            batchIou = []
            region_mask = np.ones(imgShape)
            gt_masks = genBoxFromAnnotation(boxs[0], imgShape)

            # choose the max bouding box
            iou = findMaxBox(gt_masks, region_mask)

            # the initial part
            region_image = img
            size_mask = imgShape
            offset = (0, 0)
            history_vector = torch.zeros((4, 6), device=device)
            state = get_state(region_image, history_vector, backbone, device)
            done = False

            for step in range(opt.steps):
                # Select action, the author force terminal action if case actual IoU is higher than 0.5
                if iou > 0.5:
                    action = 5
                else:
                    action = agent.select_action(state)

                # Perform the action and observe new state
                if action == 5:
                    next_state = None
                    reward = get_reward_trigger(iou)
                    done = True
                else:
                    offset, region_image, size_mask, region_mask = get_crop_image_and_mask(imgShape, offset,
                                                                    region_image, size_mask, action)
                    # Get next state
                    history_vector = update_history_vector(history_vector, action).to(device)
                    next_state = get_state(region_image, history_vector, backbone, device)
                    
                    # find the max bounding box in the region image
                    new_iou = findMaxBox(gt_masks, region_mask)
                    reward = get_reward_movement(iou, new_iou)
                    iou = new_iou

                # Move to the next state
                state = next_state

            batchIou.append(iou)
        
        print("IOU : ", np.mean(batchIou))
    
    # End training ---------------------------------------------------------
    print("End Training\n")
    
    # Save model
    torch.save(agent.dqn, save_dir)