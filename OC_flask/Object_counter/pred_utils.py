import os
import torch
import math
import cv2
import argparse
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.optim as optim
from utils import MincountLoss, PerturbationLoss
from tqdm import tqdm
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import visualize_output_and_save, select_exemplar_rois
from PIL import Image

MAPS = ['map3','map4']
Scales = [0.9, 1.1]
MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):
    denormalized = tensor.clone()
    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)
    return denormalized


def scale_and_clip(val, scale_factor, min_val, max_val):
    "Helper function to scale a value and clip it within range"

    new_val = int(round(val*scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val


def visualize_output_and_save(input_, output, boxes, save_path, figsize=(20, 12), dots=None):
    """
        dots: Nx2 numpy array for the ground truth locations of the dot annotation
            if dots is None, this information is not available
    """

    # get the total count
    pred_cnt = output.sum().item()
    boxes = boxes.squeeze(0)

    boxes2 = []
    for i in range(0, boxes.shape[0]):
        y1, x1, y2, x2 = int(boxes[i, 1].item()), int(boxes[i, 2].item()), int(boxes[i, 3].item()), int(
            boxes[i, 4].item())
        roi_cnt = output[0,0,y1:y2, x1:x2].sum().item()
        boxes2.append([y1, x1, y2, x2, roi_cnt])

    img1 = format_for_plotting(denormalize(input_))
    output = format_for_plotting(output)

    fig = plt.figure(figsize=figsize)

    # display the input image
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.imshow(img1)

    for bbox in boxes2:
        y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)

    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c='red', edgecolors='blue')
        # ax.scatter(dots[:,0], dots[:,1], c='black', marker='+')
        ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    else:
        ax.set_title("Input image")

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title("Overlaid result, predicted count: {:.2f}".format(pred_cnt))

    img2 = 0.2989*img1[:,:,0] + 0.5870*img1[:,:,1] + 0.1140*img1[:,:,2]
    ax.imshow(img2, cmap='gray')
    ax.imshow(output, cmap=plt.cm.viridis, alpha=0.5)


    # display the density map
    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(output)
    # plt.colorbar()

    ax = fig.add_subplot(2, 2, 4)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ret_fig = ax.imshow(output)
    for bbox in boxes2:
        y1, x1, y2, x2, roi_cnt = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.text(x1, y1, '{:.2f}'.format(roi_cnt), backgroundcolor='y')

    fig.colorbar(ret_fig, ax=ax)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close()


def format_for_plotting(tensor):
    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()



def predict(path ,input_image , bbox_file, need_dm=False , need_dm_details=False):
    
    output_dir = os.path.join(path,"results")
    model_path = os.path.join(path,"data","model","BoxNet.pth")
    adapt=False
    learning_rate=1e-7
    gradient_steps =100
    weight_mincount=1e-9
    weight_perturbation=1e-4

    if not torch.cuda.is_available() :
        use_gpu = False
        print("===> Using CPU mode.")
    else:
        use_gpu = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    resnet50_conv = Resnet50FPN()
    regressor = CountRegressor(6, pool='mean')

    if use_gpu:
        resnet50_conv.cuda()
        regressor.cuda()
        regressor.load_state_dict(torch.load(model_path))
    else:
        regressor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    resnet50_conv.eval()
    regressor.eval()

    image_name = os.path.basename(input_image)
    image_name = os.path.splitext(image_name)[0]

    if bbox_file is None: # if no bounding box file is given, prompt the user for a set of bounding boxes
        out_bbox_file = "{}/{}_box.txt".format(output_dir, image_name)
        fout = open(out_bbox_file, "w")

        im = cv2.imread(input_image)
        cv2.imshow('image', im)
        rects = select_exemplar_rois(im)

        rects1 = list()
        for rect in rects:
            y1, x1, y2, x2 = rect
            rects1.append([y1, x1, y2, x2])
            fout.write("{} {} {} {}\n".format(y1, x1, y2, x2))

        fout.close()
        cv2.destroyWindow("Image")
        print("selected bounding boxes are saved to {}".format(out_bbox_file))
    else:
        with open(bbox_file, "r") as fin:
            lines = fin.readlines()

        rects1 = list()
        for line in lines:
            data = line.split()
            y1 = int(data[0])
            x1 = int(data[1])
            y2 = int(data[2])
            x2 = int(data[3])
            rects1.append([y1, x1, y2, x2])

    print("Bounding boxes: ", end="")
    print(rects1)

    image = Image.open(input_image)
    image.load()
    sample = {'image': image, 'lines_boxes': rects1}
    sample = Transform(sample)
    image, boxes = sample['image'], sample['boxes']


    if use_gpu:
        image = image.cuda()
        boxes = boxes.cuda()

    with torch.no_grad():
        features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

    if not adapt:
        with torch.no_grad(): output = regressor(features)
    else:
        features.required_grad = True
        #adapted_regressor = copy.deepcopy(regressor)
        adapted_regressor = regressor
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=learning_rate)

        pbar = tqdm(range(gradient_steps))
        for step in pbar:
            optimizer.zero_grad()
            output = adapted_regressor(features)
            lCount = weight_mincount * MincountLoss(output, boxes, use_gpu=use_gpu)
            lPerturbation =weight_perturbation * PerturbationLoss(output, boxes, sigma=8, use_gpu=use_gpu)
            Loss = lCount + lPerturbation
            # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
            # So Perform gradient descent only for non zero cases
            if torch.is_tensor(Loss):
                Loss.backward()
                optimizer.step()

            pbar.set_description('Adaptation step: {:<3}, loss: {}, predicted-count: {:6.1f}'.format(step, Loss.item(), output.sum().item()))

        features.required_grad = False
        output = adapted_regressor(features)


    print('===> The predicted count is: {:6.2f}'.format(output.sum().item()))

    if need_dm:
        rslt_file = "{}/{}_out.png".format(output_dir, image_name)
        visualize_output_and_save(image.detach().cpu(), output.detach().cpu(), boxes.cpu(), rslt_file )
        print("===> Visualized output is saved to {}".format(rslt_file))

    else:
        return round(output.sum().item())
