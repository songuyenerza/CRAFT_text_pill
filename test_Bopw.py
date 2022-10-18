from itertools import count
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def test_net(net, image, text_threshold, link_threshold, low_text, cuda=True, poly = False, refine_net=None):
    t0 = time.time()
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 640, interpolation=cv2.INTER_LINEAR, mag_ratio=1)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda== True:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys, av_aggle = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    return boxes, polys, ret_score_text, av_aggle

if __name__ == '__main__':
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load("CP/craft_mlt_25k.pth", map_location='cuda')))
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
    net.eval()
    data_folder = "D:/ANlab/detect_text_pill/box_output/"
    count = 0
    for path in os.listdir(data_folder):
        count += 1
        image = loadImage(data_folder + path)
        bboxes, polys, score_text, av_aggle = test_net(net, image, text_threshold = 0.05, link_threshold = 0.3,
            low_text = 0.3, cuda  = True, poly= False, refine_net= None)
        box_img = bboxes
        img_ori = image.copy()
        if len(box_img) != 0:
            for i in range(len(box_img)):
                box = np.int0(box_img[i])
                im2 = cv2.drawContours(img_ori,[box] ,0,(100,100,100),2)
        else:
            im2 = img_ori
        (h, w) = im2.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, av_aggle, 1.0)
        im2 = cv2.warpAffine(im2, M, (w, h), flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite("D:/ANlab/detect_text_pill/CRAFT-pytorch/check/" + str(count) + ".jpg" , im2)