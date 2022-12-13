from itertools import count
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import argparse
from refinenet import RefineNet

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image, ImageOps

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

def convert_box(box_imgs):
    box_squas = []
    for box_img in box_imgs:
        box_squa1= []
        box_squa2 = []
        for box in box_img:
            box_squa1.append(box[0])
            box_squa2.append(box[1])
        box_squa = [min(box_squa1), min(box_squa2), max(box_squa1), max(box_squa2)]               
        box_squas.append(box_squa)
    return box_squas

def getBoxPanels(mask):
    scale = mask.shape[0]/256
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    # mask = cv2.dilate(mask,kernel,iterations = 1)
    # mask = cv2.erode(mask,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15*scale*scale :
            continue
        rect = cv2.minAreaRect(c)
        x, y, w, h = cv2.boundingRect(c)
        # boxes.append((x, y ,x +w , y + h))
        results.append([x, y ,x +w , y + h])
        boxes.append(rect)
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # exit()	
    return boxes, results

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def compare_iou(boxes, target , thresh_iou = 0.1):
    num_iou = 0 
    for b1 in target:
        check = False
        for b2 in boxes:
            iou = bb_intersection_over_union(b1, b2)
            print("ii " , iou)
            if iou >= thresh_iou:
                check = True
                break
        if check:
            num_iou += 1
    list_result = [num_iou , len(target) , len(boxes) - num_iou ]
    return list_result


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
    kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    # kernel = np.ones((5,5),np.float32)/25
    # gray = clahe.apply(gray)
    # img = cv2.filter2D(gray,-1,kernel)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    # img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return img

def test_net(net, image, text_threshold, link_threshold, low_text, cuda=True, poly = False, refine_net=None):
    t0 = time.time()
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 800, interpolation=cv2.INTER_LINEAR, mag_ratio=5)
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

    text_thres_img = (score_text+ score_link).copy()*255
    return boxes, polys, ret_score_text, av_aggle, text_thres_img

def mer_box(boxs):
    box0 = []
    box1 = []
    box2=[]
    box3=[]

    for box in boxs:
        # print("boxx",box)
        for b in box:
            box0.append(b[0])
            box1.append(b[1])
            # box2.append(box[2][0])
            # box3.append(box[2][1])
    x0 = min(box0)
    y0 = min(box1)
    x1 = max(box0)
    y1 = max(box1)
    
    bounding_box = [int(x0) , int (y0) , int(x1), int(y1)]
    return bounding_box
# def getBoxPanels(mask):
#     a = np.where(mask != 0)
#     boxes = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
#     return boxes

def getBoxPanels(mask):
    
    # thresh = mask
    # thresh[thresh > 20] = 255
    # thresh[thresh<= 20] = 0
    # print("thres", thresh)
    w, h = mask.shape
    # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 9))

    # thresh = cv2.erode(thresh, rect_kernel, iterations = 2)
    # dilation = cv2.dilate(thresh, rect_kernel, iterations = 7)
    # kernel = np.ones((3, 3), np.uint8)

    # kernel = cv2.morphologyEx(kernel, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imwrite("mask.png", dilation)
    dilation = Image.fromarray(mask)
    dilation = ImageOps.grayscale(dilation)
    dilation = np.array(dilation)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    results = []
    area_list = []
    for c in contours:
        area = cv2.contourArea(c)
        # if area < 0.05*h*w :
        #     continue
        rect = cv2.minAreaRect(c)
        x, y, w, h = cv2.boundingRect(c)
        # boxes.append((x, y ,x +w , y + h))
        results.append([x, y ,x +w , y + h])
        area_list.append(area)
        box = cv2.boxPoints(rect)
        # box = np.int0(box)
        boxes.append(box)
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # exit()	
    if len(area_list) > 0:
        index = np.argmax(area_list)
        box_max = boxes[index]
    else:
        box_max = []
    return boxes, results, box_max

if __name__ == '__main__':
    # refine_net = None
    # refine_net = RefineNet()
    # refine_net.load_state_dict(copyStateDict(torch.load("/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/CRAFT_text_pill/CP/craft_refiner_CTW1500.pth")))
    # refine_net = refine_net.cuda()
    # refine_net = torch.nn.DataParallel(refine_net)
    # refine_net.eval()

    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load("CP/craft_ic15_20k.pth", map_location='cuda')))
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
    net.eval()
    data_folder = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/panasonic-20221209T082601Z-001/panasonic/"
    with open('/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/panasonic-20221209T082601Z-001/panasonic/paths.txt','r') as f:
        IMAGE_PATH_DB = [line.strip('\n') for line in f.readlines()]
    count = 0
    for path in IMAGE_PATH_DB:
        count += 1
        image = loadImage(data_folder + path)
        bboxes, polys, score_text, av_aggle, text_thres_img = test_net(net, image, text_threshold = 0.2, link_threshold = 0.1,
            low_text = 0.1, cuda  = True, poly= False, refine_net= None)
        box_img = bboxes
        text_thres_img = cv2.resize(text_thres_img, (image.shape[1], image.shape[0]))

        img_ori = image.copy()
        if len(box_img) != 0:
            for i in range(len(box_img)):
                box = np.int0(box_img[i])
                # im2 = cv2.drawContours(img_ori,[box] ,0,(0,0,256),2)
                im2 = img_ori
                poly = polys[i]

                poly = np.array(poly).astype(np.int32).reshape((-1))
                poly = poly.reshape(-1, 2)
                # cv2.polylines(img_ori, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 255), thickness=2)

            box_mers, results, box_max = getBoxPanels(text_thres_img)
            for b, r in zip(results , box_mers):
            # im2 = cv2.drawContours(img_ori,[box_mer] ,0,(0,256,256),2)
                im2 = cv2.rectangle(im2, (b[0],b[1]), (b[2],b[3]), (0,256,255), 2)
                # print(r)
                im2 = cv2.drawContours(im2,[r],0,(0,0,256),2)
        else:
            im2 = img_ori
        (h, w) = im2.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, av_aggle, 1.0)
        im3 = cv2.warpAffine(im2, M, (w, h), flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE)

        # cv2.imwrite("/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/TV_L-20221208T075619Z-001/check_output/" +str(count) +"_"+  str(av_aggle) + ".jpg" , im3)
        cv2.imwrite("/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/TV_L-20221208T075619Z-001/check_output2/" + path, im2)