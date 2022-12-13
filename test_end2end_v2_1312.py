from tkinter.messagebox import RETRY
from detect_yolov5 import *
from test_Bopw_label import *
from operator import itemgetter, attrgetter
import numpy
import numpy as np
from model_unet import *
import torch
import cv2
from PIL import Image, ImageOps
import timeit

def sorted_box(box_list):

    #box_1 = topleft, box3 = topright, box4 = b_right, box2 = b_left
    box_list = sorted(box_list, key = lambda box_list: box_list[0])
    box_12 = box_list[:2]
    box_34 = box_list[2:]

    box_12 = sorted(box_12, key = lambda box_12: box_12[1])
    box1 = box_12[0]
    box2 = box_12[1]

    box_34 = sorted(box_34, key = lambda box_34: box_34[1])
    box3 = box_34[0]
    box4 = box_34[1]

    return [box1, box3, box4, box2]

def get_box_crop(box_list):
    box_x = []
    box_y = []
    for box in box_list:
        box_x.append(box[0])
        box_y.append(box[1])
    return [min(box_x), min(box_y), max(box_x), max(box_y)]

def transform_unet(image_np):
        ToPILImage = transforms.ToPILImage()
        image = ToPILImage(image_np)
        
        image = TF.to_tensor(image)

        return image

def fourPointTransform(image, rect):
    # errCode = ErrorCode.SUCCESS
    rect = ( [int(rect[0][0]), int(rect[0][1])], [int(rect[1][0]), int(rect[1][1])], [int(rect[2][0]), int(rect[2][1])], [int(rect[3][0]), int(rect[3][1])]  )
    (tl, tr, br, bl) = rect

    widthA = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # print("heightAB", heightA, heightB)
    maxHeight = max(int(heightA), int(heightB))

    rect = numpy.float32(rect)
    dst = numpy.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # return the warped image
    return  warped

def predict_unet(img_path, model_unet, thresh = 0.95):
    img = imread(img_path)[:,:,:3]
    image = img.copy()
    h0, w0 = image.shape[1],image.shape[0]

    img = resize(img, (256, 256), mode='constant', preserve_range=True)
    img = img.astype(np.uint8)
    img = transform_unet(img)
    img = img.to(device)
    img = img.unsqueeze_(0)
    mask_pred = model_unet(img.float())
    mask_pred = mask_pred.cpu()
    mask_pred = (mask_pred > thresh)
    pred = TF.to_pil_image(mask_pred.float().squeeze(0))
    mask = np.array(pred)
    kernel = np.ones((7,7),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 3)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.convert('L')
    mask_pil = mask_pil.resize((h0,w0), resample=Image.BILINEAR)
    dilation = ImageOps.grayscale(mask_pil)
    dilation = np.array(dilation)
    return dilation

if __name__ == '__main__':
    #load_model_unet == net
    device = torch.device('cuda')
    model_unet = AttU_Net()
    model_unet = model_unet.float()
    model_unet = model_unet.to(device)
    model_unet.load_state_dict(torch.load('/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/detect_point/cp/Segment_unet_cpbest_1212.pth', map_location=torch.device('cuda')))
    model_unet.eval()

    #load_model_yolov5
    weight = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/detect_point/cp/best_yolov5_121222_V3.pt" #weights_yolov5
    model, device = load_model(weights=weight)
    print("GPU Memory_____: %s" % getMemoryUsage())

    #foler_save_output
    folder_output = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/TV_L-20221208T075619Z-001/check_croped_131222/"

    #load_path_img_input
    data_folder = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/panasonic-20221209T082601Z-001/panasonic/"
    with open( data_folder +  'paths.txt','r') as f:
        IMAGE_PATH_DB = [line.strip('\n') for line in f.readlines()]

    for path in IMAGE_PATH_DB:

        img_ori = cv2.imread(data_folder + path)

        center = img_ori.shape
        h0, w0 = center[0], center[1]
        start = timeit.default_timer()
        box_img, box_image_no = detect_box(model, device, img_ori,imgsz=[800,800],conf_thres=0.8, iou_thres = 0.3)
        # print("box_image_no", box_image_no)
        img_output = folder_output + path
        stop = timeit.default_timer()

        if len(box_image_no) == 4:
            box_crop = []
            for box in box_image_no:
                b = [box[0], box[1]]
                box_crop.append(b)
            box_crop = sorted_box(box_crop)
            if 0.5 <box_crop[0][1] / box_crop[1][1] < 2:
                box_crop = np.array(box_crop)
                box_crop = np.int0(box_crop)
                img_final = cv2.drawContours(img_ori.copy(),[box_crop],0,(0,0,256),2)
                # img_final = fourPointTransform(img_ori.copy(), box_crop )

        else:
            img_path = data_folder + path
            mask_unet = predict_unet(img_path, model_unet)
            box_mers, results, box_max = getBoxPanels(mask_unet)
            if len(box_max) > 0:
                box_max = sorted_box(box_max)
                box_max = np.array(box_max)
                box_max = np.int0(box_max)
                img_final = cv2.drawContours(img_ori.copy(),[box_max],0,(0,0,256),2)

                # img_final = fourPointTransform(img_ori.copy(), box_max)

        cv2.imwrite(img_output, img_final)

        print('Time: ', stop - start)  
        # print("path_img", path)
