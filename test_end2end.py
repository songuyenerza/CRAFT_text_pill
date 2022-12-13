from detect_yolov5 import *
from test_Bopw_label import *
from operator import itemgetter, attrgetter
import numpy

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
    # print("dst = ", dst)
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # print("warped.shape = ", warped.shape)
    # return the warped image
    return  warped

if __name__ == '__main__':
    #load_model_craft == net
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load("CP/craft_ic15_20k.pth", map_location='cuda'))) #weights_craft
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
    net.eval()

    #load_model_yolov5
    weight = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/detect_point/cp/best_yolov5_121222_V2.pt" #weights_yolov5
    model, device = load_model(weights=weight)
    print("GPU Memory_____: %s" % getMemoryUsage())

    #foler_save_output
    folder_output = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/TV_L-20221208T075619Z-001/check_output2/"

    #load_path_img_input
    data_folder = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/panasonic-20221209T082601Z-001/panasonic/"
    with open( data_folder +  'paths.txt','r') as f:
        IMAGE_PATH_DB = [line.strip('\n') for line in f.readlines()]

    for path in IMAGE_PATH_DB:

        img_ori = cv2.imread(data_folder + path)

        center = img_ori.shape
        h, w = center[0], center[1]
        start = timeit.default_timer()
        box_img, box_image_no = detect_box(model, device, img_ori,imgsz=[800,800],conf_thres=0.65, iou_thres = 0.3)
        # print("box_image_no", box_image_no)
        img_output = folder_output + path
        stop = timeit.default_timer()

        if len(box_image_no) == 4:
            box_crop = []
            for box in box_image_no:
                b = [box[0], box[1]]
                box_crop.append(b)
            box_crop = sorted_box(box_crop)

            # box_crop = np.array(box_crop)
            # box_crop = np.int0(box_crop)
            # img_final = cv2.drawContours(img_ori.copy(),[box_crop],0,(0,0,256),2)
            img_final = fourPointTransform(img_ori.copy(), box_crop )
          

        else:
            image = loadImage(data_folder + path)
            bboxes, polys, score_text, av_aggle, text_thres_img = test_net(net, image, text_threshold = 0.2, link_threshold = 0.1,
                low_text = 0.1, cuda  = True, poly= False, refine_net= None)
            box_img = bboxes
            text_thres_img = cv2.resize(text_thres_img, (image.shape[1], image.shape[0]))
            # cv2.imwrite("check.png", text_thres_img)
            if len(box_img) != 0:
                for i in range(len(box_img)):
                    box = np.int0(box_img[i])
                    # im2 = cv2.drawContours(img_ori,[box] ,0,(0,0,256),2)
                    poly = polys[i]
                    poly = np.array(poly).astype(np.int32).reshape((-1))
                    poly = poly.reshape(-1, 2)
                    # cv2.polylines(img_ori, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 255), thickness=2)

                box_mers, results, box_max = getBoxPanels(text_thres_img)
              
                # for b, r in zip(results , box_mers):
                    # im2 = cv2.rectangle(im2, (b[0],b[1]), (b[2],b[3]), (0,256,255), 2)
                if len(box_max) > 0:
                    box_max = sorted_box(box_max)
                    box_max = np.array(box_max)
                    box_max = np.int0(box_max)
                    img_final = fourPointTransform(img_ori.copy(), box_max )

                    # img_final = cv2.drawContours(image.copy(),[box_max],0,(0,0,256),2)
            
                else:
                    img_final = image.copy()
            else: 
                img_final = image

        cv2.imwrite(img_output, img_final)

        print('Time: ', stop - start)  
        print("path_img", path)
