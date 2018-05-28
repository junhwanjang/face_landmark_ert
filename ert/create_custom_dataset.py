import dlib
import glob
import numpy as np
import cv2
import utils

IMG_PATH = "./dataset/"
DATASET_NAME = ["300W", "ibug", "helen", "lfpw1", "lfpw2", "300VW"]
USING_HOG = False

# Initialize Hog face detector
face_detector = dlib.get_frontal_face_detector()

image_path_list = []
rect_list = []
pts_list = []
for dataset in DATASET_NAME:
    images_list = get_paths(IMG_PATH, dataset)

    for idx, image_path in enumerate(images_list):
        # Read image
        image = cv2.imread(image_path)
        pts = utils.read_points(image_path.split(".")[0] + ".pts")
        is_valid_points = len(pts) == 68
        if is_valid_points:
            if USING_HOG:
                dets = face_detector(image)
                for k, d in enumerate(dets):
                    if d is not None:
                        top = d.top(); left = d.left(); right = d.right(); bottom = d.bottom();
                        height = bottom - top; width = right - left;
                        
                        image_path_list.append(image_path)
                        rect_list.append([left, top, height, width])
                        pts_list.append(pts)

            else:
                (left, top, right, bottom), new_size, need_pad = get_bbox_of_landmarks(image, pts, scale=1.2)
                height = bottom - top
                width = right - left
                
                if need_pad:
                    pass
                
                else:
                    image_path_list.append(image_path)
                    rect_list.append([left, top, height, width])
                    pts_list.append(pts)
        
        if idx % 10000 == 0:
            print(idx," Processing!")
        
    print(dataset, " Done!")

# Save to train info xml
utils.pts_list_to_xml(pts_list, image_path_list, rect_list, "./train_with_all.xml")