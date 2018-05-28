# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
from utils import get_bbox_of_landmarks

# You can choose using hog detector every frame or just 1st frame & tracking
USING_HOG = True
shape_predictor_path = "Your shape model path"

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)

vs = VideoStream().start()
time.sleep(2.0)

# loop over the frames from the video stream
frame_idx = 0
shape_rects = []
first_shapes = []
while True:
    frame = vs.read()

    frame = imutils.resize(frame, width=750, height=750)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    start = cv2.getTickCount()

    if USING_HOG:
        rects = detector(gray, 0)
        # loop over the face detections
        for rect in rects:
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
            shape = shape_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image except eyes and mouth indexes
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

    else:
        ######################################
        ## Step1: Detect faces in 1st frame ##
        ######################################
        if frame_idx == 0:
            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            # loop over the face detections
            for rect in rects:
                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
                first_shape = shape_predictor(gray, rect)
                first_shape = face_utils.shape_to_np(first_shape)

                for (x, y) in first_shape:
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

                # Add predicted landmarks with hog detector
                shape_rects.append(rect)
                first_shapes.append(first_shape)
        ############################################
        ## Step2: Get bounding box from landmarks ##
        ############################################
        elif frame_idx == 1:
            for shape in first_shapes:
                shape_bbox, _, _ = get_bbox_of_landmarks(gray, shape, 1.2)
                shape_rect = dlib.rectangle(shape_bbox[0], shape_bbox[1], shape_bbox[2], shape_bbox[3])
                cv2.rectangle(frame, (shape_rect.left(), shape_rect.top()), (shape_rect.right(), shape_rect.bottom()), (0, 255, 0), 2)
                shape = shape_predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                # Add shape rect for tracking
                shape_rects.append(shape_rect)
                
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        else:
            # print(len(shape_rects))
            rect = shape_rects[frame_idx - 1]
            cv2.rectangle(frame, (shape_rect.left(), shape_rect.top()), (shape_rect.right(), shape_rect.bottom()), (0, 255, 0), 2)
            shape = shape_predictor(gray, rect) 
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

            shape_bbox, _, _ = get_bbox_of_landmarks(gray, shape, 1.2)
            shape_rect = dlib.rectangle(shape_bbox[0], shape_bbox[1], shape_bbox[2], shape_bbox[3]) 
            shape_rects.append(shape_rect)

    fps_time = (cv2.getTickCount() - start)/cv2.getTickFrequency()
    cv2.putText(frame, '%.1ffps'%(1/fps_time) , (frame.shape[1]-65,15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))

    # show the frame
    cv2.imshow("6.3mb", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break