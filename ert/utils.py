import cv2
import numpy as np
import xml.etree.ElementTree as et

def get_paths(dir, dataset_name):
    # pts_paths = glob.glob(dir + dataset_name + "/*.pts")
    if dataset_name == "300W":
        img_paths = glob.glob(dir + dataset_name + "/*.png")
    elif dataset_name == "lfpw1":
        img_paths = glob.glob(dir + dataset_name + "/*.png")
    elif dataset_name == "lfpw2":
        img_paths = glob.glob(dir + dataset_name + "/*.png")
    elif dataset_name == "300VW":
        img_paths = glob.glob(dir + dataset_name + "/*/annot/*.jpg")
    else:
        img_paths = glob.glob(dir + dataset_name + "/*.jpg")
    return img_paths

def read_points(pts_path):
    with open(pts_path) as file:
        landmarks = []
        for line in file:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                x, y = line.strip().split(" ")
                landmarks.append([int(float(x)), int(float(y))])
        landmarks = np.array(landmarks)
    return landmarks

def get_bbox_of_landmarks(image, landmarks, scale):
    """
    According to landmark to generate a new bigger bbox
    Args:
        image: Numpy type
        landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
        scale_value: scale bbox in (value). eg: (1.3)
    Return:
        return new bbox and need padding info
    """
    ori_h, ori_w = image.shape[:2]
    x = int(min(landmarks[:, 0]))
    y = int(min(landmarks[:, 1]))
    w = int(max(landmarks[:, 0]) - x)
    h = int(max(landmarks[:, 1]) - y)
    
    new_size = int(max(w, h) * scale)
    
    x1 = x - (new_size - w) / 2
    y1 = y - (new_size - h) / 2
    x2 = x1 + new_size
    y2 = y1 + new_size
    
    # check if need padding
    need_pad = False
    if x1 < 0:
        need_pad = True
    if y1 < 0:
        need_pad = True
    if x2 > ori_w:
        need_pad = True
    if y2 > ori_h:
        need_pad = True

    return (x1, y1, x2, y2), new_size, need_pad

def draw_marks(image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)

def capture_frames(video_path, out_path):
    """Capture frame in video"""
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count=0; success=True;

    while success:
        success, image = vidcap.read()
        (h, w) = image.shape[:2]
        center = (w/2, h/2)

        # Rotate 90 degrees
        M = cv2.getRotationMatrix2D(center, angle=90, scale=1.0)
        rotated90 = cv2.warpAffine(image, M, (w, h))

        # If using frontal camera
        image = cv2.flip(rotated90, 1)

        # save frame image
        cv2.imwrite(out_path + "frame%d.jpg" % count, image)
        if cv2.waitKey(10) == 27:
            break
        count += 1

def pts_list_to_xml(annot_list,image_path_list,face_rect_list,xml_path,\
                    verbal=False, name='Face Dataset', comment='My comments'):
    """Create custom xml file for training"""
    xmlnode_dataset = et.Element('dataset');
    xmlnode_name = et.Element('name');
    xmlnode_comment = et.Element('comment');
    xmlnode_images = et.Element('images');

    xmlnode_name.text = name
    xmlnode_comment.text = comment

    xmlnode_dataset.append(xmlnode_name)
    xmlnode_dataset.append(xmlnode_comment)
    xmlnode_dataset.append(xmlnode_images)

    for (j,pts) in enumerate(annot_list):
        if verbal:
            if j%100==0:
                print '[%d/%d]read pts and write xml'%(j,len(annot_list))

        xmlnode_image = et.Element('image');
        xmlnode_box = et.Element('box');

        # calculate face region
        [left,top,width,height] = face_rect_list[j]    

        # write image xml(face region)
        xmlnode_image.attrib['file'] = image_path_list[j]
        xmlnode_box.attrib['top'] = str(top)
        xmlnode_box.attrib['left'] = str(left)
        xmlnode_box.attrib['width'] = str(width)
        xmlnode_box.attrib['height'] = str(height)
        
        # write part xml
        for (i,pt) in enumerate(pts):
            xmlnode_part = et.Element('part');
            xmlnode_part.attrib['name'] = "%02d" % i
            xmlnode_part.attrib['x'] = str(int(pt[0]))
            xmlnode_part.attrib['y'] = str(int(pt[1]))
            xmlnode_box.append(xmlnode_part);
        xmlnode_image.append(xmlnode_box)
        xmlnode_images.append(xmlnode_image)

    # Write XML
    et.ElementTree(xmlnode_dataset).write(xml_path,xml_declaration=True