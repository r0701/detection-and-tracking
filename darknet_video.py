from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--h', type=str, default='short_test.mp4',
                    help='use --cfg --weights --data --source')
parser.add_argument('--cfg', type=str,
                    default='short_test.mp4', help='path to cfg')
parser.add_argument('--source', type=str,
                    default='short_test.mp4', help='path to video')
parser.add_argument('--weight', type=str,
                    default='short_test.mp4', help='path to weight')
parser.add_argument('--data', type=str,
                    default='short_test.mp4', help='path to data file')
opt = parser.parse_args()


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    middle = 0
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        middle = ((pt1[0]+pt2[0])/2, (pt1[1]+pt2[1])/2)
    return img, middle


netMain = None
metaMain = None
altNames = None
# inizialize multi tracker
multiTracker = cv2.MultiTracker_create()


def YOLO():

    global metaMain, netMain, altNames, multiTracker
    configPath = opt.cfg
    weightPath = opt.weight
    metaPath = opt.data
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(opt.source)
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)

    first_detection = 0
    frame_number = 0
    detection_frame = 0
    magnitude = 0
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        if magnitude is 0 or magnitude > 20:
            detections = darknet.detect_image(
                netMain, metaMain, darknet_image, thresh=0.25)

        if first_detection is 1:
            success, boxes = multiTracker.update(frame_resized)

            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame_resized, p1, p2, (0, 0, 255), 2, 1)

        # if something is detected
        if len(detections) is not 0:
            detection_frame += 1
        if len(detections) is not 0 and first_detection is 0:
            print(detections[0])
            x, y, w, h = detections[0][2][0],\
                detections[0][2][1],\
                detections[0][2][2],\
                detections[0][2][3]
            print(f'tracker box is: {x} {y} {w} {h}')
            multiTracker.add(cv2.TrackerCSRT_create(),
                             frame_resized, (x-w/2, y-h/2, w, h))
            #multiTracker.add(cv2.TrackerMIL_create(), image, (x,y,w+5,h+5))
            first_detection = 1

        separate = False

        image, det_mid = cvDrawBoxes(detections, frame_resized)
        try:
            if det_mid is not 0:
                track_mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                print(f'middle of detection is {det_mid}')
                print(f'middle of tracker is {track_mid}')
                magnitude = int(
                    math.sqrt((det_mid[1]-track_mid[1])**2+(det_mid[0]-track_mid[0])**2))
                print(magnitude)
                separate = True
        except:
            pass
        print(1/(time.time()-prev_time))
        #image = cv2.resize(image,(1920, 1080))
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        cv2.imshow('Demo', frame_resized)
        cv2.waitKey(3)
        print(f'[FRAME] detected frames {detection_frame}/{frame_number}')
        if magnitude > 20:
            multiTracker = cv2.MultiTracker_create()
            first_detection = 0
            separate = False

        frame_number += 1
    cap.release()
    out.release()


if __name__ == "__main__":
    YOLO()
