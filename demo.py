from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import dlib
from scipy.spatial import distance as dist
import glob
x = 0
from utils.colors import bcolors
import signal
import sys
from keras import backend

def relu6(x):
    return backend.relu(x, max_value=6)

def signal_handler(sig, frame):
    global out
    global video_capture
    if out is not None:
        out.release()
        video_capture.release()
    if args["video_file"] == "0" and out is not None:
        cv2.destroyAllWindows()
    print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " Interrupt Signal Received.")
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)


# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
    help="path to trained model")
ap.add_argument("-d", "--detector", type=str, required=True,
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-p", "--shape-predictor", required=True,
         help="path to facial landmark predictor")
ap.add_argument("-o", "--out_video_path", type=str, required=True,
         help="path and name of output_video, must be .mp4 type")
ap.add_argument("-v", "--video_file", type=str, default="0",
         help="path to video_file")
ap.add_argument("-f", "--frame_rate", type=int, default=30,
         help="frame rate of the input video")
ap.add_argument("-w", "--input_img_width", type=int, default=32, help="The width of the input image.")
ap.add_argument("-he", "--input_img_height", type=int, default=32, help="The height of the input image.")
ap.add_argument("-ens", "--ensemble_flag", action='store_true', help="To use ensemble of models or not. If set, ensure to provide path to all models (including model provided in -m flag) to perform ensemble")
ap.add_argument("-ensp", "--ensemble_path", type=str, default="", help="The path to directory storing all models for ensemble usage. It should also contain a file named frame_size.txt specifying the input size of each model.\
                with lines: MODEL_FILE_NAME FRAME_WIDTH FRAME_HEIGHT")
ap.add_argument("-t", "--face_thresh", type=int, default=60,
    help="minimum size of face to undergo LivenessNet classification.")
args = vars(ap.parse_args())

width =  args["input_img_width"]
height =  args["input_img_height"]


# loading face detector from the place where we stored it
print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " loading face detector")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
#Loading the caffe model 
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
#reading data from the model.
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# loading the liveness detecting module that was trained in the training python script
print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " loading the liveness detector")
if args["ensemble_flag"] == False:
    model = load_model(args["model"], custom_objects={'relu6': relu6})
else:
    if os.path.exists(args["ensemble_path"]):
        models = []
        frame_size = []
        if not os.path.exists(args["ensemble_path"]+"/frame_size.txt"):
            print(bcolors.WARNING + "[WARNING]" + bcolors.ENDC + " frame_size.txt does not exists, the input sizes are being guessed.")
        else:
            frame_dict = dict()
            f = open(args["ensemble_path"]+"/frame_size.txt", "r")
            frame_lines = f.readlines()
            f.close()
            for line in frame_lines:
                model_name = line.split()[0]
                frame_w = int(line.split()[1])
                frame_h = int(line.split()[2])
                frame_dict[model_name] = (frame_w, frame_h)
        for model_path in glob.glob(args["ensemble_path"]+"*.h5"):
            if os.path.basename(model_path) in frame_dict:
                frame_size.append(frame_dict[os.path.basename(model_path)])
                print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " loaded "+os.path.basename(model_path)+" with frame size = ", frame_dict[os.path.basename(model_path)])
            elif "xception" in model_path:
                frame_size.append((160, 160))
            elif "ResNet50" in model_path:
                frame_size.append((224, 224))
            elif "densenet" in model_path:
                frame_size.append((224, 224))
            model = load_model(model_path, custom_objects={'relu6': relu6})
            models.append(model)
    else:
        print(bcolors.FAIL + "[Error]" + bcolors.ENDC + " Path to ensemble models are INVALID!")

#determining the facial points that are plotted by dlib
FULL_POINTS = list(range(0, 68))  
FACE_POINTS = list(range(17, 68))  
   
JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))  
   
EYE_AR_THRESH = 0.30 
EYE_AR_CONSEC_FRAMES = 2  

#initializing the parameters
COUNTER_LEFT = 0  
TOTAL_LEFT = 0  
   
COUNTER_RIGHT = 0  
TOTAL_RIGHT = 0 
#defining a function for calculating ear and then comparing with the confidence parametrs

def eye_aspect_ratio(eye):
    
    A = dist.euclidean(eye[1], eye[5])  
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])  
    ear = (A + B) / (2.0 * C)  
    return ear 

#loading the predictor for predicting
detector = dlib.get_frontal_face_detector()  

#accessing the shape predictor
predictor = dlib.shape_predictor(args["shape_predictor"])
#starting the stream
if args["video_file"] == "0":
    print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " Using camera as input.")
    video_capture = cv2.VideoCapture(0)  
    if not video_capture.isOpened():
        print(bcolors.FAIL + "[Error]" + bcolors.ENDC + " Camera cannot be opened!")
        exit(1)
    video_capture.set(3, 640)
    video_capture.set(4, 480)
else:
    video_capture = cv2.VideoCapture(args["video_file"])
    
# Define video writer parameters
output_name = args["out_video_path"]
frame_rate = args["frame_rate"]
out = None
    
print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " start processing video")
#looping over frames
start = time.time()
k = 0
while True:
    #checkpoint 1
    start_inner = time.time()
    ret, frame = video_capture.read()
    if ret:
        k+=1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        rects = detector(gray, 0)
        frame = imutils.resize(frame, width=600)
        if out == None:
            (frame_height, frame_width) = frame.shape[:2]
            out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc('X','V','I','D'), frame_rate, (frame_width,frame_height))
        for rect in rects:
            
            x = rect.left()  
            y = rect.top()  
            x1 = rect.right()  
            y1 = rect.bottom()
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])  
            left_eye = landmarks[LEFT_EYE_POINTS]  
            right_eye = landmarks[RIGHT_EYE_POINTS]  
            left_eye_hull = cv2.convexHull(left_eye)  
            right_eye_hull = cv2.convexHull(right_eye)  
            ear_left = eye_aspect_ratio(left_eye)  
            ear_right = eye_aspect_ratio(right_eye)
        
            #calculating blink wheneer the ear value drops down below the threshold
    
            if ear_left < EYE_AR_THRESH:
                COUNTER_LEFT += 1
            else:
                if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                    TOTAL_LEFT += 1  
                    # print("Left eye winked") 
                    COUNTER_LEFT = 0
                    
            if ear_right < EYE_AR_THRESH:  
                COUNTER_RIGHT += 1  
            else:
                if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES: 
                    TOTAL_RIGHT += 1  
                    # print("Right eye winked")  
                    COUNTER_RIGHT = 0

            x = TOTAL_LEFT + TOTAL_RIGHT

        (h, w) = frame.shape[:2]
        temp = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        net.setInput(temp)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            #staisfying the union need of veryfying through ROI and blink detection.  
            if confidence > args["confidence"] and x>10:
                #detect a bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                if endX - startX < args["face_thresh"] or endY - startY < args["face_thresh"]:
                    continue


                #pass the model to determine the liveness
                if args["ensemble_flag"] == False:
                    # extract the face ROI and then preproces it in the exact same manner as our training data
                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (width, height))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)
                    raw_pred = model.predict(face)
                    preds = raw_pred[0]
                    j = np.argmax(preds)
                else:
                    for (i, model) in enumerate(models):
                        # extract the face ROI and then preproces it in the exact same manner as our training data
                        face = frame[startY:endY, startX:endX]
                        face = cv2.resize(face, frame_size[i])
                        face = face.astype("float") / 255.0
                        face = img_to_array(face)
                        face = np.expand_dims(face, axis=0)
                        raw_pred = model.predict(face)
                        if i == 0:
                            preds = raw_pred[0]
                        else:
                            preds += raw_pred[0]
                    j = np.argmax(preds)
                    preds /= len(models)
                if j == 0:
                    label = "fake"
                    label = "{}: {:.4f}".format(label, preds[j])
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                else:
                    label = "real"
                    label = "{}: {:.4f}".format(label, preds[j])
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 255, 0), 2)

                # tag with the label
                
        out.write(frame)
        stop_inner = time.time()
        if args["video_file"] == "0":
            total_inner = stop_inner - start_inner
            print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " FPS = {:.2f}".format(1/total_inner), end = "\r")
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("")
                break
            
    else:
        break
if out is not None:
    out.release()
else:
    print(bcolors.FAIL + "[Error]" + bcolors.ENDC + " Input video cannot be read!")
end = time.time()
total_time = (end-start)
video_capture.release()
print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " FPS = ", k/total_time)




