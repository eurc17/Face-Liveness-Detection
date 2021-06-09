import os
import glob
import argparse

from numpy.core.multiarray import concatenate
from colors import bcolors

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_video_dir", required=True,
    help="path to directory containing input test videos")
ap.add_argument("-o", "--output_video_dir", required=True,
    help="path to directory to store output result videos")
ap.add_argument("-m", "--model_dir", required=True,
    help="path to directory containing liveness models to test")
ap.add_argument("-c", "--confidence", default=0.9,
    help="confidence threshold to detecting human face")
ap.add_argument("-f", "--frame_rate", default=30,
    help="frame rate of the output video")
args = vars(ap.parse_args())

input_video_dir = args["input_video_dir"]
out_video_dir = args["output_video_dir"]
model_dir = args["model_dir"]
confidence = float(args["confidence"])
if confidence > 1 or confidence < 0:
    confidence = 0.9
    print(bcolors.WARNING + "[WARNING]" + bcolors.ENDC + " Confidence out of range! Reset to 0.9.")

if not os.path.isdir(input_video_dir):
    print(bcolors.FAIL + "[Error]" + bcolors.ENDC + " Input video dir does not exists!")
    exit(1)
if not os.path.isdir(model_dir):
    print(bcolors.FAIL + "[Error]" + bcolors.ENDC + " Model dir does not exists!")
    exit(1)
if not os.path.exists(out_video_dir):
    os.makedirs(out_video_dir)
    
models_path = []
frame_size = []

if not os.path.exists(model_dir+"/frame_size.txt"):
    print(bcolors.WARNING + "[WARNING]" + bcolors.ENDC + " frame_size.txt does not exists, the input sizes are being guessed.")
else:
    frame_dict = dict()
    f = open(model_dir+"/frame_size.txt", "r")
    frame_lines = f.readlines()
    f.close()
    for line in frame_lines:
        model_name = line.split()[0]
        frame_w = int(line.split()[1])
        frame_h = int(line.split()[2])
        frame_dict[model_name] = (frame_w, frame_h)
for model_path in glob.glob(model_dir+"*.h5"):
    if os.path.basename(model_path) in frame_dict:
        frame_size.append(frame_dict[os.path.basename(model_path)])
        print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " loaded "+os.path.basename(model_path)+" with frame size = ", frame_dict[os.path.basename(model_path)])
    elif "xception" in model_path:
        frame_size.append((160, 160))
    elif "ResNet50" in model_path:
        frame_size.append((224, 224))
    elif "densenet" in model_path:
        frame_size.append((224, 224))
    models_path.append(model_path)

for video_path in sorted(glob.glob(input_video_dir+"/*")):
    video_base_name = os.path.basename(video_path)
    video_name = video_base_name.split(".")[0]
    for i, model_path in enumerate(models_path):
        model_name = os.path.basename(model_path).split(".")[0]
        out_video_name = "result_"+model_name+"_"+video_name+".mp4"
        out_video_path = out_video_dir+"/"+out_video_name
        (model_w , model_h)= frame_size[i]
        print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " Running "+model_name+" on "+video_base_name)
        os.system("python3 ../demo.py -m "+model_path+" -d ../ -c "+str(confidence)+" -p ../shape_predictor_68_face_landmarks.dat -o "+out_video_path+" -v "+video_path+" -f "+str(args["frame_rate"])+" -w "+str(model_w)+" -he "+str(model_h))
        