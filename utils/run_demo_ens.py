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
    help="path to directory containing liveness models to ensemble")
ap.add_argument("-c", "--confidence", default=0.9,
    help="confidence threshold to detecting human face")
ap.add_argument("-f", "--frame_rate", default=30,
    help="frame rate of the output video")
ap.add_argument("-t", "--face_thresh", type=int, default=60,
    help="minimum size of face to undergo LivenessNet classification.")
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
    

for video_path in sorted(glob.glob(input_video_dir+"/*")):
    video_base_name = os.path.basename(video_path)
    video_name = video_base_name.split(".")[0]
    out_video_name = "result_ensemble_"+video_name+"_thresh_"+str(args["face_thresh"])+".mp4"
    out_video_path = out_video_dir+"/"+out_video_name
    print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " Running ensembled model on "+video_base_name)
    os.system("python3 ../demo.py -m ./ -d ../ -c "+str(confidence)+" -p ../shape_predictor_68_face_landmarks.dat -o "+out_video_path+" -v "+video_path+" -f "+str(args["frame_rate"])+" -ens True -ensp "+model_dir +" -t "+str(args["face_thresh"]))
        