from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import os
import argparse

from torch import threshold


def extract_face(args):
    mtcnn = MTCNN(image_size=160, margin=32, keep_all=False)

    base_dir = args.input_dir
    files = os.listdir(base_dir)
    sorted_files = sorted(files)
    save_dir = args.save_dir
    
    threshold = 0.99
    for s_file in sorted_files:
        img = Image.open(base_dir+s_file)
        _boxes, _probs = mtcnn.forward(img, save_path=save_dir+s_file, return_prob=True)
        print(s_file+" is processed!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",   "--input_dir", help="Path to input frames directory", required=True)
    parser.add_argument("-o",   "--save_dir", help="Path to directory to save extracted faces", required=True)
    
    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        print("Input video does not exist!")
        exit(0)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    extract_face(args)