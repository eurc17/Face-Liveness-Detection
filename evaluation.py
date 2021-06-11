from keras.models import load_model
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from imutils import paths
from utils.colors import bcolors
from sklearn.metrics import classification_report
import argparse
import cv2
import os
import glob
import numpy as np
from keras import backend

def relu6(x):
    return backend.relu(x, max_value=6)

ap = argparse.ArgumentParser()
ap.add_argument("-dv", "--eval_dataset", required=True,
    help="path to input evaluation dataset")
ap.add_argument("-m", "--model", type=str, required=True,
    help="path to model to be evaluated.")
ap.add_argument("-ev", "--evaluation_result", type=str, default="eval_ens_result.txt",
    help="path to output evaluation result")
ap.add_argument("-w", "--input_img_width", type=int, required=True, help="The width of the input image.")
ap.add_argument("-he", "--input_img_height", type=int, required=True, help="The height of the input image.")
ap.add_argument("-b", "--bs", type=int, default=8, help="Batch size")
args = vars(ap.parse_args())

print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " loading images...")
val_image_paths = sorted(list(paths.list_images(args["eval_dataset"])))
val_data = []
val_labels = []
width =  args["input_img_width"]
height =  args["input_img_height"]
BS = args["bs"]


for val_imagePath in val_image_paths:
    # extract the class label from the filename, load the image and
    # resize it to be a fixed 32x32 pixels, ignoring aspect ratio
    label = val_imagePath.split(os.path.sep)[-2]
    image = cv2.imread(val_imagePath)
    image = cv2.resize(image, (width, height))

    # update the data and labels lists, respectively
    val_data.append(image)
    if label == 'fake':
        val_labels.append(0)
    else:
        val_labels.append(1)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
testX = np.array(val_data, dtype="float") / 255.0
testY = np_utils.to_categorical(val_labels, 2)

print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " loading model...")
model = load_model(args["model"], custom_objects={'relu6': relu6})
print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " evaluating model...")
predictions = model.predict(testX, batch_size=BS)
predicted_class_indices = np.argmax(predictions, axis=1)
print(args["model"]+" :")
print(classification_report(testY.argmax(axis=1),
    predicted_class_indices, target_names=["fake", "real"], digits=6))
with open(args["evaluation_result"], 'a') as f:
    print(args["model"]+" :", file=f)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=["fake", "real"], digits=6), file=f)
print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " Evaluation finished.")