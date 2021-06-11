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
    help="path to models to be evaluated. The models will be ensembled. Please provide frame_size.txt that contains input size information of the model.")
ap.add_argument("-ev", "--evaluation_result", type=str, default="eval_ens_result.txt",
    help="path to output evaluation result")
args = vars(ap.parse_args())

print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " loading images...")
val_image_paths = sorted(list(paths.list_images(args["eval_dataset"])))
val_data = []
val_labels = []

for val_imagePath in val_image_paths:
    # extract the class label from the filename, load the image and
    # resize it to be a fixed 32x32 pixels, ignoring aspect ratio
    label = val_imagePath.split(os.path.sep)[-2]
    image = cv2.imread(val_imagePath)

    # update the data and labels lists, respectively
    val_data.append(image)
    if label == 'fake':
        val_labels.append(0)
    else:
        val_labels.append(1)

# all pixel intensities to the range [0, 1]
testY = np_utils.to_categorical(val_labels, 2)

print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " loading models...")
if os.path.exists(args["model"]):
    models = []
    frame_size = []
    if not os.path.exists(args["model"]+"/frame_size.txt"):
        print(bcolors.WARNING + "[WARNING]" + bcolors.ENDC + " frame_size.txt does not exists, the input sizes are being guessed.")
    else:
        frame_dict = dict()
        f = open(args["model"]+"/frame_size.txt", "r")
        frame_lines = f.readlines()
        f.close()
        for line in frame_lines:
            model_name = line.split()[0]
            frame_w = int(line.split()[1])
            frame_h = int(line.split()[2])
            frame_dict[model_name] = (frame_w, frame_h)
    for model_path in glob.glob(args["model"]+"/*.h5"):
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

print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " evaluating models...")
predicted_class_indices = []
for face_raw in val_data:
    for (i, model) in enumerate(models):
        # extract the face ROI and then preproces it in the exact same manner as our training data
        face = cv2.resize(face_raw, frame_size[i])
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
    predicted_class_indices.append(j)
    
print(classification_report(testY.argmax(axis=1),
    predicted_class_indices, target_names=["fake", "real"], digits=6))
with open(args["evaluation_result"], 'a') as f:
    print("Last model:", file=f)
    print(classification_report(testY.argmax(axis=1),
        predicted_class_indices, target_names=["fake", "real"], digits=6), file=f)

print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + " Evaluation finished.")