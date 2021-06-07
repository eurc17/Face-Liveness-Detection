# USAGE
# python train_liveness.py --dataset dataset --model liveness.model --le le.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
#from pyimagesearch.livenessnet import LivenessNet
from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--train_dataset", required=True,
	help="path to input training dataset")
ap.add_argument("-dv", "--val_dataset", required=True,
	help="path to input validation dataset")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model. If path exists, the existing model is loaded and will resuming training.")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-ev", "--evaluation_result", type=str, default="result.txt",
	help="path to output evaluation result")
ap.add_argument("-r", "--lr", type=float, default=1e-4, help="Initial Learning Rate")
ap.add_argument("-b", "--bs", type=int, default=8, help="Batch size")
ap.add_argument("-e", "--epochs", type=int, default=50, help="Number of Training Epochs")
ap.add_argument("-w", "--input_img_width", type=int, default=160, help="The width of the input image.")
ap.add_argument("-he", "--input_img_height", type=int, default=160, help="The height of the input image.")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = args["lr"]
BS = args["bs"]
EPOCHS = args["epochs"]
width =  args["input_img_width"]
height =  args["input_img_height"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
val_image_paths = sorted(list(paths.list_images(args["val_dataset"])))
val_data = []
val_labels = []

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


# construct the training image generator for data augmentation
print("[INFO] Constructing ImageDataGenerator")
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest", rescale=1/255.)

no_aug = ImageDataGenerator(rescale=1/255.)

# initialize the optimizer and model
print("[INFO] compiling model...")
if not os.path.exists(args["model"]):
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model = LivenessNet.build(width=width, height=height, depth=3,
		classes=2)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])
else:
    model = load_model(args["model"])
    print("[INFO] Loaded an existing model:", args["model"])

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
train_generator = aug.flow_from_directory(args["train_dataset"], target_size=(width, height), class_mode= "categorical", batch_size=BS)
val_generator = no_aug.flow_from_directory(args["val_dataset"], target_size=(width, height), class_mode= "categorical", batch_size=BS)

H = model.fit_generator(train_generator,
	validation_data=val_generator, steps_per_epoch=train_generator.n // train_generator.batch_size, validation_steps=val_generator.n//val_generator.batch_size,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
predicted_class_indices = np.argmax(predictions, axis=1)

print(classification_report(testY.argmax(axis=1),
	predicted_class_indices, target_names=["fake", "real"]))
with open(args["evaluation_result"], 'w') as f:
	print(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=["fake", "real"]), file=f)
# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])