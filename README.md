# Face Liveness Detection+

Note: This is a forked version of the [code](https://github.com/sakethbachu/Face-Liveness-Detection).

---
## Description
A deep-learning pipeline capable of spotting fake vs legitimate faces and performing anti-face spoofing in face recognition systems. It is built with the help of Keras, Tensorflow, and OpenCV. A sample dataset is uploaded in the sample_dataset_folder.

## Method
The problem of detecting fake faces vs real/legitimate faces is treated as a binary classification task. Basically, given an input image, we’ll train a Convolutional Neural Network capable of distinguishing real faces from fake/spoofed faces. There are 4 main steps involved in the task:
 1. Build the image dataset itself.
 2. Implement a CNN capable of performing liveness detector(LivenessNet).
 3. Train the liveness detector network.
  4. Create a Python + OpenCV script capable of taking our trained liveness detector model and apply it to real-time video.

## Contents of this repository
1. sample_liveness_data : contains the sample dataset.
3. demo.py : Our demonstration script will fire up your webcam to grab frames to conduct face liveness detection in real-time if input_video path not provided. Else, it will take an input video and output the result as a video.
3. deploy.prototxt : Support file for pretrained face detector. 
7. livenessnet.py : The python file containing the model structure.
8. res10_300x300_ssd_iter_140000.caffemodel: Pretrained face detector.
9. train_liveness.py: The python script to train the model.
7. resource/ : Contains resources to show on Github page.

## Working flow

### Training

![training workflow](https://github.com/sakethbachu/liveness_detection/blob/master/sample_liveness_data/Desc%20info/workflow.png "Logo Title Text 1")

### Inference

![inference workflow](https://github.com/eurc17/Face-Liveness-Detection/blob/master/resource/demo_flow.png "Logo Title Text 1")

## Environment Setup

Tested with the following python package under python 3.5.2:

```
Keras==2.1.5
matplotlib==2.1.2
opencv-python==4.4.0.42
scikit-learn==0.19.1
tensorflow==1.6.0
imutils==0.5.4
scipy==1.0.0
dlib==19.22.0
h5py==2.10.0
```

## Training

1. Prepare the dataset training and validation with structure like the following:

```
sample_liveness_data_training/
├── fake
│   ├── 0.png
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── real
    ├── 0.png
    ├── 1.png
    ├── 199.png
    ├── 2.jpg
    └── ...
```

2. Train with the following command:

```bash
python3 train_liveness.py -d [PATH_TO_TRAINING_DATASET] -dv [PATH_TO_VAL_DATASET] --model [PATH_TO_SAVE_MODEL]
```

Optional Arguments:

```
  -h, --help            show this help message and exit
  -d TRAIN_DATASET, --train_dataset TRAIN_DATASET
                        path to input training dataset
  -dv VAL_DATASET, --val_dataset VAL_DATASET
                        path to input validation dataset
  -m MODEL, --model MODEL
                        path to trained model. If path exists, the existing
                        model is loaded and will resuming training.
  -p PLOT, --plot PLOT  path to output loss/accuracy plot
  -ev EVALUATION_RESULT, --evaluation_result EVALUATION_RESULT
                        path to output evaluation result
  -r LR, --lr LR        Initial Learning Rate
  -b BS, --bs BS        Batch size
  -e EPOCHS, --epochs EPOCHS
                        Number of Training Epochs
  -w INPUT_IMG_WIDTH, --input_img_width INPUT_IMG_WIDTH
                        The width of the input image.
  -he INPUT_IMG_HEIGHT, --input_img_height INPUT_IMG_HEIGHT
                        The height of the input image.
  -ckpt CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                        path to save intermediate model checkpoints.
```





## Evaluation

1. Prepare the trained model

2. Prepare the evaluation dataset with the structure like the following:

   ```
   sample_liveness_data_val/
   ├── fake
   │   ├── 0.png
   │   ├── 1.png
   │   ├── 2.png
   │   └── ...
   └── real
       ├── 0.png
       ├── 1.png
       ├── 199.png
       ├── 2.jpg
       └── ...
   ```

3. Evaluate single model with the following command:

   ```
   python3 evaluation.py -dv [PATH_TO_EVAL_DATASET_DIR] -m [PATH_TO_MODEL] -ev [PATH_TO_SAVE_EVAL_RESULT_TXT] -w [INPUT_IMG_WIDTH] -he [INPUT_IMG_HEIGHT] -b [BATCH_SIZE]

4. Evaluate ensembled models:

   a.  Prepare a folder containing models to be evaluated, with a `frame_size.txt` to define the input image size of each model. The structure of the folder should be like the following:

   ```
   ./ensemble_models/
   ├── densenet_base.h5
   ├── frame_size.txt
   ├── inceptionv3_last.h5
   ├── mobilenet_best_acc.h5
   ├── ResNet50.h5
   └── xception_base.h5
   ```

   The `frame_size.txt` should contain information with the format: [MODEL_NAME] [INPUT_IMG_WIDTH] [INPUT_IMG_HEIGHT], separated by a space. Example `frame_size.txt`:

   ```
   ResNet50.h5 224 224
   xception_base.h5 160 160
   densenet_base.h5 224 224
   mobilenet_best_acc.h5 224 224
   inceptionv3_last.h5 160 160
   ```

   b. Evaluate with the following command:

   ```
   python3 evaluation_ens.py -dv [PATH_TO_EVAL_DATASET_DIR] -m [PATH_TO_ENS_MODEL_DIR] -ev [PATH_TO_SAVE_EVAL_RESULT_TXT]
   ```

   

## Inference

1. Download dlib shape_detector from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2).

2. Decompress the downloaded file with the following command

   ```bash
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```

3. Run the following command:

   ```bash
   python3 demo.py -m [PATH_TO_LIVENESS_NET_MODEL] -c [CONFIDENCE_THRESHOLD] -p shape_predictor_68_face_landmarks.dat -d [PATH_TO_FOLDER_CONTAINING_deploy.prototxt] -o [OUTPUT_VIDEO_PATH] -v [INPUT_VIDEO_PATH] -f [FRAME_RATE]
   ```

   Optional Arguments:

   ```
     -h, --help            show this help message and exit
     -m MODEL, --model MODEL
                           path to trained model
     -d DETECTOR, --detector DETECTOR
                           path to OpenCV's deep learning face detector
     -c CONFIDENCE, --confidence CONFIDENCE
                           minimum probability to filter weak detections
     -p SHAPE_PREDICTOR, --shape-predictor SHAPE_PREDICTOR
                           path to facial landmark predictor
     -o OUT_VIDEO_PATH, --out_video_path OUT_VIDEO_PATH
                           path and name of output_video, must be .mp4 type
     -v VIDEO_FILE, --video_file VIDEO_FILE
                           path to input video_file. If not provided, the webcam will be used as input source.
     -f FRAME_RATE, --frame_rate FRAME_RATE
                           frame rate of the input video
     -w INPUT_IMG_WIDTH, --input_img_width INPUT_IMG_WIDTH
                           The width of the input image.
     -he INPUT_IMG_HEIGHT, --input_img_height INPUT_IMG_HEIGHT
                           The height of the input image.
     -ens ENSEMBLE_FLAG, --ensemble_flag ENSEMBLE_FLAG
                           To use ensemble of models or not. If set to True,
                           ensure to provide path to all models (including model
                           provided in -m flag) to perform ensemble
     -ensp ENSEMBLE_PATH, --ensemble_path ENSEMBLE_PATH
                           The path to directory storing all models for ensemble
                           usage. It should also contain a file named
                           frame_size.txt specifying the input size of each
                           model. with lines: MODEL_FILE_NAME FRAME_WIDTH
                           FRAME_HEIGHT
   ```

4. There are scripts for running demo on several videos in a folder. Please checkout the README file in [utils](https://github.com/eurc17/Face-Liveness-Detection/tree/master/utils).

## Further work

1. Gathering data having a larger set of ethnicity and different types of fake/spoofed photos.
2. Update code to use Tensorflow 2.0 as backend.

## Reference

[Shape Detector Github Link](https://github.com/davisking/dlib-models)

