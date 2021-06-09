# Utilities

The utilities are for FinTech Project in NTU.

## mtcnn.py

Prerequisite: [facenet-pytorch](https://github.com/timesler/facenet-pytorch)

Description: Takes a folder of images and performs MTCNN face detection on each image and saves the cropped faces in each image to another directory.

Usage:

```bash
python3 mtcnn.py [-h] -i INPUT_DIR -o SAVE_DIR
```

Optional Arguments:

```
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Path to input frames directory
  -o SAVE_DIR, --save_dir SAVE_DIR
                        Path to directory to save extracted faces
```

## transform_dataset.py

Notes:

* This is for **CASIA-CBSR** dataset only.
* `train_label.txt` should be in the same directory as the `train` folder.

Description:  Takes the `train_label.txt` and copy fake images to a folder and copy real images to another folder.

Usage:

```bash
python3 transform_dataset.py [-h] -i LABEL_TXT -f FAKE_DIR -r REAL_DIR
```

Optional Arguments:

```
  -h, --help            show this help message and exit
  -i LABEL_TXT, --label_txt LABEL_TXT
                        Path to train_label.txt
  -f FAKE_DIR, --fake_dir FAKE_DIR
                        Path to directory to store fake images
  -r REAL_DIR, --real_dir REAL_DIR
                        Path to directory to store real images
```

## run_demo.py

Description: Run demo.py with several different backbone liveness models on a folder of videos. 

Note: In the folder containing models, please provide a file named `frame_size.txt` containing the model file name and the width and height of the input image.

Example:

File structure:

```
./models_to_test/
├── densenet_base.h5
├── frame_size.txt
├── ResNet50.h5
└── xception_base.h5
```

frame_size.txt: (format: MODEL_NAME	INPUT_WIDTH	INPUT_HEIGHT)

```
ResNet50.h5 224 224
xception_base.h5 160 160
densenet_base.h5 224 224
```

Usage:

```bash
python3 run_demo.py -i [PATH_TO_INPUT_VIDEO_DIR] -o [PATH_TO_DIR_SAVING_OUT_VIDEO] -m [PATH_TO_MODEL_DIR]
```

Optional Arguments:

```
  -h, --help            show this help message and exit
  -i INPUT_VIDEO_DIR, --input_video_dir INPUT_VIDEO_DIR
                        path to directory containing input test videos
  -o OUTPUT_VIDEO_DIR, --output_video_dir OUTPUT_VIDEO_DIR
                        path to directory to store output result videos
  -m MODEL_DIR, --model_dir MODEL_DIR
                        path to directory containing liveness models to test
  -c CONFIDENCE, --confidence CONFIDENCE
                        confidence threshold to detecting human face
  -f FRAME_RATE, --frame_rate FRAME_RATE
                        frame rate of the output video
```

## run_demo_ens.py

Description: Run demo.py with several different backbone liveness models ensembled on a folder of videos. 

Note: In the folder containing models to be ensembled, please provide a file named `frame_size.txt` containing the model file name and the width and height of the input image.

Example:

File structure:

```
./models_to_ensemble/
├── densenet_base.h5
├── frame_size.txt
├── ResNet50.h5
└── xception_base.h5
```

frame_size.txt: (format: MODEL_NAME	INPUT_WIDTH	INPUT_HEIGHT)

```
ResNet50.h5 224 224
xception_base.h5 160 160
densenet_base.h5 224 224
```

Usage:

```bash
python3 run_demo_ens.py -i [PATH_TO_INPUT_VIDEO_DIR] -o [PATH_TO_DIR_SAVING_OUT_VIDEO] -m [PATH_TO_MODEL_ENS_DIR]
```

Optional Arguments:

```
  -h, --help            show this help message and exit
  -i INPUT_VIDEO_DIR, --input_video_dir INPUT_VIDEO_DIR
                        path to directory containing input test videos
  -o OUTPUT_VIDEO_DIR, --output_video_dir OUTPUT_VIDEO_DIR
                        path to directory to store output result videos
  -m MODEL_DIR, --model_dir MODEL_DIR
                        path to directory containing liveness models to
                        ensemble
  -c CONFIDENCE, --confidence CONFIDENCE
                        confidence threshold to detecting human face
  -f FRAME_RATE, --frame_rate FRAME_RATE
                        frame rate of the output video
```

