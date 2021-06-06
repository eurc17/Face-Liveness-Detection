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

