FROM tensorflow/tensorflow:1.6.0-devel-py3

# get basic updates and upgrades
RUN apt-get update -y
# install basic packages
RUN apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y
RUN apt-get install ffmpeg libsm6 libxext6 -y


#install python packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install keras==2.1.5
RUN python3 -m pip install --upgrade setuptools
RUN python3 -m pip install matplotlib==2.1.2 opencv-python==4.4.0.42 scikit-learn==0.19.1 imutils==0.5.4 scipy==1.0.0 h5py==2.10.0
RUN python3 -m pip install dlib==19.17.0
