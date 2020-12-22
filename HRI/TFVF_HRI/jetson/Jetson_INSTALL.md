# Xiaodu-Hi Model on Jetson

## JetPack Install & Flash

Download the latest [JetPack](https://developer.nvidia.com/embedded/jetpack) and run the installer, choose the following options to be installed and flashed into your Jetson TX1/TX2:

```txt
Package                                                Action
>── Target - Jetson TX1/TX2
    >── Linux for Tegra Host Side Image Setup..........no action
    │   >── ...
    >── Flash OS Image to Target.......................install
    >── Install on Target
        >── VisionWorks Pack...........................no action
        │   >── ...
        >── CUDA Toolkit...............................install
        >── Compile CUDA Samples.......................no action
        >── TensorRT...................................install
        >── OpenCV for Tegra...........................install
        >── Multimedia API package.....................install
        >── cuDNN Package..............................install
```

## Change the Performance Setting

```sh
sudo rm /etc/rc.local
set +H
sudo sh -c "echo '#!/bin/sh\n/home/ubuntu/jetson_clocks.sh\nnvpmodel -m 0\nexit 0\n' >> /etc/rc.local"
set -H
sudo chmod a+x /etc/rc.local
sudo chmod a+x /home/ubuntu/jetson_clocks.sh
```

## Prerequisites

### Base

```sh
sudo apt-get update
sudo apt-get -y install build-essential cmake wget libboost-all-dev libgflags-dev libgoogle-glog-dev uuid-dev libboost-filesystem-dev libboost-system-dev libboost-thread-dev ncurses-dev libssl-dev
```

### Blas

```sh
sudo apt-get -y install libatlas-base-dev libopenblas-base libopenblas-dev liblapack-dev liblapack3
```

### Codecs

```sh
sudo apt-get -y install libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libpng-dev libtiff-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libtheora-dev libxvidcore-dev x264 v4l-utils gstreamer1.0 gstreamer1.0-tools gstreamer1.0-plugins-ugly libturbojpeg libvorbis-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev
```

### Update Eigen3

```sh
cd /home/ubuntu
wget --no-check-certificate https://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz
tar -zxvf 3.3.7.tar.gz
cd eigen
mkdir build
cd build
cmake ..
sudo make install
```

### Update OpenCV

```
cd /home/ubuntu
mkdir OpenCV && cd OpenCV
wget --no-check-certificate https://github.com/opencv/opencv/archive/4.2.0.tar.gz
tar -zxvf 4.2.0.tar.gz && rm 4.2.0.tar.gz
mv opencv-4.2.0 opencv
wget --no-check-certificate https://github.com/opencv/opencv_contrib/archive/4.2.0.tar.gz
tar -zxvf 4.2.0.tar.gz && rm 4.2.0.tar.gz
mv opencv_contrib-4.2.0 opencv_contrib
mkdir build_opencv && cd build_opencv
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON \
    -DWITH_CUDA=ON -DCUDA_FAST_MATH=ON -DWITH_CUBLAS=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda \
    -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
    ../opencv
make -j
```

### Build Paddle Inference Library

Check [official document](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/build_and_install_lib_cn.html) for details.

Assume you build paddle with `export PADDLE_ROOT=/home/ubuntu/paddle-capi`.

### Build gRPC

Check [official document](https://grpc.io/docs/languages/cpp/quickstart/) for details.

## Run

Assume the code is in folder `/home/ubuntu/jetson/`.

Open `run.sh`, config variables as following:

```txt
WITH_GPU=ON
USE_TENSORRT=ON

INFER_NAME=infer_v1
LIB_DIR=/home/ubuntu/paddle-capi
OPENCV_DIR=/home/ubuntu/OpenCV/build_opencv
CUDA_LIB_DIR=YOUR_CUDA_LIB_DIR
CUDNN_LIB_DIR=YOUR_CUDNN_LIB_DIR
MODEL_DIR=/home/ubuntu/jetson/xiaodu_hi_v1
```

Attach a USB camera, then compile and run `infer_v1.cpp` using:

```sh
sh run.sh
```
