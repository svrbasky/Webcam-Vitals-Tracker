#!/bin/bash

cd ~
git clone https://github.com/opencv/opencv.git
cd opencv

mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D BUILD_TESTS=OFF \
-D CMAKE_OSX_ARCHITECTURES=x86_64 \
-D INSTALL_C_EXAMPLES=OFF \
-D Qt5_DIR=/usr/local/Cellar/qt/5.15.1 \
-D WITH_QT=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_EXAMPLES=ON ..

make -j4
sudo make install

echo 'export OpenCV_DIR=~/opencv/build' >> ~/.bash_profile
source ~/.bash_profile
