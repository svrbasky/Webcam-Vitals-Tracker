
# Remote PPG App v1
Project files for c++ opencv version of rPPG-v1

## Directory Structure
The below tree proposes directory structure for the project

```
rPPG
│---README.md
│---CMakeLists.txt
|---opencv_install_mac.sh
|
│───src
│   └───opencv_ex.cpp                           ## project main file
│   └───vita.hpp                                ## supplimentary functions
│
│───data
│   └───spd.png                                 ## SP Digital logo
|   └───shape_predictor_81_face_landmarks.dat   ## DLIB face detector model file
│
│───dlib                                        ## DLIB include folder referenced in CMAkeLists.txt
```

## Dependancies

### On mac-os
* Install Qt-creator using 
` brew install qt `

* Install Opencv by running the shell script
`./opencv_install.sh`

* Install librealsense2
Follow instruction given in the parent repository : **vita_ppg** 

## Build & Run Project
### Build
```
mkdir build
cd build
rm -rf *
cmake ..
make -j4
```
### Run
`./rPPG-v1`
