# CONFIGURING AWS EC2 FOR IMAGE ANALYSIS

Manuel Gonzalez-Rivero 02/05/2017

Machines are here configured using the Bitfusion Caffe image from the AWS Marketplace. While caffe is preinstalled, it was compiled without PYTHON_LAYER active, which means that we cannot train the machine from the OS calling at python to setup the layers. 

## Compile caffe in C++
###Install CMake

	cd ~
	wget https://cmake.org/files/v3.8/cmake-3.8.0.tar.gz
	tar xzf cmake-3.8.0.tar.gz
	cd cmake-3.8.0
	./configure --prefix=/opt/cmake
	make
	sudo make install
	
Check that the installation was OK:

	/opt/cmake/bin/cmake -version
	
The output should be somthing like this:

	cmake version 3.8.0
	
###Install OpenCV

	sudo apt-get install libopencv-dev python-opencv
	sudo apt-get install liblapacke-dev checkinstall

Get install-open-cv.sh from here:
https://github.com/milq/milq/blob/master/scripts/bash/install-opencv.sh#L1-L1

	sudo bash install-open--cv.sh
	
NOTE: 
If you find the following error:

	NOTFOUND/lapacke.h: No such file or directory #include "LAPACKE_H_PATH-NOTFOUND/lapacke.h"

The fix is to replace the line:

	cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -	DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON $

with:

	cmake -DBUILD_WITH_DYNAMIC_IPP=ON -D WITH_LAPACK=OFF -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON -$

### Install Google Protobuf

Follow the steps in:

https://github.com/google/protobuf/blob/master/src/README.md

###Compile Caffe
Need to modify Makefile.config and uncomment 'WITH_PYTHON_LAYER' unsing an editor (e.g., nano)

	sudo cp /usr/lib/x86_64-linux-gnu/libhdf5_hl.so /usr/local/lib/libhdf5_hl.so
	sudo cp /usr/lib/x86_64-linux-gnu/libhdf5.so /usr/local/lib/libhdf5.so

	cd ~/caffe
	make clean
	rm -R ~/caffe/build
	mkdir ~/caffe/build
	cd build
	cmake  ..
	make all
	make install
	make runtest
	
Try importing caffe module from python to make sure it is all good. 

## Clone GitHub libraries 

Python libraries from Oscar Beijbom. Most of these are wrappers to work with Caffe in Python as well as other computer vision tools.

	cd ~
	git clone https://github.com/beijbom/beijbom_vision_lib.git

Python libraries from Manuel Gonzalez-Rivero and Oscar Beijbom. These are mainly wrappers to process the images the way we do at the XL CSS. 

	cd ~
	git clone https://github.com/mgonzalezrivero/catlin_deeplearning.git
	
## Add essential paths to bash profile

	nano ~/.bashrc
	
Add the following line at the end:

	export PYTHONPATH=$PYTHONPATH:/home/ubuntu/caffe/python:/home/ubuntu:/home/ubuntu/catlin_deeplearning/beijbom


## Mount data drive

At this point caffe is fully operational. Now you need to make sure a volume for data is attached to the machine. Check this in the AWs console.

Check the volume name:

	lsblk

New volumes are raw block devices, and you need to create a file system on them before you can mount and use them:

	sudo file -s /dev/xvdb

If the output of the previous command shows simply data for the device, then there is no file system on the device.

	sudo mkfs -t ext4 device_name
	#In my case:
	sudo mkfs -t ext4 /dev/xvdb

WARNING: THIS WILL ERASE THE DATA FROM THE DISK IS IT HAD DATA.

Make mounting point:

	sudo mkdir /media/data_caffe
	
Mount drive:

	sudo mount /dev/xvdb /media/data_caffe/
	
NOTE: The drive will have to be mounted everytime the EC2 is initiated. The reason why I am not adding this tot fstab for automounting is because, if the device is unattached and reattached from the AWS console, it will change the name and cause conflicts in the fstab system. 

Therefore, you will have to run the line above everytime the machine is activated. 
	
 
	
