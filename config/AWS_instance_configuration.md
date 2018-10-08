# CONFIGURING AWS EC2 FOR IMAGE ANALYSIS

Machines were configured using the [Bitfusion Caffe image](https://aws.amazon.com/marketplace/pp/B01B52CMSO?qid=1533256793574&sr=0-2&ref_=srh_res_product_title) from the AWS Marketplace. While caffe is preinstalled, it was compiled without PYTHON_LAYER active, which means that we cannot train the machine calling the OS from Python to setup the layers. So, a bit of extra work needs to be done to confire the machine properly.  The following steps will take you through the configuration of the AWS instance using the BitFusion image for Caffe. 

Alternatively, you can use the [Docker image for caffe](https://hub.docker.com/r/bvlc/caffe/) or newer images in the AWS marketplace. However, you will have to test the best option. Here we present one way to do it using the Bitfusion Caffe image.

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

`https://github.com/google/protobuf/blob/master/src/README.md`

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

Python libraries from Manuel Gonzalez-Rivero and Oscar Beijbom. These are mainly wrappers to process the images the way we did for the publication. 

	cd ~
	git clone https://github.com/mgonzalezrivero/reef_learning.git
	
## Add essential paths to bash profile

	nano ~/.bashrc
	
Add the following line at the end:

	export PYTHONPATH=$PYTHONPATH:<PATH TO YOUR HOME DIR>/reef_learning:<PATH TO YOUR HOME DIR>/beijbom_vision_lib


## Mount data drive

At this point caffe is fully operational. Now you need to make sure that a volume for data is attached to the machine. Create a Volume from the AWS console and attach it to the EC2 instance you are using.

Check the volume name:

	lsblk

New volumes are raw block devices, and you will need to create a file system on them before you can mount and use them:

	sudo file -s /dev/xvdb

If the output of the previous command shows simply data for the device, then there is no file system on the device.

	sudo mkfs -t ext4 device_name
	#In my case:
	sudo mkfs -t ext4 /dev/xvdb
	
<aside class="warning">
<span style="color:red">**WARNING.**</span> THIS WILL ERASE ANY DATA FROM THE DISK.
</aside>

Make mounting point:  

>This is directory where the volume will be mounted and we will use this path in the scrpts when running the training, tests and deployment of Nets. 

	sudo mkdir /media/data_caffe
	
Mount drive:

	sudo mount /dev/xvdb /media/data_caffe/
	
>NOTE: The drive will have to be mounted everytime the EC2 is initiated. The reason why I am not adding this to fstab for auto-mounting is because if the device is unattached and reattached from the AWS console, it will change the name and cause conflicts in the fstab system. Therefore, you will have to run the line above everytime the machine is activated. Also, this will give you the flexibility of mounting different volumes depending on the containers where you data is stored in AWS.
	
 
	
