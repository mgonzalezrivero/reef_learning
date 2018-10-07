# CONFIGURING DOCKER CONTAINER

Docker images for caffe are available [here]([Docker image for caffe](https://hub.docker.com/r/bvlc/caffe/)). These images are preconfigured and ready to work with pycaffe. 



## Troubleshooting
### Protobuf version

Caffe will give you an error if the Google Protocol Buffer version is below 3.2. If you run into this problem, follow the steps [here](https://github.com/google/protobuf/tree/master/src) and recompile caffe. 

When compiling protobuf use the following flags in when runing the configure file:

	./configure --prefix=/usr --with-protoc=protoc   
	

To compile caffe follow the installation [instructions](http://caffe.berkeleyvision.org/installation.html) from Caffe. At this point, an compiler version error may appear:  
	
>/usr/include/c++/5/bits/c++0x_warning.h:32:2: error: #error This file requires compiler and library support for the ISO C++ 2011 standard. This support must be enabled with the -std=c++11 or -std=gnu++11 compiler options.
	
You can solve this problem by modifying `CMakeLists.txt` from the caffe folder, typically found in `/opt/caffe`.

**Original**

	# ---[ Flags
		if(UNIX OR APPLE)
  		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} 	-fPIC -Wall")
		endif()if()  

**Modify**

	# ---[ Flags
		if(UNIX OR APPLE)
  		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} 	-fPIC -Wall -std=c++11")
		endif()if()
		
		
###pycaffe import error

After compiling caffe with the upgraded version for protoc, the python protobuf version may be out of date and this gives you the following error when importing the python caffe module:

> File "caffe/proto/caffe_pb2.py", line 23, in <module>
    ld\x18\x37 \x01(\x02:\x03\x30.5\x12\x1d\n\x0f\x64\x65t_fg_fraction\x18\x38 \x01(\x02:\x04\x30.25\x12\x1a\n\x0f\x64\x65t_context_pad\x18: \x01(\r:\x01\x30\x12\x1b\n\rdet_crop_mode\x18; \x01(\t:\x04warp\x12\x12\n\x07new_num\x18< \x01(\x05:\x01\x30\x12\x17\n\x0cnew_channels\x18= \x01(\x05:\x01\x30\x12\x15\n\nnew_height\x18> \x01(\x05:\x01\x30\x12\x14\n\tnew_width\x18? \x01(\x05:\x01\x30\x12\x1d\n\x0eshuffle_images\x18@ \x01(\x08:\x05\x66\x61lse\x12\x15\n\nconcat_dim\x18\x41 \x01(\r:\x01\x31\x12\x36\n\x11hdf5_output_param\x18\xe9\x07 \x01(\x0b\x32\x1a.caffe.HDF5OutputParameter\".\n\nPoolMethod\x12\x07\n\x03MAX\x10\x00\x12\x07\n\x03\x41VE\x10\x01\x12\x0e\n\nSTOCHASTIC\x10\x02\"W\n\x0ePReLUParameter\x12&\n\x06\x66iller\x18\x01 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x1d\n\x0e\x63hannel_shared\x18\x02 \x01(\x08:\x05\x66\x61lse*\x1c\n\x05Phase\x12\t\n\x05TRAIN\x10\x00\x12\x08\n\x04TEST\x10\x01')
TypeError: __init__() got an unexpected keyword argument 'serialized_options'

To solve this problem, check that the protoc and python protobuf versions are the same:

	protoc --version
	pip info protobuf 
	
If the versions are different upgrade protobusuign:  
	
	pip install protobuf==<your protoc version>

in our case the protoc and protobuf version we are using is 3.6




	
