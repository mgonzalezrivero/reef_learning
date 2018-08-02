# Reef Learning:
### *Deep learning application for automated image analyses of reef benthos*

## Scope
This repository contains a series to tools and wrappers for implementing Deep Learning in ecology as a tool to automate the annotation of images and estimate the abundance of organisms, *sensu* the point-count method. 

Machine learning is widely used in a number of applications in ecology. The intention of this work is to facilitate the uptake of this technology for ecological studies that rely on the analysis of images (i.e., photoquadrats) to extract ecologically relevant metrics (e.g., cover). While this work is particularly centred on benthic ecology of coral reefs, the same approach can be applied to plant ecology, temperate marine ecology, etc. 

This repository derives from the publication bellow, a study to evaluate the applications of automated image annotation for coral reef monitoring. Based on this, this README will guide you through on the configuration and code execution, using our data as examples, to:

1. Train Convoluted Neural Networks (CNN) to recognise organisms from images 
2. QAQC and select CNN models 
3. Deploy selected CCN model to new classify images and estimate percent cover

>### IMPORTANT NOTE. 
>This repository provides a series of scripts easy to implement and designed to automate estimations of cover from images and to reproduce the results from the publication below. Should you prefer to avoid configuring and executing this application locally, deep learning has also been implemented in two online and open access annotation platform designed by 
*Oscar Beijbom* and *Mat Wyatt*, co-authors of this work. The platforms, [CoralNet](https://coralnet.ucsb.edu) and [BenthoBox](https://benthobox.com), offer automated image annotation capabilities for extracting benthic cover data, as well as, a series of helper tools to facilitate the annotations, evaluation and data synthesis. As such, CoralNet and BenthoBox are stand-alone and user-friendly applications readily available for automated image annotations of benthic images.
>
>Importantly, while CoralNet and BenthoBox use Deep Learning for automated image classification, there are technical differences in the approach, behind the scenes. Therefore, we do not expect that the classification from these platforms will reproduce the same results from this publication, but similar. Some advantages from using the source code provided here are:  
>
>* Flexibility for further development.  
>* Adaptability to specific needs.  
>* Runs locally, which may be an advantage depending on the number of images you need to process. 

## Citation:
#### Method:
*González-Rivero M, Beijbom O, Rodriguez-Ramirez A, Bryant DEP, Ganase A, Gonzalez-Marrero Y, Herrera-Reveles A, Kennedy EV, Kim C, Lopez-Marcano S, Markey K, Neal BP, Osborne K, Reyes-Nivia C, Sampayo EM, Stolberg K, Taylor A, Vercelloni J, Wyatt M and Hoegh-Guldberg O.* (2018) Cost-effective monitoring of coral reefs using artificial intelligence. **Methods in Ecology and Evolution** *in review*

#### Data: 
[Dryad Digital Repository](http://xxxxxx) (González-Rivero et al., 2018)

## The Approach

This work is based on [Caffe](http://caffe.berkeleyvision.org/), a deep learning architecture designed for automated image classification, developed by the Berkeley AI Research group. Based on this architecture, we have written a series of functions and wrappers, written in Python, that execute caffe for training and predictions.  For more information about Caffe and Deep Learning networks, have a look a [this nice tutorial/overview](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.g129385c8da_651_0) from Caffe.  

Caffe architecture is designed for scene classification, i.e. assigning a class to the whole image. Here, however, we are interested in learning to automate random point annotation, i.e. the assignment of one class to a particular location in an image. This method, also referred as random point count, is commonly used in many population estimation applications using photographic records. In random point annotations, the relative cover or abundance of each class is defined by the number of points classified as such relative to the total number of observed points on the image. In this study, the relative cover of benthic groups or labels was estimated using 50 points per image.  

In other to archive automated random point annotation, we converted each image to a set of patches (224 x 224 pixels) cropped out around each given point location. Each patch was classified independently and the relative abundance for each of the benthic classifications was estimated by the ratio between the number of patches classified for a given class by the total number of patches evaluated in an image.

## Data specifications:
This work is based on two type of data:

* *Images:* these are conceived here as what we typically refer to *photo-quadrats* in Ecology: scaled images of a substrate used to estimate the abundance and size of organisms and substrate types. Something like this: 

![](http://grr-images-au.s3.amazonaws.com/15025/0150250068-quadrat2.jpg)

* *Annotations:* For this work, we are using point-based annotations from each image to train the CNN to recognise organisms and substrate type. For this to worki within this framework, each annotated images that we use either for training or testing the CNNs need to have information about the location of each annotation point (in pixels) and the label assigned to it. For example:

	| File Name | X | Y | Label |
	|-----------|---|---|-------|
	|reefA_0001.jpg|345|250|*Acropora* spp.|
	|reefA_0001.jpg|50|150|*Pocillopora* spp.|
	|reefA_0001.jpg|945|50| Turf algae|


## Hardware requirements:
GPU capabilties are highly recomended for this work. However, caffe can be configured to run for only using CPU. 

As a refrence, we have processed all images for this work in Amazon Web Services, using [AWS Cloud Computing p2-large instances](https://aws.amazon.com/ec2/instance-types/p2/). Also, the same work was replicated locally using NVIDIA Titan X graphic cards. 

## Getting started: An overview

Here is an overview on how to do automated image analysis of photoquadrats to estimate abundance (i.e., cover). More detailed steps will be detailed in documents within the repositorie and refered in this summary. 

1. 	__Configure the machine:__ Prepare either your local machine or the AWS instance by installing Caffe and other dependencies needed to excuted this code. Please refer to: `config.md` and follow each step.
2. __Download data:__ Here we are using the data from the publication above as an example. However, you can use your own dataset. If this is the case, follow the data structure of our data repository. You can download our dataset from [here](http://xxxxxx).  

>__Note:__ The data provided in the Dryad repository includes the entire dataset for all global regions of this paper. For each region we trained and tested a different Deep Learning network. If you only want to try this work, you will only need to download data for one of the regions

Once the machine and data have been configured. You can follow the `protocol.md` file to process the images. Here is a brief descriptions of the steps that will be required to automatically extract cover data from your images:

1. __Train Deep Learning Nets:__ Using the training images and annotations, this step will perform a number of iterations to fine-tune the base Net `model_zoo/VGG_ILSVRC_16_layers.caffemodel`. This base Net is a VGG-16 network initilized or trainned with the ImageNet dataset. For more details about this Net, please refer to the [Caffe Model Zoo documentation](https://github.com/BVLC/caffe/wiki/Model-Zoo) and the developers [arxiv paper](http://arxiv.org/pdf/1409.1556). The provided code will fine-tune this base Net using the training images and annotations provided. In the process of training the net, a couple of parameters are recommended to be optimised (e.g., number of iterations, base Learning Rate, Image Scale). In this code, an optimization function trains multiple net with a range of parameter values (grid search approach). This step is called here "Experiments", and the resulting nets from these experiments are then screened and the best network is selected based on their performance (step below). 
 
2. __Screen and select best Net configuration:__ 
3. __Evaluate the ecological performance of the Net:__ Once the best Net is selected, the final step is to evaluate the performance of the Net by using the error metric described in the publication. This error is a measure of the abolute difference between automated and manual estimations of cover by the machine and expert human, respectively. For this work, we are no longer looking at the accuracy of the machine at the point level (as determined by the confusion matrix), but rather the precision of the automated estimations of benthic cover for each label within a sample unit. In our case, a sample unit is a 50m transect containing a number of images. To evaluate the performance error a set of transects are set aside, called "test-cells". The images and manual annotations from this test-cells are stored in the test folder, as downloaded from our dataset. The predicted cover estimations on those images from the resulting Net are then contrasted with the manual cover estimations from the `test/test.csv` using the following R script: `dfd.R`
5. __Train final Net:__ Having optimised the Net configuration, defined the final labelset and evaluated the peformance of this Net, the test and training data can be merged to create the final or deployment Net that you will use for classifing new images.
6. __Classify new images:__ 
7. __Aggregate classifications into cover data:__


## Contents:

## Contact:

We hope this work will be useful for your work. Feel free to contact me if you need any help or want to discuss the applications of this code to your work. 

----
*Manuel Gonzalez-Rivero*   
Australian Institute of Marine Science.  
[m.gonzalezrivero@aims.gov.au](mailto:m.gonzalezrivero@aims.gov.au)





