# Reef Learning:
### *Deep learning application for automated image analyses of reef benthos*

## Scope
This repository contains a series of tools and wrappers for implementing Deep Learning in ecology as a tool to automate the annotation of images and estimate the abundance of organisms, *sensu* the point-count method. 

Machine learning is increasingly being implemented in a number of applications in ecology. The intention of this work is to facilitate the uptake of this technology for ecological studies that rely on the analysis of images (i.e., photoquadrats) to extract abundance of organism (e.g., cover). While this work is particularly centred on benthic ecology of coral reefs, the same approach can be applied to plant ecology, temperate marine ecology, and other similar fields. 

The repository derives from the publication below, a study designed to evaluate the applications of automated image annotation for coral reef monitoring. This README file will guide you through the configuration and code execution, using our data as examples, to:

1. Train Convoluted Neural Networks (CNN) to recognise organisms from images 
2. Test and select CNN models 
3. Deploy selected CCN model to classify new images and estimate percent cover

>### IMPORTANT NOTE. 
>The repository provides a series of scripts easy to implement and designed to automate estimations of cover from images and to reproduce the results from the publication below. Should you prefer to avoid configuring and executing this application, deep learning has also been implemented in two online and open access annotation platforms, designed by 
*Oscar Beijbom* and *Mat Wyatt*, co-authors of this work. The platforms, [CoralNet](https://coralnet.ucsb.edu) and [BenthoBox](https://benthobox.com), offer automated image annotation capabilities for extracting benthic cover data, as well as, a series of helper tools to facilitate the annotations, evaluation and data synthesis. As such, CoralNet and BenthoBox are stand-alone and user-friendly applications readily available for automated image annotations of benthic images.
>
>Importantly, while CoralNet and BenthoBox use Deep Learning for automated image classification, there are technical differences in the approach (e.g., Net architecture, initialization). Therefore, we do not expect that the classification from these platforms will reproduce exactly the same results from this publication, although similar. Some advantages from using the source code provided here are:  
>
>* Flexibility for further development.  
>* Adaptability to specific needs.  
>* Runs locally, which may be an advantage depending on the number of images you need to process. 

## Citation:
#### Method:
*González-Rivero M, Beijbom O, Rodriguez-Ramirez A, Bryant DEP, Ganase A, Gonzalez-Marrero Y, Herrera-Reveles A, Kennedy EV, Kim C, Lopez-Marcano S, Markey K, Neal BP, Osborne K, Reyes-Nivia C, Sampayo EM, Stolberg K, Taylor A, Vercelloni J, Wyatt M and Hoegh-Guldberg O.* (2019) Cost-effective monitoring of coral reefs using artificial intelligence. **Scientific Reports** *in review*

#### Data: 
[Queensland Research and Innovation Services Cloud (QRIS CLoud) Repository](http:// https://nextcloud.qriscloud.org.au/index.php/s/YMgU7ZpdxSjPwpu) 
## The Approach

This work is based on [Caffe](http://caffe.berkeleyvision.org/), a deep learning architecture designed for automated image classification, developed by the Berkeley AI Research group. Based on this architecture, we have written a series of functions and wrappers, written in Python, that execute caffe for training and predictions.  For more information about Caffe and Deep Learning networks, have a look a [this comprenhensive overview](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.g129385c8da_651_0) from Caffe.  

Caffe architecture is designed for scene classification, i.e. assigning a class to the whole image. Here, however, we are interested in learning to automate random point annotation, i.e. the assignment of one class to a particular location in an image. This method, also referred as random point count, is commonly used in many population estimation applications using photographic records. In random point annotations, the relative cover or abundance of each class is defined by the number of points classified as such relative to the total number of observed points on the image. In this study, the relative cover of benthic groups or labels was estimated using 50 points per image.  

In order to achieve automated random point annotation, we converted each image to a set of patches (224 x 224 pixels) cropped out around each given point location. Each patch was classified independently and the relative abundance for each of the benthic classifications was estimated by the ratio between the number of patches classified for a given class by the total number of patches evaluated in an image (see figure below):

![methods](https://github.com/mgonzalezrivero/reef_learning/blob/master/figures/method.png?raw=true)


## Data specifications:
This work is based on two type of data:

* *Images:* these are conceived here as what we typically refer to *photo-quadrats* in ecology: scaled images of a substrate used to estimate the abundance and size of organisms and substrate types. Here is an example from our data: 

<div style="text-align:center"><img src ="http://grr-images-au.s3.amazonaws.com/15025/0150250068-quadrat2.jpg" /></div>

* *Annotations:* For this work, we are using point-based annotations from each image to train the CNN to recognise organisms and substrate type. For this to work within this framework, each annotated image that we use either for training or testing the CNNs need to have information about the location of each annotation point (Row and Columns in pixels) and the label assigned to it. Here is a sample from the annotation data used here:

    |id|image id|Date annotated|Annotator|Row|Column|Label Name|shortcode|func_group |file_name|method|
    |-|-|-|-|-|-|-|-|-|-|-|
    |1|38001004401|2017-06-20 01:33:56+00:00|Obs 1|234|250|Epilithical Algal Matrix|EAM_DHC|Algae|38001004401.jpg|random
    |2|38001004402|2017-06-20 01:33:56+00:00|Obs 2|235|450|Tabular Acropora|TB_Acr|Hard Coral|38001004402.jpg|random

>**Note:** Make sure your dataset follows the same structure as shown above, as these scripts are hard-coded to this structure. 

## Hardware requirements:
GPU capabilties are highly recommended for this work. However, caffe can be configured to run for only using CPU. 

As a reference, we have processed all images for this work in Amazon Web Services, using [AWS Cloud Computing p2-large instances](https://aws.amazon.com/ec2/instance-types/p2/). Also, the same work was replicated locally using NVIDIA Titan X graphic cards. 

## Network architecture:
This work is based on using a VGG-16 network architecture. Instead of having many hyper parameters, VGG-16 use a much simpler network, just having convoluted layers that are just 3 x 3 filters with stride, using the same padding. While VGG-16 is a very deep network, with about 138 million parameters, VGG is a very simplified architecture.

Below you can see a VGG diagram representing the 16 layers (more details [here](https://blog.heuritech.com/2016/02/29/a-brief-report-of-the-heuritech-deep-learning-meetup-5/)):

<div style="text-align:center"><img src ="https://cdn-images-1.medium.com/max/1600/0*qrMVR8XCPceU7dnP.png" /></div>


## Getting started: 

Here is an overview on how to do automated image analysis of photoquadrats to estimate abundance (i.e., cover). More detailed steps will be detailed in documents within the repository and referred to in this summary. 

1. __Configure the machine:__ Prepare either your local machine or the AWS instance by installing Caffe and other dependencies needed to execute this code. Please refer to Caffe installation instructions [here](http://caffe.berkeleyvision.org/installation.html). We have also created a configurations guide for the AWS instance using a Bitfusion imafe. (Details in  `config.md` )
2. __Download data:__ Here we are using the data from the publication above as an example. However, you can use your own dataset. If this is the case, follow the data structure of our data repository. You can download our dataset from [here](https://nextcloud.qriscloud.org.au/index.php/s/YMgU7ZpdxSjPwpu).

>__Note:__ The data provided includes the entire dataset for all the global regions analysed in this paper. For each region, or country within a region, we trained and tested a different Deep Learning network. 

Once the AWS instance and data have been configured, the Jupyter Notebooks, in the `notebooks` folder, will provide a guide and an example to run the source code from this repository. Here is a brief description of each notebook to automatically extract cover data from your images:

1. __Train Deep Learning Nets (`train_nets.pynb`):__  
Using the training images and annotations, this step will perform a number of iterations to fine-tune the base Net `model_zoo/VGG_ILSVRC_16_layers.caffemodel`. This base Net is a VGG-16 network initialized or trained with the ImageNet dataset (1M images and 1K classification units or labels). For more details about this Net, please refer to the [Caffe Model Zoo documentation](https://github.com/BVLC/caffe/wiki/Model-Zoo) and the developers [arxiv paper](http://arxiv.org/pdf/1409.1556). This code will fine-tune the base Net using the training images and annotations provided. In the process of training the net, the last layer of the base net gets replaces by the classifciation units of our data (labels) and the hyperparameters of the network will be adjusted by back-propagation. During the training, two more hyperparameters are recommended calibrate (e.g., number of iterations, base Learning Rate, Image Scale). to calibrate such parameters, the code will train  multiple nets using a range of parameter values. This step is refered here as "Experiments", and the resulting nets from these experiments will then be screened to select the best network.  

2. __Screen and select best Net configuration on their performance (`compare_nets.pynb`):__  
All Nets produced in the step above, using different configuration parameters, will be contrasted in this step. Different performance metrics will be produced based on the net classification on manually classified images. 
Once the best Net is selected, the final step is to evaluate the performance of the Net by using the error metric described in the publication. This error is a measure of the absolute difference between automated and manual estimations of cover by the machine and expert human, respectively. For this work, we are no longer looking at the accuracy of the machine at the point level (as determined by the confusion matrix), but rather the precision of the automated estimations of benthic cover for each label within a sample unit. In our case, a sample unit is a 50m transect containing a number of images. To evaluate the performance error, a set of transects are set aside, called "test-cells". The images and manual annotations from these test-cells are stored in the test folder, as downloaded from our dataset. The predicted cover estimations will then be contrasted with the manual cover estimations on the test images.
3. __Classify new images (`deploy_net.pynb`):__ 
The selected Net will be used to predict labels on a given number of random points from a specified set of images that are contained in a folder called `data`. After running the classifier on the new images, the scripts will produce a folder called `coverages` that will contain a text file for every new image with the location and classification of each point. Use these files to estimate benthic coverage as described in the manuscript associated to this repository

## Contents:
>Note: This repository is designed as a python module that contains sub-modules for executing caffe using `pycaffe`.

* **config:** Contains markdown files to guide you through the configuration of the AWS instance. More intructions for local installation [here](http://caffe.berkeleyvision.org/installation.html)
* **deeplearning_wrappers:** Python modules that include wrapper functions to train, test and deploy Deep learning for automated estimations of benthic cover from images.
* **experiments:** Python module designed as a sweeper to train the NET under a range of values for base learning rate as well as image size.
* **notebooks:** Jupyter notebooks that will guide you through the execution of the source code for automated image analysis. 
* **toolbox:** source code and helper functions


## Contact:

We hope this repo can be useful for your work. Feel free to contact us if you need any help or want to discuss the applications of this code to your specific work. 

*Manuel Gonzalez-Rivero*   
Australian Institute of Marine Science.  
[m.gonzalezrivero@aims.gov.au](mailto:m.gonzalezrivero@aims.gov.au)
