import csv
import json
import re
import glob
import os
import shutil
import os.path as osp

import reef_learning.toolbox.beijbom_caffe_tools as bct
import reef_learning.toolbox.beijbom_misc_tools as bmt
import reef_learning.deeplearning_wrappers.catlin_caffe_tools as cct
import reef_learning.deeplearning_wrappers.catlin_classify as cc

from beijbom_vision_lib.misc.tools import psave, pload


def scale_experiment_wrapper(region):
    r=region
    et ='scaleratio_sweeper'
    methods = ['scale','ratio', 'scale', 'scale', 'ratio']
    factors = [ 2.0, 22, 1.0, 3.0, 30]
    for method, factor in zip(methods, factors):
        set_experiment(method,factor,et,r)

def learning_experiment_wrapper(region):
    etype='learningrate'
    methods = 'scale'
    factors =  2.0 
    csize = 224
    lrates=['0.00001', '0.0001', '0.001','0.01']
    lratepol='"fixed"'
    for lrate in lrates:
        set_experiment(methods, factors, etype, region, lrate, lratepol)

def set_experiment(method, factor, experiment_type, region, lrate='0.001', lratepol='"fixed"'):
    #r=region
    #print(r)
    #this_path=osp.join('/media/data_caffe',r)
    cycles=30
    cyclesize=1000
    Split='val'
    dest_folder='{}_{}_{}_{}'.format(experiment_type, method, factor, lrate)
    #workdir = this_path+'/{}_{}_{}_{}'.format(experiment_type, method, factor, lrate)
    #os.makedirs(workdir)            


    #im_mean, labelset = cct.setup_data(workdir,r)

    #cct.write_solver(workdir, lr=lrate, lrp=lratepol)
    
    #cct.write_net(workdir, im_mean, len(labelset), scaling_method = method, scaling_factor = factor, cropsize= 224)

    #shutil.copyfile('/media/data_caffe/model_zoo/VGG_ILSVRC_16_layers.caffemodel', osp.join(workdir, 'vgg_initial.caffemodel'))
    cc.run_css(region, lrate, method,factor, cycles, cyclesize, Split, dest_folder,  gpuid=0)
    #cct.run_css(workdir, 30, 1000)
