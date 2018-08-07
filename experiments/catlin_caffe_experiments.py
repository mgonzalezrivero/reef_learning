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

def set_experiment(basedir,method, factor, experiment_type, region, lrate='0.001', lratepol='"fixed"'):
    cycles=30
    cyclesize=1000
    Split='val'
    dest_folder='{}_{}_{}_{}'.format(experiment_type, method, factor, lrate)
    cc.run_css(basedir,region, lrate, method,factor, cycles, cyclesize, Split, dest_folder,  gpuid=0)

