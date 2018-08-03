"""
Series of miscelaneous functions for handling images and data required to deploy net on the CSS imagery
""" 

import caffe
import json
from caffe import layers as L

import numpy as np
import os.path as osp
import scipy 

from PIL import Image

import catlin_deeplearning.beijbom.beijbom_caffe_tools as bct
from beijbom_vision_lib.caffe.nets.vgg import vgg_core
from beijbom_vision_lib.misc.tools import crop_and_rotate, psave

def crop_patch(im, crop_size, scale, point_anns, height_cm):
    patchlist = []

    # Pad the boundaries
    im = np.pad(im, ((crop_size*2, crop_size*2),(crop_size*2, crop_size*2),(0,0)), mode='reflect') 
        
    # Extract patches
    for (row, col,label) in point_anns:
        center_org = np.asarray([row, col])
        center = np.round(crop_size*2 + center_org * scale).astype(np.int)
        patchlist.append(crop_and_rotate(im, center, crop_size, 0, tile = False))

    return patchlist

def coral_image_resize(im, scaling_method, scaling_factor, height_cm):
    """
    resizes the image according to convention used in the data layers.
    """

    if scaling_method == 'scale':
        scale = float(scaling_factor) # here scaling_factor is the desired image scaling.
    elif scaling_method == 'ratio':
        scale = float(scaling_factor) * height_cm / im.shape[0] # here scaling_factor is the desited px_cm_ratio.
    im = scipy.misc.imresize(im, scale)
    return (im, scale)

def rotate_with_PIL(im, angle):
    im = Image.fromarray(im)
    im.rotate(angle)
    return np.asarray(im)
    
       
def load_annotation_file(file_, labelset):
    imdict = {}
    imlist = set()
    lines = [line.rstrip('\n') for line in open(file_)]
    for line in lines[1:]:
        parts = line.split(',')
        parts = [p.strip() for p in parts]
        imname = parts[1].strip('"') + '.jpg'
        imlist.add(imname)
        if not imname in imdict.keys():
            imdict[imname] = ([], 100)

        imdict[imname][0].append([int(parts[4]), int(parts[5]), labelset.index(parts[7].strip('"'))])

    return list(imlist), imdict
     

def load_data(split, labelset, region):
    assert split in ['train', 'val', 'test', 'deploytrain', 'deployval']

    imlist, imdict = load_annotation_file(osp.join('/media/data_caffe',region, 'train','annotations.csv'), labelset)
    imlist = [osp.join('/media/data_caffe', region,'train/images', im) for im in imlist]
    vallist = imlist[::6]
    trainlist = list(set(imlist) - set(vallist))
    if split == 'train':
        return trainlist, imdict
    elif split == 'val':
        return vallist, imdict
    
    # If we are still goind, let's load the test data.
    if split == 'test':
        imlist_test, imdict_test = load_annotation_file(osp.join('/media/data_caffe',region, 'test','annotations.csv'), labelset)
        imlist_test = [osp.join('/media/data_caffe', region,'test/images', im) for im in imlist_test]
        return imlist_test, imdict_test
    
    # If we are still going we are doing the deployment stuff. Let's start by merging everything.
    imdict.update(imdict_test)
    imlist.extend(imlist_test)
    
    vallist = imlist[::6]
    trainlist = list(set(imlist) - set(vallist))
    if split == 'deploytrain':
        return trainlist, imdict
    elif split == 'deployval':
        return vallist, imdict

    
    
def write_split(workdir, region, split, labelset, scale_method, scale_factor):

    imlist, imdict = load_data(split, labelset, region)
    im_mean = bct.calculate_image_mean(imlist[::10])

    with open(osp.join(workdir, '{}list.txt'.format(split)), 'w') as fp:
        for im in imlist:
            fp.write(im + '\n')

    with open(osp.join(workdir, '{}dict.json'.format(split)), 'w') as outfile:
        json.dump(imdict, outfile)

    pyparams =  dict(
                batch_size = 45, 
                imlistfile = '{}list.txt'.format(split), 
                imdictfile = '{}dict.json'.format(split),
                imgs_per_batch = 3,
                crop_size = 224,
                scaling_method = scale_method,
                scaling_factor = scale_factor,       
                rand_offset = 1,
                random_seed = 100,
                im_mean = list(im_mean),
                dalevel = 'cheap')

    psave(pyparams, osp.join(workdir, split + 'pyparams.pkl'))
    
    #net = caffe.NetSpec()
   
    #net['data'], net['label'] = L.Python(module = 'beijbom_caffe_data_layers', layer = 'RandomPointDataLayer', ntop=2, param_str=str(pyparams))

    #net = vgg_core(net)
    #net.score = L.InnerProduct(net.fc7, num_output=len(labelset), param=[dict(lr_mult=5, decay_mult=1), dict(lr_mult=10, decay_mult=0)])
    #net.loss = L.SoftmaxWithLoss(net.score, net.label)
 
    with open(osp.join(workdir, '{}net.prototxt'.format(split)), 'w') as w:
        n = bct.vgg(pyparams, 'RandomPointDataLayer', len(labelset))
        #w.write(str(net.to_proto()))
        w.write(str(n.to_proto()))

def get_labelset(region):
    codes = [str(line.rstrip('\n').split(',')[7]).strip('"').strip() for line in open(osp.join('/media/data_caffe',region,'train/annotations.csv'))]
    testcodes = [str(line.rstrip('\n').split(',')[7]).strip('"').strip() for line in open(osp.join('/media/data_caffe',region,'test/annotations.csv'))]

    codes.extend(testcodes)
    codes = set(codes)
    codes.remove('shortcode')

    return list(codes)
