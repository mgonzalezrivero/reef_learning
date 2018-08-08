"""
Deploy classification net and predict on CSS imagery within regions
""" 

import glob
import shutil
import os
import scipy 
import numpy as np
import os.path as osp
#import math

from PIL import Image
import reef_learning.toolbox.beijbom_caffe_tools as bct
#import coral_lib.patch.tools as cpt ## new functions adapted from bmt to replace coral_lib fucntions

#from caffe import layers as L, params as P ## Need to change this gits folder
from beijbom_vision_lib.misc.tools import crop_and_rotate, pload 
from beijbom_vision_lib.caffe.nets.vgg import vgg_core 
import reef_learning.deeplearning_wrappers.catlin_caffe_tools as cct
import reef_learning.deeplearning_wrappers.catlin_tools as ct
 
## set up deployment net by training with both train and test datasets

def run_css(basedir,region, lr, scale_method,scale_factor, cycles, cyclesize, Split, dest_folder, gpuid=0):
    workdir = osp.join(basedir,region,dest_folder)

    if osp.isdir(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

    labelset=ct.get_labelset(basedir,region)
    
    imlist, imdict = ct.load_data(basedir,Split,labelset,region)
    im_mean = bct.calculate_image_mean(imlist[::10])
 
    for split in ['train', 'val']:
        ct.write_split(basedir,workdir, region, split, labelset, scale_method,scale_factor)
   
    solver = bct.CaffeSolver(onlytrain=True) 
    solver.sp['base_lr'] = lr
    solver.sp['lr_policy'] = '"fixed"'
    #for f in ['test_net', 'test_iter', 'test_interval', 'test_initialization']:
    #            del solver.sp[f]
    solver.write(osp.join(workdir, 'solver.prototxt')) 


    pyparams = pload(osp.join(workdir, 'valpyparams.pkl'))

    cct.write_net(workdir, im_mean, len(labelset), scaling_method = scale_method, scaling_factor = scale_factor, cropsize= 224, onlytrain=True)
    
    shutil.copyfile(osp.join(basedir,'/model_zoo/VGG_ILSVRC_16_layers.caffemodel'), osp.join(basedir,workdir, 'vgg_initial.caffemodel'))
    for i in range(cycles):
        bct.run(workdir, gpuid=gpuid,nbr_iters= cyclesize, onlytrain=True)
        _ = bct.classify_from_patchlist_wrapper(imlist, imdict, pyparams, workdir, gpuid = gpuid, save = True, net_prototxt = 'trainnet.prototxt')

## Classify random points from a given image and net model 
def classify_image(imname, pyparams, npoints, net):

    transformer = bct.Transformer(pyparams['im_mean'])
    
    # Load image
    im = np.asarray(Image.open(imname))        
    # create annlist
    height_cm = 100
    point_anns = []
    [nrows, ncols, _] = im.shape
    rows = np.random.choice(np.linspace(50, nrows - 50, nrows - 99), npoints, replace = False)
    cols = np.random.choice(np.linspace(50, ncols - 50, ncols - 99), npoints, replace = False)
    for row, col in zip(rows, cols):
        point_anns.append((int(row), int(col), 0))
    
    # Resize & Crop
    (im, scale) = ct.coral_image_resize(im, pyparams['scaling_method'], pyparams['scaling_factor'], height_cm) 
    patchlist = ct.crop_patch(im, pyparams['crop_size'], scale, point_anns, height_cm)

    # Classify
    [estlist, _] = bct.classify_from_imlist(patchlist, net, transformer, pyparams['batch_size'])

    return estlist, rows, cols

## Deploy final Net to predict labels from all images within each expedition of a given region
def classify_allexp(basedir,modeldir,region,gpuid=0, npoints=50, force_rewrite = False):
    """
    At the moment, this is using the net produced for test. Need to merge training and test-cell images to produce the deployment net  
    """
    bestiter, _ = cct.find_best_iter(modeldir)
    caffemodel = 'snapshot_iter_{}.caffemodel'.format(bestiter)
    net = bct.load_model(modeldir, caffemodel, gpuid = gpuid, net_prototxt = 'deploy.prototxt')
    pyparams = pload(osp.join(modeldir, 'deploytrainpyparams.pkl'))
    labelset=ct.get_label(region)
    indir_root = osp.join(basedir,region,'data') ## Need ssh link this one to qcloud
    for indir in glob.glob(indir_root + 'exp*'):
        outdir = indir + '/coverages'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        imgs = glob.glob(indir + '/images/*.jpg')
        print "Starting", indir, len(imgs)
        for img in imgs:
            imname = osp.basename(img)
            covname = osp.join(outdir, imname + '.points.csv')
            if osp.isfile(covname) and not force_rewrite:
                continue
            estlist, rows, cols = classify_image(img, pyparams, npoints, net)
            f = open(covname, 'w')
            f.write(imname + '\n')
            f.write('row, col, labelcode\n')
            for row, col, est in zip(rows, cols, estlist):
                f.write('{}, {}, {}\n'.format(int(row), int(col), labelset[est]))
            f.close()
        print "Done", indir, len(imgs)

## Deploy final Net to predict labels from all images in a given expedition and region
def classify_exp(basedir,region,expdir, modeldir, npoints=50, gpuid=0, force_rewrite = False):

    bestiter, _ = cct.find_best_iter(modeldir)
    caffemodel = 'snapshot_iter_{}.caffemodel'.format(bestiter)
    net = bct.load_model(modeldir, caffemodel, gpuid = gpuid, net_prototxt = 'trainnet.prototxt')
    pyparams = pload(osp.join(modeldir, 'trainpyparams.pkl'))
    labelset=ct.get_labelset(basedir,region)
    indir_root = osp.join(basedir,region, 'data')
    indir = osp.join(indir_root, expdir)
    outdir = indir + '/coverages'

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    imgs = glob.glob(indir + '/images/*.jpg')
    print "Starting", indir, len(imgs)

    for img in imgs:
        imname = osp.basename(img)
        covname = osp.join(outdir, imname + '.points.csv')
        if osp.isfile(covname) and not force_rewrite:
            continue

        estlist, rows, cols = classify_image(img, pyparams, npoints, net)
        f = open(covname, 'w')
        f.write(imname + '\n')
        f.write('row, col, labelcode\n')
        
        for row, col, est in zip(rows, cols, estlist):

            f.write('{}, {}, {}\n'.format(int(row), int(col), labelset[est]))

        f.close()

    print "Done", indir, len(imgs)
    
def classify_test(basedir,region,test_folder,modeldir, gpuid=0, force_rewrite=False, height_cm=100):
    '''
    Run classification of test images using specific previously annotated points
    '''
      
    bestiter, _ = cct.find_best_iter(modeldir)
    caffemodel = 'snapshot_iter_{}.caffemodel'.format(bestiter)
    net = bct.load_model(modeldir, caffemodel, gpuid = gpuid, net_prototxt = 'trainnet.prototxt')
    pyparams = pload(osp.join(modeldir, 'trainpyparams.pkl'))
    labelset=ct.get_labelset(basedir,region)
    indir_root = osp.join(basedir,region)
    indir = osp.join(indir_root, test_folder)
    outdir = indir + '/coverages'
    height_cm = 100
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

   # load test-data
    imlist, imdict = ct.load_annotation_file(osp.join(basedir,region, test_folder,'annotations.csv'), labelset)
    imlist = [osp.join(basedir, region, test_folder,'images', im) for im in imlist]

    for img in imlist:
        transformer = bct.Transformer(pyparams['im_mean'])
        imname = osp.basename(img)
        covname = osp.join(outdir, imname + '.points.csv')
        if osp.isfile(covname) and not force_rewrite:
            continue
        
        point_anns = []
        rows = [y for y,_,_ in imdict[imname][0]]
        cols = [y for _,y,_ in imdict[imname][0]]
        
        for row, col in zip(rows, cols):
            point_anns.append((int(row), int(col), 0))
    
        # Resize & Crop
        im = np.asarray(Image.open(img))
        (im, scale) = ct.coral_image_resize(im, pyparams['scaling_method'], pyparams['scaling_factor'], height_cm) 
        patchlist = ct.crop_patch(im, pyparams['crop_size'], scale, point_anns, height_cm)

        # Classify
        [estlist, _] = bct.classify_from_imlist(patchlist, net, transformer, pyparams['batch_size'])
        
        f = open(covname, 'w')
        f.write(imname + '\n')
        f.write('row, col, labelcode\n')

        for row, col, est in zip(rows, cols, estlist):
            f.write('{}, {}, {}\n'.format(int(row), int(col), labelset[est]))

        f.close()

    print "Done", indir, len(imlist)
