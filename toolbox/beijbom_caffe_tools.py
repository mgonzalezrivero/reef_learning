import glob, os, math, colorsys, scipy, caffe, re, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import beijbom_misc_tools as bmt
import beijbom_confmatrix as confmatrix
from pylab import *
from copy import deepcopy, copy
import cPickle as pickle
from tqdm import tqdm
from settings import CAFFEPATH
from caffe import layers as L, params as P
from reef_learning.toolbox.beijbom_misc_tools import coral_image_resize, crop_and_rotate, pload, psave
import json
import gc
"""
beijbom_caffe_tools (bct) contains classes and wrappers for caffe.
"""

class Transformer:
    """
    Transformer is a class for preprocessing and deprocessing images according to the vgg16 pre-processing paradigm (scaling and mean subtraction.)
    """

    def __init__(self, mean = [0, 0, 0]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occuring in the vgg16 caffe prototxt.
        """
    
        im = np.float32(im)
        im = im[:, :, ::-1] #change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))
        
        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1] #change to RGB
        
        return np.uint8(im)



class CaffeSolver:
    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default values and can export a solver parameter file.
    Note that all parameters are stored as strings. For technical reasons, the strings are stored as strings within strings.
    """

    def __init__(self, testnet_prototxt_path = "testnet.prototxt", trainnet_prototxt_path = "trainnet.prototxt", debug = False, onlytrain=False):
        
        self.sp = {}

        # critical:
        self.sp['base_lr'] = '0.001'
        self.sp['momentum'] = '0.9'
        
        # speed:
        self.sp['test_iter'] = '100'
        self.sp['test_interval'] = '250'
        
        # looks:
        self.sp['display'] = '25'
        self.sp['snapshot'] = '2500'
        self.sp['snapshot_prefix'] = '"snapshot"' # string withing a string!
        
        # learning rate policy
        self.sp['lr_policy'] = '"fixed"'

        # important, but rare:
        self.sp['gamma'] = '0.1'
        self.sp['weight_decay'] = '0.0005'
        self.sp['train_net'] = '"' + trainnet_prototxt_path + '"'
        self.sp['test_net'] = '"' + testnet_prototxt_path + '"'

        # pretty much never change these.
        self.sp['max_iter'] = '100000'
        self.sp['test_initialization'] = 'false'
        self.sp['average_loss'] = '25' # this has to do with the display.
        self.sp['iter_size'] = '1' #this is for accumulating gradients

        if (debug):
            self.sp['max_iter'] = '12'
            self.sp['test_iter'] = '1'
            self.sp['test_interval'] = '4'
            self.sp['display'] = '1'

        if (onlytrain):
            #print 'deleting test info'
            for f in ['test_net', 'test_iter', 'test_interval', 'test_initialization']:
                del self.sp[f]

    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()
        return 1

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
        return 1



def run(workdir = None, caffemodel = None, gpuid = 0, solverfile = 'solver.prototxt', log = 'train.log', snapshot_prefix = 'snapshot', caffepath = CAFFEPATH, restart = False, nbr_iters = None, onlytrain=False):
    """
    run is a simple caffe wrapper for training nets. It basically does two things. (1) ensures that training continues from the most recent model, and (2) makes sure the output is captured in a log file.

    Takes
    workdir: directory where the net prototxt lives.
    caffemodel: name of a stored caffemodel.
    solverfile: name of solver.prototxt [this refers, in turn, to net.prototxt]
    log: name of log file
    snapshot_prefix: snapshot prefix. 
    caffepath: path the caffe binaries. This is required since we make a system call to caffe.
    restart: determines whether to restart even if there are in the directory.

    """

    # find initial caffe model
    if not caffemodel:
        caffemodel = glob.glob(os.path.join(workdir, "*initial.caffemodel"))
        if caffemodel:
            caffemodel = os.path.basename(caffemodel[0])

    # finds the latest snapshots
    snapshots = glob.glob(os.path.join(workdir, "{}*.solverstate".format(snapshot_prefix)))
    if snapshots:
        _iter = [int(f[f.index('iter_')+5:f.index('.solverstate')]) for f in snapshots]
        max_iter = np.max(_iter)
        latest_snapshot = os.path.basename(snapshots[np.argmax(_iter)])
    else:
        max_iter = 0

    # update solver with new max_iter parameter (if asked for)
    if not nbr_iters is None: 
        solver = CaffeSolver(onlytrain=onlytrain)
        solver.add_from_file(os.path.join(workdir, solverfile))
        solver.sp['max_iter'] = str(max_iter + nbr_iters)
        solver.sp['snapshot'] = str(1000000) #disable this, don't need it.
        solver.write(os.path.join(workdir, solverfile))

    # by default, start from the most recent snapshot
    if snapshots and not(restart): 
        print "Running {} from iter {}.".format(workdir, np.max(_iter))
        runstring  = 'cd {}; {} train -solver {} -snapshot {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solverfile, latest_snapshot, gpuid, log)

    # else, start from a pre-trained net defined in caffemodel
    elif(caffemodel): 
        if(os.path.isfile(os.path.join(workdir, caffemodel))):
            print "Fine tuning {} from {}.".format(workdir, caffemodel)
            runstring  = 'cd {}; {} train -solver {} -weights {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solverfile, caffemodel, gpuid, log)

        else:
            raise IOError("Can't fine intial weight file: " + os.path.join(workdir, caffemodel))

    # Train from scratch. Not recommended for larger nets.
    else: 
        print "No caffemodel specified. Running {} from scratch!!".format(workdir)
        runstring  = 'cd {}; {} train -solver {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solverfile, gpuid, log)
    os.system(runstring)


def classify_imagedatalayer(net, n_testinstances = 50, batch_size = 50, scorelayer = 'score'):
    scorelist = []
    for test_itt in tqdm(range(n_testinstances//batch_size + 1)):
        scorelist.extend(list(copy(net.blobs[scorelayer].data).astype(np.float)))
        net.forward()
    scorelist = scorelist[:n_testinstances]
    return scorelist

def cycle_runs(workdir, cycle_size = 1000, ncycles = 10, gpuid = 1, batch_size_test = 50, n_testinstances = 50, testnet_prototxt = 'testnet.prototxt', snapshot_prefix = 'snapshot', scorelayer = 'score'):
    """
    cycle_runs is a wrapper around run and classify methods. It cycles through the various experiments, thus running them in "parrallell". After training net i for cycle_sizes[i] iterations, it will run through the TEST set of all *net.prototxt files in the directory and store these to disk. It will then move on to the next experiment, and cycle though all for ncycles.

    Takes:
    stuff 

    """
    for cycle in range(ncycles):
        run(workdir = workdir, nbr_iters = cycle_size, gpuid = gpuid)
        caffemodel = find_latest_caffemodel(workdir, snapshot_prefix = snapshot_prefix)
        net = load_model(workdir, caffemodel, gpuid = gpuid, net_prototxt = testnet_prototxt, phase = caffe.TEST)
        scorelist = classify_imagedatalayer(net, n_testinstances = n_testinstances, batch_size = batch_size_test, scorelayer = scorelayer)
        pickle.dump((scorelist), open(os.path.join(workdir, 'predictions_using_' + caffemodel +  '.p'), 'wb'))
        del net


def load_model(workdir, caffemodel, gpuid = 0, net_prototxt = 'net.prototxt', phase = caffe.TEST):
    """
    changes current directory to INPUT workdir and loads INPUT net_prototxt.
    """
    os.chdir(workdir)
    caffe.set_device(gpuid)
    caffe.set_mode_gpu()
    net = caffe.Net(net_prototxt, caffemodel, phase)
    net.forward() #one forward to initialize the net
    return net


def nbr_lines(fname):
    """
    Opens INPUT file fname and returns the number of lines in the file.
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def sac(im, net, Transformer, scorelayer, target_size = [1024, 1024], padcolor = [126, 148, 137], startlayer = 'conv1_1'):
    """
    sac (slice and classify) slices the input image, feed each piece to the
    caffe net object, and then stitch the output back together to an output image

    Takes
    im: input numpy array.
    net: Caffe net object.
    transformer: transformer object as defined above.
    scorelayer: string defining the name of the score layer.
    target_size: size of each slice.
    padcolor: the RGB values used when padding the image.
    startlayer: string defining the name of first convolutional layer.

    Gives
    (est, scores) tuple, where est is an integer image of the same size as the input, and scores is a multi-layer image encoding the score of each class in each layer.

    """

    input_size = im.shape[:2]
    (imlist, ncells) = bmt.slice_image(im, target_size = target_size, padcolor = padcolor)
    imcounter = -1
    for row in range(ncells[0]):
        for col in range(ncells[1]):
            imcounter += 1
            net.blobs['data'].data[...] = transformer.preprocess(imlist[imcounter])
            net.forward(start = startlayer)
            scores_slice = np.float32(np.squeeze(net.blobs[scorelayer].data.transpose(2, 3, 1, 0)))
            gc.collect()
            if col == 0:
                scores_row = copy(scores_slice)
            else:
                scores_row = np.concatenate((scores_row, scores_slice), axis = 1) # Build one row (along the columns)
        if row == 0:
            scores = scores_row
        else:
            scores = np.concatenate((scores, scores_row), axis = 0) # Concatenate the rows
    scores = scores[:input_size[0], :input_size[1], :] # Crop away the padding.
    est = np.argmax(scores, axis = 2) # For convenience, get the predictions.
    return (est, scores)

def classify_imlist(im_list, net, Transformer, batch_size, scorelayer = 'score', startlayer = 'conv1_1'):
    """
    classify_imlist classifies a list of images and returns estimated labels and scores. Only support classification nets (not FCNs).

    Takes
    im_list: list of images to classify (each stored as a numpy array).
    net: caffe net object
    Transformer: transformer object as defined above.
    batch_size: batch size for the net.
    scorelayer: name of the score layer.
    startlayer: name of first convolutional layer.
    """

    nbatches = int(math.ceil(float(len(im_list)) / batch_size))
    scorelist = []
    pos = -1
    for b in range(nbatches):
        for i in range(batch_size):
            pos += 1
            if pos < len(im_list):
                net.blobs['data'].data[i, :, :, :] = Transformer.preprocess(im_list[pos])
        net.forward(start = startlayer)
        scorelist.extend(list(copy(net.blobs[scorelayer].data).astype(np.float)))
        
    scorelist = scorelist[:len(im_list)]
    estlist = [np.argmax(s) for s in scorelist]
    
    return(estlist, scorelist)


def classify_from_patchlist(imlist, imdict, pyparams, net, scorelayer = 'score', startlayer = 'conv1_1'):

    estlist, scorelist, gtlist = [], [], []
    transformer = Transformer(pyparams['im_mean'])
    for imname in imlist:
        
        patchlist = []
        (point_anns, height_cm) = imdict[os.path.basename(imname)]

        # Load image
        im = np.asarray(Image.open(imname))
        (im, scale) = coral_image_resize(im, pyparams['scaling_method'], pyparams['scaling_factor'], height_cm) #resize.

        # Pad the boundaries
        im = np.pad(im, ((pyparams['crop_size']*2, pyparams['crop_size']*2),(pyparams['crop_size']*2, pyparams['crop_size']*2), (0, 0)), mode='reflect') 
        
        # Extract patches
        for (row, col, label) in point_anns:
            center_org = np.asarray([row, col])
            center = np.round(pyparams['crop_size']*2 + center_org * scale).astype(np.int)
            patchlist.append(crop_and_rotate(im, center, pyparams['crop_size'], 0, tile = False))
            gtlist.append(label)

        # Classify and append
        [this_estlist, this_scorelist] = classify_imlist(patchlist, net, transformer, pyparams['batch_size'], scorelayer = scorelayer, startlayer = startlayer)
        estlist.extend(this_estlist)
        scorelist.extend(this_scorelist)
        
    return (gtlist, estlist, scorelist)

def classify_from_imlist(im_list, net, transformer, batch_size, scorelayer = 'score', startlayer = 'conv1_1'):
    """
    classify_from_imlist classifies a list of images and returns estimated labels and scores. Only support classification nets (not FCNs).

    Takes
    im_list: list of images to classify (each stored as a numpy array).
    net: caffe net object
    transformer: transformer object as defined above.
    batch_size: batch size for the net.
    scorelayer: name of the score layer.
    startlayer: name of first convolutional layer.
    """

    scorelist = []
    for b in range(len(im_list) / batch_size + 1):
        for i in range(batch_size):
            pos = b * batch_size + i
            if pos < len(im_list):
                net.blobs['data'].data[i, :, :, :] = transformer.preprocess(im_list[pos])
        net.forward(start = startlayer)
        scorelist.extend(list(copy(net.blobs[scorelayer].data).astype(np.float)))

    scorelist = scorelist[:len(im_list)]
    estlist = [np.argmax(s) for s in scorelist]  
    
    return estlist, scorelist

def classify_from_patchlist_wrapper(imlist, imdict, pyparams, workdir, scorelayer = 'score', startlayer = 'conv1_1', net_prototxt = 'testnet.prototxt', gpuid = 0, snapshot_prefix = 'snapshot', save = False, caffemodel = None):

    # Preliminaries    
    if not caffemodel:
        caffemodel = find_latest_caffemodel(workdir, snapshot_prefix = snapshot_prefix)
    net = load_model(workdir, caffemodel, gpuid = gpuid, net_prototxt = net_prototxt)
    
    print "Classifying {} images in {} using {}".format(len(imlist), workdir, caffemodel)
    (gtlist, estlist, scorelist) = classify_from_patchlist(imlist, imdict, pyparams, net, scorelayer = scorelayer, startlayer = startlayer)
       
    if (save):
        pickle.dump((gtlist, estlist, scorelist), open(os.path.join(workdir, 'predictions_using_' + caffemodel +  '.p'), 'wb'))
    return (gtlist, estlist, scorelist)

def patchlist_cycle_runs(workdir, gpuid, ncycles, nbr_iters, imdictpath = '../imdict.json', testlistpath = '../testlist.txt', scorelayer = 'score'):
    """
    This is a custom wrapper for the patchlist_wrapper that relies heavily on a certain file structure.
    """
    imlist = [line.strip() for line in open(os.path.join(workdir, testlistpath))] 
    pyparams = pload(os.path.join(workdir, 'pyparams.pkl'))
    with open(os.path.join(workdir, imdictpath)) as s:
        imdict = json.load(s)
    for cycle in range(ncycles):
        run(workdir, gpuid = gpuid, nbr_iters = nbr_iters)
        classify_from_patchlist_wrapper(imlist, imdict, pyparams, workdir, scorelayer = scorelayer, gpuid = gpuid, save = True)
    

def find_latest_caffemodel(workdir, snapshot_prefix = 'snapshot'):
    
    caffemodels = glob.glob("{}*.caffemodel".format(os.path.join(workdir, snapshot_prefix)))
    if caffemodels:
        _iter = [int(f[f.index('iter_')+5:f.index('.caffemodel')]) for f in caffemodels]
        return os.path.basename(caffemodels[np.argmax(_iter)])
    else:
        print "Can't find a trained model in " + workdir + " using prefix: " + snapshot_prefix + "."
        return None


def calculate_image_mean(imlist): 
    """
    Returns mean channel intensity across the images in imlist.
    NOTE: returns mean in in BGR order.
    """
    mean = np.zeros(3).astype(np.float32)
    for imname in imlist:
        im = np.asarray(Image.open(imname))
        if len(im.shape) == 2:
            im = np.mean(im, axis = 0)
            im = np.mean(im, axis = 0)
            mean = mean + [im, im, im]
        else:   
            im = im[:, :, ::-1] #change to BGR
            im = np.mean(im, axis = 0)
            im = np.mean(im, axis = 0)
            mean = mean + im
    mean /= len(imlist)
    print mean
    return mean


def clean_workdirs(workdirs):
    for workdir in workdirs:
        for file_ in glob.glob(os.path.join(workdir, 'snapshot*')):
            if os.path.isfile(file_):
                os.remove(file_)
        for file_ in glob.glob(os.path.join(workdir, 'predictions_*')):
            if os.path.isfile(file_):
                os.remove(file_)
        for file_ in glob.glob(os.path.join(workdir, '*.log')):
            if os.path.isfile(file_):
                os.remove(file_)
    for file_ in glob.glob(os.path.join(workdir, '*.testlog')):
            if os.path.isfile(file_):
                os.remove(file_)


def vgg_core(n, nclasses, learn = True, scorelayer_name = 'score', acclayer = False, softmax = False, scorelayer = True):
    # first input 'n', must have layer member: n.data

    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, learn = learn)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, learn = learn)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, learn = learn)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, learn = learn)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, learn = learn)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, learn = learn)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, learn = learn)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, learn = learn)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, learn = learn)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, learn = learn)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, learn = learn)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, learn = learn)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, learn = learn)
    n.pool5 = max_pool(n.relu5_3)

    if learn:
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)]


    n.fc6 = L.InnerProduct(n.pool5, num_output=4096,
        param = param)

    n.relu6 = L.ReLU(n.fc6, in_place=True)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7 = L.InnerProduct(n.fc6, num_output=4096,
        param = param)

    n.relu7 = L.ReLU(n.fc7, in_place=True)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    if scorelayer:
        n[scorelayer_name] = L.InnerProduct(n.fc7, num_output=nclasses,
            param=[dict(lr_mult=5, decay_mult=1), dict(lr_mult=10, decay_mult=0)]) #always learn this layer. Else it's no fun!

    if softmax:
        n.loss = L.SoftmaxWithLoss(n[scorelayer_name], n.label)

    if acclayer:
        n.accuracy = L.Accuracy(n[scorelayer_name], n.label)

    return n


def vgg(pydata_params, data_layer, nclasses, ntop = 2, acclayer = False, learn = True, scorelayer_name = 'score'):
#NOTE: I am changing the dir so the machine can import hte data_layers module. I need to change this, but for now it is to test whether it works

    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'beijbom_caffe_data_layers', layer = data_layer,
            ntop=ntop, param_str=str(pydata_params))

    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, learn = learn)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, learn = learn)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, learn = learn)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, learn = learn)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, learn = learn)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, learn = learn)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, learn = learn)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, learn = learn)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, learn = learn)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, learn = learn)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, learn = learn)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, learn = learn)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, learn = learn)
    n.pool5 = max_pool(n.relu5_3)

    if learn:
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)]


    n.fc6 = L.InnerProduct(n.pool5, num_output=4096,
        param = param)

    n.relu6 = L.ReLU(n.fc6, in_place=True)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7 = L.InnerProduct(n.fc6, num_output=4096,
        param = param)

    n.relu7 = L.ReLU(n.fc7, in_place=True)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n[scorelayer_name] = L.InnerProduct(n.fc7, num_output=nclasses,
        param=[dict(lr_mult=5, decay_mult=1), dict(lr_mult=10, decay_mult=0)]) #always learn this layer. Else it's no fun!

    n.loss = L.SoftmaxWithLoss(n[scorelayer_name], n.label)
    
    if acclayer:
        n.accuracy = L.Accuracy(n[scorelayer_name], n.label)
    return n

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, learn=True):
    if learn:
        param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]

    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, param=param)
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def conv_bn(bottom, nout, ks = 3, stride=1, pad = 0, learn = True):
    if learn:
        param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, param = param, weight_filler=dict(type="msra"), bias_filler=dict(type="constant"))
    bn = L.BatchNorm(conv)
    lrn = L.LRN(bn)
    return conv, bn, lrn


def residual_standard_unit(n, nout, s, newdepth = False):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    bottom = n.__dict__['tops'][n.__dict__['tops'].keys()[-1]] #find the last layer in netspec
    stride = 2 if newdepth else 1

    n[s + 'conv1'], n[s + 'bn1'], n[s + 'lrn1'] = conv_bn(bottom, ks = 3, stride = stride, nout = nout, pad = 1)
    n[s + 'relu1'] = L.ReLU(s + 'lrn1', in_place=True)
    n[s + 'conv2'], n[s + 'bn2'], n[s + 'lrn2'] = conv_bn(s + 'relu1', ks = 3, stride = 1, nout = nout, pad = 1)
   
    if newdepth: 
        n[s + 'conv_expand'], n[s + 'bn_expand'], n[s + 'lrn_expand'] = conv_bn(bottom, ks = 1, stride = 2, nout = nout, pad = 0)
        n[s + 'sum'] = L.Eltwise(s + 'lrn2', s + 'lrn_expand')
    else:
        n[s + 'sum'] = L.Eltwise(s + 'lrn2', bottom)

    n[s + 'relu2'] = L.ReLU(s + 'sum', in_place=True)
    

def residual_bottleneck_unit(n, nout, s, newdepth = False):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    
    bottom = n.__dict__['tops'].keys()[-1] #find the last layer in netspec
    stride = 2 if newdepth and nout > 64 else 1

    n[s + 'conv1'], n[s + 'bn1'], n[s + 'lrn1'] = conv_bn(n[bottom], ks = 1, stride = stride, nout = nout, pad = 0)
    n[s + 'relu1'] = L.ReLU(n[s + 'lrn1'], in_place=True)
    n[s + 'conv2'], n[s + 'bn2'], n[s + 'lrn2'] = conv_bn(n[s + 'relu1'], ks = 3, stride = 1, nout = nout, pad = 1)
    n[s + 'relu2'] = L.ReLU(n[s + 'lrn2'], in_place=True)
    n[s + 'conv3'], n[s + 'bn3'], n[s + 'lrn3'] = conv_bn(n[s + 'relu2'], ks = 1, stride = 1, nout = nout * 4, pad = 0)
   
    if newdepth: 
        n[s + 'conv_expand'], n[s + 'bn_expand'], n[s + 'lrn_expand'] = conv_bn(n[bottom], ks = 1, stride = stride, nout = nout * 4, pad = 0)
        n[s + 'sum'] = L.Eltwise(n[s + 'lrn3'], n[s + 'lrn_expand'])
    else:
        n[s + 'sum'] = L.Eltwise(n[s + 'lrn3'], n[bottom])

    n[s + 'relu3'] = L.ReLU(n[s + 'sum'], in_place=True)

def residual_net(total_depth, data_layer_params, num_classes = 1000, acclayer = True):
    """
    Generates nets from "Deep Residual Learning for Image Recognition". Nets follow architectures outlined in Table 1. 
    """
    # figure out network structure
    net_defs = {
        18:([2, 2, 2, 2], "standard"),
        34:([3, 4, 6, 3], "standard"),
        50:([3, 4, 6, 3], "bottleneck"),
        101:([3, 4, 23, 3], "bottleneck"),
        152:([3, 8, 36, 3], "bottleneck"),
    }
    assert total_depth in net_defs.keys(), "net of depth:{} not defined".format(total_depth)

    nunits_list, unit_type = net_defs[total_depth] # nunits_list a list of integers indicating the number of layers in each depth.
    nouts = [64, 128, 256, 512] # same for all nets

    # setup the first couple of layers
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'beijbom_caffe_data_layers', layer = 'ImageNetDataLayer',
                ntop = 2, param_str=str(data_layer_params))
    n.conv1, n.bn1, n.lrn1 = conv_bn(n.data, ks = 7, stride = 2, nout = 64, pad = 3)
    n.relu1 = L.ReLU(n.lrn1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, stride = 2, kernel_size = 3)
    
    # make the convolutional body
    for nout, nunits in zip(nouts, nunits_list): # for each depth and nunits
        for unit in range(1, nunits + 1): # for each unit. Enumerate from 1.
            s = str(nout) + '_' + str(unit) + '_' # layer name prefix
            if unit_type == "standard":
                residual_standard_unit(n, nout, s, newdepth = unit is 1 and nout > 64)
            else:
                residual_bottleneck_unit(n, nout, s, newdepth = unit is 1)
                
    # add the end layers                    
    n.global_pool = L.Pooling(n.__dict__['tops'][n.__dict__['tops'].keys()[-1]], pooling_param = dict(pool = 1, global_pooling = True))
    n.score = L.InnerProduct(n.global_pool, num_output = num_classes,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    if acclayer:
        n.accuracy = L.Accuracy(n.score, n.label)

    return n            

