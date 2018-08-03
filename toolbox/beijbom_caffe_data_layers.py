# standard python imports
import os.path
import json
import time
from random import shuffle
from threading import Thread
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import scipy.misc
import skimage.io

# own class imports
import caffe
from beijbom_misc_tools import crop_and_rotate, tile_image, coral_image_resize
from beijbom_caffe_tools import Transformer


# ==============================================================================
# ==============================================================================
# =========================== RANDOM POINT PATCH LAYER =========================
# ==============================================================================
# ==============================================================================

class RandomPointDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        params = eval(self.param_str)
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'imlistfile' in params.keys(), 'Params must include imlistfile.'
        assert 'imdictfile' in params.keys(), 'Params must include imdictfile.'
        assert 'imgs_per_batch' in params.keys(), 'Params must include imgs_per_batch.'
        assert 'crop_size' in params.keys(), 'Params must include crop_size.'
        assert 'scaling_method' in params.keys(), 'Params must include scaling_method'
        assert 'scaling_factor' in params.keys(), 'Params must include scaling_factor'
        assert 'im_mean' in params.keys(), 'Params must include im_mean.'
        assert 'rand_offset' in params.keys(), 'Params must include rand_offset.'
        
        self.batch_size = params['batch_size']

        # === Check some of the input variables
        imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        assert len(imlist) >= params['imgs_per_batch'], 'Image list must be longer than the number of images you ask for per batch.'
        assert params['scaling_method'] in ('ratio', 'scale')

        # === set up thread and batch advancer ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = PatchBatchAdvancer(self.thread_result, params)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, params['crop_size'], params['crop_size'])
        top[1].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        """ happens during setup """
        pass

    def forward(self, bottom, top):
        #print time.clock() - self.t0, "seconds since last call to forward."
        if self.thread is not None:
            self.join_worker()

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class PatchBatchAdvancer():
    """
    The PatchBatchAdvancer is a helper class to RandomPointDataLayer. It is called asychronosly and prepares the tops.
    """
    def __init__(self, result, params):
        self._cur = 0
        self.result = result
        self.params = params
        self.imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        with open(params['imdictfile']) as f:
            self.imdict = json.load(f)
        self.transformer = TransformerWrapper(params['im_mean'])
        shuffle(self.imlist)

        print "DataLayer initialized with {} images, {} imgs per batch, and {}x{} pixel patches".format(len(self.imlist), params['imgs_per_batch'], params['crop_size'], params['crop_size'])

    def __call__(self):
        t1 = timer()
        self.result['data'] = []
        self.result['label'] = []

        if self._cur + self.params['imgs_per_batch'] >= len(self.imlist):
            self._cur = 0
            shuffle(self.imlist)
        
        # Grab images names from imlist
        imnames = self.imlist[self._cur : self._cur + self.params['imgs_per_batch']]

        # Figure out how many patches to grab from each image
        patches_per_image = self.chunkify(self.params['batch_size'], self.params['imgs_per_batch'])

        # Make nice output string
        output_str = [str(npatches) + ' from ' + os.path.basename(imname) + '(id ' + str(itt) + ')' for imname, npatches, itt in zip(imnames, patches_per_image, range(self._cur, self._cur + self.params['imgs_per_batch']))]
        
        # Loop over each image
        for imname, npatches in zip(imnames, patches_per_image):
            self._cur += 1

            # randomly select the rotation angle for each patch             
            angles = np.random.choice(360, size = npatches, replace = True)

            # randomly select whether to flip this particular patch.
            flips = np.round(np.random.rand(npatches))*2-1

            # get random offsets
            rand_offsets = np.round(np.random.rand(npatches, 2) * (self.params['rand_offset'] * 2)  - self.params['rand_offset'])

            # Randomly permute the patch list for this image. Sampling is done with replacement 
            # so that if we ask for more patches than is available, it still computes.
            (point_anns, height_cm) = self.imdict[os.path.basename(imname)] # read point annotations and image height in centimeters.
            point_anns = [point_anns[pp] for pp in np.random.choice(len(point_anns), size = npatches, replace = True)]

            # Load image
            im = np.asarray(Image.open(imname))
            (im, scale) = coral_image_resize(im, self.params['scaling_method'], self.params['scaling_factor'], height_cm) #resize.

            # Pad the boundaries
            crop_size = self.params['crop_size'] #for convenience, store this value locally                        
            im = np.pad(im, ((crop_size * 2, crop_size * 2),(crop_size * 2, crop_size * 2), (0, 0)), mode='reflect')        
            for ((row, col, label), angle, flip, rand_offset) in zip(point_anns, angles, flips, rand_offsets):
                center_org = np.asarray([row, col])
                center = np.round(crop_size * 2 + center_org * scale + rand_offset).astype(np.int)
                patch = self.transformer(crop_and_rotate(im, center, crop_size, angle, tile = False))
                self.result['data'].append(patch[::flip, :, :])
                self.result['label'].append(label)

    def chunkify(self, k, n):
        """ 
        Returns a list of n integers, so that the sum of the n integers is k.
        The list is generated so that the n integers are as even as possible
        """
        lst = range(k)
        return [ len(lst[i::n]) for i in xrange(n) ]
        

class TransformerWrapper(Transformer):
    def __init__(self, mean):
        Transformer.__init__(self, mean)
    def __call__(self, im):
        return self.preprocess(im)



# ==============================================================================
# ==============================================================================
# =========================== IMAGENET LAYER ===================================
# ==============================================================================
# ==============================================================================

class ImageNetDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        params = eval(self.param_str)
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'imlistfile' in params.keys(), 'Params must include imlistfile.'
        assert 'imdictfile' in params.keys(), 'Params must include imdictfile.'
        assert 'crop_size' in params.keys(), 'Params must include crop_size.'
        assert 'im_mean' in params.keys(), 'Params must include im_mean.'
        
        self.batch_size = params['batch_size']

        # === set up thread and batch advancer ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = ImageNetPatchBatchAdvancer(self.thread_result, params)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, params['crop_size'], params['crop_size'])
        top[1].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        """ happens during setup """
        pass

    def forward(self, bottom, top):
        if self.thread is not None:
            self.join_worker()

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class ImageNetPatchBatchAdvancer():
    """
    The ImageNetPatchBatchAdvancer is a helper class to ImageNetDataLayer. It is called asychronosly and prepares the tops.
    """
    def __init__(self, result, params):
        self._cur = 0
        self.result = result
        self.params = params
        self.imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        with open(params['imdictfile']) as f:
            self.imdict = json.load(f)
        self.transformer = TransformerWrapper(params['im_mean'])
        shuffle(self.imlist)

        print "DataLayer initialized with {} images".format(len(self.imlist))

    def __call__(self):
        self.result['data'] = []
        self.result['label'] = []

        if self._cur + self.params['batch_size'] >= len(self.imlist):
            self._cur = 0
            shuffle(self.imlist)
        
        # Loop over each image
        for imname in self.imlist[self._cur : self._cur + self.params['batch_size']]:
            self._cur += 1

            im = Image.open(imname) # Load image
            im = im.convert("RGB") # make sure it's 3 channels
            im = self.scale_augment(im) # scale augmentation

			# random crop
            (width, height) = im.size
            left = np.random.choice(width - 224)
            upper = np.random.choice(height - 224)
            im = im.crop((left, upper, left + 224, upper + 224))
            im = np.asarray(im)
           
			# random flip 
            flip = np.random.choice(2)*2-1
            im = im[:, ::flip, :]
                
            self.result['data'].append(self.transformer(im)            )
            self.result['label'].append(self.imdict[os.path.basename(imname)])

    def scale_augment(self, im):
        (width, height) = im.size
        width, height = float(width), float(height)
        if width <= height:
            wh_ratio = height / width
            new_width = int(np.random.choice(480-256) + 256)
            im = im.resize((new_width, int(new_width * wh_ratio)))
        else:
            hw_ratio = width / height
            new_height = int(np.random.choice(480-256) + 256)
            im = im.resize((int(new_height * hw_ratio), new_height))
        return im
    
class TransformerWrapper(Transformer):
    def __init__(self, mean):
        Transformer.__init__(self, mean)
    def __call__(self, im):
        return self.preprocess(im)









# ==============================================================================
# ==============================================================================
# ============================== REGRESSION LAYER ==============================
# ==============================================================================
# ==============================================================================

class RandomPointRegressionDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # === Read input parameters ===
        params = eval(self.param_str)
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'imlistfile' in params.keys(), 'Params must include imlistfile.'
        assert 'imdictfile' in params.keys(), 'Params must include imdictfile.'
        assert 'im_scale' in params.keys(), 'Params must include im_scale.'
        assert 'im_mean' in params.keys(), 'Params must include im_mean.'

        self.t0 = 0
        self.t1 = 0
        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']
        self.nclasses = params['nclasses']
        imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        with open(params['imdictfile']) as f:
            imdict = json.load(f)

        transformer = TransformerWrapper()
        transformer.set_mean(params['im_mean'])
        transformer.set_scale(params['im_scale'])

        print "Setting up RandomPointRegressionDataLayer with batch size:{}".format(self.batch_size)

        # === set up thread and batch advancer ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = RegressionBatchAdvancer(self.thread_result, self.batch_size, imlist, imdict, transformer, self.nclasses, self.im_shape)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, self.nclasses)

    def reshape(self, bottom, top):
        """ happens during setup """
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, self.nclasses)
        #pass

    def forward(self, bottom, top):
        # print time.clock() - self.t0, "seconds since last call to forward."
        if self.thread is not None:
            self.t1 = timer()
            self.join_worker()
            # print "Waited ", timer() - self.t1, "seconds for join."

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 
        self.t0 = time.clock()
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class RegressionBatchAdvancer():
    """
    The RegressionBatchAdvancer is a helper class to RandomPointRegressionDataLayer. It is called asychronosly and prepares the tops.
    """
    def __init__(self, result, batch_size, imlist, imdict, transformer, nclasses, im_shape):
        self.result = result
        self.batch_size = batch_size
        self.imlist = imlist
        self.imdict = imdict
        self.transformer = transformer
        self._cur = 0
        self.nclasses = nclasses
        self.im_shape = im_shape
        shuffle(self.imlist)

        print "RegressionBatchAdvancer is initialized with {} images".format(len(imlist))

    def __call__(self):
        
        t0 = timer()
        self.result['data'] = []
        self.result['label'] = []

        if self._cur == len(self.imlist):
            self._cur = 0
            shuffle(self.imlist)
        
        imname = self.imlist[self._cur]

        # Load image
        im = np.asarray(Image.open(imname))
        im = scipy.misc.imresize(im, self.im_shape)
        point_anns = self.imdict[os.path.basename(imname)][0]

        class_hist = np.zeros(self.nclasses).astype(np.float32)
        for (row, col, label) in point_anns:
            class_hist[label] += 1
        class_hist /= len(point_anns)

                
        self.result['data'].append(self.transformer.preprocess(im))
        self.result['label'].append(class_hist)
        self._cur += 1
        # print "loaded image {} in {} secs.".format(self._cur, timer() - t0)


# ==============================================================================
# ==============================================================================
# ============================== MULTILABEL LAYER ==============================
# ==============================================================================
# ==============================================================================

class RandomPointMultiLabelDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        params = eval(self.param_str)
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'imlistfile' in params.keys(), 'Params must include imlistfile.'
        assert 'imdictfile' in params.keys(), 'Params must include imdictfile.'
        assert 'im_scale' in params.keys(), 'Params must include im_scale.'
        assert 'im_mean' in params.keys(), 'Params must include im_mean.'

        self.t0 = 0
        self.t1 = 0
        self.batch_size = params['batch_size']
        self.nclasses = params['nclasses']
        self.im_shape = params['im_shape']
        imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        with open(params['imdictfile']) as f:
            imdict = json.load(f)

        transformer = TransformerWrapper()
        transformer.set_mean(params['im_mean'])
        transformer.set_scale(params['im_scale'])

        print "Setting up RandomPointRegressionDataLayer with batch size:{}".format(self.batch_size)

        # === set up thread and batch advancer ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = MultiLabelBatchAdvancer(self.thread_result, self.batch_size, imlist, imdict, transformer, self.nclasses, self.im_shape)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, self.nclasses)

    def reshape(self, bottom, top):
        """ happens during setup """
        pass

    def forward(self, bottom, top):
        # print time.clock() - self.t0, "seconds since last call to forward."
        if self.thread is not None:
            self.t1 = timer()
            self.join_worker()
            # print "Waited ", timer() - self.t1, "seconds for join."

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 
        self.t0 = time.clock()
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class MultiLabelBatchAdvancer():
    """
    The MultiLabelBatchAdvancer is a helper class to RandomPointRegressionDataLayer. It is called asychronosly and prepares the tops.
    """
    def __init__(self, result, batch_size, imlist, imdict, transformer, nclasses, im_shape):
        self.result = result
        self.batch_size = batch_size
        self.imlist = imlist
        self.imdict = imdict
        self.transformer = transformer
        self._cur = 0
        self.nclasses = nclasses
        self.im_shape = im_shape
        shuffle(self.imlist)

        print "MultiLabelBatchAdvancer is initialized with {} images".format(len(imlist))

    def __call__(self):
        
        t0 = timer()
        self.result['data'] = []
        self.result['label'] = []

        if self._cur == len(self.imlist):
            self._cur = 0
            shuffle(self.imlist)
        
        imname = self.imlist[self._cur]

        # Load image
        im = np.asarray(Image.open(imname))
        im = scipy.misc.imresize(im, self.im_shape)
        point_anns = self.imdict[os.path.basename(imname)][0]

        class_in_image = np.zeros(self.nclasses).astype(np.float32)
        for (row, col, label) in point_anns:
            class_in_image[label] = 1

                
        self.result['data'].append(self.transformer.preprocess(im))
        self.result['label'].append(class_in_image)
        self._cur += 1
        # print "loaded image {} in {} secs.".format(self._cur, timer() - t0)
