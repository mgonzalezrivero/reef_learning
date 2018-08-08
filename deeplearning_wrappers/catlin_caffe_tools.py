import csv
import json
import re
import glob
import os
import os.path as osp
import reef_learning.toolbox.beijbom_caffe_tools  as bct
import reef_learning.toolbox.beijbom_misc_tools as bmt
from reef_learning.toolbox.beijbom_misc_tools import psave, pload
import reef_learning.deeplearning_wrappers.catlin_tools as ct

def write_imlist(datadir, listpath, imlist):
    with open(listpath, 'w') as f:
        for img in imlist:
            f.write(osp.join(datadir, img)+'\n')


def setup_data(basedir, workdir, r):
    # WRITE TRAIN AND VALDATA

    labelset=ct.get_labelset(basedir,r)


    lines = [line.rstrip() for line in open(basedir+r+'/train/img_file.csv')][1:]
    valimgs = lines[::5]
    trainimgs = list(set(lines) - set(valimgs))

    write_imlist(basedir+r+'/train/images', osp.join(workdir, 'trainlist.txt'), trainimgs)    
    write_imlist(basedir+r+'/train/images', osp.join(workdir, 'vallist.txt'), valimgs)    

    imdict = coralnet_export_to_imdict(basedir+r+'/train/annotations.csv', labelset)
    with open(os.path.join(workdir, 'traindict.json'), 'w') as outfile:
        json.dump(imdict, outfile)

    with open(os.path.join(workdir, 'valdict.json'), 'w') as outfile:
        json.dump(imdict, outfile)
        
    # WRITE TESTDATA
    testimgs = [line.rstrip() for line in open(basedir+r+'/test/img_file.csv')][1:]
    write_imlist(basedir+r+'/test/images', osp.join(workdir, 'testlist.txt'), testimgs)    

    imdict = coralnet_export_to_imdict(basedir+r+'/test/annotations.csv', labelset)
    with open(os.path.join(workdir, 'testdict.json'), 'w') as outfile:
        json.dump(imdict, outfile)

    im_mean = bct.calculate_image_mean([osp.join(basedir+r+'/train/images', trainimg) for trainimg in trainimgs[::10]])

    return im_mean, labelset


def coralnet_export_to_imdict(csvfile_path, labelset):
    imdict = {}
    with open(csvfile_path, 'rb') as csvfile:
        lines = csv.reader(csvfile, delimiter = ',', quotechar='"')
        next(lines)
        for parts in lines:
            filename = parts[9].strip('"')
            if not filename in imdict.keys():
                imdict[filename] = ([], 100)
            labelstr = parts[7].strip('"').strip(' ')
            imdict[filename][0].append( ( int(parts[4]), int(parts[5]), labelset.index(labelstr) ) )
    return imdict


def write_net(workdir, im_mean, nbr_classes, scaling_method = 'scale', scaling_factor = 1.0, cropsize=224, onlytrain=False): 
    if (onlytrain):
        splits=['train','val']
    else:
        splits=['train', 'val', 'test']

    for split in splits: 
     
        pyparams = dict( 
            batch_size = 45,  
            imlistfile = osp.join(workdir, split + 'list.txt'),  
            imdictfile = osp.join(workdir, split + 'dict.json'), 
            imgs_per_batch = 5, 
            crop_size = cropsize,  
            scaling_method = scaling_method, 
            scaling_factor = scaling_factor, 
            rand_offset = 1, 
            im_mean = list(im_mean)) 
     
        psave(pyparams, osp.join(workdir, split + 'pyparams.pkl')) 

        with open(osp.join(workdir, split + 'net.prototxt'), 'w') as f: 
            n = bct.vgg(pyparams, 'RandomPointDataLayer', nbr_classes) 
            f.write(str(n.to_proto())) 
 
def write_solver(workdir, lr='0.0001', lrp='"fixed"'): 
    """ 
    Defines the standard solver used throughout. 
    """ 
    solver = bct.CaffeSolver() 
    solver.sp['base_lr'] = lr
    solver.sp['test_interval'] = '60000' 
    solver.sp['lr_policy'] = lrp
    solver.write(osp.join(workdir, 'solver.prototxt')) 



def catlin_run(workdir, cycles, cyclesize, gpuid = 0, max_testimages = 100000000000):
    """
    Default run script including valset.
    """

    # load val-data for evaluation
    pyparams = pload(osp.join(workdir, 'valpyparams.pkl'))
    imlist = [line.rstrip() for line in open(osp.join(workdir, 'vallist.txt'))][:max_testimages]
    with open(osp.join(workdir, 'valdict.json')) as f:
        imdict = json.load(f)

    # run
    for i in range(cycles):
        bct.run(workdir, gpuid = gpuid, nbr_iters = cyclesize)
        _ = bct.classify_from_patchlist_wrapper(imlist, imdict, pyparams, workdir, gpuid = gpuid, save = True, net_prototxt = 'valnet.prototxt')

    # find and load optimal model
    bestiter, _ = find_best_iter(workdir)
    caffemodel = 'snapshot_iter_{}.caffemodel'.format(bestiter)
    net = bct.load_model(workdir, caffemodel, gpuid = gpuid, net_prototxt = 'testnet.prototxt')

    # load test-data
    pyparams = pload(osp.join(workdir, 'testpyparams.pkl'))
    imlist = [line.rstrip() for line in open(osp.join(workdir, 'testlist.txt'))][:max_testimages]
    with open(osp.join(workdir, 'testdict.json')) as f:
        imdict = json.load(f)

    # run on test-data
    (gtlist, estlist, scorelist) = bct.classify_from_patchlist(imlist, imdict, pyparams, net)
    psave((gtlist, estlist, scorelist), osp.join(workdir, 'predictions_on_test.p'))

def find_best_iter(workdir):
    testtoken = 'predictions_using_snapshot_iter_*.caffemodel.p'
    iterlist = []
    for testname in glob.glob(osp.join(workdir, testtoken)):
        [gtlist, estlist, scorelist] = pload(osp.join(workdir, testname))
        acc = bmt.acc(gtlist, estlist)
        m = re.search('iter_([0-9]*).caffemodel.p', testname)
        iter_ = int(m.group(1))
        iterlist.append((iter_, acc))
    iterlist = sorted(iterlist)
    bestacc = -1
    bestiter = []
    for iter_, acc in iterlist:
        if acc > bestacc:
            bestiter = iter_
            bestacc = acc
    return bestiter, iterlist
