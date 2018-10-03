import itertools
import scipy
import scikits.bootstrap as bootstrap
import reef_learning.deeplearning_wrappers.catlin_tools as ct
import pandas as pd
import numpy as np
import os.path as osp
from reef_learning.toolbox.beijbom_misc_tools import pload


def MAE_plot(basedir,region,sample_unit_file, labelset_map_file,selected_net, Split='test'):
    
    # load reference files
    sunits=pd.DataFrame.from_csv(osp.join(basedir,region,sample_unit_file))
    lsmap=pd.DataFrame.from_csv(osp.join(basedir,region,labelset_map_file)).reset_index()

    #Get image list from test split
    labelset=ct.get_labelset(basedir,region)
    imlist, imdict = ct.load_data(basedir,Split,labelset,region)
    imlist=list(itertools.chain.from_iterable(itertools.repeat(x, 50) for x in imlist))
    
    
    #Use selected Net to extract predictions
    selNet=osp.dirname(''.join(selected_net))
    (gtlist, estlist, scorelist) = pload(selNet+'/predictions_on_test.p')
    
    #Create merged and summarised dataframe
    df=pd.DataFrame([map(osp.basename, imlist),
                     [labelset[i] for i in gtlist], [labelset[i] for i in estlist]]).T
    df.columns=['image','observed','predicted']
    df=df.merge(sunits,left_on='image',right_on='image', how='left')
    df=pd.melt(df, id_vars=['sampleunit','image'], 
               value_vars=['observed','predicted'],
               var_name='method', value_name='label')
    df=df.groupby(['sampleunit','method','label']).size().reset_index(name='count')
    df=df.groupby(['sampleunit','method','label']).agg({'count': 'sum'})
    df=df.groupby(level=['sampleunit','method']).apply(lambda x: 100 * x / float(x.sum())).reset_index()
    df=df.merge(lsmap, on='label', how='left')

    df=df.groupby(['sampleunit','method','tier3_name'])['count'].agg({'count':np.sum}).reset_index()
    df=df.rename(index=str, columns={"tier3_name": "label"})
    df=df.pivot_table(index=['sampleunit','label'], columns='method',
                      values='count').reset_index().fillna(value=0)

    df['error']=abs(df['observed']-df['predicted'])
    df=df.groupby('label')['error'].agg({'mean': np.mean, 
                                     'std': np.std, 
                                     'cilow': lambda x: bootstrap.ci(x, statfunction=scipy.mean)[0],
                                     'cimax':lambda x: bootstrap.ci(x, statfunction=scipy.mean)[1],
                                   }).reset_index()
    
    #Plot Mean Absolute Error as the absolute diference between machine predictions and manual observations from test images.
    cierror=[df['cilow'],df['cimax']]
    plot = df.plot(kind='bar',
               y='mean',
               x='label',
               yerr=cierror,
               color='DarkGreen',
               edgecolor='black',
               grid=False,
               figsize=(8,2),
               position=0.45,
               error_kw=dict(ecolor='black',elinewidth=0.5),
               width=0.8,
              legend=False,
               rot=90,
              fontsize=9)
    plot.set_xlabel('Labels', fontsize=12)
    plot.set_ylabel('Mean Absolute Error (%)', fontsize=12)
    plot.xaxis.set_tick_params('labelcenter')
