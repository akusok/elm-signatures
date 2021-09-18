#!/usr/bin/env python
# coding: utf-8
from __future__ import division
import warnings

import time
import os
import shutil
from random import shuffle

import io
import pkg_resources
import math as m
import numpy as np
from PIL import Image, ImageFile
from io import BytesIO
import pandas as pd
import pickle

import decimal
from fastparquet import write

import mxnet as mx
from joblib import Parallel, delayed


from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

Image.MAX_IMAGE_PIXELS = 1000000000  



import matplotlib.pyplot as plt
from mxnet import gluon, nd

#get_ipython().run_line_magic('matplotlib', 'inline')

# ## This part is for getting the 1024 features 



def _prepare(img):
    
    if img is None:
        return None
    
    img = img.resize((224, 224), resample=Image.LANCZOS)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    
    # Inception21k specific normalization
    img = img.astype(np.float32) - 117
    return img




class WIBC_processor(object):
    
    def __init__(self, bsize=20, gpu=True, verbose=False):
        sym, arg_params, aux_params = mx.model.load_checkpoint('./00_Inception21k/Inception21k', 9)
        all_layers = sym.get_internals()
        sym3 = all_layers['flatten_output']
        
        context = mx.gpu(0) if gpu else mx.cpu()
        
        # To get the 1024 features
        mod3 = mx.mod.Module(symbol=sym3, label_names=None, context=context)
        mod3.bind(for_training=False, data_shapes=[('data', (bsize,3,224,224))])
        mod3.set_params(arg_params, aux_params)
        
        self.bsize = bsize
        self.verbose = verbose
        self.mod3 = mod3

    
    def process(self, imglist, Nmax=1000000):

        feats = []           
        img_batch = []
        count = 0
        last_iter = False
                
        for fimg in imglist:
            #img = _prepare(fimg)
            if fimg is None:
                continue
                
            img_batch.append(fimg)
                
            count += 1
            if count >= Nmax:
                last_iter = True

            # process batch
            if len(img_batch) >= self.bsize or last_iter:
                data = np.vstack(img_batch)
                self.mod3.forward(Batch([mx.nd.array(data)]))
                feats.append(self.mod3.get_outputs()[0].asnumpy())
                img_batch = []

            if last_iter:
                break
        
        layer_features = np.vstack(feats) 
        
        return layer_features           



# # This part is for getting the features from the resized images

def prepare21k(img):
    
    if img is None:
        return None
    
    img = img.resize((224, 224), resample=Image.LANCZOS)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    
    # Inception21k specific normalization
    img = img.astype(np.float32) - 117
    return img


# In[6]:


from PIL import Image, ImageChops

def trim(filename):
    im = Image.open(filename)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0,0)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


# In[7]:


def white_background_thumbnail(path_to_image, thumbnail_size=(224,224)):
    background = Image.new('RGB', thumbnail_size, "white")    
    source_image = trim(path_to_image)
    source_image.thumbnail(thumbnail_size)
    (w, h) = source_image.size
    background.paste(source_image, (int((thumbnail_size[0] - w) / 2), int((thumbnail_size[1] - h) / 2 )))
    return background



# ## This part is for creating the sliding windows



def get_sliding_image_overlap(filename,N = 256,d=10,save_img=False):
    '''This functions returns a list of the coordinates for the sliding window
    including and overlap variable d between the windows'''

    img = Image.open(filename)
    xsize, ysize = img.size
    
    #print(xsize,ysize)
    
    # estimation of how many squares to take in each direction
    nxs = m.ceil(xsize/(N-d)) 
    nys = m.ceil(ysize/(N-d))
    
    #print(nxs,nys)

    
    x1 = [ix for ix in range(0,N*nxs,N-d) if ix < xsize] 
    y1 = [iy for iy in range(0,N*nys,N-d) if iy < ysize]

    x2 = [ix+N for ix in range(0,N*nxs,N-d) if ix+N < xsize] 
    y2 = [iy+N for iy in range(0,N*nys,N-d) if iy+N < ysize]
    
    x2.append(xsize)    
    y2.append(ysize)
    
    x1l = len(x1)
    x2l = len(x2)
    y1l = len(y1)
    y2l = len(y2)
    
    # IN principle x1,x2 and y1,y2 should have the same lenght but in the
    # case is not, just make them iqual
    
    if ((x1l != x2l) | (y1l != y2l)):
        if x1l < x2l:
            x1 = x1[:x1l]
            x2 = x2[:x1l]
        else:
            x1 = x1[:x2l]
            x2 = x2[:x2l]

        if y1l < y2l:
            y1 = y1[:y1l]
            y2 = y2[:y1l]
        else:
            y1 = y1[:y2l]
            y2 = y2[:y2l]
    
    x1x2 = [l for l in zip(x1,x2)]
    y1y2 = [l for l in zip(y1,y2)]
    
    sliding=[(xx1,yy1,xx2,yy2)  for yy1,yy2 in y1y2 for xx1,xx2 in x1x2]
        
    #shuffle(sliding)
    
    for a,b,c,d in sliding:
        yield _prepare(img.crop((a,b,c,d))),(a,b,c,d), (nxs, nys , len(x1), len(y1))
    



# In[10]:


def get_weapon_probs(filename,n=256,D=5,Nmax=100):
    '''Here we get the probability matrix for each snapshot, the coordinates and the segmented
       images in a list'''

    generator = get_sliding_image_overlap(filename,N=n,d=D,save_img=False)

    batch = 400
    tmp_list = []
    coor_list = []
    imgs_list = []
    features_list = []

    for i, elements in enumerate(generator):
        tmp_list.append(elements)
         
        new_batch = min(list(tmp_list[0][2]))

        imgs_list = [i[0] for i in tmp_list]
        
        if new_batch < batch:
            batch = new_batch
        
    
        if len(imgs_list) >= batch:
            coor_list.extend([i[1] for i in tmp_list])    
 
            wp = WIBC_processor(bsize=batch, verbose=True)
            features = wp.process(imgs_list)
            features_list.extend(features)
        
            imgs_list = []
            tmp_list = []
        
        if i > Nmax:
            print("Maximun reached {}".format(N))
            break
    
    filename = "data/randomforest1024_weapon.pkl"
    load_model = pickle.load(open(filename,'rb'))
    
    results = load_model.predict(features_list)
    probs = load_model.predict_proba(features_list)
    
    return probs[:, 1], coor_list


# In[11]:


def get_weapon_features(filename,n=256,D=5,Nmax=10000):
    '''Here we get the probability matrix for each snapshot, the coordinates and the segmented
       images in a list'''

    generator = get_sliding_image_overlap(filename,N=n,d=D,save_img=False)

    batch = 400
    tmp_list = []
    coor_list = []
    imgs_list = []
    features_list = []

    print("Processing file {}, window size {}, overlap of {} pixels.".format(filename.split('/')[-1],n,D))
    
    for i, elements in enumerate(generator):
                
        tmp_list.append(elements)
        imgs_list = [i[0] for i in tmp_list]
       
        new_batch_max = max(tmp_list[0][2][2],tmp_list[0][2][3])
        new_batch_min = min(tmp_list[0][2][2],tmp_list[0][2][3])
 
        #new_batch_max = max(list(tmp_list[0][2]))
        #new_batch_min = min(list(tmp_list[0][2]))
                
        if new_batch_max <= batch:
            batch = new_batch_max
        else:
            if new_batch_min <= batch:
                batch = new_batch_min
        
        #print(batch)
        if len(imgs_list) >= batch:
            coor_list.extend([i[1] for i in tmp_list])    
 
            wp = WIBC_processor(bsize=batch, verbose=True)
            features = wp.process(imgs_list)
            features_list.extend(features)
        
            imgs_list = []
            tmp_list = []
            
        #else:
        #    bad_batch = 1
        #    coor_list.extend([i[1] for i in tmp_list])    
        #    wp = WIBC_processor(bsize=bad_batch, verbose=True)
        #    features = wp.process(imgs_list)
        #    features_list.extend(features)
       # 
       #     imgs_list = []
       #     tmp_list = []

                
        if i > Nmax:
            print("Maximum number images reached: {}.".format(Nmax))
            break
            
        #print(batch)
    
    return features_list, coor_list



# # Some plot utilities


import cv2
import random

def get_draw_boxes(image_name,coord_list):
    img = cv2.imread(image_name)
    for elm in coord_list:
        xmin,ymin,xmax,ymax = elm
        r = lambda: random.randint(0,255)
        img_rec = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(r(),r(),r()),3)

    #Because cv2 uses BGR and PIL RGB.
    img2 = cv2.cvtColor(img_rec, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img2)



# In[13]:


def get_weapons_coordinates(probs_list,coordinates,threshold=0.5):
    weapons_coordinates = []
    for i,j in enumerate(probs_list):
        if j > threshold:
            weapons_coordinates.append(coordinates[i])
    return weapons_coordinates



# In[14]:


def get_all_boxes_4windows_size(filename, windows = [256],min_prob=0.2,overlap = 10):
    all_coordinates = []
    
    for window in windows:
        matrix_probs, coordinates, imgs = get_weapon_probs(filename,window,D=overlap)
        probs_list = matrix_probs #.flatten('F')
        weapons_coordinates = get_weapons_coordinates(probs_list,coordinates,threshold=min_prob)
        all_coordinates.extend(weapons_coordinates)
        
    return get_draw_boxes(filename,all_coordinates)





# In[15]:


def get_heatmap_plot(filename, window = 256, min_prob=0.2, overlap=10):
    matrix_probs, coordinates, imgs = get_weapon_probs(filename,window,D=overlap)
    probs_list = matrix_probs.flatten('F')
    return [(coord + (prob,)) for coord,prob in zip(coordinates,probs_list)]




# In[16]:


def get_draw_boxes_fromfile(imagefile, filename, Nmax):

    df = pd.read_csv(filename)
    df_filter = df.loc[df['prob']>Nmax]
    df_coord = df_filter[['xmin','ymin','xmax','ymax']]
    fcoordinates = [tuple(elm) for elm in df_coord.values]
    return get_draw_boxes(imagefile,fcoordinates)



# In[17]:


def get_draw_boxes_from_features_file(imagefile, filename, Wsize=256):

    df = pd.read_csv(filename)
    df_filter = df.loc[df['wsize']==Wsize]
    df_coord = df[['xmin','ymin','xmax','ymax']]
    fcoordinates = [tuple(elm) for elm in df_coord.values]    
    return get_draw_boxes(imagefile,fcoordinates)


# # Saving info into files or moving



# In[18]:


def get_snapshot_info_file(filename,wsizes=[256]):
    textname = filename.split('.')[0]
    
    with open('{}.txt'.format(textname),'a+') as f:
            f.write('#wsize,prob,xmin,ymin,xmax,ymax\n')
            for wsize in wsizes:
                prob, coor, _ = get_weapon_probs(filename,n=wsize)
                for i in zip(prob,coor):
                    f.write('{},{},{},{},{},{}\n'.format(wsize,i[0],i[1][0],i[1][1],i[1][2],i[1][3]))



# In[19]:


def get_automxnet(x):
    #if os.path.getsize(x) <= 20000:
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    #else:
    #    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"


# In[20]:


def get_snapshot_features_file(filename,wsizes=[256],overlap=5,WMax=100):
    
    get_automxnet(filename)
    
    textname = filename.split('.')[0]
    df_full = pd.DataFrame()

    for wsize in wsizes:  
        # The overlap is in pixels but is introduced in percentages, so here we change it to pixels
        overlap_pixels = int(0.01 * wsize * overlap)
        
        features = get_weapon_features(filename,n=wsize,D=overlap_pixels,Nmax=WMax)
        #print(features)
        df_coor = pd.DataFrame(np.array(features[1]), columns = ["xmin","ymin","xmax","ymax"])
                
        window_size = len(features[1])* [wsize]  
        df_coor.insert(loc=0, column='wsize', value=window_size)
        df_coor.insert(loc=1, column='overlap', value=overlap)

        df_features = pd.DataFrame(data=features[0])#[0:,0:]) 
                
        df_all = df_coor.join(df_features)
            
        df_full=df_full.append(df_all,ignore_index=True)
        
        #df_full.to_csv("{}_features.csv.gz".format(textname),compression='gzip')
        

    df_full.columns = df_full.columns.astype(str)
    df_full.to_parquet("{}_features.parquet.gz".format(textname),compression='gzip',engine='fastparquet')
    #write("{}_features.parq".format(textname),df_full,compression='GZIP')

# In[21]:


import shutil,glob

def get_move_files(source_dir,dest_dir,ext="*.csv"):
    files = glob.iglob(os.path.join(source_dir, ext))
    
    print("moving files with extension: {}\n".format(ext))
    for file in files:
        if os.path.isfile(file):
            shutil.move(file, dest_dir)




def get_restart_files(fpath, imglist_all, overlaps, windows):

    print("restarting feature extraction for {} folder".format(fpath.split('/')[-1]))
    
    rlist = [os.path.join(fpath, f) for f in os.listdir(fpath)  if f.endswith('.csv.gz')]
    
    for file in rlist:
        df = pd.read_csv(file,usecols=['wsize','%overlap'])
        wsizes = df['wsize']
        ovrlps = df['%overlap']
        
        if len(list(set(list(wsizes)))) is not len(windows):
            last_file = file
            print(file)
    
            if last_file:
                print('restarting from file: {}'.format(file))        
                rlist.pop(file)
                os.remove(file)
    
    sovrlp = list(set(list(ovrlps))) 
    rflist = [f.replace('_features.csv.gz','.png') for f in rlist]
    print('found {} features files'.format(len(rflist)))
    
    rimglist_all = [f for f in imglist_all if f not in set(rflist)]
    
    print('restarting for the remaining {} files'.format(len(imglist_all)-len(rflist)))

    roverlaps = overlaps[overlaps.index(ovrlps[0]):]
    print('restarting from overlap {}p. List overlaps {}'.format(sovrlp[0],roverlaps))    
    
    return  rimglist_all,roverlaps



### MAIN code


path = '/home/data/MyBackUP/Projects/ELM/02_Signatures/GPDSS10000'

folders = [imgdir for imgdir in os.listdir(path)]

restart_list = []
windows = [128, 256, 512] 

move_small = False
restart = False

for elm in folders:
    
    overlaps = [10, 25, 50, 90] # percentages 
   
    if elm in restart_list:
        restart = True
 
    fpath = os.path.join(path,elm)
    print("Folder path: {} ".format(fpath))
    imglist_all = [os.path.join(root, f) for root,_,files in os.walk(fpath) for f in files if f.endswith('.png')]
    
    if move_small is True:
        imglist_small = [f for f in imglist_all if f.endswith('.png') if os.path.getsize(f) < 10000]
        imglist = [f for f in imglist_all if f.endswith('.png') if os.path.getsize(f) >= 10000]
    
        print("{} images in total here".format(len(imglist_all)))
        print("skipping {} images (less than 10Kb)".format(len(imglist_small)))
        print("Processing {} images\n".format(len(imglist)))
    
        small_dir = os.path.join(fpath,'less10kb')
    
        if len(imglist_small) >0:
            if not os.path.isdir(small_dir):
                os.makedirs(small_dir)
    
            for small in imglist_small:
                try:
                    shutil.move(small, small_dir)
                except:
                    pass

    if restart is True:
        imglist_res , overlaps_res =  get_restart_files(fpath,imglist_all,overlaps, windows)
        overlaps = overlaps_res
        #restart = False
    else:    
        imglist = imglist_all

    
    for ovlp in overlaps:

        if restart is True:
            imglist = imglist_res
            restart = False   
        else:
            print('\nNow Using the whole list of images, ie restart = False')
            imglist = imglist_all

        print("getting features from snapshots of window sizes: {}. Overlap {}%".format(windows,ovlp))
        
        [get_snapshot_features_file(snapshot,wsizes=windows,overlap=ovlp,WMax=1000000) for snapshot in imglist]
        
        dest_dir = os.path.join(fpath,"overlap{}p".format(ovlp))
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        get_move_files(fpath,dest_dir,ext='*.gz')

print("done.\n\n")     






