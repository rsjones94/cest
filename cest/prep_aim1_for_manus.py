#!/usr/bin/env python3

import os
import glob
import shutil
import datetime
import operator
import sys
import copy
import glob

import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage import feature
from matplotlib import ticker
import cv2
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes
import scipy
from scipy.signal import butter, lfilter, freqz
from matplotlib.colors import LightSource
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import scipy


import preprocess as pp
import analyze as az

mpl.rcParams.update({'errorbar.capsize': 2})

# input
working_folder = '/Users/skyjones/Desktop/cest_processing/data/working/'
bulk_folder = '/Users/skyjones/Desktop/cest_processing/data/bulk/'

#output
output_folder = '/Users/skyjones/Desktop/cest_processing/data/aim1_for_manus/'



    
    
aim1_processed_folder = os.path.join(working_folder, 'aim1', 'processed')
data_folders = glob.glob(os.path.join(aim1_processed_folder, '*/'))

for f in data_folders:
    bn = os.path.basename(os.path.normpath(f))
    
    
    dest = os.path.join(output_folder, bn)
    shutil.copytree(f, dest)
    
    split = bn.split('_')
    mrid = '_'.join(split[:2])
    masking_folder = os.path.join(bulk_folder, 'aim1', mrid, 'masking')
    masking_dest = os.path.join(dest, 'masking')
    
    shutil.copytree(masking_folder, masking_dest)