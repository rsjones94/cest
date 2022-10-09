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
from matplotlib.colors import LightSource
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import scipy

import analyze as az
import preprocess as pp


target_folder = '/Users/skyjones/Desktop/hiv_processing/for_jared/'

data_folder = '/Users/skyjones/Desktop/hiv_processing/data/working/aim5/'


proc = os.path.join(data_folder, 'processed')
raw_folders = glob.glob(os.path.join(data_folder, 'raw_*/'))


proc_folders = glob.glob(os.path.join(proc, '*/'))


cest_patterns = ['_C1_', '_C4_', 'OCEST_APT_water', 'OCEST_NOE_water', 'OCEST_ZSPECTRUM_water']

for pf in proc_folders:
    
    pf_base = os.path.basename(os.path.normpath(pf))
    
    out_folder = os.path.join(target_folder, pf_base)
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    # first, grab the cest data. all of these should exit
    for cpat in cest_patterns:
        basename = f'{pf_base}{cpat}.nii.gz'
        fromname = os.path.join(pf, basename)
        toname = os.path.join(out_folder, basename)
        shutil.copyfile(fromname, toname)
        
    # now grab the raw dicoms. some of these may not exist
    dicom_folder = os.path.join(out_folder, 'dicom')
    
    if not os.path.exists(dicom_folder):
        os.mkdir(dicom_folder)
    for raw_f in raw_folders:
        folderbase_d = os.path.basename(os.path.normpath(raw_f))
        
        folderbase_pattern = folderbase_d.replace('raw_', '')
        basename_d = f"{pf_base.replace('cestdixon', folderbase_pattern)}.DCM"
        
        fromname_d = os.path.join(raw_f, basename_d)
        toname_d = os.path.join(dicom_folder, basename_d)
        
        try:
            shutil.copyfile(fromname_d, toname_d)
        except FileNotFoundError:
            pass