#!/usr/bin/env python3

import os
import glob
import shutil
import datetime
import operator
import sys
import copy

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
from scipy import stats

# input
master_folder = '/Users/skyjones/Desktop/cest_processing/data/working/'
aim1_bulk_folder = '/Users/skyjones/Desktop/cest_processing/data/bulk/aim1/'
cest_spreadsheet = '/Users/skyjones/Desktop/cest_processing/data/bcrl_scan_ids.xlsx'

#output
data_out = '/Users/skyjones/Desktop/cest_processing/data/working/results_aim1.csv'
figure_folder = '/Users/skyjones/Desktop/cest_processing/data/working/figures/'

do_prep = True
do_figs = True
do_analysis = True

##########

if do_prep:
    cest_df = pd.read_excel(cest_spreadsheet, 'clean')
    cest_df['mri_id'] = cest_df['mri_id'].astype(str)
    
    out_df = pd.DataFrame()
    
    subs = glob.glob(os.path.join(master_folder, '*/'))
    subs = [sub for sub in subs if 'aim' in sub.lower()]
    
    for sub in subs:
        aim_folder = sub[:-1].split('/')[-1]
        
        if aim_folder != 'aim1':
            continue
        
        aim_figure_folder = os.path.join(figure_folder, aim_folder)
        if os.path.exists(aim_figure_folder):
            shutil.rmtree(aim_figure_folder)
        os.mkdir(aim_figure_folder)
        
        processed_folder = os.path.join(sub, 'processed')
        data_folders = glob.glob(os.path.join(processed_folder, '*/'))
        
        for fol in data_folders:
            base = os.path.basename(fol[:-1])
            splitter = base.split('_')
            cest_id = '_'.join(splitter[:-2])
            num_id = '_'.join(splitter[1:-2])
            arm = splitter[-1]
            
            # get the rows in the cest_df that correspond to this mri_id
            
            the_rows = cest_df[cest_df['mri_id'] == num_id]
            
            #if num_id != '146291':
            #    continue
            
            print(f'{aim_folder}: {num_id}, {arm}')
            
            
            
            water_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C1_.nii.gz')
            fw_inphase = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C2_.nii.gz')
            fw_outphase = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C3_.nii.gz')
            fat_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C4_.nii.gz')
            t2star_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C5_.nii.gz')
            b0_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C6_.nii.gz')
            apt_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_APT.nii.gz')
            noe_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_OPPAPT.nii.gz')
            zspec_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_ZSPECTRUM.nii.gz')
            
            analysis_folder = os.path.join(fol, 'analysis')
            if os.path.exists(analysis_folder):
                shutil.rmtree(analysis_folder)
            #os.mkdir(analysis_folder)
            
            id_figure_folder = os.path.join(aim_figure_folder, num_id)
            if not os.path.exists(id_figure_folder):
                os.mkdir(id_figure_folder)
            
            
            fat_image_loaded = nib.load(fat_image)
            fat_image_data = fat_image_loaded.get_fdata()
            fat_image_data = np.mean(fat_image_data, 3) # time average
            fat_image_voxelvol = np.product(fat_image_loaded.header['pixdim'][1:4])
            
            water_image_loaded = nib.load(water_image)
            water_image_data = water_image_loaded.get_fdata()
            water_image_data = np.mean(water_image_data, 3) # time average
            water_image_voxelvol = np.product(fat_image_loaded.header['pixdim'][1:4])
            
            t2star_image_loaded = nib.load(t2star_image)
            t2star_image_data = t2star_image_loaded.get_fdata()
            t2star_image_data = np.mean(t2star_image_data, 3) # time average
            
            apt_image_loaded = nib.load(apt_image)
            apt_image_data = apt_image_loaded.get_fdata()
            
            noe_image_loaded = nib.load(noe_image)
            noe_image_data = noe_image_loaded.get_fdata()
            
            zspec_image_loaded = nib.load(zspec_image)
            zspec_image_data = zspec_image_loaded.get_fdata() # no averaging - it's a spectrum
    
            # now see if we can find the mask
            
            masking_folder = os.path.join(aim1_bulk_folder, f'Donahue_{num_id}', 'masking')
            #the mask file will have the word "mask" in addition to "masking" in it
            globber = os.path.join(masking_folder, "*_mask.nii.gz")
            cands = [f for f in glob.glob(globber)]
            
            try:
                mask_image = cands[0]
            except IndexError:
                continue
                
            mask_image_loaded = nib.load(mask_image)
            mask_image_data = mask_image_loaded.get_fdata()
            
            mask_voxel_vol = np.product(mask_image_loaded.header.get_zooms())
            
            nodes = [int(i) for i in np.unique(mask_image_data) if not np.isclose(i,0)]            
                        
            # time to parse the zspec data
            
            #central_index = 55 # chem shift = 0ppm here. note in matlab code this index is 56, but python is 0-indexed
            # actually I think the central index is literally just the middle index
            central_index = int(np.ceil(zspec_image_data.shape[3] / 2)) - 1
            shift_step = 0.1 # change in chemical shift between each index
            n_steps = zspec_image_data.shape[3]
            
            start_shift = -(central_index * shift_step)
            end_shift = (n_steps - central_index) * shift_step
            
            shifts = np.arange(start_shift, end_shift - shift_step, shift_step)
            shifts = [round(i,1) for i in shifts] # round to nearest tenth. thx floating point imprecision
            
            #for idx in the_rows.index:
            #    cest_df.at[idx, f'chemical_shifts_ppm'] = str(shifts)
            
            bulk_means = []
            bulk_stds = []
            bulk_asyms = []
            bulk_difs = []
            for node in nodes:
                
                ser = pd.Series()
                
                n_vox = np.sum(mask_image_data == node)
                
                # calculate the zspec
                means = []
                stds = []
                for i, shift in enumerate(shifts):
                    boxed = zspec_image_data[:,:,:,i]
                    boxed_copy = boxed.copy()
                    boxed_copy[mask_image_data != node] = np.nan # mask it
                    
                    the_mean = np.nanmean(boxed_copy)
                    the_std = np.nanstd(boxed_copy)
                    
                    means.append(the_mean)
                    stds.append(the_std)
                    
                # now calculate the asymmetry
                # f(x) = [v(-x) - v(x)] / x
                asym = []
                for i, (shift, mean) in enumerate(zip(shifts, means)):
                    try:
                        neg_i = shifts.index(-shift)
                    except ValueError: # occurs when there is no symmetric shift val
                        asym.append(np.nan)
                        continue
                        
                    
                    asym_mean = means[neg_i]
                    try:
                        val = (mean - asym_mean) / shift
                    except ZeroDivisionError:
                        asym.append(np.nan)
                        continue
                        
                    asym.append(val)
                    
                # now calculate the equilibrium difference
                # f(x) = v_0 - v(x)
                dif = []
                for i, (shift, mean) in enumerate(zip(shifts, means)):
                    
                    dif_val = mean - means[0]
                    dif.append(dif_val)
                
                for c in the_rows.columns:
                    ser[c] = the_rows.iloc[0][c]
                
                ser[f'chemical_shifts_ppm'] = str(shifts)
                ser[f'signals_mean'] = str(means)
                ser[f'signals_std'] = str(stds)
                ser[f'signals_asym'] = str(asym)
                ser[f'signals_eqdif'] = str(dif)
                    
                if node > 0:
                    node_type = 'normal'
                elif node < 0:
                    node_type = 'malignant'
                else:
                    raise Exception('Node is 0')
                    
                ser[f'node_type'] = node_type
                ser[f'node_num'] = node
                ser[f'node_vol_mm3'] = mask_voxel_vol * n_vox
                
                bulk_means.append(means)
                bulk_stds.append(stds)
                bulk_asyms.append(asym)
                bulk_difs.append(dif)
                
                out_df = out_df.append(ser, ignore_index=True)
                
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 12))

                
                water_slice = water_image_data[:,:,5]
                axs[0][0].imshow(water_slice, cmap='gray')
                axs[0][0].set_title('water-weighted')   
                
                roi_slice = mask_image_data[:,:,5]
                axs[0][1].imshow(roi_slice, cmap='Accent')
                axs[0][1].set_title('lymph nodes')
                
                fat_slice = fat_image_data[:,:,5]
                axs[1][0].imshow(fat_slice, cmap='gray')
                axs[1][0].set_title('fat-weighted')
    
                apt_slice = apt_image_data[:,:,5]
                axs[1][1].imshow(apt_slice, cmap='inferno', interpolation='nearest')
                axs[1][1].set_title('APT')
                axs[1][1].imshow(roi_slice, cmap='Accent', alpha=0.5)
                                
                
                inspect_image_name = os.path.join(analysis_folder, id_figure_folder, f'{num_id}_roi_{arm}')
                
                plt.tight_layout()
                plt.savefig(inspect_image_name)
                plt.close(fig)
            
            
                
                 
        out_df.to_csv(data_out, index=False)
        

###### figs
if do_figs:
    print('figuring')
    in_df = pd.read_csv(data_out)
    
    normal_sub = in_df[in_df['node_type'] == 'normal']
    malignant_sub = in_df[in_df['node_type'] == 'malignant']
    
    target_aim_folder = os.path.join(figure_folder, f'aim1')
    zspec_comp_fig_name = os.path.join(target_aim_folder, f'zspec_comp_aim1.png')
    
    
    sig_types = ['signals_mean', 'signals_asym', 'signals_eqdif']
    sig_names = ['Mean signal', 'Signal asymmetry', 'Equilibrium difference']
    
    figure, axs = plt.subplots(nrows=len(sig_types), ncols=1, figsize=(8,16))
    
    ncolor = 'blue'
    mcolor = 'red'
    
    for sigt, sign, ax in zip(sig_types, sig_names, axs):
        
        
        shifts = [float(j) for j in in_df.iloc[0]['chemical_shifts_ppm'][1:-1].split(',')] # you better hope this column is consistent
        
        normal_nodes_all = np.array([np.array([float(j) for j in i[1:-1].split(',')]) for i in normal_sub[sigt].dropna()])
        malignant_nodes_all = np.array([np.array([float(j) for j in i[1:-1].split(',')]) for i in malignant_sub[sigt].dropna()])

        normal_signals = normal_nodes_all.mean(0)
        malignant_signals = malignant_nodes_all.mean(0)
        
        ax.set_title(sign)
        
        if len(normal_nodes_all) > 0:
            for nn in normal_nodes_all:
                ax.plot(shifts, nn, color=ncolor, alpha=1/len(normal_nodes_all)*3)
        
        if len(malignant_nodes_all) > 0:
            for mn in malignant_nodes_all:
                ax.plot(shifts, mn, color=mcolor, alpha=1/len(malignant_nodes_all)*3)
        
        if len(normal_nodes_all) > 0:
            ax.plot(shifts, normal_signals, color=ncolor, label='Normal nodes')
        if len(malignant_nodes_all) > 0:
            ax.plot(shifts, malignant_signals, color=mcolor, label='Cancerous nodes')
        
        ax.legend()
        ax.set_xlabel('Chemical shift (ppm)')
        ax.set_ylabel('Signal (a.u.)')
        
    plt.tight_layout()
    plt.savefig(zspec_comp_fig_name)
    
    
if do_analysis:
    print('Analyzing')    
    in_df = pd.read_csv(data_out)
    
    normal_sub = in_df[in_df['node_type'] == 'normal']
    malignant_sub = in_df[in_df['node_type'] == 'malignant']
    
    
    
    
        
        
    
    
    