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
master_folder = '/Users/skyjones/Desktop/cest_processing/data/working/'
bulk_folder = '/Users/skyjones/Desktop/cest_processing/data/bulk/'
cest_spreadsheet = '/Users/skyjones/Desktop/cest_processing/check.csv'

#output
figure_folder = '/Users/skyjones/Desktop/cest_processing/data/working/figures/'

do_prep = False
coord_restore = False

do_figs = True
bulk_figs = False



master_node_df_outname = os.path.join(master_folder, 'aim1_node_data.xlsx')
byperson_df_outname = os.path.join(master_folder, 'aim1_node_data_byperson.xlsx')

##########

'''
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 6
fs = 10    # sample rate, Hz
cutoff = 0.8 # desired cutoff frequency of the filter, Hz


b, a = butter_lowpass(cutoff, fs, order)

T = 8  # total sample time (or in our case, chemical shift range)
n = int(T * fs) # total number of samples

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(cleaned, cutoff, fs, order)

fig,axs=plt.subplots(2,1)

axs[0].plot(shi, cleaned, 'b-', label='data')
axs[0].plot(shi, y, 'g-', linewidth=2, label='filtered data')
axs[0].set_xlabel('Chemical shift')
axs[0].set_ylabel('Signal')
axs[0].legend()

pg = scipy.signal.periodogram(cleaned,fs=fs)

axs[1].plot(pg[0], pg[1])
axs[1].set_xlabel('Frequency')
axs[1].set_ylabel('Density')
axs[1].axvline(cutoff,color='red')


plt.tight_layout()

'''



def _lorentzian_mono(x, amp1, cen1, wid1, offset1):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) + offset1

#weight_types = ['fat', 'water']
weight_types = ['water']

if do_prep:
    cest_df = pd.read_csv(cest_spreadsheet)
    #cest_df['mri_id'] = cest_df['mri_id'].astype(str)
    
    subs = glob.glob(os.path.join(master_folder, '*/'))
    subs = [sub for sub in subs if 'aim' in sub.lower()]
    
    for sub in subs:
        
        master_node_df = pd.DataFrame()
        #master_node_df_flatweighted = pd.DataFrame()
        #master_node_df_volweighted = pd.DataFrame()
        
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
            
            #scan_id = '_'.join(splitter[:2])
            scan_id = cest_id
            
            
            
            # get the rows in the cest_df that correspond to this mri_id
            
            #the_rows = cest_df[cest_df['mri_id'] == num_id]
            the_rows = cest_df[cest_df['scan_id'] == scan_id]
            the_row = the_rows.iloc[0]
            
            is_control = the_row['n_nodes_metastatic'] == 0
            n_nodes_metastatic = the_row['n_nodes_metastatic']
            n_nodes_removed = the_row['n_nodes_removed']
            
            adj_ther_raw = int(the_row['adj_therapy'])
            neoadj_ther_raw = int(the_row['neoadj_therapy'])
            both_ther = adj_ther_raw and neoadj_ther_raw
            
            adj_ther = adj_ther_raw and not both_ther
            neoadj_ther = neoadj_ther_raw and not both_ther
            
            cancer_proportion = n_nodes_metastatic / n_nodes_removed
            #race = the_row['race']
            #sex = the_row['sex']
            #age = the_row['age']
            
            #if num_id != '146291':
            #    continue
            
            print(f'{aim_folder}: {num_id}, {arm}')
                        
            if coord_restore:
                overwrite_coords = False
                
                txtmarker = os.path.join(fol, 'coords_restored.txt')
                
                if not os.path.exists(txtmarker) or overwrite_coords:
                    # step 1: make sure we restore the coordinate info
                    subims = glob.glob(os.path.join(fol, '*.nii.gz'))
                    
                    
                    sourcer = os.path.join(sub, 'raw', f'{os.path.basename(os.path.normpath(fol))}.DCM')
                    pp.dcm_to_nii(sourcer)
                    
                    convd = sourcer.replace('DCM', 'nii.gz')
                    
                    if not os.path.exists(convd):
                        convd = convd.replace('.nii.gz', '_1.nii.gz')
                    
                    
                    for sim in subims:
                        pp.restore_coordspace_from_source(nif=sim,
                                                          source=convd) # source = sourcer
                        with open(txtmarker, 'w') as f:
                            f.write('')
                            
                    os.remove(convd)
                            
                else:
                    pass
                        
                        
            
            
            water_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C1_.nii.gz')
            fw_inphase = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C2_.nii.gz')
            fw_outphase = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C3_.nii.gz')
            fat_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C4_.nii.gz')
            t2star_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C5_.nii.gz')
            #b0_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C6_.nii.gz') # this is actually just the water-weighted b0
            
            node_image_globber = os.path.join(bulk_folder, aim_folder, cest_id, 'masking', f'*mask.nii.gz')
            
            node_image_candidates = glob.glob(node_image_globber)
            try:
                node_image = node_image_candidates[0]
            except IndexError:
                node_image = None
                print(f'\t\tNO NODE IMAGE')
                continue
            
            analysis_folder = os.path.join(fol, 'analysis')
            if os.path.exists(analysis_folder):
                shutil.rmtree(analysis_folder)
            #os.mkdir(analysis_folder)
            
            id_figure_folder = os.path.join(aim_figure_folder, num_id)
            if not os.path.exists(id_figure_folder):
                os.mkdir(id_figure_folder)
            
            
            fat_image_loaded = nib.load(fat_image)
            fat_image_data = fat_image_loaded.get_fdata()
            fat_image_data = np.mean(fat_image_data, 3) # zspec average
            fat_image_voxelvol = np.product(fat_image_loaded.header['pixdim'][1:4])
            
            water_image_loaded = nib.load(water_image)
            water_image_data = water_image_loaded.get_fdata()
            water_image_data = np.mean(water_image_data, 3) # zspec average
            water_image_voxelvol = np.product(fat_image_loaded.header['pixdim'][1:4])
            
            node_image_loaded = nib.load(node_image)
            node_image_data = node_image_loaded.get_fdata()
            node_image_voxelvol = np.product(node_image_loaded.header['pixdim'][1:4])
            
            node_image_data = node_image_data.astype(int)
            node_image_data = morphology.label(node_image_data) # comment this out if the image is prelabeled. but this line is needed if the mask is simple binary
            
            
            for weight_type in weight_types:
                print(f'\tWeighting: {weight_type}')

                
                apt_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_APT_{weight_type}.nii.gz')
                noe_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_NOE_{weight_type}.nii.gz')
                zspec_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_ZSPECTRUM_{weight_type}.nii.gz')
                s0_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_S0_{weight_type}.nii.gz')
                uncorrected_zspec_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_uncorrected_{weight_type}.nii.gz')
                
                apt_image_loaded = nib.load(apt_image)
                apt_image_data = apt_image_loaded.get_fdata()
                
                noe_image_loaded = nib.load(noe_image)
                noe_image_data = noe_image_loaded.get_fdata()
                
                zspec_image_loaded = nib.load(zspec_image)
                zspec_image_data = zspec_image_loaded.get_fdata()
                zspec_image_data[np.isclose(zspec_image_data,0)] = np.nan
                
                uncorrected_zspec_image_loaded = nib.load(uncorrected_zspec_image)
                uncorrected_zspec_image_data = uncorrected_zspec_image_loaded.get_fdata()
                #note that we don't need to set 0 pixels to nan in the uncorrected as the uncorrected image gives signal everywhere
                
                s0_image_loaded = nib.load(s0_image)
                s0_image_data = s0_image_loaded.get_fdata()
                
                s0_image_data = np.where(np.isclose(s0_image_data,0), np.nan, s0_image_data)
                
                zspec_image_data_norm = zspec_image_data.copy()
                for i in range(zspec_image_data_norm.shape[-1]):
                    zspec_image_data_norm[:,:,:,i] = zspec_image_data_norm[:,:,:,i] / s0_image_data
                
                apt_image_data_norm = apt_image_data / s0_image_data
                
                noe_image_data_norm = noe_image_data / s0_image_data
        

                node_nums = np.unique(node_image_data)
                
                '''
                what we want to do is save a df of zspec data for each node -- one df per participant
                then afterwards we will collate everything into a master csv with aggregate stats for each participant
                '''
                #individual_signals = []
                i#ndividual_apts = []
                #individual_noes = []
                #individual_vols = []
                
                for node_num in node_nums:
                    if node_num == 0:
                        continue
                    
                    print(f'\tNode num: {node_num}')
                    node_mask = node_image_data == node_num
                    
                    node_n_voxels = node_mask.sum()
                    node_vol = node_n_voxels * node_image_voxelvol
                    
                    node_ser = pd.Series()
                    node_ser['cest_id'] = cest_id
                    node_ser['control'] = is_control
                    node_ser['n_nodes_metastatic'] = n_nodes_metastatic
                    #node_ser['race'] = race
                    #node_ser['age'] = age
                    #node_ser['sex'] = sex
                    node_ser['num_id'] = num_id
                    node_ser['arm'] = arm
                    node_ser['weight_type'] = weight_type
                    node_ser['node_num'] = node_num
                    node_ser['node_vol'] = node_vol
                    node_ser['node_voxels'] = node_n_voxels
                    
                    #individual_vols.append(node_vol)
                    
                    # bulk signal
                    
                    sigs = [apt_image_data_norm, noe_image_data_norm]
                    sig_names = ['apt', 'noe']
            
                    for sig, sig_name in zip(sigs, sig_names):
                        sig_copy = sig.copy()
                        sig_copy[node_mask != 1] = np.nan # mask it
                        sig_copy[sig_copy == 0] = np.nan # remove zero signal wierdery
                        
                        the_mean = np.nanmean(sig_copy)
                        the_std = np.nanstd(sig_copy)
                        
                        node_ser[f'mean_{sig_name}'] = the_mean
                        node_ser[f'std_{sig_name}'] = the_std
                    
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
                    
                    # also set up the uncorrected spectrum
                    central_index_uncor = int(np.ceil(uncorrected_zspec_image_data.shape[3] / 2)) - 1
                    shift_step_uncor = 0.1 # change in chemical shift between each index
                    n_steps_uncor = uncorrected_zspec_image_data.shape[3]
                    
                    start_shift_uncor = -(central_index_uncor* shift_step_uncor)
                    end_shift_uncor = (n_steps_uncor - central_index_uncor) * shift_step_uncor
                    
                    shifts_uncor = np.arange(start_shift_uncor, end_shift_uncor - shift_step_uncor, shift_step_uncor)
                    shifts_uncor = [round(i,1) for i in shifts_uncor] # round to nearest tenth. thx floating point imprecision
                    
                    node_ser[f'chemical_shifts_ppm'] = np.array(shifts)
                    node_ser.at[f'chemical_shifts_uncor_ppm'] = np.array(shifts_uncor)
                    
                    #individual_noes.append(node_ser['mean_noe'])
                    #individual_apts.append(node_ser['mean_apt'])
                        
                    # calculate the zspec
                    means = []
                    stds = []
                    for i, shift in enumerate(shifts):
                        boxed = zspec_image_data[:,:,:,i]
                        boxed_copy = boxed.copy()
                        boxed_copy[node_mask != 1] = np.nan # mask it
                        
                        the_mean = np.nanmean(boxed_copy)
                        the_std = np.nanstd(boxed_copy)
                        
                        means.append(the_mean)
                        stds.append(the_std)
                        
                    # now normalize the means
                    means_norm = []
                    stds_norm = []
                    for i, shift in enumerate(shifts):
                        boxed = zspec_image_data_norm[:,:,:,i]
                        boxed_copy = boxed.copy()
                        boxed_copy[node_mask != 1] = np.nan # mask it
                        #boxed_copy = boxed_copy / norm_coef
                        
                        the_mean_norm = np.nanmean(boxed_copy)
                        the_std_norm = np.nanstd(boxed_copy)
                        
                        means_norm.append(the_mean_norm)
                        stds_norm.append(the_std_norm)
                        
                    
                    #individual_signals.append(means_norm)
                    
                    # now get the uncorrected means
                    means_uncor = []
                    stds_uncor = []
                    for i, shift in enumerate(shifts_uncor):
                        boxed = uncorrected_zspec_image_data[:,:,:,i]
                        boxed_copy = boxed.copy()
                        boxed_copy[node_mask != 1] = np.nan # mask it
                        #boxed_copy = boxed_copy / norm_coef
                        
                        the_mean_uncor = np.nanmean(boxed_copy)
                        the_std_uncor = np.nanstd(boxed_copy)
                        
                        means_uncor.append(the_mean_uncor)
                        stds_uncor.append(the_std_uncor)
                        
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
                        
                    # optimiziation initial params for lorentzian
                    if weight_type == 'water':
                        amp_guess = -1
                        center_guess = 0
                        width_guess = 2
                        offset_guess = 1
                    elif weight_type == 'fat':
                        amp_guess = -0.5
                        center_guess = -3
                        width_guess = 2
                        offset_guess = 1
                    
                    try:
                        popt_lor, pcov_lor = scipy.optimize.curve_fit(_lorentzian_mono, shifts, means_norm, [amp_guess, center_guess, width_guess, offset_guess], maxfev=int(1e4))
                        perr_lor = np.sqrt(np.diag(pcov_lor))
                        
                        opt_sigs = _lorentzian_mono(np.array(shifts), popt_lor[0], popt_lor[1], popt_lor[2], popt_lor[3])

                        node_ser[f'lorentzian_sigs'] = np.array(opt_sigs)
                        node_ser[f'lorentzian_amp'] = popt_lor[0]
                        node_ser[f'lorentzian_center'] = popt_lor[1]
                        node_ser[f'lorentzian_width'] = popt_lor[2]
                        node_ser[f'lorentzian_offset'] = popt_lor[3]
                    except ValueError:
                        pass
                        
                    
    
                    
                    node_ser[f'signals_mean'] = np.array(means)
                    node_ser[f'signals_std'] = np.array(stds)
                    node_ser[f'signals_asym'] = np.array(asym)
                    node_ser[ f'signals_eqdif'] = np.array(dif)
                    node_ser[ f'signals_meannorm'] = np.array(means_norm)
                    node_ser[f'signals_stdnorm'] = np.array(stds_norm)
                    node_ser[ f'signals_mean_uncor'] = np.array(means_uncor)
                    node_ser[f'signals_std_uncor'] = np.array(stds_uncor)
                    node_ser[f'cancer_proportion'] = cancer_proportion
                    
                    
                    node_ser[f'adjuvant_therapy'] = adj_ther
                    node_ser[f'neoadjuvant_therapy'] = neoadj_ther
                    node_ser[f'both_therapy'] = both_ther

                        
                    if np.isnan(node_ser['mean_noe']):
                        print('Node has no data in ROI')
                    else:
                        master_node_df = master_node_df.append(node_ser, ignore_index=True)

        
    # now lets get unweighted averages of signals for each pt
    unique_ids = master_node_df['cest_id'].unique()
    arm_types = ['aff', 'cont']
    incl_cols = [i for i in master_node_df.columns if 'lorentz' not in i and 'node' not in i and 'sig' not in i and 'std' not in i]
    
    node_df_byperson = pd.DataFrame()
    for wt in weight_types:
        pare_df = master_node_df[master_node_df['weight_type'] == wt]
        for at in arm_types:
            sub_pare_df = pare_df[pare_df['arm'] == at]
            for the_id in unique_ids:
                
                sub_df = sub_pare_df[sub_pare_df['cest_id'] == the_id]
                if len(sub_df) == 0:
                    continue
                
                r1 = sub_df.iloc[0]
                
                ser = pd.Series()
                
                for co in incl_cols:
                    ser[co] = r1[co]
                ser['signals_meannorm'] = np.average(np.array(sub_df['signals_meannorm']), axis=0)
                ser['mean_apt'] = np.average(np.array(sub_df['mean_apt']))
                ser['mean_noe'] = np.average(np.array(sub_df['mean_noe']))
                ser['n_nodes'] = len(sub_df)
                ser['vol_nodes'] = np.sum(sub_df['node_vol'])
                ser['n_node_voxels'] = np.sum(sub_df['node_voxels'])
                ser['cancer_proportion'] = sub_df['cancer_proportion'].iloc[0]
                
                # optimiziation initial params for lorentzian
                if wt == 'water':
                    amp_guess = -1
                    center_guess = 0
                    width_guess = 2
                    offset_guess = 1
                elif wt == 'fat':
                    amp_guess = -0.5
                    center_guess = -3
                    width_guess = 2
                    offset_guess = 1
                    
                try:
                    popt_lor, pcov_lor = scipy.optimize.curve_fit(_lorentzian_mono, ser['chemical_shifts_ppm'], ser['signals_meannorm'],
                                                                  [amp_guess, center_guess, width_guess, offset_guess], maxfev=int(1e4))
                    perr_lor = np.sqrt(np.diag(pcov_lor))
                    
                    opt_sigs = _lorentzian_mono(np.array(shifts), popt_lor[0], popt_lor[1], popt_lor[2], popt_lor[3])

                    ser[f'lorentzian_sigs'] = np.array(opt_sigs)
                    ser[f'lorentzian_amp'] = popt_lor[0]
                    ser[f'lorentzian_center'] = popt_lor[1]
                    ser[f'lorentzian_width'] = popt_lor[2]
                    ser[f'lorentzian_offset'] = popt_lor[3]
                except ValueError:
                    pass
                node_df_byperson = node_df_byperson.append(ser, ignore_index=True)

    master_node_df.to_excel(master_node_df_outname)
    
    node_df_byperson.to_excel(byperson_df_outname)
    
                    

    

###### figs
if do_figs:
    print('figuring')
    
        
    processed_folder = os.path.join(master_folder, 'aim1', 'processed')

            

    master_node_df = pd.read_excel(master_node_df_outname)
    node_df_byperson = pd.read_excel(byperson_df_outname)
    
    #weight_types = ['water', 'fat']
    
    tiks = np.arange(-4, 4.5, step=0.5)
    tiks = [round(i,1) for i in tiks]
    
    #plt.rc('xtick', labelsize=6)
    
    
    for wt in weight_types:
        
        if bulk_figs:
                
            for ttype in ['muscle', 'fat']:
                alt_df = pd.read_csv('/Users/skyjones/Desktop/cest_processing/data/working/results_aims234.csv')
                alt_df = alt_df[alt_df['aim']==1]
                
                
                node_df_byperson = pd.read_excel(byperson_df_outname)
                node_df_byperson['bulk_sigs'] = None
                
                for i,row in node_df_byperson.iterrows():
                    the_id = row['cest_id']
                    cor_row = alt_df[alt_df['scan_id']==the_id].iloc[0]
                    
                    node_df_byperson.at[i,'bulk_sigs'] = cor_row[f'{ttype}_aff_signals_meannorm_{wt}weighted']
                    
            
                the_df = node_df_byperson[node_df_byperson['weight_type']==wt]
                control_df = the_df[the_df['control']==1]
                cancer_df = the_df[the_df['control']==0]
                prop_thresh = 0.6
                lowcancer_df = cancer_df[cancer_df['cancer_proportion']<prop_thresh]
                highcancer_df = cancer_df[cancer_df['cancer_proportion']>=prop_thresh]
                control_color = 'orange'
                cancer_color = 'red'
                highcancer_color = 'purple'
                
                
                # aggregate
                
                spectras = []
                fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
                for sub_df, ctype, pcol in zip([control_df, lowcancer_df, highcancer_df],
                                               ['No metastasis', f'Low metastasis ratio (<{prop_thresh})', f'High metastasis ratio (>={prop_thresh})'],
                                               [control_color, cancer_color, highcancer_color]):
                    
                    cleaned_spectra = []
                    spectra = sub_df['bulk_sigs']
                    shifts = sub_df['chemical_shifts_ppm']
                    for spec, shifter in zip(spectra,shifts):
                        
                        
                        cleaned = az.reconvert_nparray_from_string(spec)
                        if np.isnan(cleaned[0]):
                            continue
                        
                        cleaned_spectra.append(cleaned)
                        shi = az.reconvert_nparray_from_string(shifter)
                        axs[0].plot(shi, cleaned, color=pcol, alpha=0.1)
                        
                    
                    
                    cleaned_spectra = np.array(cleaned_spectra)
                    

                    mean_spectra = np.mean(cleaned_spectra, axis=0)
                    spectras.append(mean_spectra)
                    
                    adj_prop = sum(sub_df['adjuvant_therapy']) / len(sub_df)
                    neoadj_prop = sum(sub_df['neoadjuvant_therapy']) / len(sub_df)
                    both_prop = sum(sub_df['both_therapy']) / len(sub_df)
                    
                    axs[0].plot(shi, mean_spectra, color=pcol, alpha=0.8, label=f'{ctype} (n={len(cleaned_spectra)})\nadj:{round(adj_prop,2)},neoadj:{round(neoadj_prop,2)},both:{round(both_prop,2)}')
                    #if ttype == 'muscle' and ctype=='No metastasis':
                    #    sys.exit()
                
                    
                    
                    
                axs[0].set_xticks(tiks)
                axs[0].set_title(f'Bulk tissue spectra ({ttype})')
                axs[0].set_ylim(0,1.2)
                axs[0].set_xlim(-4,4)
                axs[0].legend()
                axs[0].set_xlabel('Chemical shift (ppm)')
                axs[0].set_ylabel('Signal (a.u.)')
                
                axs[0].invert_xaxis()
                    
                spectras = np.array(spectras)
                #spectral_diff = np.diff(spectras, axis=0)[0]
                spectral_diff = spectras[2] - spectras[0]
                
                axs[1].plot([-100,100], [0,0], color='red', alpha=0.8)
                axs[1].plot(shi, spectral_diff, color='black', alpha=0.8)
                
                axs[1].set_xticks(tiks)
                axs[1].set_title('Spectral difference (high metastasis - no metastasis)')
                axs[1].set_ylim(-0.5,0.5)
                axs[1].set_xlim(-4,4)
                axs[1].legend()
                axs[1].set_xlabel('Chemical shift (ppm)')
                axs[1].set_ylabel('Signal difference')
                
                axs[1].invert_xaxis()
                 
                
                
                difference_figure_name = os.path.join(figure_folder, 'aim1', f'{ttype}BULK_differences_{wt}weighted.png')
                plt.tight_layout()
                plt.savefig(difference_figure_name, dpi=300)
                
                
                
                
                
                
                
            
        else:
    
            the_df = node_df_byperson[node_df_byperson['weight_type']==wt]
            master_df = master_node_df[master_node_df['weight_type']==wt]
            
            
            control_df = the_df[the_df['control']==1]
            cancer_df = the_df[the_df['control']==0]
            
            prop_thresh = 0.6
            
            lowcancer_df = cancer_df[cancer_df['cancer_proportion']<prop_thresh]
            highcancer_df = cancer_df[cancer_df['cancer_proportion']>=prop_thresh]
            
            
            
            control_color = 'orange'
            cancer_color = 'red'
            highcancer_color = 'purple'
            
            # aggregate
            node_weightings = [None, 'n_nodes', 'vol_nodes']
            node_weightings_clean = ['equal participant weighting', 'equal node weighting', 'weighted by voxel']
            for nw, nwc in zip(node_weightings, node_weightings_clean):
                
                spectras = []
                fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
                for sub_df, ctype, pcol in zip([control_df, lowcancer_df, highcancer_df],
                                               ['No metastasis', f'Low metastasis ratio (<{prop_thresh})', f'High metastasis ratio (>={prop_thresh})'],
                                               [control_color, cancer_color, highcancer_color]):
                    
                    cleaned_spectra = []
                    spectra = sub_df['signals_meannorm']
                    shifts = sub_df['chemical_shifts_ppm']
                    for spec, shifter in zip(spectra,shifts):
                        
                        shi = az.reconvert_nparray_from_string(shifter)
                        
                        cleaned = az.reconvert_nparray_from_string(spec)
                        cleaned_spectra.append(cleaned)
                        axs[0].plot(shi, cleaned, color=pcol, alpha=0.07)
                        
                    
                    if nw is not None:
                        weight_total = sum(sub_df[nw])
                    else:
                        weight_total = len(cleaned_spectra)
                    
                    cleaned_spectra = np.array(cleaned_spectra)
                    
                    if nw is not None:
                        weight_vals = sub_df[nw]
                        mean_spectra = np.average(cleaned_spectra, axis=0, weights=weight_vals)
                    else:
                        mean_spectra = np.mean(cleaned_spectra, axis=0)
                    
                    
                    spectras.append(mean_spectra)
                    
                    adj_prop = sum(sub_df['adjuvant_therapy']) / len(sub_df)
                    neoadj_prop = sum(sub_df['neoadjuvant_therapy']) / len(sub_df)
                    both_prop = sum(sub_df['both_therapy']) / len(sub_df)
                    axs[0].plot(shi, mean_spectra, color=pcol, alpha=0.8, label=f'{ctype} (n={len(cleaned_spectra)})\nadj:{round(adj_prop,2)},neoadj:{round(neoadj_prop,2)},both:{round(both_prop,2)}')
                    
                    
                    
                axs[0].set_xticks(tiks)
                axs[0].set_title(f'Lymph node spectra ({nwc})')
                axs[0].set_ylim(0,1.2)
                axs[0].set_xlim(-4,4)
                axs[0].legend()
                axs[0].set_xlabel('Chemical shift (ppm)')
                axs[0].set_ylabel('Signal (a.u.)')
                
                axs[0].invert_xaxis()
                    
                spectras = np.array(spectras)
                #spectral_diff = np.diff(spectras, axis=0)[0]
                spectral_diff = spectras[2] - spectras[0]
                
                axs[1].plot([-100,100], [0,0], color='red', alpha=0.8)
                axs[1].plot(shi, spectral_diff, color='black', alpha=0.8)
                
                axs[1].set_xticks(tiks)
                axs[1].set_title('Spectral difference (high metastasis - no metastasis)')
                axs[1].set_ylim(-0.5,0.5)
                axs[1].set_xlim(-4,4)
                axs[1].legend()
                axs[1].set_xlabel('Chemical shift (ppm)')
                axs[1].set_ylabel('Signal difference')
                
                axs[1].invert_xaxis()
                 
                
                
                difference_figure_name = os.path.join(figure_folder, 'aim1', f'differences_{wt}weighted_{str(nw)}weighting.png')
                plt.tight_layout()
                plt.savefig(difference_figure_name, dpi=300)
            
            # individual
            
            for cid in master_node_df['cest_id'].unique():
                sub_df = master_df[master_df['cest_id']==cid]
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
                    
                metastasis_ratio = sub_df['cancer_proportion'].iloc[0]
            
                if sub_df.iloc[0]['control'] == 1:
                    pcol = control_color
                    fileadder = 'control'
                else:
                    pcol = cancer_color
                    fileadder = 'meta'
                    
                    if metastasis_ratio>=prop_thresh:
                        pcol = highcancer_color
                        fileadder = 'highmeta'
                        
                
                cleaneds = []
                for k,row in sub_df.iterrows():
                        
                    shi = az.reconvert_nparray_from_string(row['chemical_shifts_ppm'])
                    cleaned = az.reconvert_nparray_from_string(row['signals_meannorm'])
                    cleaneds.append(cleaned)
                    
                    ax.plot(shi, cleaned, color=pcol, alpha=0.15)
                    
                cleaneds = np.array(cleaneds)
                meaneds = np.mean(cleaneds, axis=0)
                ax.plot(shi, meaneds, color=pcol, alpha=0.8, label=f'Mean signal (n={len(cleaneds)})')
                
                ax.legend()
                ax.set_xticks(tiks)
                ax.set_title(f'Lymph node spectra, {cid}, metastasis ratio={metastasis_ratio:#.2g} ({wt} weighted)')
                ax.set_ylim(0,1.2)
                ax.set_xlim(-4,4)
                ax.set_xlabel('Chemical shift (ppm)')
                ax.set_ylabel('Signal (a.u.)')
                
                single_outname = os.path.join(figure_folder, 'aim1', f'nodesignals_{fileadder}_{cid}_{wt}weighted.png')
                plt.tight_layout()
                plt.savefig(single_outname, dpi=300)
        
                    
                
                    
            
    
    