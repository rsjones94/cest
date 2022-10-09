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

mpl.rcParams.update({'errorbar.capsize': 2})

# input
master_folder = '/Users/skyjones/Desktop/hiv_processing/data/working/'
bulk_folder = '/Users/skyjones/Desktop/hiv_processing/data/bulk/'
cest_spreadsheet = '/Users/skyjones/Desktop/hiv_processing/workbook.xlsx'

#output
data_out = '/Users/skyjones/Desktop/hiv_processing/data/working/results.csv'
figure_folder = '/Users/skyjones/Desktop/hiv_processing/data/working/figures/'

do_prep = True
coord_restore = True

do_figs = False



master_node_df_outname = os.path.join(master_folder, f'node_data.xlsx')
byperson_df_outname = os.path.join(master_folder, f'node_data_byperson.xlsx')
delta_df_outname = os.path.join(master_folder, f'node_data_delta.xlsx')
##########
def _lorentzian_mono(x, amp1, cen1, wid1, offset1):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) + offset1

#weight_types = ['fat', 'water']
weight_types = ['water']

if do_prep:
    cest_df = pd.read_excel(cest_spreadsheet)
    #cest_df['mri_id'] = cest_df['mri_id'].astype(str)
    
    subs = glob.glob(os.path.join(master_folder, '*/'))
    subs = [sub for sub in subs if 'aim' in sub.lower()]
    
    for sub in subs:
        
        master_node_df = pd.DataFrame()
        #master_node_df_flatweighted = pd.DataFrame()
        #master_node_df_volweighted = pd.DataFrame()
        
        aim_folder = sub[:-1].split('/')[-1]

        
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
            
            is_control = the_row['Control']
            race = the_row['Race']
            gender = the_row['Gender']
            ethnicity = the_row['Ethnicity']
            is_followup = the_row['followup']
            
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
            
            node_image_globber = os.path.join(bulk_folder, aim_folder, cest_id, f'*mask*{arm}.nii.gz')
            
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
                noe_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_OPPAPT_{weight_type}.nii.gz')
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
                    node_ser['followup'] = is_followup
                    node_ser['race'] = race
                    node_ser['ethnicity'] = ethnicity
                    node_ser['gender'] = gender
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

                        
                    if np.isnan(node_ser['mean_noe']):
                        print('Node has no data in ROI')
                    else:
                        master_node_df = master_node_df.append(node_ser, ignore_index=True)
                    
                
                '''
                individual_signals = np.array(individual_signals)
                individual_apts = np.array(individual_apts)
                individual_noes = np.array(individual_noes)
                individual_vols = np.array(individual_vols)
                
                weightings = [None, individual_vols]
                
                study_row_flatweighted = pd.Series()
                study_row_volweighted = pd.Series()
                
                sers = [study_row_flatweighted, study_row_volweighted]
                
                for the_ser, the_weighting in zip(sers, weightings):
                    
                    the_ser['cest_id'] = cest_id
                    the_ser['control'] = is_control
                    the_ser['race'] = race
                    the_ser['ethnicity'] = ethnicity
                    the_ser['gender'] = gender
                    the_ser['num_id'] = num_id
                    the_ser['arm'] = arm
                    the_ser['weight_type'] = weight_type
                    
                    avg_signals = np.average(individual_signals, axis=0, weights=the_weighting)
                    avg_apt = np.average(individual_apts, weights=the_weighting)
                    avg_noe = np.average(individual_noes, weights=the_weighting)
                    total_vol = individual_vols.sum()
                    n_nodes = len(individual_vols)
                    
                    
                    the_ser['signals_meannorm'] = str(avg_signals)
                    the_ser['chemical_shifts_ppm'] = str(shifts)
                    the_ser['mean_noe'] = avg_noe
                    the_ser['mean_apt'] = avg_apt
                    
                    the_ser['total_node_vol'] = total_vol
                    the_ser['n_nodes'] = n_nodes
                    
                    try:
                        popt_lor, pcov_lor = scipy.optimize.curve_fit(_lorentzian_mono, shifts, avg_signals, [amp_guess, center_guess, width_guess, offset_guess], maxfev=int(1e4))
                        perr_lor = np.sqrt(np.diag(pcov_lor))
                        
                        opt_sigs = _lorentzian_mono(np.array(shifts), popt_lor[0], popt_lor[1], popt_lor[2], popt_lor[3])

                        the_ser[f'lorentzian_sigs'] = str(opt_sigs)
                        the_ser[f'lorentzian_amp'] = str(popt_lor[0])
                        the_ser[f'lorentzian_center'] = str(popt_lor[1])
                        the_ser[f'lorentzian_width'] = str(popt_lor[2])
                        the_ser[f'lorentzian_offset'] = str(popt_lor[3])
                    except ValueError:
                        pass

                
                master_node_df_flatweighted = master_node_df_flatweighted.append(study_row_flatweighted, ignore_index=True)
                master_node_df_volweighted = master_node_df_volweighted.append(study_row_volweighted, ignore_index=True)
                '''


        
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

    # now let's calculate the pre-post difference
    node_df_delta = pd.DataFrame()
    incl_cols = [i for i in master_node_df.columns if 'lorentz' not in i and 'node' not in i
                 and 'sig' not in i and 'std' not in i and 'cest_id' not in i and 'followup' not in i]
    for wt in weight_types:
        pare_df = node_df_byperson[node_df_byperson['weight_type'] == wt]
        for at in arm_types:
            sub_pare_df = pare_df[pare_df['arm'] == at]
            for the_id in unique_ids:
                ser = pd.Series()
                sub_df = sub_pare_df[sub_pare_df['cest_id'] == the_id]
                if len(sub_df) == 0:
                    continue
                
                
                r1 = sub_df.iloc[0] # post
                
                partner_name = r1['followup']
                if type(partner_name) == float:
                    continue
                
                partner_df = sub_pare_df[sub_pare_df['cest_id'] == partner_name]
                r2 = partner_df.iloc[0] # pre
                
                for co in incl_cols:
                    ser[co] = r1[co]
                    
                rows = [r2, r1]
                row_temporals = ['initial', 'followup']
                
                for coln in sub_df.columns:
                    if coln in incl_cols:
                        continue
                    for ro, rot in zip(rows, row_temporals):
                        ser[f'{rot}_{coln}'] = ro[coln]
                
                ser['signals_meannorm_delta'] = r1['signals_meannorm'] - r2['signals_meannorm']
                ser['mean_noe_delta'] = r1['mean_noe'] - r2['mean_noe']
                ser['mean_apt_delta'] = r1['mean_apt'] - r2['mean_apt']
                ser['mean_lorwidth_delta'] = r1['lorentzian_width'] - r2['lorentzian_width']
                
                node_df_delta = node_df_delta.append(ser, ignore_index=True)
                
                        
    master_node_df.to_excel(master_node_df_outname)
    
    node_df_byperson.to_excel(byperson_df_outname)
    
    node_df_delta.to_excel(delta_df_outname)
                    

    

###### figs
if do_figs:
    
    processed_folder = os.path.join(master_folder, 'aim5', 'processed')
    
    master_node_df = pd.read_excel(master_node_df_outname)
    node_df_byperson = pd.read_excel(byperson_df_outname)
    node_df_delta = pd.read_excel(delta_df_outname)
    
    arm_types = ['aff', 'cont']
    weight_types = ['water']
    temporals = ['initial', 'followup']
    temporal_colors = ['blue', 'green']
    
    delta_figure_folder = os.path.join(figure_folder, 'deltas')
    if not os.path.exists(delta_figure_folder):
        os.mkdir(delta_figure_folder)
        
    print('figuring')
    tiks = np.arange(-4, 4.5, step=0.5)
    tiks = [round(i,1) for i in tiks]
    
    plt.rc('xtick', labelsize=6)
    for at in arm_types:
        print(f'Arm is {at}')
        sub_df = node_df_delta[node_df_delta['arm']==at]
        for wt in weight_types:
            print(f'\tWeighting is {wt}')
            sub_sub_df = sub_df[sub_df['weight_type']==wt]
            for i, row in sub_sub_df.iterrows():
                print(f'\t\tRow {i+1} of {len(node_df_delta)}')
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
                
                is_control = row['control']
                if is_control:
                    ct = 'control'
                else:
                    ct = 'hiv'
                    
                delta_apt = row['mean_apt_delta']
                delta_noe = row['mean_noe_delta']
                delta_fwhm = row['mean_lorwidth_delta']
                
                id_list = []
                shifts = az.reconvert_nparray_from_string(row['chemical_shifts_ppm'])
                for tempo, tempo_col in zip(temporals, temporal_colors):
                    signal = az.reconvert_nparray_from_string(row[f'{tempo}_signals_meannorm'])
                    lorentz = az.reconvert_nparray_from_string(row[f'{tempo}_lorentzian_sigs'])
                    the_id = row[f'{tempo}_cest_id']
                    id_list.append(the_id)
                    
                    
                    axs[0][0].plot(shifts, signal, alpha=0.7, color=tempo_col, label=f'{tempo} signal')
                    axs[0][0].plot(shifts, lorentz, alpha=0.3, color=tempo_col, label=f'{tempo} lorentzian', ls='dashed')
                    
                axs[0][0].set_xlabel('Chemical shift (ppm)')
                axs[0][0].set_ylabel('Normalized signal (a.u.)')
                    
                axs[0][0].legend()
                axs[0][0].set_title(f'CEST response')
                axs[0][0].set_xlim(-4,4)
                axs[0][0].set_ylim(0,1.2)
    
                axs[0][0].set_xticks(tiks)
                
                delta = az.reconvert_nparray_from_string(row[f'signals_meannorm_delta'])
                axs[0][1].plot([-10,10], [0,0], alpha=0.7, color='red')
                axs[0][1].plot(shifts, delta, alpha=0.7, color='black', label=f'Signal delta')
                
                axs[0][1].set_xlabel('Chemical shift (ppm)')
                axs[0][1].set_ylabel('Normalized signal delta (a.u.)')
                    
                axs[0][1].legend()
                
                axs[0][1].set_title(f'\n\nRelative signal change')
                axs[0][1].set_xlim(-4,4)
                axs[0][1].set_ylim(-0.5,0.5)
                
                axs[0][1].set_xticks(tiks)
                
                for i, tempo in enumerate(temporals):
                    the_ax = axs[1][i]
                    cest_id = row[f'{tempo}_cest_id']
                    n_nodes = row[f'{tempo}_n_nodes']
                    v_nodes = int(row[f'{tempo}_n_node_voxels'])
                    
                    fol = os.path.join(processed_folder, f'{cest_id}_cestdixon_{at}')
                    water_image = os.path.join(fol, f'{cest_id}_cestdixon_{at}_C1_.nii.gz')
                    node_image_globber = os.path.join(bulk_folder, 'aim5', cest_id, f'*mask*{at}.nii.gz')
                    node_image_candidates = glob.glob(node_image_globber)
                    node_image = node_image_candidates[0]
            
                    water_image_loaded = nib.load(water_image)
                    water_image_data = water_image_loaded.get_fdata()
                    water_image_data = np.mean(water_image_data, 3) # zspec average
            
                    node_image_loaded = nib.load(node_image)
                    node_image_data = node_image_loaded.get_fdata()
                    labeled_node_data = morphology.label(node_image_data)
                    
                    n_slices = labeled_node_data.shape[2]
                    n_unique = 0
                    win_slice = 0
                    for ix in range(n_slices):
                        sli = labeled_node_data[:,:,ix]
                        unique_in_slice = len(np.unique(sli))
                        if unique_in_slice > n_unique:
                            n_unique = unique_in_slice
                            win_slice = ix
                            
                    the_ax.imshow(water_image_data[:,:,win_slice], cmap='gray')
                    the_ax.imshow(node_image_data[:,:,win_slice], cmap='Reds', alpha=0.5)
                    
                    the_ax.axis('off')
                    the_ax.set_title(f'{tempo}: {cest_id}\n{n_nodes} nodes ({n_unique-1} visible), total voxels {v_nodes}')
                        
                            
                    
                plt.suptitle(f'{ct}, {at} arm, {wt} weighted\n$\Delta$apt={round(delta_apt,2)}, $\Delta$noe={round(delta_noe,2)}, $\Delta$FWHM={round(delta_fwhm,2)}', fontweight='bold')
                plt.tight_layout()
                
                delta_outname = os.path.join(delta_figure_folder, f'delta_{ct}_{"_to_".join(id_list)}_{wt}weighted_{at}arm.png')
                fig.savefig(delta_outname, dpi=300)
    
                    
                
            
    