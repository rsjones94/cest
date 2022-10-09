#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 15:20:47 2021

@author: skyjones
"""

import os
import sys
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import redcap

check_df_outname = '/Users/skyjones/Desktop/cest_processing/check.csv'

path_to_bulk_data = '/Users/skyjones/Desktop/cest_processing/data/bulk/'
path_to_processed_data = '/Users/skyjones/Desktop/cest_processing/data/working/'


aim1_token_loc = '/Users/skyjones/Documents/repositories/lymph_aim1_token.txt'
aim2_token_loc = '/Users/skyjones/Documents/repositories/lymph_aim2_token.txt'
aim3_token_loc = '/Users/skyjones/Documents/repositories/lymph_aim3_token.txt'
aim4_token_loc = '/Users/skyjones/Documents/repositories/lymph_aim4_token.txt'

comb_adj_col = 'neoadjuvant_or_adjuvant'
neoadj_col = 'neo_adjuvent_or_adjuvent___1'
adj_col = 'neo_adjuvent_or_adjuvent___2'

prefix = 'Donahue'


api_url = 'https://redcap.vanderbilt.edu/api/'

print('Contacting Aim 1 REDCap database...')

aim1_token = open(aim1_token_loc).read()
project = redcap.Project(api_url, aim1_token)

aim1_data_raw = project.export_records(export_survey_fields=False, export_data_access_groups=True)
aim1_data = pd.DataFrame(aim1_data_raw)

aim1_data = aim1_data.drop(aim1_data[aim1_data['mri_scan_id1'] == ''].index)

aim1_ids = aim1_data['mri_scan_id1']
aim1_dates = aim1_data['study_date']
aim1_sids = aim1_data['study_id']
aim1_nodes_removed = aim1_data['number_of_lymph_nodes_remo']
aim1_nodes_metastatic = aim1_data['number_of_metastatic_lymph']
aim1_nodes_comments = aim1_data['removal_comments']
aim1_ages = aim1_data['age1']
aim1_races = aim1_data['race']
aim1_sexes = aim1_data['sex']
#aim1_prefixes = aim1_data['mri_forename']
aim1_full_ids = [f'{prefix}_{i}' for i in aim1_ids]

aim1_neoadj_therapy = [1 if i=='1' or i=='3' else 0 for i in aim1_data[comb_adj_col]]
aim1_adj_therapy = [1 if i=='2' or i=='3' else 0 for i in aim1_data[comb_adj_col]]

#####

print('Contacting Aim 2 REDCap database...')

aim2_token = open(aim2_token_loc).read()
project = redcap.Project(api_url, aim2_token)

aim2_data_raw = project.export_records(export_survey_fields=False, export_data_access_groups=True)
aim2_data = pd.DataFrame(aim2_data_raw)

aim2_data = aim2_data.drop(aim2_data[aim2_data['mri_scan_id1'] == ''].index)

aim2_ids = aim2_data['mri_scan_id1']
aim2_dates = aim2_data['study_date']
aim2_visits = aim2_data['redcap_event_name']
aim2_repeats = aim2_data['redcap_repeat_instrument']
aim2_sids = aim2_data['study_id']

aim2_ages = aim2_data['age12']
aim2_races = aim2_data['race']
aim2_sexes = aim2_data['sex']


aim2_ids = [i for i, v, r in zip(aim2_ids, aim2_visits, aim2_repeats) if v == 'visit_1_enrollment_arm_1' and r != 'photographs']
aim2_dates = [i for i, v, r in zip(aim2_dates, aim2_visits, aim2_repeats) if v == 'visit_1_enrollment_arm_1' and r != 'photographs']
aim2_sids = [i for i, v, r in zip(aim2_sids, aim2_visits, aim2_repeats) if v == 'visit_1_enrollment_arm_1' and r != 'photographs']


aim2_races = [i for i, v, r in zip(aim2_races, aim2_visits, aim2_repeats) if v == 'visit_1_enrollment_arm_1' and r != 'photographs']
aim2_ages = [i for i, v, r in zip(aim2_ages, aim2_visits, aim2_repeats) if v == 'visit_1_enrollment_arm_1' and r != 'photographs']
aim2_sexes = [i for i, v, r in zip(aim2_sexes, aim2_visits, aim2_repeats) if v == 'visit_1_enrollment_arm_1' and r != 'photographs']
aim2_neoadj_raw = [i for i, v, r in zip(aim2_data[neoadj_col], aim2_visits, aim2_repeats) if v == 'visit_1_enrollment_arm_1' and r != 'photographs']
aim2_adj_raw = [i for i, v, r in zip(aim2_data[adj_col], aim2_visits, aim2_repeats) if v == 'visit_1_enrollment_arm_1' and r != 'photographs']

aim2_neoadj_therapy = [1 if i=='1' else 0 for i in aim2_neoadj_raw]
aim2_adj_therapy = [1 if i=='1' else 0 for i in aim2_adj_raw]

#aim2_prefixes = aim2_data['mri_forename']
aim2_full_ids = [f'Donahue_{i}' for i in aim2_ids]


#####


print('Contacting Aim 3 REDCap database...')

treatment_dict = {'visit_1_enrollment_arm_1': ('cdt_alone', 'pre', 'cdt_alone_then_cdt_and_lt'),
                  'visit_1_enrollment_arm_2': ('cdt_and_lt','pre', 'cdt_and_lt_then_cdt_alone'),
                  'visit_11_final_vis_arm_1': ('cdt_alone','post', 'cdt_alone_then_cdt_and_lt'),
                  'visit_11_final_vis_arm_2': ('cdt_and_lt','post', 'cdt_and_lt_then_cdt_alone'),
                  
                  'visit_1_enrollment_arm_3': ('cdt_and_lt', 'pre', 'cdt_alone_then_cdt_and_lt'),
                  'visit_1_enrollment_arm_4': ('cdt_alone','pre', 'cdt_and_lt_then_cdt_alone'),
                  'visit_11_final_vis_arm_3': ('cdt_and_lt','post', 'cdt_alone_then_cdt_and_lt'),
                  'visit_11_final_vis_arm_4': ('cdt_alone','post', 'cdt_and_lt_then_cdt_alone')
                  }

aim3_token = open(aim3_token_loc).read()
project = redcap.Project(api_url, aim3_token)

aim3_data_raw = project.export_records(export_survey_fields=False, export_data_access_groups=True)
aim3_data = pd.DataFrame(aim3_data_raw)
aim3_ids = aim3_data['mri_scan_id1']
aim3_dates = aim3_data['study_date']
aim3_visits = aim3_data['redcap_event_name']
aim3_sids = aim3_data['study_id']
aim3_ldexes = aim3_data['ldex']
aim3_psfss = aim3_data['psfs_1_value']
aim3_uefis = aim3_data['uefi_score']
aim3_dashes = aim3_data['dash_work_score']
aim3_bmi = aim3_data['bmi']


aim3_ages = aim3_data['age13']
aim3_races = aim3_data['race']
aim3_sexes = aim3_data['sex']


aim3_ids_orig = [i.strip('*') for i in aim3_ids]
aim3_sids = [i.strip('*') for i in aim3_sids]

aim3_ids = [i for i, v, j in zip(aim3_ids, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']
aim3_dates = [i for i, v, j in zip(aim3_dates, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']
aim3_ldexes = [i for i, v, j in zip(aim3_ldexes, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']
aim3_sids = [i for i, v, j in zip(aim3_sids, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']
aim3_psfss = [i for i, v, j in zip(aim3_psfss, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']
aim3_uefis = [i for i, v, j in zip(aim3_uefis, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']
aim3_dashes = [i for i, v, j in zip(aim3_dashes, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']
aim3_bmi = [i for i, v, j in zip(aim3_bmi, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']

aim3_ages = [i for i, v, j in zip(aim3_ages, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']
aim3_races = [i for i, v, j in zip(aim3_races, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']
aim3_sexes = [i for i, v, j in zip(aim3_sexes, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']


aim3_visits = [i for i, v, j in zip(aim3_visits, aim3_visits, aim3_ids_orig) if ('enrollment_arm' in v or 'final_vis' in v) and j != '']


aim3_neoadj_therapy = [1 if i=='1' and ('enrollment_arm' in v or 'final_vis' in v) and j != '' else 0 for i, v, j in zip(aim3_data[neoadj_col], aim3_visits, aim3_ids_orig)]
aim3_adj_therapy = [1 if i=='1' and ('enrollment_arm' in v or 'final_vis' in v) and j != '' else 0 for i, v, j in zip(aim3_data[adj_col], aim3_visits, aim3_ids_orig)]

arm3_treatment_type = [treatment_dict[i][0] for i in aim3_visits]
arm3_treatment_status = [treatment_dict[i][1] for i in aim3_visits]
arm3_treatment_order = [treatment_dict[i][2] for i in aim3_visits]

#aim3_prefixes = aim3_data['mri_forename']
aim3_full_ids = [f'Donahue_{i}' for i in aim3_ids]

####

print('Contacting Aim 4 REDCap database...')
aim4_token = open(aim4_token_loc).read()
project = redcap.Project(api_url, aim4_token)

aim4_data_raw = project.export_records(export_survey_fields=False, export_data_access_groups=True)
aim4_data = pd.DataFrame(aim4_data_raw)

aim4_data = aim4_data.drop(aim4_data[aim4_data['mri_scan_id1'] == ''].index)


aim4_ages = aim4_data['age1']
aim4_races = aim4_data['race']
aim4_sexes = aim4_data['sex']

aim4_ids = aim4_data['mri_scan_id1']
aim4_dates = aim4_data['study_date']
aim4_sids = aim4_data['study_id']

#aim4_prefixes = aim4_data['mri_forename']
aim4_full_ids = [f'{prefix}_{i}' for i in aim4_ids]


aim4_neoadj_therapy = [1 if i=='1' or i=='3' else 0 for i in aim4_data[comb_adj_col]]
aim4_adj_therapy = [1 if i=='2' or i=='3' else 0 for i in aim4_data[comb_adj_col]]


####

all_full_ids = []
for i in [aim1_full_ids, aim2_full_ids, aim3_full_ids, aim4_full_ids]:
    all_full_ids.extend(i)
    
all_dates = []
for i in [aim1_dates, aim2_dates, aim3_dates, aim4_dates]:
    all_dates.extend(i)    
    
all_sids = []
for i in [aim1_sids, aim2_sids, aim3_sids, aim4_sids]:
    all_sids.extend(i)       
    
all_ages = []
for i in [aim1_ages, aim2_ages, aim3_ages, aim4_ages]:
    all_ages.extend(i)      
    
all_races = []
for i in [aim1_races, aim2_races, aim3_races, aim4_races]:
    all_races.extend(i)      
    
all_sexes = []
for i in [aim1_sexes, aim2_sexes, aim3_sexes, aim4_sexes]:
    all_sexes.extend(i) 
    
all_adj_therapy = []
for i in [aim1_adj_therapy, aim2_adj_therapy, aim3_adj_therapy, aim4_adj_therapy]:
    all_adj_therapy.extend(i)
    
all_neoadj_therapy = []
for i in [aim1_neoadj_therapy, aim2_neoadj_therapy, aim3_neoadj_therapy, aim4_neoadj_therapy]:
    all_neoadj_therapy.extend(i)
    
all_aims = []
for j in [[1 for i in aim1_full_ids], [2 for i in aim2_full_ids], [3 for i in aim3_full_ids], [4 for i in aim4_full_ids]]:
    all_aims.extend(j)

all_treatment_type = []
for j in [[None for i in aim1_full_ids], [None for i in aim2_full_ids], [i for i in arm3_treatment_type], [None for i in aim4_full_ids]]:
    all_treatment_type.extend(j)
    
all_treatment_status = []
for j in [[None for i in aim1_full_ids], [None for i in aim2_full_ids], [i for i in arm3_treatment_status], [None for i in aim4_full_ids]]:
    all_treatment_status.extend(j)
    
all_treatment_order = []
for j in [[None for i in aim1_full_ids], [None for i in aim2_full_ids], [i for i in arm3_treatment_order], [None for i in aim4_full_ids]]:
    all_treatment_order.extend(j)
    
all_nodes_removed = []
for j in [[i for i in aim1_nodes_removed], [None for i in aim2_full_ids], [None for i in arm3_treatment_order], [None for i in aim4_full_ids]]:
    all_nodes_removed.extend(j)
        
all_nodes_metastatic = []
for j in [[i for i in aim1_nodes_metastatic], [None for i in aim2_full_ids], [None for i in arm3_treatment_order], [None for i in aim4_full_ids]]:
    all_nodes_metastatic.extend(j)    
    
all_nodes_comments = []
for j in [[i for i in aim1_nodes_comments], [None for i in aim2_full_ids], [None for i in arm3_treatment_order], [None for i in aim4_full_ids]]:
    all_nodes_comments.extend(j)   

all_ldex = []
for j in [[None for i in aim1_full_ids], [None for i in aim2_full_ids], [i for i in aim3_ldexes], [None for i in aim4_full_ids]]:
    all_ldex.extend(j)
    
all_psfs= []
for j in [[None for i in aim1_full_ids], [None for i in aim2_full_ids], [i for i in aim3_psfss], [None for i in aim4_full_ids]]:
    all_psfs.extend(j)
    
all_uefi = []
for j in [[None for i in aim1_full_ids], [None for i in aim2_full_ids], [i for i in aim3_uefis], [None for i in aim4_full_ids]]:
    all_uefi.extend(j)
    
all_dash = []
for j in [[None for i in aim1_full_ids], [None for i in aim2_full_ids], [i for i in aim3_dashes], [None for i in aim4_full_ids]]:
    all_dash.extend(j)
    
all_bmi = []
for j in [[None for i in aim1_full_ids], [None for i in aim2_full_ids], [i for i in aim3_bmi], [None for i in aim4_full_ids]]:
    all_bmi.extend(j)
    
    
check_df = pd.DataFrame()
check_df['scan_id'] = all_full_ids
check_df['study_id'] = all_sids
check_df['scan_date'] = all_dates
check_df['age'] = all_ages
check_df['race'] = all_races
check_df['sex'] = all_sexes
check_df['aim'] = all_aims
check_df['n_nodes_removed'] = all_nodes_removed
check_df['n_nodes_metastatic'] = all_nodes_metastatic
check_df['biopsy_comments'] = all_nodes_comments
check_df['treatment_type'] = all_treatment_type
check_df['treatment_status'] = all_treatment_status
check_df['treatment_order'] = all_treatment_order
check_df['ldex'] = all_ldex
check_df['psfs'] = all_psfs
check_df['uefi'] = all_uefi
check_df['dash'] = all_dash
check_df['bmi'] = all_bmi
check_df['adj_therapy'] = all_adj_therapy
check_df['neoadj_therapy'] = all_neoadj_therapy




#### now that we've assembled the dataframe, check to see if there are folders with the data
print('Checking if data has been downloaded....')

for i, row in check_df.iterrows():
    scan_id = row['scan_id']
    aim = row['aim']
    
    
    bulk_data_folder = os.path.join(path_to_bulk_data, f'aim{aim}')
    if os.path.exists(os.path.join(bulk_data_folder, scan_id)):
        has_bulk = 1
    else:
        has_bulk = 0
      
    '''
    processed_data_folder = os.path.join(path_to_processed_data, f'aim{aim}')
    if os.path.exists(os.path.join(processed_data_folder, scan_id)):
        has_processed = 1
    else:
        has_processed = 0
    '''
        
    check_df.at[i, 'has_bulk'] = has_bulk
    #check_df.at[i, 'has_processed'] = has_processed
    
    
    print(f'\t{scan_id} (Aim {aim})\n\t\tHas bulk: {has_bulk}')
    
    

check_df.to_csv(check_df_outname)