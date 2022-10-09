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

df_outname_base = '/Users/skyjones/Desktop/cest_processing/aim3_bedsides'


aim_token_loc = '/Users/skyjones/Documents/repositories/lymph_aim3_token.txt'

prefix = 'Donahue'


api_url = 'https://redcap.vanderbilt.edu/api/'
print('Contacting REDCap database...')

aim_token = open(aim_token_loc).read()
project = redcap.Project(api_url, aim_token)

aim_data_raw = project.export_records(export_survey_fields=False, export_data_access_groups=True)
aim_data = pd.DataFrame(aim_data_raw)


treatment_dict = {'visit_1_enrollment_arm_1': ('cdt_alone', 'pre', 'cdt_alone_then_cdt_and_lt'),
                  'visit_1_enrollment_arm_2': ('cdt_and_lt','pre', 'cdt_and_lt_then_cdt_alone'),
                  'visit_11_final_vis_arm_1': ('cdt_alone','post', 'cdt_alone_then_cdt_and_lt'),
                  'visit_11_final_vis_arm_2': ('cdt_and_lt','post', 'cdt_and_lt_then_cdt_alone'),
                  
                  'visit_1_enrollment_arm_3': ('cdt_and_lt', 'pre', 'cdt_alone_then_cdt_and_lt'),
                  'visit_1_enrollment_arm_4': ('cdt_alone','pre', 'cdt_and_lt_then_cdt_alone'),
                  'visit_11_final_vis_arm_3': ('cdt_and_lt','post', 'cdt_alone_then_cdt_and_lt'),
                  'visit_11_final_vis_arm_4': ('cdt_alone','post', 'cdt_and_lt_then_cdt_alone')
                  }

treatment_arms = [
                  [['visit_1_enrollment_arm_1', 'visit_1_enrollment_arm_4'],
                  ['visit_11_final_vis_arm_1', 'visit_11_final_vis_arm_4'],],
                  
                  [['visit_1_enrollment_arm_2', 'visit_1_enrollment_arm_3'],
                  ['visit_11_final_vis_arm_2', 'visit_11_final_vis_arm_3']]
                  ]

treatment_times = ['pre',
                   'post']

treatment_modalities = ['cdt',
                        'cdt_and_lt']

treatment_orders = ['cdt_then_dual', 'dual_then_cdt']



id_col = 'study_id'

demos = [id_col,
         'age13',
         'race',
         'ethnicity',
         'sex',
         'neo_adjuvent_or_adjuvent___1',
         'neo_adjuvent_or_adjuvent___2',
         'date_of_ln_removal',
         'no_ln_removed'
         ]

demo_names = [id_col,
              'age',
              'race',
              'ethnicity',
              'sex',
              'neoadj_therapy',
              'adj_therapy',
              'date_of_ln_removal',
              'n_nodes_removed'
              ]



cols_of_interest = [id_col,
                    'redcap_event_name',
                    'mri_scan_id1',
                    'study_date',
                    'bcrl_risk_side2',
                    'bmi',
                    't_affected_vol_1a',
                    't_unaffected_vol_1a',
                    'true_vol_diff_1a',
                    'true_vol_diff_calc_1a',
                    'ldex',
                    'psfs_1_value',
                    'uefi_score',
                    'dash_work_score'
                    ]

cols_of_interest_names = [id_col,
                          'treatment_arm',
                          'mri_scan_id',
                          'study_date',
                          'lymphedema_affected_side',
                          'bmi',
                          'perometer_volumetric_reading_affected',
                          'perometer_volumetric_reading_unaffected',
                          'perometer_LEFTminusRIGHT_volume_difference',
                          'perometer_LEFTminusRIGHT_volume_difference_percent',
                          'ldex',
                          'psfs',
                          'uefi',
                          'dash_work_score'
                          ]

all_ids = set(aim_data[id_col].unique())

df_outname = f'{df_outname_base}.xlsx'


df_list = []
df_tab_names = []

# first make the demographics sheet
demo_renamer = {i:j for i,j in zip(demos,demo_names)}

pre_arms1 = treatment_arms[0][0]
pre_arms2 = treatment_arms[1][0]
cut_df1 = aim_data[(aim_data['redcap_event_name']==pre_arms1 [0]) | (aim_data['redcap_event_name']==pre_arms1 [1])]
cut_df2 = aim_data[(aim_data['redcap_event_name']==pre_arms2[0]) | (aim_data['redcap_event_name']==pre_arms2[1])]

cut_df1 = cut_df1.set_index(id_col)
cut_df2 = cut_df2.set_index(id_col)

cut_df = cut_df1.combine_first(cut_df2)

n_index = np.arange(len(cut_df))
cut_df['index'] = n_index
cut_df[id_col] = cut_df.index
cut_df = cut_df.set_index('index')

cut_df = cut_df[demos]
cut_df = cut_df.rename(demo_renamer, axis='columns')

cut_df_ids = set(cut_df[id_col].unique())
missing_ids = all_ids - cut_df_ids
for mid in missing_ids:
    row = pd.Series({id_col:mid})
    cut_df = cut_df.append(row, ignore_index=True)

cut_df = cut_df.sort_values('study_id', 'index')
df_list.append(cut_df.copy())
df_tab_names.append('demographics')


for modality, ta in zip(treatment_modalities, treatment_arms):
    
    # first make the demographics sheet
    demo_renamer = {i:j for i,j in zip(demos,demo_names)}
    
    pre_arms = ta[0]
    cut_df = aim_data[(aim_data['redcap_event_name']==pre_arms[0]) | (aim_data['redcap_event_name']==pre_arms[1])]
    
    cut_df = cut_df[demos]
    cut_df = cut_df.rename(demo_renamer, axis='columns')
    
    cut_df_ids = set(cut_df[id_col].unique())
    missing_ids = all_ids - cut_df_ids
    for mid in missing_ids:
        row = pd.Series({id_col:mid})
        cut_df = cut_df.append(row, ignore_index=True)
    
    cut_df = cut_df.sort_values('study_id', 'index')
    df_list.append(cut_df.copy())
    df_tab_names.append('demographics')

    col_renamer = {i:j for i,j in zip(cols_of_interest, cols_of_interest_names)}
    for treat_time, arm_list in zip(treatment_times, ta):
        cut_df = aim_data[(aim_data['redcap_event_name']==arm_list[0]) | (aim_data['redcap_event_name']==arm_list[1])]
        cut_df = cut_df[cols_of_interest]
        cut_df = cut_df.rename(col_renamer, axis='columns')
        
        
        cut_df_ids = set(cut_df[id_col].unique())
        missing_ids = all_ids - cut_df_ids
        for mid in missing_ids:
            row = pd.Series({id_col:mid})
            cut_df = cut_df.append(row, ignore_index=True)
    
        cut_df = cut_df.sort_values('study_id', 'index')
        
        df_list.append(cut_df.copy())
        df_tab_names.append(f'{treat_time}_{modality}')



    writer = pd.ExcelWriter(df_outname, engine='xlsxwriter')
    for df, tname in zip(df_list,df_tab_names):
        df.to_excel(writer, sheet_name=tname)
    
    writer.save()
