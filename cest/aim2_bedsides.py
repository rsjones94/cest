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

df_outname = '/Users/skyjones/Desktop/cest_processing/aim2_bedsides.xlsx'


aim_token_loc = '/Users/skyjones/Documents/repositories/lymph_aim2_token.txt'

prefix = 'Donahue'


api_url = 'https://redcap.vanderbilt.edu/api/'
print('Contacting REDCap database...')

aim_token = open(aim_token_loc).read()
project = redcap.Project(api_url, aim_token)

aim_data_raw = project.export_records(export_survey_fields=False, export_data_access_groups=True)
aim_data = pd.DataFrame(aim_data_raw)
aim_data = aim_data[aim_data['redcap_repeat_instrument']!='photographs']

timepoints = ['visit_1_enrollment_arm_1',
                'visit_2_arm_1',
                'visit_3_arm_1',
                'visit_4_arm_1',
                'visit_5_arm_1']


id_col = 'study_id'

demos = [id_col,
         'age12',
         'race',
         'ethnicity',
         'sex',
         'neo_adjuvent_or_adjuvent___1',
         'neo_adjuvent_or_adjuvent___2',
         'is_patient_showing_signs_o',
         'date_of_ln_removal',
         'no_ln_removed',
         'number_of_metastatic_lymph'
         ]

demo_names = [id_col,
              'age',
              'race',
              'ethnicity',
              'sex',
              'neoadj_therapy',
              'adj_therapy',
              'signs_of_lymphedema',
              'date_of_ln_removal',
              'n_nodes_removed',
              'n_nodes_metastatic']

cols_of_interest = [id_col,
                    'mri_forename_id',
                    'mri_scan_id1',
                    'study_date',
                    'bcrl_risk_side2',
                    'bmi',
                    'ua_circum_involv',
                    'ua_circum_contra',
                    't_affected_vol_1a',
                    't_unaffected_vol_1a',
                    'true_vol_diff_1a',
                    'true_vol_diff_calc_1a',
                    'ldex'
                    ]

cols_of_interest_names = [id_col,
                          'mri_forename_id',
                          'mri_scan_id',
                          'study_date',
                          'lymphedema_affected_side',
                          'bmi',
                          'tape_circumference_upperarm_affected',
                          'tape_circumference_upperarm_unaffected',
                          'perometer_volumetric_reading_affected',
                          'perometer_volumetric_reading_unaffected',
                          'perometer_LEFTminusRIGHT_volume_difference',
                          'perometer_LEFTminusRIGHT_volume_difference_percent',
                          'ldex'
                          ]


all_ids = set(aim_data[id_col].unique())

out_df = pd.DataFrame()
df_list = []
df_tab_names = []

# first make the demographics sheet
demo_renamer = {i:j for i,j in zip(demos,demo_names)}
cut_df = aim_data[aim_data['redcap_event_name']==timepoints[0]]
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

# now make the timepoint data
col_renamer = {i:j for i,j in zip(cols_of_interest, cols_of_interest_names)}
for tp in timepoints:
    cut_df = aim_data[aim_data['redcap_event_name']==tp]
    cut_df = cut_df[cols_of_interest]
    cut_df = cut_df.rename(col_renamer, axis='columns')
    
    cut_df_ids = set(cut_df[id_col].unique())
    missing_ids = all_ids - cut_df_ids
    for mid in missing_ids:
        row = pd.Series({id_col:mid})
        cut_df = cut_df.append(row, ignore_index=True)
    
    cut_df = cut_df.sort_values('study_id', 'index')
    
    df_list.append(cut_df.copy())
    df_tab_names.append(tp)


writer = pd.ExcelWriter(df_outname, engine='xlsxwriter')
for df, tname in zip(df_list,df_tab_names):
    df.to_excel(writer, sheet_name=tname)
    
writer.save()
