
import os
import glob
import shutil
import datetime
import operator
import sys

import pandas as pd

bulk_folder = '/Users/skyjones/Desktop/cest_processing/data/bulk/'
target_folder = '/Users/skyjones/Desktop/cest_processing/data/working/'



######

sigs = ['angio']
excls = ['vwip']

organize = True


bulkers = glob.glob(os.path.join(bulk_folder, '*/'))


for sig in sigs:
    
    log_txt = ''
    map_txt = ''
    
    print(f'\t\tSignature: {sig}!!!!!')

    for bulkf in bulkers:
        
        aim_folder = os.path.basename(os.path.normpath(bulkf))
        
        if aim_folder != 'aim3':
            continue
        
        raw_fol = os.path.join(target_folder, aim_folder, f'raw_{sig}')
        if not os.path.exists(raw_fol):
            os.mkdir(raw_fol)
        
        print(f'-----On {aim_folder}-----')
        
        subfolders = glob.glob(os.path.join(bulkf, '*'))
        for sf in subfolders:
            
            pt_id = sf.split('/')[-1]
            print(f'\t{pt_id}')
            dicom_folder = os.path.join(sf, 'DICOM')
            files = [f.name for f in os.scandir(dicom_folder)]
            
            
            
            arms = ['aff', 'cont']
            
            for arm in arms:
                
            
                if arm == 'aff':
                    cestdixon_searcher = [f for f in files if (sig in f.lower() and ('cont' not in f.lower() and 'unaff' not in f.lower()) and 'm0' not in f.lower() and 'source' not in f.lower())]
                elif arm == 'cont':
                    cestdixon_searcher = [f for f in files if (sig in f.lower() and ('cont' in f.lower() or 'unaff' in f.lower()) and 'm0' not in f.lower() and 'source' not in f.lower())]
                
                cestdixon_searcher = [i for i in cestdixon_searcher if not any([j in i.lower() for j in excls])]
                
                if len(cestdixon_searcher) == 1:
                    cestdixon_root = cestdixon_searcher[0]
                    cestdixon_file = os.path.join(dicom_folder, cestdixon_root)
                    cestdixon_newname = os.path.join(raw_fol, f'{pt_id}_{sig}_{arm}.DCM')
                    shutil.copy(cestdixon_file, cestdixon_newname)
                    
                    map_txt = map_txt + f'\n{aim_folder}, {pt_id}, {arm}, {cestdixon_root}'
                    
                else:
                    addon = ''
                    if len(cestdixon_searcher) > 1:
                        # if there is more than option, pick the one with the latest timestamp
                        cestdixon_stamps = [f.split('/')[-1].split('.')[3] for f in cestdixon_searcher]
                        cestdixon_times = [datetime.datetime.strptime(i, '%H-%M-%S') for i in cestdixon_stamps]
                        min_index, min_value = min(enumerate(cestdixon_times), key=operator.itemgetter(1))
                        
                        addon = f' (index {min_index} selected)'
                        cestdixon_root = cestdixon_searcher[min_index]
                        cestdixon_file = os.path.join(dicom_folder, cestdixon_root)
                        cestdixon_newname = os.path.join(target_folder, aim_folder, 'raw', f'{pt_id}_{sig}_{arm}.DCM')
                        shutil.copy(cestdixon_file, cestdixon_newname)
                        
                        map_txt = map_txt + f'\n{aim_folder}, {pt_id}, {arm}, {cestdixon_root}'
                    else:
                        if arm == 'cont':
                            cestdixon_searcher_potential = [f for f in files if (sig in f.lower()and 'm0' not in f.lower() and 'source' not in f.lower())] #  and 'cont' not in f.lower() and 'unaff' not in f.lower() 
                            cestdixon_searcher_potential = [i for i in cestdixon_searcher_potential if not any([j in i.lower() for j in excls])]
                            print(f'Contralateral: no matches. (aff matches: {len(cestdixon_searcher_potential)})')
                            if len(cestdixon_searcher_potential) > 1:
                                cestdixon_searcher = cestdixon_searcher_potential
                                # if there are no cont/unaff files, the last of the "affected" files could be it
                                cestdixon_stamps = [f.split('/')[-1].split('.')[3] for f in cestdixon_searcher]
                                cestdixon_times = [datetime.datetime.strptime(i, '%H-%M-%S') for i in cestdixon_stamps]
                                max_index, max_value = max(enumerate(cestdixon_times), key=operator.itemgetter(1))
                                
                                addon = f' (index {max_index} selected) [ALTERNATE MATCHING]'
                                cestdixon_root = cestdixon_searcher[max_index]
                                cestdixon_file = os.path.join(dicom_folder, cestdixon_root)
                                cestdixon_newname = os.path.join(target_folder, aim_folder, 'raw', f'{pt_id}_{sig}_{arm}.DCM')
                                shutil.copy(cestdixon_file, cestdixon_newname)
                                
                                map_txt = map_txt + f'\n{aim_folder}, {pt_id}, {arm}, {cestdixon_root}'
                            else:
                                map_txt = map_txt + f'\n{aim_folder}, {pt_id}, {arm}, NONE'
                    
                        else:
                            map_txt = map_txt + f'\n{aim_folder}, {pt_id}, {arm}, NONE'
                        
                        
                    # log the problem
                    log_txt = log_txt + f'\n{aim_folder}, {pt_id}, {arm}, {len(cestdixon_searcher)}' + addon
                    for potential_file in cestdixon_searcher:
                        log_txt = log_txt + f'\n\t{potential_file}'
                        
                
        
    # write log
    log_file = os.path.join(bulk_folder, f'prep_log_{sig}.txt')
    with open(log_file, 'w+') as logger:
        logger.write(log_txt)
        
    # write map
    map_file = os.path.join(bulk_folder, f'prep_map_{sig}.txt')
    with open(map_file, 'w+') as mapper:
        mapper.write(map_txt)



if organize:
    
    org_folder = '/Users/skyjones/Desktop/cest_processing/data/working/aim3/lyangio_package/'
    excel_name = '/Users/skyjones/Desktop/cest_processing/data/working/figures/aim3/response.xlsx'
    raw_folder = '/Users/skyjones/Desktop/cest_processing/data/working/aim3/raw_angio/'
    
    the_df = pd.read_excel(excel_name)
    
    for i,row in the_df.iterrows():
        sid = row['study_id']
        
        sid_path = os.path.join(org_folder, sid)
        
        times = ['pre', 'post']
        arms = ['aff', 'cont']
        treats = ['cdt', 'dual']
        
        os.mkdir(sid_path)
        
        
        for treat in treats:
            treat_folder = os.path.join(sid_path, treat)
            os.mkdir(treat_folder)
            for arm in arms:
                arm_folder = os.path.join(treat_folder, arm)
                os.mkdir(arm_folder)
                for time in times:
                    time_folder = os.path.join(arm_folder, time)
                    os.mkdir(time_folder)
                    
                    scan_id = row[f'{treat}_{time}_names']
                    orig_file = os.path.join(raw_folder, f'{scan_id}_angio_{arm}.DCM')
                    target_file = os.path.join(time_folder, f'{scan_id}_angio_{arm}.DCM')
                    
                    if os.path.exists(orig_file):
                        shutil.copyfile(orig_file, target_file)
        
        
        
        
        
        