
import os
import glob
import shutil
import datetime
import operator


bulk_folder = '/Users/skyjones/Desktop/hiv_processing/data/bulk/'
target_folder = '/Users/skyjones/Desktop/hiv_processing/data/working/'



######

sigs = ['star', 't1w_highres', 't2w_highres']
alt_sigs = [None, 'vista', None]


bulkers = glob.glob(os.path.join(bulk_folder, '*/'))


for sig, asig in zip(sigs, alt_sigs):
    
    
    log_txt = ''
    map_txt = ''
    
    print(f'\t\tSignature: {sig}!!!!!')

    for bulkf in bulkers:
        
        aim_folder = os.path.basename(os.path.normpath(bulkf))
        
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
            
            
            arms = ['ipsi', 'cont']
            
            for arm in arms:
                
            
                if arm == 'ipsi':
                    cestdixon_searcher = [f for f in files if (sig in f.lower() and ('cont' not in f.lower() and 'unaff' not in f.lower()) and 'm0' not in f.lower() and 'source' not in f.lower())]
                elif arm == 'cont':
                    cestdixon_searcher = [f for f in files if (sig in f.lower() and ('cont' in f.lower() or 'unaff' in f.lower()) and 'm0' not in f.lower() and 'source' not in f.lower())]
                    
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
                        cestdixon_newname = os.path.join(raw_fol, f'{pt_id}_{sig}_{arm}.DCM')
                        shutil.copy(cestdixon_file, cestdixon_newname)
                        
                        map_txt = map_txt + f'\n{aim_folder}, {pt_id}, {arm}, {cestdixon_root}'
                    else:
                        
                        if asig is None:
                            map_txt = map_txt + f'\n{aim_folder}, {pt_id}, {arm}, NONE'
                            
                        else:
                            
                            if arm == 'ipsi':
                                cestdixon_searcher_potential = [f for f in files if (asig in f.lower() and ('cont' not in f.lower() and 'unaff' not in f.lower()) and 'm0' not in f.lower() and 'source' not in f.lower())]
                            elif arm == 'cont':
                                cestdixon_searcher_potential = [f for f in files if (asig in f.lower() and ('cont' in f.lower() or 'unaff' in f.lower()) and 'm0' not in f.lower() and 'source' not in f.lower())]
                            
                            print(f'{arm}: no matches.')
                            if len(cestdixon_searcher_potential) >= 1:
                                cestdixon_searcher = cestdixon_searcher_potential
                                
                                
                                addon = f' (index 0 selected) [SECONDARY SIGNATURE {asig}]'
                                cestdixon_root = cestdixon_searcher[0]
                                cestdixon_file = os.path.join(dicom_folder, cestdixon_root)
                                cestdixon_newname = os.path.join(raw_fol, f'{pt_id}_{sig}_{arm}.DCM')
                                shutil.copy(cestdixon_file, cestdixon_newname)
                                
                                map_txt = map_txt + f'\n{aim_folder}, {pt_id}, {arm}, {cestdixon_root}'
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
        