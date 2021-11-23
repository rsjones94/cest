
import os
import glob
import shutil
import datetime
import operator


#bulk_folder = '/Users/skyjones/Desktop/cest_processing/data/bulk/'
#target_folder = '/Users/skyjones/Desktop/cest_processing/data/working/'
bulk_folder = '/Users/skyjones/Desktop/hiv_processing/data/bulk/'
target_folder = '/Users/skyjones/Desktop/hiv_processing/data/working/'

form = 'hiv' # hiv or lymph



######




bulkers = glob.glob(os.path.join(bulk_folder, '*'))
log_txt = ''
map_txt = ''

for bulkf in bulkers:
    
    aim_folder = bulkf.split('/')[-1]
    
    print(f'-----On {aim_folder}-----')
    
    subfolders = glob.glob(os.path.join(bulkf, '*'))
    for sf in subfolders:
        
        pt_id = sf.split('/')[-1]
        print(f'\t{pt_id}')
        dicom_folder = os.path.join(sf, 'DICOM')
        files = [f.name for f in os.scandir(dicom_folder)]
        
        
        arms = ['aff', 'cont']
        
        if form == 'lymph':
            arms_n = arms.copy()
        elif form == 'hiv':
            arms_n = ['ipsi', 'cont']
        
        for arm, arm_n in zip(arms, arms_n):
            
            if arm == 'cont' and aim_folder == 'aim1':
                continue
        
            if arm == 'aff':
                cestdixon_searcher = [f for f in files if ('cest' in f.lower() and 'cont' not in f.lower() and 'unaff' not in f.lower() and 'b1' not in f.lower() and 'low' not in f.lower())]
            elif arm == 'cont':
                cestdixon_searcher = [f for f in files if ('cest' in f.lower() and ('cont' in f.lower() or 'unaff' in f.lower()) and 'b1' not in f.lower() and 'low' not in f.lower())]
                
            if len(cestdixon_searcher) == 1:
                cestdixon_root = cestdixon_searcher[0]
                cestdixon_file = os.path.join(dicom_folder, cestdixon_root)
                cestdixon_newname = os.path.join(target_folder, aim_folder, 'raw', f'{pt_id}_cestdixon_{arm}.DCM')
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
                    cestdixon_newname = os.path.join(target_folder, aim_folder, 'raw', f'{pt_id}_cestdixon_{arm}.DCM')
                    shutil.copy(cestdixon_file, cestdixon_newname)
                    
                    map_txt = map_txt + f'\n{aim_folder}, {pt_id}, {arm}, {cestdixon_root}'
                else:
                    if arm == 'cont':
                        cestdixon_searcher_potential = [f for f in files if ('cest' in f.lower()and 'b1' not in f.lower() and 'low' not in f.lower())] #  and 'cont' not in f.lower() and 'unaff' not in f.lower() 
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
                            cestdixon_newname = os.path.join(target_folder, aim_folder, 'raw', f'{pt_id}_cestdixon_{arm}.DCM')
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
log_file = os.path.join(bulk_folder, 'prep_log.txt')
with open(log_file, 'w+') as logger:
    logger.write(log_txt)
    
# write map
map_file = os.path.join(bulk_folder, 'prep_map.txt')
with open(map_file, 'w+') as mapper:
    mapper.write(map_txt)
        