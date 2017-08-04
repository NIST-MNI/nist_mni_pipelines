from scoop import futures, shared

import iplScoopGenerateModel as gm
import os

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'

if __name__ == '__main__':
  # setup data for parallel processing
  
  gm.regress_ldd_csv(
    'subjects_lin.lst',
    work_prefix='tmp_regress_LCC_lin_4mm',
    options={
             'protocol': [
                          {'iter':20,   'level':4, 'blur_int': None, 'blur_vel': None },
                          #{'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 2 },
                          #{'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 1 },
                         ],
             
             'parameters': {'smooth_update':2,
                            'smooth_field':2,
                            'conf': { 32:40, 16:40, 8:40, 4:40, 2:40 },
                            'hist_match':True,
                            'max_step':  2.0,
                            'LCC':True },
             
             'start_level': 8,
             'refine': True,
             'cleanup': False,
             'debug': True,
             'debias': True,
             'qc': True,
             'incremental': False
            },
    #regress_model=['data/object_0_4.mnc'],
    model='data/object_0_4.mnc',
    mask='data/mask_0_4.mnc',
    int_par_count=1,
  )
