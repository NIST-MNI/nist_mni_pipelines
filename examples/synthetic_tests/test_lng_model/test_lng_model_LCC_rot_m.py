from scoop import futures, shared
import os
import iplScoopGenerateModel as gm
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'

if __name__ == '__main__':
  # setup data for parallel processing
  
  gm.regress_ldd_csv(
    'subjects_rot_2.lst',
    work_prefix='tmp_regress_LCC_rot_inc_2ba_std',
    options={
             'protocol': [
                          {'iter':16,   'level':4, 'blur_int': None, 'blur_vel': None },
                          #{'iter':2,   'level':4, 'blur_int': None, 'blur_vel': None },
                          #{'iter':16,  'level':2, 'blur_int': None, 'blur_vel': None },
                          #{'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 2 },
                          #{'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 1 },
                         ],
             
             'parameters': {'smooth_update':2,
                            'smooth_field':2,
                            'conf': { 32:40, 16:40, 8:40, 4:40, 2:40 },
                            'LCC':False },
             
             'start_level': 4,
             'refine':  True,
             'cleanup': False,
             'debug':   True,
             'debias':  True,
             'qc': True,
             'incremental': True
            },
    #regress_model=['data/object_0_4.mnc'],
    model='data_rot/object_0_4.mnc',
    mask='data_rot/mask_0_4.mnc',
    int_par_count=1,
  )
