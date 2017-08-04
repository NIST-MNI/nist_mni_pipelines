from scoop import futures, shared

import iplScoopGenerateModel as gm

if __name__ == '__main__':
  # setup data for parallel processing
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  
  gm.regress_ldd_csv(
    'subjects_1.lst',
    work_prefix='tmp_regress_LCC_1',
    options={
             'protocol': [
                          {'iter':4,  'level':4, 'blur_int': None, 'blur_vel': None },
                          {'iter':4,  'level':2, 'blur_int': None, 'blur_vel': None },
                          {'iter':4,  'level':1, 'blur_int': None, 'blur_vel': None },
                          #{'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 2 },
                          #{'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 1 },
                         ],
             
             'parameters': {'smooth_update':2,
                            'smooth_field':2,
                            'conf': { 32:40, 16:40, 8:40, 4:40, 2:40 },
                            'hist_match':False,
                            'max_step':  2.0,
                            'LCC':True },
             
             'start_level': 8,
             'refine': False,
             'cleanup': False,
             'debug': True,
             'debias': False,
             'qc': True,
             'incremental': False
            },
    #regress_model=['data/object_0_4.mnc'],
    model='data/object_0_4.mnc',
    mask='data/mask_0_4.mnc',
  )
