import os
import ray


from ipl.model_ldd.regress_ldd  import regress_ldd_csv

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'

if __name__ == '__main__':
  # setup data for parallel processing
  ray.init()
  
  regress_ldd_csv(
    'subjects_ldd.lst',
    work_prefix='tmp_regress_LCC_ldd_sym2',
    options={
             'protocol': [
                          #{'iter':10,   'level':8, 'blur_int': None, 'blur_vel': None },
                          {'iter':10,   'level':4, 'blur_int': None, 'blur_vel': None },
                          #{'iter':16,  'level':2, 'blur_int': None, 'blur_vel': None },
                          #{'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 2 },
                          #{'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 1 },
                         ],
             
             'parameters': {'smooth_update':2,
                            'smooth_field':2,
                            'conf': { 32:200, 16:200, 8:200, 4:200, 2:40 },
                            'LCC':True },
             
             'start_level': 8,
             'refine': True,
             'cleanup':False,
             'debug':  True,
             'debias': False,
             'qc':     True,
             'incremental': True,
             'remove0':True,
             'sym':True,
            },
    #regress_model=['data/object_0_4.mnc'],
    model='data_ldd/object_0_4.mnc',
    mask='data_ldd/mask_0_4.mnc',
    int_par_count=1,
  )
