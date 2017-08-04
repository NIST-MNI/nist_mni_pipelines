from scoop import futures, shared
import os
import iplScoopGenerateModel as gm

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'

if __name__ == '__main__':
  # setup data for parallel processing
  
  gm.generate_nonlinear_model_csv(
    'subjects_ldd_cut.lst',
    work_prefix='tmp_avg_std',
    options={
             'protocol': [
                          {'iter':4,   'level':8, },
                          {'iter':4,   'level':4, },
                          #{'iter':16,  'level':2, 'blur_int': None, 'blur_vel': None },
                          #{'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 2 },
                          #{'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 1 },
                         ],
             
             'start_level': 8,
             'refine': True,
             'cleanup':False,
             'debug':  True,
             'debias': True,
             'qc':     True,
             'incremental': True
            },
    #regress_model=['data/object_0_4.mnc'],
    model='data_ldd/object_0_0.mnc',
    mask='data_ldd/mask_0_0.mnc',
  )
