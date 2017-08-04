from scoop import futures, shared

import iplScoopGenerateModel as gm
import os

if __name__ == '__main__':
  
  # limit number of threads
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  # setup data for parallel processing
  gm.generate_nonlinear_model_csv('subjects.lst',
    work_prefix='tmp_nl_dd',
    options={'symmetric':False,
             
             'protocol': [{'iter':4,'level':4},
                          {'iter':4,'level':2},
                          ],
             
             'cleanup': True,
             'use_dd': True,
             
             'parameters': 
              {
                'conf': { 32:40,16:40,8:40,4:40,2:40 },
                'smooth_update':2,
                'smooth_field':1,
                'update_rule':0,
                'grad_type':0,
                'max_step':1.0,
                'hist_match':True 
              }
            },
    model='ref.mnc',
    mask='mask.mnc'
            
  )
