from scoop import futures, shared

import iplScoopGenerateModel as gm
import os

if __name__ == '__main__':
  
  # limit number of threads
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  # setup data for parallel processing
  gm.generate_nonlinear_model_csv('subjects.lst',
    work_prefix='tmp_nl_ants_nr',
    options={'symmetric':False,
             
             'protocol': [{'iter':4,'level':4},
                          {'iter':4,'level':2},
                          ],
             
             'cleanup': True,
             'use_ants': True,
             
             'parameters': 
              {
                'conf':   { 32:40,16:40,8:40,4:40,2:40 },
                'shrink': { 32:2,16:2,8:2,4:2,2:2 },
                'blur':   { 32:2,16:2,8:2,4:2,2:2 },
              },
              
              'refine': False
            },
    model='test_data/ellipse_1.mnc',
    mask='test_data/mask.mnc'
            
  )
