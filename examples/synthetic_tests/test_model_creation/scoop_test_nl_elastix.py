from scoop import futures, shared

import iplScoopGenerateModel as gm
import os

if __name__ == '__main__':
  
  # limit number of threads
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  # setup data for parallel processing
  gm.generate_nonlinear_model_csv('subjects.lst',
    work_prefix='tmp_nl_elastix',
    options={'symmetric':False,
             
             'protocol': [{'iter':4,'level':16},
                          {'iter':8,'level':8},
                          ],
             
             'cleanup': False,
             'use_elastix': True,
             
             'parameters': 
              {
                '16': {'grid_spacing': 16,'resolutions':1,'pyramid':'8 8 8', 'downsample_grid':4 },
                '8':  {'grid_spacing': 8, 'resolutions':1,'pyramid':'4 4 4', 'downsample_grid':2 }
              },
              
              'refine': True,
              'use_median': True,
            },
    model='test_data/ellipse_1.mnc',
    mask='test_data/mask.mnc'
            
  )
