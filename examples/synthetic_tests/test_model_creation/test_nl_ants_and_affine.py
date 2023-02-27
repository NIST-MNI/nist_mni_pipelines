import ray
import os

from ipl.model_ants.generate_nonlinear  import generate_nonlinear_model_csv

if __name__ == '__main__':
  
  # limit number of threads
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  ray.init(num_cpus=9)
  # setup data for parallel processing
  generate_nonlinear_model_csv('subjects.lst',
    work_prefix='tmp_nl_ants_and_affine',
    options={'symmetric':False,
             'protocol': [{'iter':8,'level':4},
                          ],
             'cleanup': False,
             'parameters':
              {
                'convergence':'1.e-9,10',
                'cost_function': 'CC',
                'cost_function_par': '1,4',
                'transformation': 'SyN[ .1, 3, 0 ]',
                'use_histogram_matching': True,
                'winsorize_intensity': {'low':0.01, 'high':0.99},
             },
             'lin_parameters':{'only_rigid': False},
             'start_level':16,
             'grad_step':0.1,
             'refine': False,
             'qc':False
            },
    model='test_data/ellipse_1.mnc',
    mask='test_data/mask.mnc'
  )
