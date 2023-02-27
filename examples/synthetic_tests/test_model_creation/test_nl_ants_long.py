import ray
import os

from ipl.model_ants.generate_nonlinear  import generate_nonlinear_model_csv

if __name__ == '__main__':
  
  # limit number of threads
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  ray.init(num_cpus=8)
  # setup data for parallel processing
  generate_nonlinear_model_csv('subjects2.lst',
    work_prefix='tmp_nl_asym_ants2',
    options={'symmetric':False,
             'protocol': [{'iter':24,'level':4},
                          ],
             'cleanup': True,
             'parameters':
              {
                'convergence':'1.e-9,10',
                'cost_function': 'CC',
                'cost_function_par': '1,4',
                'transformation': 'SyN[ .1, 3, 0 ]',
                'use_histogram_matching': True,
                'winsorize_intensity': {'low':0.01, 'high':0.99},

                'conf':  {'8':100, '4':100,'2':50,'1':20},
                'shrink':{'8': 8, '4':4, '2':2, '1':1},
                'blur':  {'8': 6, '4':3, '2':2, '1':0},
             },
             'lin_parameters':{'only_rigid': True},
             'start_level':8,
             'grad_step':0.1,
             'refine': False,
             'qc':False
            },
    model='test_data2/ref.mnc'
  )
