import os
import ray

from ipl.model.generate_nonlinear  import generate_nonlinear_model_csv

if __name__ == '__main__':
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  ray.init(num_cpus=4)
  # setup data for parallel processing
  generate_nonlinear_model_csv('subjects.lst',
    work_prefix='nl_model',
    options={'symmetric':False,
             'protocol': [{'iter':4,'level':16},
                          {'iter':4,'level':8},
                          ],
             'cleanup': True
            },
    model='test_data/ellipse_1.mnc',
    mask='test_data/mask.mnc',
  )
