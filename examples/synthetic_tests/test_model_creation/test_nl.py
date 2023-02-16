import ray
import os

from ipl.model.generate_nonlinear  import generate_nonlinear_model_csv

if __name__ == '__main__':
  # setup data for parallel processing
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  ray.init(num_cpus=9)

  generate_nonlinear_model_csv('subjects.lst',
    work_prefix='tmp_nl_asym',
    options={'symmetric':False,
             'protocol': [{'iter':4,'level':16},
                          {'iter':4,'level':8}, ],
            'cleanup':True
            },
    model='test_data/ellipse_1.mnc',
    mask='test_data/mask.mnc'
  )
