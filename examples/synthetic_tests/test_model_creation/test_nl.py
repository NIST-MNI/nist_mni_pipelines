import ray

from ipl.model.generate_nonlinear  import generate_nonlinear_model_csv

if __name__ == '__main__':
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
    #stop_early=4,
    #skip=0,
  )
