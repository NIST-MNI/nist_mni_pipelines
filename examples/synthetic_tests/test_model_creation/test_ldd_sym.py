import os
import ray


from ipl.model_ldd.generate_nonlinear_ldd  import generate_ldd_model_csv

if __name__ == '__main__':
  # setup data for parallel processing
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  ray.init()
    
  generate_ldd_model_csv('subjects.lst',
    work_prefix='tmp_ldd_sym',
    options={'symmetric':True,
             'refine':True,
             'protocol': [{'iter':4,'level':16},
                          {'iter':4,'level':8},
                          {'iter':4,'level':4},
                         ],
             'parameters': {'smooth_update':2,
                            'smooth_field':2,
                            'conf': { 32:20,16:20,8:20,4:20,2:20 } }
            },
    model='test_data/ellipse_1.mnc',
    mask='test_data/mask.mnc'
            
  )
