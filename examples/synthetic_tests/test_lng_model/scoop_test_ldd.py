import os
import ray


from ipl.model_ldd.generate_nonlinear_ldd  import generate_ldd_model_csv


if __name__ == '__main__':
  ray.init()
  # setup data for parallel processing
  generate_ldd_model_csv('subjects_cut.lst',
    work_prefix='tmp_ldd',
    options={'symmetric':False,
             'refine':True,
             'protocol': [{'iter':4,'level':8},
                          {'iter':4,'level':4},
                         ],
             'parameters': {'smooth_update':2,
                            'smooth_field':2,
                            'conf': { 32:20,16:20,8:20,4:20,2:20,1:20 } }
            }
            
  )
