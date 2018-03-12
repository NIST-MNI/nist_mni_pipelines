from scoop import futures, shared

import iplScoopGenerateModel as gm

if __name__ == '__main__':
  # setup data for parallel processing
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  
  gm.generate_ldd_model_csv('subjects.lst',
    work_prefix='tmp_ldd_nr',
    options={'symmetric':False,
             'refine':False,
             'protocol': [{'iter':4,'level':4},
                          {'iter':4,'level':2},
                         ],
             
             'parameters': {'smooth_update':2,
                            'smooth_field':1,
                            'conf': { 32:40,16:40,8:40,4:40,2:40 } }
            },
    model='test_data/ellipse_1.mnc',
    mask='test_data/mask.mnc'
            
  )
