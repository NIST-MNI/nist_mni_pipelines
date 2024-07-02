import os
import ray

from ipl.model.generate_linear  import generate_linear_model_csv

if __name__ == '__main__':
  # setup data for parallel processing
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  ray.init()
  
  generate_linear_model_csv('subjects.lst',
    work_prefix='tmp_lsq6_sym_skip',
    options={'symmetric':True, 
             'reg_type':'-lsq6', 
             'objective':'-xcorr', 
             'iterations':4,
             'refine':True, 
             'cleanup':False, 
             'linreg':'bestlinreg_20180117' },
    model='test_data/ellipse_1.mnc',
    mask='test_data/mask.mnc',
    skip=0,stop_early=3)
    
  generate_linear_model_csv('subjects.lst',
    work_prefix='tmp_lsq6_sym_skip',
    options={'symmetric':True, 
             'reg_type':'-lsq6', 
             'objective':'-xcorr', 
             'iterations':4,
             'refine':True, 
             'cleanup':False, 
             'linreg':'bestlinreg_20180117' },
    model='test_data/ellipse_1.mnc',
    mask='test_data/mask.mnc',
    skip=2,stop_early=10)
    
