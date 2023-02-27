import ray

import iplScoopGenerateModel as gm

if __name__ == '__main__':
  # setup data for parallel processing
  
  #j0=gm.generate_linear_model_csv.remote('subjects.lst',
    #work_prefix='tmp_lsq9_std',
    #options={'symmetric':False, 'reg_type':'-lsq9', 'objective':'-xcorr', 'iterations':4, 'refine':True},
    #model='test_data/ref.mnc',
    #mask='test_data/mask.mnc')
    
  j1=gm.generate_linear_model_csv.remote('subjects.lst',
    work_prefix='tmp_lsq9_20171229_2',
    options={'symmetric':False,'reg_type':'-lsq9', 'objective':'-xcorr', 'iterations':4, 'refine':True, 'linreg':'bestlinreg_20171223'},
    model='test_data/ref.mnc',
    mask='test_data/mask.mnc')

  
  ray.wait([j1])
