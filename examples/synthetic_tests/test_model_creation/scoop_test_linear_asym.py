import ray

import iplScoopGenerateModel as gm

if __name__ == '__main__':
  # setup data for parallel processing
  
  j0=gm.generate_linear_model_csv.remote('subjects1.lst',
    work_prefix='tmp_lsq6',
    options={'symmetric':False,'reg_type':'-lsq6','objective':'-xcorr','iterations':4,'refine':True},
    model='ref.mnc',
    mask='mask.mnc')
    
  j1=gm.generate_linear_model_csv.remote('subjects1.lst',
    work_prefix='tmp_lsq9',
    options={'symmetric':False,'reg_type':'-lsq9','objective':'-xcorr','iterations':4,'refine':True},
    model='ref.mnc',
    mask='mask.mnc')

  j2=gm.generate_linear_model_csv.remote('subjects2.lst',
    work_prefix='tmp_lsq12',
    options={'symmetric':False,'reg_type':'-lsq12','objective':'-xcorr','iterations':4,'refine':True},
    model='ref.mnc',
    mask='mask.mnc')
  
  ray.wait([j0,j1,j2],num_returns=3)
