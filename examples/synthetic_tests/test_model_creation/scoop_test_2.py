from scoop import futures, shared

import iplScoopGenerateModel as gm

if __name__ == '__main__':
  # setup data for parallel processing
  
  j0=futures.submit( gm.generate_linear_model_csv,'subjects1.lst',
    work_prefix='tmp_lsq6',
    options={'symmetric':False,'reg_type':'-lsq6','objective':'-xcorr','iterations':4,'refine':True},
    model='ref.mnc',
    mask='mask.mnc')
    
  j1=futures.submit( gm.generate_linear_model_csv,'subjects1.lst',
    work_prefix='tmp_lsq9',
    options={'symmetric':False,'reg_type':'-lsq9','objective':'-xcorr','iterations':4,'refine':True},
    model='ref.mnc',
    mask='mask.mnc')

  j2=futures.submit( gm.generate_linear_model_csv,'subjects2.lst',
    work_prefix='tmp_lsq12',
    options={'symmetric':False,'reg_type':'-lsq12','objective':'-xcorr','iterations':4,'refine':True},
    model='ref.mnc',
    mask='mask.mnc')
  
  futures.wait([j0,j1,j2], return_when=futures.ALL_COMPLETED)
