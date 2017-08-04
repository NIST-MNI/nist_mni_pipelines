from scoop import futures, shared

import iplScoopGenerateModel as gm

if __name__ == '__main__':
  # setup data for parallel processing
  
  j0=futures.submit( gm.generate_linear_model_csv,'subjects.lst',
    work_prefix='tmp_lsq6_downsample',
    options={'symmetric':False, 'reg_type':'-lsq6', 'objective':'-xcorr', 'iterations':4, 'downsample':1,'cleanup':True,'refine':False},
    model='test_data/ref.mnc',
    mask='test_data/mask.mnc')
    
  futures.wait([j0], return_when=futures.ALL_COMPLETED)
