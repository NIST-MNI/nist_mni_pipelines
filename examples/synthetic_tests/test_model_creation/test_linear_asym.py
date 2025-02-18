import os
import ray


from ipl.model.generate_linear  import generate_linear_model_csv

@ray.remote
def generate_linear_model_csv_r(*args, **kwargs):
    return generate_linear_model_csv(*args, **kwargs)


if __name__ == '__main__':
  # setup data for parallel processing
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  ray.init()
  j0=generate_linear_model_csv_r.remote('subjects1.lst',
    work_prefix='tmp_lsq6',
    options={'symmetric':False,'reg_type':'-lsq6','objective':'-xcorr','iterations':4,'refine':True},
    model='ref.mnc',
    mask='mask.mnc')
    
  j1=generate_linear_model_csv_r.remote('subjects1.lst',
    work_prefix='tmp_lsq9',
    options={'symmetric':False,'reg_type':'-lsq9','objective':'-xcorr','iterations':4,'refine':True},
    model='ref.mnc',
    mask='mask.mnc')

  j2=generate_linear_model_csv_r.remote('subjects2.lst',
    work_prefix='tmp_lsq12',
    options={'symmetric':False,'reg_type':'-lsq12','objective':'-xcorr','iterations':4,'refine':True},
    model='ref.mnc',
    mask='mask.mnc')
  
  ray.wait([j0,j1,j2], num_returns=3)

