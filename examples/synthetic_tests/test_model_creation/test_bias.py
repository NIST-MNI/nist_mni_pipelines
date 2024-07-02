import os
import ray


from ipl.model.generate_linear  import generate_linear_model_csv

if __name__ == '__main__':
  # setup data for parallel processing
  os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
  ray.init()

  generate_linear_model_csv(
   'big_subjects.lst',
    work_prefix=           'tmp_bias_lsq6',
    options={'symmetric':  False,
             'reg_type':   '-lsq6',
             'objective':  '-xcorr',
             'iterations': 4,
             'biascorr':   True,
             'biasdist':   40,
           })
