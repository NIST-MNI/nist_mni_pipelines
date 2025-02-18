from scoop import futures, shared

import iplScoopGenerateModel as gm

if __name__ == '__main__':
  # setup data for parallel processing
  gm.generate_linear_model_csv('big_subjects.lst',
    work_prefix='tmp_nobias',
    options={'symmetric':False,
             'reg_type':'-lsq6',
             'objective':'-xcorr',
             'iterations':4,
             'biascorr':False})
