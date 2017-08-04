from scoop import futures, shared

import iplScoopGenerateModel as gm

if __name__ == '__main__':
  # setup data for parallel processing
  gm.regress_ldd_csv('subjects_lim.lst',
    work_prefix='tmp_regress_lim_nr_v2',
    options={
             'protocol': [
                          {'iter':4,  'level':4, 'blur_int': None, 'blur_vel': 4 },
                          {'iter':4,  'level':4, 'blur_int': None, 'blur_vel': 2 },
                          {'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 2 },
                          {'iter':4,  'level':2, 'blur_int': None, 'blur_vel': 1 },
                         ],
             'parameters': {'smooth_update':2,
                            'smooth_field':2,
                            'conf': { 32:40, 16:40, 8:40, 4:40, 2:40, 1:40 },
                            'hist_match':True,
                            'max_step':  4.0 },
             'start_level':16,
             'refine': False,
             'cleanup': False,
             'debug': True,
             'debias': True,
             'qc': True,
            },
    #regress_model=['data/object_0_4.mnc'],
    model='data/object_0_4.mnc',
    mask='data/mask_0_4.mnc',
    int_par_count=1,
  )
