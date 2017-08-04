from iplMincTools      import mincTools,mincError
import traceback
import os

from scoop import futures, shared

import iplScoopGenerateModel as gm
# setup data for parallel processing
# have to be at global level

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'

if __name__ == '__main__':
    try:

        res=gm.regress_csv(
            'subjects.lst',
            work_prefix='tmp_regress_std_dd_nr_nd',
            options={
                    'protocol': [
                                {'iter':4,  'level':4, 'blur_int': None, 'blur_def': None },
                                {'iter':4,  'level':2, 'blur_int': None, 'blur_def': None },
                                {'iter':4,  'level':1, 'blur_int': None, 'blur_def': None },
                                ],
                    'start_level':8,
                    'refine':  False,
                    'cleanup': False,
                    'debug':   True,
                    'debias':  False,
                    'qc':      True,
                    'nl_mode': 'dd',
                    },
            #regress_model=['data/object_0_4.mnc'],
            model='data/object_0_4.mnc',
            mask='data/mask_0_4.mnc',
            int_par_count=1,
        )
        
        # 
        
        
    except mincError as e:
        print "Exception in regress_csv:{}".format(str(e))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print "Exception in regress_csv:{}".format(sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        raise


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on
