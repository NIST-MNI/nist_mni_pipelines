#! /usr/bin/env python
import os
from iplPatient import *  # class to store all the patient data
import argparse
import re
import copy
import csv

def load_volume_std(template,in_file,sep=',',exclude=[]):
    
    vols=copy.deepcopy(template)
    # load all data first
    with open(in_file) as f:
        for l in f:
            ll=l.rstrip("\n").split(sep)
            if ll[0] in vols:
                vols[ll[0]]=float(ll[1])
            else:
                if not ll[0] in exclude:
                    print(("{} unexpected key found:{}".format(in_file,ll[0])))
    return vols

def list_to_dict(lll):
    return {i:None for i in lll}
    

def query_volumes(options):
    output_dir=options.prefix
    patients = {}
    
    with open(options.list) as p:
        for line in p:
            
            sp = line[:-1].split(',')

            size = len(sp)  # depending the number of items not all information was given
            if size < 3:
                print(' -- Line error: ' + str(len(sp)))
                print('     - Minimum format is :  id,visit,t1')
                continue

            id = sp[0]  # set id
            visit = sp[1]  # set visit
            patientdir="{}/{}/".format(output_dir,id)
            
            #vol_file="{0}/{1}/{2}/vol/vol_{1}_{2}.txt".format(output_dir,id,visit)
            #deep_file="{0}/{1}/{2}/vol/deep_{1}_{2}.txt".format(output_dir,id,visit)
            
            #if os.path.exists(vol_file):
                #print_volumes(vol_file,id,visit)
            #else:
                #print("Missing:{}".format(vol_file))
                
            if id not in patients:  # search key in the dictionary
                # check if the pickle file exists
                pickle_file=patientdir + id + '.pickle'
                if not os.path.exists(pickle_file):
                    print(("{} file is missing!".format(pickle_file)))
                else:
                    patients[id]=LngPatient.read(pickle_file)
    # now we have all the patients info loadade in memory
    
    
    
    # figure out what we have 
    
    exclude=["id","visit"]
    
    lob_headers=[
        "ScaleFactor","T1_SNR",
        "Age","ICC_vol","CSF_vol","GM_vol","WM_vol","scale","parietal_right_gm",
        "lateral_ventricle_left",
        "occipital_right_gm",
        "parietal_left_gm","occipital_left_gm",
        "lateral_ventricle_right",
        "globus_pallidus_right","globus_pallidus_left",
        "putamen_left","putamen_right",
        "frontal_right_wm","brainstem","subthalamic_nucleus_right", "fornix_left",
        "frontal_left_wm", "subthalamic_nucleus_left", 
        "caudate_left",
        "occipital_right_wm","caudate_right",
        "parietal_left_wm","temporal_right_wm","cerebellum_left",
        "occipital_left_wm","cerebellum_right",
        "temporal_left_wm",
        "thalamus_left", "parietal_right_wm",
        "thalamus_right","frontal_left_gm",
        "frontal_right_gm","temporal_left_gm","temporal_right_gm","3rd_ventricle","4th_ventricle","fornix_right",
        "extracerebral_CSF"    
    ]
    
    deep_headers=[
        'ScaleFactor',
        'putamen_left',
        'putamen_right',
        'pons',
        'middle_cerebellar_peduncle_left',
        'middle_cerebellar_peduncle_right',
        'cerebellum',
        'medulla',
        'globus_pallidus_left',
        'globus_pallidus_right',
        'fornix_left',
        'fornix_right',
        'caudate_left',
        'caudate_right',
        'thalamus_left',
        'thalamus_right']
    
    hc_headers  =['ScaleFactor','rHC','lHC']
    am_headers  =['ScaleFactor','rAM','lAM']
    vent_headers=['ScaleFactor','Ventricles']
    
    templates={
        'lobes':list_to_dict(lob_headers),
        'lng_lobes':list_to_dict(lob_headers),
        'deep':list_to_dict(deep_headers),
        'vent':list_to_dict(vent_headers),
        'hc':list_to_dict(hc_headers),
        'am':list_to_dict(am_headers),
        'id':{'id':None,'visit':None}
        }
    
    measurements=[]
    kk={'id':True, 'lobes':False,'lng_lobes':False,'deep':False,'vent':False,'hc':False,'am':False}
    print("Found following datasets:")
    for i,p in patients.items():
        #print("{} - {}".format(p.id,' '.join(p.keys())))
        for j,t in p.items():
            #print(t.vol.keys())
            vols={'id':{'id':i,'visit':j}}
            
            for k in list(kk.keys()):
                if k in t.vol:
                    pp=t.vol[k]
                    
                    # replace path
                    if options.path_subst is not None:
                        pp=re.sub(options.path_subst[0],options.path_subst[1],pp)
                        
                    if os.path.exists(pp):
                        
                        kk[k]=True
                        t.vol[k]=pp
                        sep=','
                        if k=='lobes' or k=='lng_lobes': sep=' '
                        
                        vols[k]=load_volume_std(templates[k],pp,sep=sep,exclude=exclude)
                    else:
                        print(("Missing:{}".format(pp)))
                        t.vol[k]=None
                        vols[k]=None
            measurements.append(vols)
    print((repr(kk)))
    # flatten structure and write out csv file
    # going to overwrite with more precise measurements , if available
    
    output=[]
    for i,j in enumerate(measurements):
        o={}
        for k in ['id','lobes','lng_lobes','deep','vent','hc','am']:
            if kk[k]:
                o.update(templates[k]) # make sure we have basic entries
                if k in j:
                    o.update(j[k])
        output.append(o)
    # now output is flat structure with the same entries everywhere
    # will write out into csv file
    with open(options.output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(output[0].keys()))
        writer.writeheader()
        for o in output:
            writer.writerow(o)
            
def parse_options():
    
    usage = """
       The list have this structure:
          id,visit,t1w(,t2w,pdw,sex,age,geot1,geot2,lesions)
      
          - id,visit,t1w are mandatory.
          - if the data do not exist, no space should be left
              id,visit,t1w,,,sex,age -> to include sex and age in the pipeline
    
    """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Gather volumes',usage=usage)
    parser.add_argument("--debug",
                    action="store_true",
                    dest="debug",
                    default=False,
                    help="Print debugging information" )

    parser.add_argument("list",
                        help="Input list file")

    parser.add_argument("prefix",
                        help="Processing prefix")
    
    parser.add_argument("output",
                        help="Output file")
    
    parser.add_argument('--path_subst',
                        nargs=2,
                        help="Substitute path")
    
    options = parser.parse_args()

    if options.debug:
        print((repr(options)))

    return options

if __name__ == '__main__':
    
    options = parse_options()
    
    query_volumes(options)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;tab-width 4
