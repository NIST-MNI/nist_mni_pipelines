# IPL Longitudinal pipeline
created by Daniel Garcia-Lorenzo, Nicolas Guizard, Berengere Aubert-Broche and Vladimir S. Fonov

## Introduction
This collection of python and perl scripts is intended to serve as automatic image processing pipeline.
With the main goal to perform automatic analysis on longitudinally acquired high-resoltion (1x1x1mm) T1w scans of multiple subjects. 
This work relies on various scripts and tools developed during NIHPD project by Vladimir S. Fonov, Matthew Kitching, Andrew Janke and Larry Baer
with later additions by Berengere Aubert-Broche and Daniel Garcia-Lorenzo

## References
The results obtained with this pipeline have been published in the following article:
[B. Aubert-Broche et all 2012 "A New Framework for Analyzing Structural Volume Changes of Longitudinal Brain MRI Data"](http://dx.doi.org/10.1007/978-3-642-33555-6_5)

## Installation
Pipeline uses several well-known minc tools, as well as several custom made ones. 
It also need [scoop python module](https://github.com/vfonov/scoop)


### Dependencies
Pipeline depends on the minc tools from [minc-toolkit package](http://www.bic.mni.mcgill.ca/ServicesSoftware/ServicesSoftwareMincToolKit)

```shell
$ conda create -n mni --file ./conda_specfile.txt
$ conda activate mni
```

*TODO: add description of dependencies*

### Installing pipeline scripts
*TODO*

### Running test case to make sure all the tools are installed properly
*TODO*

## Running

### Anatomical Templates
By default, the [ICBM 2009c Nonlinear Symmetric template](http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009#icbm152_nlin_sym_09c) will be used.
The location of this template is `/opt/minc/share/icbm152_model_09c`

### Input File Formats
The pipeline needs a text file describing the input data in the following format:

    <subject>,<visit>,<t1w.mnc>[,<t2w.mnc>,[<pdw.mnc>,[<sex>[,<age>]]]]

Where
* `<subject>` unique subject identifier, could be any text, *mondatory*
* `<visit>`   unique identifier, could be any text, *mondatory*
* `<t1w.mnc>` a path to T1w scan , can be relative to the directory where the script will be running, *mondatory*
* `<t2w.mnc>` a path to T2w scan, *optional*
* `<t2w.mnc>` a path to PDw scan, *optional*
* `<sex>`     subject's gender, *optional*
* `<age>`     subject's age at the give visit, *modatory* if longitudinal regularization (see below) is going to be used, *optional* otherwise
Empty fields are allowed, so for a subject with only T1w scan but with known age following example is acceptable:
```
    JohnDoe,m00,native/JohnDoe_m00_t1w.mnc,,,Male,58
    JohnDoe,m12,native/JohnDoe_m12_t1w.mnc,,,Male,59
    MaryAnn,m00,native/MaryAnn_m00_t1w,,,Female,67
    MaryAnn,m06,native/MaryAnn_m00_t1w,,,Female,67.5
```

### Command line arguments
Example:
```
    python -m scoop iplLongitudinalPipeline.py \
    -l subjects.lst -o proc -L -D -S \
    --model-dir=/opt/minc/share/icbm152_model_09c \
    --model-name=mni_icbm152_t1_tal_nlin_sym_09c -r \
    --beast-dir=/opt/minc/share/beast-library-1.1
```
will  execute pipeline using all avaialable cores

Example for submisssion on guillimin (Compute canada cluster) with jobs distributed acreoss `<N>` cores, on arbitrary number of nodes, with time limit of `X` hrs:

`qsub -o <log_file> -N <job_name> run_pipeline.sh`
Contents of run_pipeline.sh:
```sh
#! /bin/sh
#PBS -l nodes=<N>
#PBS -l pmem=4000m
#PBS -l walltime=<X>:00:00
#PBS -A <insert your team id here>
#PBS -j oe
#PBS -o logfile.log
#PBS -V

# change dir to work directory
if [ ! -z $PBS_O_WORKDIR ];then
cd $PBS_O_WORKDIR
fi

# limit number of threads used by ITK and by openMP programs
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export OMP_NUM_THREADS=1

# start execution of the pipeline
# scoop should be able to determine how many cores we have available
python -m scoop \
    -l <subjects_data>.lst -o <output> -L -D -S \
    --model-dir=<prefix of model>/icbm152_model_09c \
    --model-name=mni_icbm152_t1_tal_nlin_sym_09c -r \
    --beast-dir=<prefix of beast>/beast-library-1.1
```

### Output Files
The pipeline produces following files:

    <subject>/<subject>.pickle
    <subject>/<subject>.sge.log
    <subject>/qc/qc_stx_t1_<subject>_<visit>.jpg
    <subject>/qc/qc_stx_mask_<subject>_<visit>.jpg
    <subject>/qc/qc_t1t2_<subject>_<visit>.jpg
    <subject>/qc/qc_stx_t1_<subject>_v02.jpg
    <subject>/qc/qc_stx_mask_<subject>_v02.jpg
    <subject>/qc/qc_t1t2_<subject>_v02.jpg
    <subject>/qc/qc_stx_t1_<subject>_v03.jpg
    <subject>/qc/qc_stx_mask_<subject>_v03.jpg
    <subject>/qc/qc_t1t2_<subject>_v03.jpg
    <subject>/qc/qc_stx_t1_<subject>_v04.jpg
    <subject>/qc/qc_stx_mask_<subject>_v04.jpg
    <subject>/qc/qc_t1t2_<subject>_v04.jpg
    <subject>/qc/qc_lin_template_<subject>.jpg
    <subject>/qc/qc_stx2_t1_<subject>_v04.jpg
    <subject>/qc/qc_stx2_t1_<subject>_<visit>.jpg
    <subject>/qc/qc_stx2_t1_<subject>_v03.jpg
    <subject>/qc/qc_stx2_t1_<subject>_v02.jpg
    <subject>/qc/qc_stx2_mask_<subject>_<visit>.jpg
    <subject>/qc/qc_stx2_mask_<subject>_v02.jpg
    <subject>/qc/qc_stx2_mask_<subject>_v03.jpg
    <subject>/qc/qc_stx2_mask_<subject>_v04.jpg
    <subject>/qc/qc_nl_template_<subject>.jpg
    <subject>/qc/qc_cls_<subject>_<visit>.jpg
    <subject>/qc/qc_lob_<subject>_<visit>.jpg
    <subject>/qc/qc_cls_<subject>_v02.jpg
    <subject>/qc/qc_lob_<subject>_v02.jpg
    <subject>/qc/qc_cls_<subject>_v03.jpg
    <subject>/qc/qc_lob_<subject>_v03.jpg
    <subject>/qc/qc_cls_<subject>_v04.jpg
    <subject>/qc/qc_lob_<subject>_v04.jpg
    <subject>/qc/qc_lngcls_<subject>_v04.jpg
    <subject>/qc/qc_lngcls_<subject>_<visit>.jpg
    <subject>/qc/qc_lngcls_<subject>_v03.jpg
    <subject>/qc/qc_lngcls_<subject>_v02.jpg
    <subject>/<visit>/clp/clp_<subject>_<visit>_t1.mnc
    <subject>/<visit>/clp/clp_<subject>_<visit>_mask.mnc
    <subject>/<visit>/clp/clp_<subject>_<visit>_t2.mnc
    <subject>/<visit>/clp/clp_<subject>_<visit>_t2t1.xfm
    <subject>/<visit>/clp/clp_<subject>_<visit>_pdt1.xfm
    <subject>/<visit>/clp/clp_<subject>_<visit>_pd.mnc
    <subject>/<visit>/clp2/clp2_<subject>_<visit>_t1.mnc
    <subject>/<visit>/clp2/clp2_<subject>_<visit>_mask.mnc
    <subject>/<visit>/stx/stx_<subject>_<visit>_t1.xfm
    <subject>/<visit>/stx/stx_<subject>_<visit>_t1.mnc
    <subject>/<visit>/stx/nsstx_<subject>_<visit>_t1.xfm
    <subject>/<visit>/stx/nsstx_<subject>_<visit>_t1.mnc
    <subject>/<visit>/stx/stx_<subject>_<visit>_mask.mnc
    <subject>/<visit>/stx/nsstx_<subject>_<visit>_mask.mnc
    <subject>/<visit>/stx/stx_<subject>_<visit>_t2.xfm
    <subject>/<visit>/stx/stx_<subject>_<visit>_t2.mnc
    <subject>/<visit>/stx/nsstx_<subject>_<visit>_t2.xfm
    <subject>/<visit>/stx/stx_<subject>_<visit>_pd.xfm
    <subject>/<visit>/stx/stx_<subject>_<visit>_pd.mnc
    <subject>/<visit>/stx/stx_<subject>_<visit>_t2les.mnc
    <subject>/<visit>/stx/stx_<subject>_<visit>_masknoles.mnc
    <subject>/<visit>/stx/nsstx_<subject>_<visit>_masknoles.mnc
    <subject>/<visit>/stx2/stx2_<subject>_<visit>_t1.xfm
    <subject>/<visit>/stx2/stx2_<subject>_<visit>_t2.xfm
    <subject>/<visit>/stx2/stx2_<subject>_<visit>_pd.xfm
    <subject>/<visit>/stx2/stx2_<subject>_<visit>_t1.mnc
    <subject>/<visit>/stx2/stx2_<subject>_<visit>_t2.mnc
    <subject>/<visit>/stx2/stx2_<subject>_<visit>_pd.mnc
    <subject>/<visit>/stx2/stx2_<subject>_<visit>_t2les.mnc
    <subject>/<visit>/stx2/stx2_<subject>_<visit>_mask.mnc
    <subject>/<visit>/stx2/stx2_<subject>_<visit>_masknoles.mnc
    <subject>/<visit>/nl/nl_<subject>_<visit>_grid_0.mnc
    <subject>/<visit>/nl/nl_<subject>_<visit>.xfm
    <subject>/<visit>/vbm/vbm_imp_csf_<subject>_<visit>.mnc
    <subject>/<visit>/vbm/vbm_imp_gm_<subject>_<visit>.mnc
    <subject>/<visit>/vbm/vbm_imp_wm_<subject>_<visit>.mnc
    <subject>/<visit>/vbm/vbm_jac_<subject>_<visit>.mnc
    <subject>/<visit>/cls/cls_<subject>_<visit>.mnc
    <subject>/<visit>/cls/clsem_<subject>_<visit>.mnc
    <subject>/<visit>/cls/lob_<subject>_<visit>.mnc
    <subject>/<visit>/cls/lobem_<subject>_<visit>.mnc
    <subject>/<visit>/cls/deep_<subject>_<visit>.mnc
    <subject>/<visit>/cls/lngcls_<subject>_<visit>.mnc
    <subject>/<visit>/cls/lnglob_<subject>_<visit>.mnc
    <subject>/<visit>/vol/vol_<subject>_<visit>.txt
    <subject>/<visit>/vol/volem_<subject>_<visit>.txt
    <subject>/<visit>/vol/deep_<subject>_<visit>.txt
    <subject>/<visit>/vol/lngvol_<subject>_<visit>.txt
    <subject>/<visit>/lng/lng_<subject>_<visit>_t1.mnc
    <subject>/<visit>/lng/lng_<subject>_<visit>_t1_grid_0.mnc
    <subject>/<visit>/lng/lng_<subject>_<visit>_t1.xfm
    <subject>/lng_template/lin_template_<subject>_t1.mnc
    <subject>/lng_template/lin_template_<subject>_mask.mnc
    <subject>/lng_template/lin_template_<subject>_t1.xfm
    <subject>/lng_template/nl_template_<subject>_t1.mnc
    <subject>/lng_template/nl_template_<subject>_mask.mnc
    <subject>/lng_template/nl_<subject>.xfm
    <subject>/lng_template/nl_<subject>_grid_0.mnc
    <subject>/<subject>.commands
    <subject>/<subject>.log

in short , `qc` directory will contain various jpeg files for quality control
`stx2*` files are going to be files in stereotaxic space co-registered between time-points
and `lngcls_*`  - tissue classification with longitudinal restriction, 
`lnglob_*` - lobe segmentation also with longitudinal restrictions. 
`lng_template`  will contain subject-specific non-linear average in stereotaxic space.

## Optional variants

### Distortion correction
*TODO*

### Brain Extraction (BEaST)
*TODO*

### Inter-modality co-registration
*TODO*

## Performing statistical analysis on the results
*TODO*

### Using glim_image
How-to use glim_image for statistical analysis *TODO*

### Using RMINC
How-to use RMINC *TODO*

### Using pyminc
How-to use pyminc *TODO*

## Making changes to the pipeline

### Adding new functionality
It is recommended to familiarize yourself with git version control system, 
the short guide is available here: http://githowto.com/ , another guide is here: http://gitimmersion.com/
or here: http://stackoverflow.com/questions/315911/git-for-beginners-the-definitive-practical-guide


## Editing this document
This document is written using markdown [see syntax](http://daringfireball.net/projects/markdown/syntax). 
