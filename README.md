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


### Dependencies
Pipeline depends on the minc tools from [minc-toolkit-v2 package](https://github.com/BIC-MNI/minc-toolkit-v2)

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

    <subject>,<visit>,<t1w.mnc>[,<t2w.mnc>,[<pdw.mnc>,[,<age>,[<sex>]]]]

Where
* `<subject>` unique subject identifier, could be any text, *mondatory*
* `<visit>`   unique identifier, could be any text, *mondatory*
* `<t1w.mnc>` a path to T1w scan , can be relative to the directory where the script will be running, *mondatory*
* `<t2w.mnc>` a path to T2w scan, *optional*
* `<t2w.mnc>` a path to PDw scan, *optional*
* `<age>`     subject's age at the give visit, *modatory* if longitudinal regularization (see below) is going to be used, *optional* 
* `<sex>`     subject's sex, *optional*

Empty fields are allowed, so for a subject with only T1w scan but with known age following example is acceptable:
```
    JohnDoe,m00,native/JohnDoe_m00_t1w.mnc,,,58,Male
    JohnDoe,m12,native/JohnDoe_m12_t1w.mnc,,,59,Male
    MaryAnn,m00,native/MaryAnn_m00_t1w,,,67,Female
    MaryAnn,m06,native/MaryAnn_m00_t1w,,,67.5,Female
```

### Command line arguments
Example:
```
    python iplLongitudinalPipeline.py \
    -l subjects.lst -o output  \
    --cleanup \
    --model-dir=/opt/minc/share/icbm152_model_09c \
    --model-name=mni_icbm152_t1_tal_nlin_sym_09c \
    --ray_start 4 \
    --threads 4 \
    --nl_ants \
    --nl_step 1.0 \
    --nl_cost_fun CC \
    --bison_pfx /opt/models/ipl_bison_1.3.0 \
    --bison_method  HGB1 \
    --wmh_bison_pfx /opt/models/wmh_bison_1.3.0/t1_15T \
    --wmh_bison_atlas_pfx /opt/models/wmh_bison_1.3.0/15T_2009c_ \
    --wmh_bison_method HGB1 \
    --synthstrip_onnx /opt/models/synthstrip/synthstrip.1.onnx 

```
will  execute pipeline using 4 cores 

### Output Files

In short:

- `<subject>/qc`  - various jpeg files for quality control
- `<subject>/<timepoint>/stx2` - files in stereotaxic space co-registered between time-points
- `<subject>/<timepoint>/cls` - classification results
- `<subject>/<timepoint>/nl` - nonlinear registration results
- `<subject>/<timepoint>/vol` - summary results in text format
- `<subject>/lng_template`  - subject-specific non-linear average in stereotaxic space.


The pipeline produces following files:

* Quality control files in jpeg format for quickly inspecting various stages of the pipeline

    `<subject>/qc/qc_stx_t1_<subject>_<visit>.jpg`  - T1w linearly registerd to template , 1st stage
    `<subject>/qc/qc_stx_mask_<subject>_<visit>.jpg` - brain mask 
    `<subject>/qc/qc_t1t2_<subject>_<visit>.jpg` - T1w and T2w coregistration
    `<subject>/qc/qc_lin_template_<subject>.jpg` - average linear T1w
    `<subject>/qc/qc_stx2_t1_<subject>_<visit>.jpg` - T1w linearly registerd to template , 2nd stage
    `<subject>/qc/qc_stx2_mask_<subject>_<visit>.jpg` - brain mask in 2nd stage
    `<subject>/qc/qc_nl_template_<subject>.jpg` - average non-linear T1w
    `<subject>/qc/qc_cls_<subject>_<visit>.jpg` - tissue classification
    `<subject>/qc/qc_lob_<subject>_<visit>.jpg` - lobe segmentations

* Files in the native space, after intensity normalization

    `<subject>/<visit>/clp2/clp2_<subject>_<visit>_t1.mnc` - T1w scan 
    `<subject>/<visit>/clp2/clp2_<subject>_<visit>_mask.mnc` - brain mask 

* Files in stereotaxic space

    `<subject>/<visit>/stx/stx_<subject>_<visit>_t1.xfm` - transformaton parameters T1w native to template space, 1st stage
    `<subject>/<visit>/stx/nsstx_<subject>_<visit>_t1.xfm` - transformaton parameters T1w native to template space, 1st stage, no scaling
    `<subject>/<visit>/stx/stx_<subject>_<visit>_t2.xfm` - transformaton parameters T2w native to template space, 1st stage
    `<subject>/<visit>/stx/nsstx_<subject>_<visit>_t2.xfm` - transformaton parameters T2w native to template space, 1st stage, no scaling
    `<subject>/<visit>/stx/stx_<subject>_<visit>_pd.xfm` - transformaton parameters PDw native to template space, 1st stage
    `<subject>/<visit>/stx2/stx2_<subject>_<visit>_t1.xfm` - transformaton parameters T1w native to template space, 2nd stage
    `<subject>/<visit>/stx2/stx2_<subject>_<visit>_t2.xfm` - transformaton parameters T2w native to template space, 2nd stage
    `<subject>/<visit>/stx2/stx2_<subject>_<visit>_pd.xfm` - transformaton parameters PDw native to template space, 2nd stage
    `<subject>/<visit>/stx2/stx2_<subject>_<visit>_t1.mnc` - T1w scan in stereotaxic space after linear registration, 2nd stage
    `<subject>/<visit>/stx2/stx2_<subject>_<visit>_t2.mnc` - T2w scan in stereotaxic space after linear registration, 2nd stage
    `<subject>/<visit>/stx2/stx2_<subject>_<visit>_pd.mnc` - PDw scan in stereotaxic space after linear registration, 2nd stage
    `<subject>/<visit>/stx2/stx2_<subject>_<visit>_mask.mnc` - brain mask in stereotaxic space, 2nd stage

* Nonlinear registration parameters

    `<subject>/<visit>/nl/nl_<subject>_<visit>.xfm` - nonlinear transformation from linear stereotaxic space to template
    `<subject>/<visit>/nl/nl_<subject>_<visit>_grid_0.mnc` - corresponding vector field
    `<subject>/<visit>/lng/lng_<subject>_<visit>_t1.xfm` - nonlinear transformation from current timepoint to subject-specific average
    `<subject>/<visit>/lng/lng_<subject>_<visit>_t1_grid_0.mnc`  - corresponding vector field
    `<subject>/<visit>/lng/lng_<subject>_<visit>_t1_inv.xfm` - inverse nonlinear transformation from current timepoint to subject-specific average
    `<subject>/<visit>/lng/lng_<subject>_<visit>_t1_inv_grid_0.mnc`  - corresponding vector field

* Voxel-based morphometry outputs

    `<subject>/<visit>/vbm/vbm_xfm_<subject>_<visit>.xfm` - nonlinear transformation to the template space for the purpose of VBM (different then `<subject>/<visit>/nl/nl_<subject>_<visit>.xfm` )
    `<subject>/<visit>/vbm/vbm_xfm_<subject>_<visit>_grid_0.mnc` - corresponding vector field
    `<subject>/<visit>/vbm/vbm_idet_<subject>_<visit>.mnc` - jacobian of the inverse transformartion 
    `<subject>/<visit>/vbm/vbm_imp_csf_<subject>_<visit>.mnc` - CSF probility map modulated by jacobian (weight preserving), can be above 1.0
    `<subject>/<visit>/vbm/vbm_imp_gm_<subject>_<visit>.mnc` - GM probility map modulated by jacobian (weight preserving), can be above 1.0
    `<subject>/<visit>/vbm/vbm_imp_wm_<subject>_<visit>.mnc` - WM probility map modulated by jacobian (weight preserving), can be above 1.0

* Tissue clasification results

    `<subject>/<visit>/cls/cls_<subject>_<visit>.mnc` - tissue classification results: 1 - CSF, 2- GM , 3- WM 
    `<subject>/<visit>/cls/lob_<subject>_<visit>.mnc` - lobe segmentation results
    `<subject>/<visit>/cls/wmh_<subject>_<visit>.mnc` - white matter hyperintensities segmentation

* Subject-specific anatomical average

    `<subject>/lng_template/lin_template_<subject>_t1.mnc` - average t1 , linearly coregister
    `<subject>/lng_template/lin_template_<subject>_mask.mnc` - brain mask of average linear t1
    `<subject>/lng_template/lin_template_<subject>_t1.xfm`  - linear transformation to the template
    `<subject>/lng_template/nl_template_<subject>_t1.mnc` - average t1 , non-linearly coregister
    `<subject>/lng_template/nl_template_<subject>_mask.mnc` - brain mask of average non-linear t1

    `<subject>/lng_template/nl_<subject>.xfm` - non-linear transformation to the template
    `<subject>/lng_template/nl_<subject>_grid_0.mnc` - correspinding vector field

    `<subject>/lng_template/nl_subject43_inverse.xfm` - inverse non-linear transformation to the template
    `<subject>/lng_template/nl_<subject>_inverse_grid_0.mnc` - correspinding vector field

* Summary files

    `<subject>/<visit>/vol/vol_<subject>_<visit>.txt` -  summary measurements in text format (two columns separated by space)
    `<subject>/<visit>/vol/vol_<subject>_<visit>.json` - summary measurements in json format
    `<subject>/<visit>/vol/vol_<subject>_<visit>.csv` - summary measurements in .csv format, (header and one line)
