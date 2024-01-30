#! /bin/sh

if [[ ! -e subject43 ]];then
echo "Downloading test dataset"
wget https://github.com/vfonov/nist_mni_pipelines/releases/download/release-0.1.00/ipl_subject43.tar.gz
tar zxf ipl_subject43.tar.gz
fi

if [[ ! -e nist_pipeline_0.1.00.sif ]];then
echo "Downloading container"
wget https://github.com/vfonov/nist_mni_pipelines/releases/download/release-0.1.00/nist_pipeline_0.1.00.sif
fi
