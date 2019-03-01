#!/bin/bash 

#download dataset
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_bdl_arctic-0.95-release.zip -P /tmp
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_clb_arctic-0.95-release.zip -P /tmp
# wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_rms_arctic-0.95-release.zip -P /tmp
# wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.zip -P /tmp

#unzip
unzip -q /tmp/cmu_us_bdl_arctic-0.95-release.zip -d /tmp/bdl
unzip -q /tmp/cmu_us_clb_arctic-0.95-release.zip -d /tmp/clb
# unzip -q /tmp/cmu_us_rms_arctic-0.95-release.zip -d /tmp/rms
# unzip -q /tmp/cmu_us_slt_arctic-0.95-release.zip -d /tmp/slt

repo=$PWD

# process bdl
cd /tmp/bdl/cmu_us_bdl_arctic/wav
for f in `ls`;
do 
    new_name="bdl_$f"
    mv $f $new_name
done

# process clb
cd /tmp/clb/cmu_us_clb_arctic/wav
for f in `ls`;
do 
    new_name="clb_$f"
    mv $f $new_name
done

# return repo dir
cd $repo

cp "/tmp/clb/cmu_us_clb_arctic/wav/"* "parallel_data/wavs/"
cp "/tmp/bdl/cmu_us_bdl_arctic/wav/"* "parallel_data/wavs/"

ls "parallel_data/wavs"|grep "clb"|tail -n +101|sort > /tmp/source.txt
ls "parallel_data/wavs"|grep "bdl"|tail -n +101|sort > /tmp/target.txt

ls "parallel_data/wavs"|grep "clb"|head -n 100|sort > "parallel_data/test.lst"
paste -d\| /tmp/source.txt /tmp/target.txt > "parallel_data/metadata.csv"

