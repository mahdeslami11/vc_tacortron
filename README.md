# VC_Tacotron

Voice Conversion using Tacotron Model.

### Quick Start
#### 1. Download dataset and Preprocess
```shell
./download_data.sh
python preprocess.py
```


#### 2. Training
``` shell
python train.py
```

#### 3. Evaluation
```shell
./eval.sh
```

### Results
#### 1. Audio samples
clb -> dbl
```
./samples
```
#### 2. Alignments
![alignments](https://github.com/vBaiCai/vc_tacotron/raw/master/figs/align.png)

## Reference

* [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf),

* [keithito's tacotron implementation](https://github.com/keithito/tacotron)
* [cmu_arctic dataset](http://festvox.org/cmu_arctic/)