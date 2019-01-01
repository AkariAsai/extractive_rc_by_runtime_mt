## Neural Machine Translation module
This is a README file for NMT module of our proposed system.  
We implemented our NMT model with [PyTorch](https://pytorch.org/) (0.4.0).  
Our NMT model is based on [Luong et al. (2015)](http://aclweb.org/anthology/D15-1166) and use a beam search method prospoed in [Oda et al. (2017)](http://www.aclweb.org/anthology/W17-5712).

### The structure of this directory
```
.
├── tools/
      └── multi-bleu.perl     # a perl script to evaluate BLEU score. 
├── data.py                   # Preprocess and load NMT corpora.
├── model.py                  # Define models.
├── test.py                   # Test the pretrained models.
├── train.py                  # Train new models.
├── utils.py                  # Define Utility functions.
└── README.md
```

### Train
To train new models, you need to set `--sourceDevFile`, `--targetDevFile`, `--sourceTrainFile` and `--targetTrainFile` options. 

```sh
$ cd nmt
$ python train.py \
--sourceDevFile [PATH TO YOUR DEV SOURCE] --targetDevFile [PATH TO YOUR DEV TARGET] \
--sourceTrainFile [PATH TO YOUR TRAIN SOURCE] --targetTrainFile  [PATH TO YOUR TRAIN TARGET]
```

For the details of other command line options, see [`nmt/train.py`](https://github.com/AkariAsai/extractive_rc_by_runtime_mt/blob/master/nmt/train.py#L252#L294).  

**Note: We are currently refactoring the codes and also update them for newer version of PyTorch, and it might produce errors in your environment.**

### Evaluate
To train new models, you need to set `--trans_source`(path to a source file you would like to translate), `--trans_target`(path to a target file you would like to translate), `--train_source`(path to a source file you used for training) and `--train_target`(path to a target file you used for training) options, besides pretrained embedding(`--embedding_params`) and params (`--encdec_params`)

```sh
$ cd nmt
$ python test.py \
--trans_source [PATH TO YOUR DEV SOURCE] --trans_target [PATH TO YOUR DEV TARGET] \
--train_source [PATH TO YOUR TRAIN SOURCE] --train_target  [PATH TO YOUR TRAIN TARGET] \
--embedding_params [PATH TO YOUR TRAINED EMBEDDING FILE]
--encdec_params [PATH TO YOUR TRAINED PARAM FILES]
```

For the details of other command line options, see [`nmt/test.py`](https://github.com/AkariAsai/extractive_rc_by_runtime_mt/blob/master/nmt/test.py#L361#L382).  
