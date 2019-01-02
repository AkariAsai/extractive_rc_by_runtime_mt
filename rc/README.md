
## Neural Reading Comprehension (RC) module
This is a README file for RC module of our proposed system.  
We implemented our RC models with [PyTorch](https://pytorch.org/) (0.4.0) and [AllenNLP](https://github.com/allenai/allennlp) (0.5.0).  
We implemented BiDAF ([Seo et al., 2017](https://arxiv.org/abs/1611.01603)) and BiDAF + Self Attention + ELMo ([Peters et al., 2018](https://arxiv.org/pdf/1802.05365.pdf)).

### The structure of this directory
```
.
├── training_config/                              # Training configuration files for RC models. 
  ├── bidaf_self_elmo_300d_training_config.json       # BiDAF + Self Attention + ELMo (pre-trained word embeddings dimension = 300)
  ├── bidaf_self_elmo_training_config.json            # BiDAF + Self Attention + ELMo (pre-trained word embeddings dimension = 100)
  └── bidaf_training_config.json                      # BiDAF 
├── .env                                          # If you use Google Translate/Bing Translator, you need to set API Keys here.
├── model.py                                      # Define models.
├── data_ml.py                                    # Define datareader classes for multingual SQuAD.
├── evaluate.py                                   # Evaluate performance on the original SQuAD dataset.
├── evaluate_mlqa.py                              # Evaluate performance on the multilingual SQuAD dataset.
├── main.py   
├── squad.py                                      # Evaluate performance on the original SQuAD dataset.
├── squad_ml_answer_retreival.py                  # Retreive answer using attention-based alignment from NMT models. 
├── tri_lnear_attention.py                        # Define customized attention class for self attention.
├── white_space_word_tokenizer.py                 # Define customized tokenizer for the multilingual SQuAD dataset.
├── utils.py                                      # Define utility functions.
└── README.md
```

### Train
To train your own model, you need to set the config json file path and model name (`self` or `bidaf`). Optionally you can set serialization directory name. 

For example, you can train a BiDAF + Self Attention + ELMo model from scratch by runnig the command below. 

```sh
python main.py train training_config/bidaf_self_elmo_300d_training_config.json -s bidaf_self_elmo_output -m self
```

For more details on command line options, see [`rc/train.py`](https://github.com/AkariAsai/extractive_rc_by_runtime_mt/blob/master/rc/train.py#L33#L60).

### Evaluate
To evaluate your trained model on the original SQuAD dataset, you can run the command below:

```sh
python main.py evaluate bidaf_self_elmo_output -s bidaf_self_elmo_output \
--evaluation-data-file https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json \
--unziped_archive_directory bidaf_self_elmo_output
```
