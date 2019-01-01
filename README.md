# Multilingual Extractive Reading Comprehension by Runtime Machine Translation
We introduce the first extractive RC systems for  non-English languages, without using language-specific RC training data, but instead by using an English RC model and an attention-based Neural Machine Translation (NMT) model [1].  

![The Overview](https://github.com/AkariAsai/extractive_rc_by_runtime_mt/blob/master/overview.png)

## Contents
1. [Code](#code)
2. [Datasets](#datasets)
3. [Benchmarks](#benchmarks)
4. [Reference](#reference)
5. [Contact](#contact)

## Code
We implemented our NMT and extractive RC models (BiDAF, BiDAF + Self Attention + ELMo) in [PyTorch](https://pytorch.org/).  

### Installation
The installation steps are as follows:

```bash
git clone https://github.com/AkariAsai/extractive_rc_by_runtime_mt.git
cd extractive_rc_by_runtime_mt
pip install -r requirements.txt
```

For preprocessing steps, we used [StanfordCoreNLP](https://stanfordnlp.github.io/CoreNLP/) for English, [mosesdecoder's scripts](https://github.com/moses-smt/mosesdecoder) for French and [MeCab](http://taku910.github.io/mecab/) for Japanese.   
In our implementation, we call StanfordCoreNLP from [py-corenlp](https://github.com/smilli/py-corenlp), moses' tokenizer from [its original perl scrips](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl), and MeCab through [mecab-python3](https://pypi.org/project/mecab-python3/).  

##### For English
For py-corenlp, first make sure you have the Stanford CoreNLP server running. See [this instruction](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started).

##### For Japanese
For mecab-python3, we use [Neologism dictionary for MeCab](https://github.com/neologd/mecab-ipadic-neologd/), instead of default Ochasen.   
Please install the dictionary beforehand following the instruction on [official page](https://github.com/neologd/mecab-ipadic-neologd/).

##### For French
You can install mosesdecoder's tokenizer by following the commands listed below. 
```sh
wget https://github.com/moses-smt/mosesdecoder/archive/master.zip
unzip master.zip # files are to be extracted under extractive_rc_by_runtime_mt/mosesdecoder-master/
rm master.zip
```
For easy set up, we will replace the codes for tokenization process with [mosestokenizer](https://pypi.org/project/mosestokenizer/), the python wrapper for mose's tokenizer, sometime soon. 

### Train
The details of training RC and NMT models, see [README.md]() under `rc` directory and [README.md]() under `nmt` directory.


### Evaluation 
To run the evaluation on multilingual SQuAD dataset, you first need to train your own model for RC and NMT, or use pre-trained models.  
You can download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1mqz_L5B4uiQ7TPu8F8ytHsIFeMGPtlW1?usp=sharing).

For example, you can evaluate on French SQuAD using the pre-trained models by following the processes instructions below.

1. Create `params` directory right under the home directory. 
2. Download the zipped files `fren_nmt_params.tar.gz` and `rc_params.tar.gz` into `extractive_rc_by_runtime_mt/params` directory, 
and extract necessary files.

```sh
tar -xzf params/fren_nmt_params.tar.gz && rm -f params/fren_nmt_params.tar.gz
tar -xzf params/rc_params.tar.gz && rm -f params/rc_params.tar.gz
```

3. Run the evaluation based on the command below. 

```sh
tar -xzf params/fren_nmt_params.tar.gz && rm -f params/fren_nmt_params.tar.gz
tar -xzf params/rc_params.tar.gz && rm -f params/rc_params.tar.gz
cd rc
python main.py evaluate_mlqa ../params/rc_params \
--evaluation-data-file https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json \
--unziped_archive_directory ../params/rc_params --elmo \
--trans_embedding_model ../params/fren_nmt_params/embedding.bin  \
--trans_encdec_model  ../params/fren_nmt_params/encdec.bin \
--trans_train_source ../params/fren_nmt_params/train_fr_lower_1.txt \
--trans_train_target ../params/fren_nmt_params/train_en_lower_1.txt \
-l Fr -v5 --beam
```
For the details of the command line options, see [rc/evaluate_mlqa.py](https://github.com/AkariAsai/extractive_rc_by_runtime_mt/blob/master/rc/evaluate_mlqa.py)

## Datasets
We provide the two datasets, (1) multilingual SQuAD Datasets (Japanese, French) and (2) {Japanese, French}-to-English bilingual corpora to train our NMT models for the extractive RC system.

#### Multilingual SQuAD Datasets
The Japanese and French datasets are created by manually translating the original [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (v1.1) development dataset into Japanese and French.  
These datasets contains 327 questions for each.  
More details can be found in Section 3.3 (SQuAD Test Data for Japanese and French) in [1].

| | Multilingual SQuAD Datasets       |
| ------------- |:-------------:|
| Japanese    | [japanese_squad.json](datasets/squad_japanese_test.json) |
| French | [french_squad.json](datasets/squad_french_test.json) |


#### {Japanese, French}-to-English bilingual Corpora
##### Wikipedia-based {Japanese, French}-to-English bilingual corpora
To train the NMT model for specific language directions, we take advantage of constantly growing web resources to automatically construct parallel corpora, rather than assuming the availability of high quality parallel corpora of the target domain.  
We constructed bilingual corpora from Wikipedia articles, using its inter-language links and [hunalign](https://github.com/danielvarga/hunalign), a sentence-level aligner.  
More details can be found in Supplementary Material Section A (Details of Wikipedia-based Bilingual CorpusCreation) in [1].  

To download preprocessed Wikipedia-based {Japanese, French}-to-English bilingual corpora, please run the commands below.  

```sh
cd datasets
wget http://www.hal.t.u-tokyo.ac.jp/~asai/datasets/extractive_rc_by_runtime_mt/wiki_corpus.tar.gz
tar -xvf wiki_corpus.tar.gz && rm -f wiki_corpus.tar.gz
cd ..
```

The downloaded wikipedia-based corpora will be placed under `datasets/wiki_corpus` folder.  
The training data includes 1,000,000 pairs, and
the development data includes 2,000 pairs.

##### Manually translated question sentences
In our experiment, we also found that adding a small number of manually translated question sentences could further improve the extractive RC performance.   
Here, we also provide the translated question sentences we actually used to train our NMT models.  
The details pf the creation of these small parallel questions datasets can be found in Supplementary Material Section C (Details of Manually Translated SQuAD DatasetQuestions Creation) in [1].

| | question sentences        |
| ------------- |:-------------:|
| Japanese     | [questions.ja](datasets/questions_jaen.ja), [questions.en](datasets/questions_jaen.en) |
| French  | [questions.fr](datasets/questions_fren.fr), [questions.en](datasets/questions_jaen.en) |


## Benchmarks
We provide the results of our proposed method on multilingual SQuAD datasets.
- Japanese

| methods|F1          | EM  |
| ------------- |:-------------:| :-----:|
| Our Method| **52.19** | **37.00** |
| back-translation baseline| 42.60|24.77|

- French

| methods |F1          | EM  |
| ------------- |:-------------:| :-----:|
| Our Method | **61.88** | **40.67** |
| back-translation baseline | 44.02 | 23.54|



## Reference
Please cite [1] if you found the resources in this repository useful.

[1] Akari Asai, Akiko Eriguchi, Kazuma Hashimoto, and Yoshimasa Tsuruoka. 2018. "[Multilingual Extractive Reading Comprehension by Runtime Machine Translation](https://arxiv.org/abs/1809.03275)".

## Contact
Please direct any questions to [Akari Asai](https://akariasai.github.io/) at akari-asai@g.ecc.u-tokyo.ac.jp.
