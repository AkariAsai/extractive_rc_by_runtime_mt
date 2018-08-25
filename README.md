# Extractive Reading Comprehension by Runtime Machine Translation
We have presented the first extractive Reading Comprehension (RC) systems for non-English languages without additional language-specific RC training data, but instead by using an English RC model and an attention-based Neural Machine Translation (NMT) model ("Extractive Reading Comprehension by Runtime Machine Translation"[1]).

![The Overview](https://github.com/AkariAsai/extractive_rc_by_runtime_mt/blob/master/overview.png)

## Contents
1. [Code](#code)
2. [Datasets](#datasets)
3. [Benchmarks](#benchmarks)
4. [Reference](#reference)
5. [Contact](#contact)

## Code
Code will be released soon. 
We implemented our NMT/extractive RC modules in [PyTorch](https://pytorch.org/).

## Datasets
We provide multilingual SQuAD datasets, and the Wikipedi-based {French,Japanese}-to-English bilingual corpora. 

#### Multilingual SQuAD Datasets
The datasets are created by translating the original [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (v1.0) development dataset into French and Japanese. 
These datasets contains 321 questions for each.

#### {French,Japanese}-to-English bilongual Corpus
We constacted bilingual copora from Wikipedia articles, using its inter-language links and [hunalign](https://github.com/danielvarga/hunalign), a sentence-level aligner.

| | train          | dev  |
| ------------- |:-------------:| :-----:|
| Japanese     | [train.ja](),[train.en]() | [train.ja](),[dev.en]() |
| French  | [train.fr](),[train.en]() | [dev.fr](),[dev.en]() |

## Benchmarks
We provide the results of our proposed method on multingual SQuAD datasets. 
- Japanese

| methods|F1          | EM  |
| ------------- |:-------------:| :-----:|
| Ours| **50.09** | **37.37** |
| back-translation baseline| 40.57|22.77|

- French

| methods |F1          | EM  |
| ------------- |:-------------:| :-----:|
| Ours | **54.73** | **39.39** |
| back-translation baseline |51.25 | 23.76|



## Reference
Please cite [1] if you found the resources in this repository useful.

[1] "Extractive Reading Comprehensio by Runtime Machine Translation"

```
    @InProceedings{,
      author    = {},
      title     = {},
    }
```
## Contact
Please direct any questions to [Akari Asai](https://akariasai.github.io/) at akari-asai@g.ecc.u-tokyo.ac.jp.
