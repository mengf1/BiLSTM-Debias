# Welcome to BiLSTM+Debias
---
## Introduction

This source code is the basis of the following paper:
> [Learning when to trust distant supervision: An application to low-resource POS tagging using cross-lingual projection](http://www.aclweb.org/anthology/K/K16/K16-1018.pdf), CoNLL 2016

## Building
It's developed on clab/cnn toolkit.
- Install clab/cnn following [clab/cnn](https://github.com/clab/cnn-v1).
- Add the source code to folder *cnn/examples* and add bilstm-dn to *CMakeLists.txt*.
- Make again.

## Data format
The format of input data is as follows:
```
Tok_1 Tok_2 ||| Tag_1 Tag_2
Tok_1 Tok_2 Tok_3 ||| Tag_1 Tag_2 Tag_3
...
```

## How to run
```sh
./bilstm-dn gold_data_file projected_data_file dev_data_file test_data_file max_epochs
```


