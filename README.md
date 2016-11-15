# Welcome to BiLSTM+Debias
---
## Introduction

This source code is the basis of the following paper
> Learning when to trust distant supervision: An application to low-resource POS tagging using cross-lingual projection, CoNLL 2016

## Building
It's developed on clab/cnn toolkit.
- Install clab/cnn following [clab/cnn](https://github.com/clab/cnn-v1).
- Add source code to folder *cnn/examples* and add bilstm-dn to *CMakeLists.txt*
- Make again.

## How to run
```shell
./bilstm-dn gold_data_file projected_data_file dev_file test_file max_epochs
```


