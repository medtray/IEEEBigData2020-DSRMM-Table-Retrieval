# IEEEBigData2020-DSRMM-Table-Retrieval

This repository contains resources developed within the following paper: M. Trabelsi, Z. Chen, B. D. Davison and J. Heflin, "A Hybrid Deep Model for Learning to Rank Data Tables", 2020 IEEE International Conference on Big Data (Big Data).

# Requirements

- Python 3.7
- pytorch 0.4.1
- nltk 3.5
- pandas
- Glove embedding (http://nlp.stanford.edu/data/glove.6B.zip)
- trec_eval is used to calculate evaluation metrics (https://github.com/usnistgov/trec_eval)

# Data

- WikiTables (http://websail-fe.cs.northwestern.edu/TabEL/tables.json.gz)

# Table retrieval results

- Run five fold cross validation on wikiTables without STR features:
python3 train_test_dsrmm.py --max_query_len 6 --max_meta_len 50 --max_attributes_len 30 --max_values_len 20 --emsize 300 --k1 20 --k2 20 --k3 20 --k4 200 --sem_feature 100 --nbins 5 --device 0 --batch_size 5 --epochs 15 --data_path wikiTables --inter_folder inter_folder --values_file data_fields_with_struct_values2.json --use_max_ndcg --lr 0.001 --word_embedding glove.6B/glove.6B.300d.txt

- Run five fold cross validation on wikiTables with STR features:
python3 train_test_dsrmm.py --max_query_len 6 --max_meta_len 50 --max_attributes_len 30 --max_values_len 20 --emsize 300 --k1 20 --k2 20 --k3 20 --k4 200 --sem_feature 100 --nbins 5 --device 0 --batch_size 5 --epochs 15 --data_path wikiTables --inter_folder inter_folder --values_file data_fields_with_struct_values2.json --use_max_ndcg --lr 0.001 --word_embedding /home/mohamed/PycharmProjects/glove.6B/glove.6B.300d.txt --STR

# Citation

@INPROCEEDINGS{dsrmm,
  author={M. {Trabelsi} and Z. {Chen} and B. D. {Davison} and J. {Heflin}},
  booktitle={2020 IEEE International Conference on Big Data (Big Data)}, 
  title={A Hybrid Deep Model for Learning to Rank Data Tables}, 
  year={2020},
  volume={},
  number={},
  pages={},}
  
  # Contact
  
  if you have any questions, please contact Mohamed Trabelsi at mot218@lehigh.edu
  
