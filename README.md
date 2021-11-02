# DTP
Dependency Tree Pruning for Retaining Information in WIC Tasks
================================================================
This Repository contains the codes needed to Run and Replicate our NLP Project, which makes use of Dependency Trees and Their Pruning to retain information in Word in Context(WIC) Tasks. 

The datasets used to achieve our results are in xlwic_datasets.rar file. The Description of the data can be read in the following papers:

https://arxiv.org/abs/2010.06478
https://raw.githubusercontent.com/SapienzaNLP/mcl-wic/master/SemEval_2021_Task_2__Multilingual_and_Cross_lingual_Word_in_Context_Disambiguation__MCL_WiC___Paper_.pdf

In order to train and evaluate the model on the data, use Model_1_Train.py and Model_1_Evaluation.py respectively. Use the path to data with different prune levels to achieve our training results while training. Use the evaluation data with different prune levels to achieve our training results while evaluating. 

In order to prune the data, feed the data in the text form to dep_tree_reducer. The output will be a csv with data pruned to some level T. 
The data should be in CSV form for the model to train\evaluate. The code for it is available in the first section of training. change it accordingly for the data that you have. 

For analysis purposes, Word_Level_Finder.py can be used on the data to find the distance of a word from the target word in the Dependency Tree. To replicate the analysis on the report, combine this with Eraser method explained in the following paper. 

https://arxiv.org/abs/1911.03429
