# Deep Discriminative Features Learning and Sampling for Imbalanced Data Problem
Deep Discriminative Features Learning and Sampling for Imbalanced Data Problem

## Using the script
Comment example : python3 train_model.py DFBS_csv 0 0,1 2 1 False

Comment parameters : 

1. Model: ex. DFBS_csv or DFBS_image 
2. Img_Resize: ex. 28 or 0(csv)
3. class_ID: ex. 0,1,2,3,4
4. class_num: ex. 5
5. imb_ratio: for balanced data to create imbalanced data 
6. Restore_ckpt: True or False

## Short discription of the code 

1.	train_model.py <br />
Main script who read the dataset and call the DFBS model.

2.	datas.py <br />
To read the dataset.

3.	tfrecord.py <br />
To transform the original data to the Tfrecord data type.

4.	nets.py <br />
Network architectures, like CNN, DNN and autoencoder.

5.	models.py <br />
The setting about model training. Ex: Loss functions, Optimizer, Learning rate, and Training epochs。

6.	tuning_and_eval.py <br />
Applying the classifier(Logistic regression) on the final outputs, synthetic feature vectors and orignal vectors.

## Download the raw data
We do the preprocessing to make the dataset become the binary class.<br />
Pima : https://www.kaggle.com/uciml/pima-indians-diabetes-database<br />
Haberman's : https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival<br />
Satimage : https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)<br />
E.Coil : https://archive.ics.uci.edu/ml/datasets/ecoli<br />
Shuttle : https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)<br />
ionosphere : https://archive.ics.uci.edu/ml/datasets/ionosphere<br />
GiveMeSomeCredit : https://www.kaggle.com/c/GiveMeSomeCredit<br />
