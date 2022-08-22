
# KDGN
KDGN : Knowledge-Enhanced Dual Graph Neural Network for Robust Medicine Recommendation

## Overview
This repository contains code necessary to run KDGN model. KDGN is an end-to-end model mainly based on graph convolutional networks (GCN) and attention nerual networks. Paitent history information and medicine knowledge are utilized to provide safe and personalized recommendation of medication combination. KDGN is tested on real-world clinical dataset [MIMIC-III](https://mimic.physionet.org/) and outperforms several state-of-the-art deep learning methods in heathcare area in all effectiveness measures.

## Folder Specification

### data(Including the data set used in the experiment)
PRESCRIPTIONS.csv, DIAGNOSES_ICD.csv, PROCEDURES_ICD.csv: These three files are the patient's clinical records, which contain the patient's diagnosis, procedure and medication information, etc.

drug-atc.csv: this is a CID-ATC file, which gives the mapping from CID code to detailed ATC code. This file is obtained from https://github.com/sjy1203/GAMENet.

drugbank_drugs_info.csv: drug information table downloaded from drugbank here https://www.dropbox.com/s/angoirabxurjljh/drugbank_drugs_info.csv?dl=0, which is used to map drug name to drug SMILES string.

drug-DDI.csv: this file contains the drug DDI information. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing

### codes
layers.py: This file describes a graph neural network.

models.py: This file contains the overall network architecture of KDGN, including clinical records, domain knowledge processing and confidence generation networks.

new.py : This file defines the graph transformer neural network architecture.

train_KDGN.py : This file trains the KDGN model.

util.py: This file contains some defined functions.

## Requirements
-pandas: 1.3.0
-dill: 0.3.4
-torch: 1.11.0+cu111
-rdkit: 2021.03.4
-scikit-learn: 0.24.2
-numpy: 1.21.1
- Python >=3.5

## Running the code
### Data preprocessing

1. First, you should go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset (You may need to get the certificate); After that, save the three patient clinical records(DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv) into the data folder.
2.  After that, you need to download the DDI dataset [DDI data](https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0) ,put it in data folder.
3. Finally, you can process the data to get a complete records_final.pkl.
  
 
 ### train the KDGN model
 ```
 python train_KDGN.py 

### You can also save the trained model for quick testing:
 python train_KDGN.py --model_name KDGN --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval # testing with domain knowledge

```
Partial credit to https://github.com/sjy1203/GAMENet.
