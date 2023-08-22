# Bean leaf health screening using autoencoder

Methodology discussed in [report](report.pdf)
# File structure

|File | description|
|:---: |:---:| 
|main.py |Common Code for CNN and Autoencoder models|
|cnn_classification_model.pt| Saved CNN model|
|CNN_blind_prediction.csv| Blind Test predictions made by trained CNN model|
|autoencoder_model.pt | Saved Autoencoder model|
|autoencoder_classification_model.pt | Saved Autoencoder classification model|
|autoencoder_blind_prediction.csv |Blind Test predictions made by trained|

# Results


| Model | Train accuracy | Valid accuracy|
|:---: |:---:| :---:|
|Autoencoder| 98.13| 93|
|CNN |93.4 |90|

## References:
1. This implementation is done as an assignment for the following MSAI course (Winter 2023): [Northwestern University - MSAI 437: Deep Learning](https://www.mccormick.northwestern.edu/artificial-intelligence/curriculum/descriptions/msai-437.html). The instructor for this course is: [Prof. David Demeter](https://scholar.google.com/citations?user=TUnj2lIAAAAJ&hl=en).
2. Datasets:
```
a. https://github.com/AI-Lab-Makerere/ibean 
```

