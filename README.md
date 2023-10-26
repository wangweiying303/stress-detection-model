# stress-detection-model
This repository includes data collection, data preprocessing, construction, training, and testing for a syllable-stress detection model.

The model is based on the self-attention mechanism. There are two versions of the model: one takes numerical features measured for each syllable as input, and another one takes numerical and categorical features.
The former is included in "self_attention_numerical.py" and the latter in "self_attention_all_features.py".

The "data collection.ipnb" includes code blocks for data collection and preprocessing.
The "training and testing.ipynb" includes further codes for data processing, model initialization, training, and testing.

The folder "pronunciation dictionary" contains the pronunciation dictionary, which is publicly available at http://www.speech.cs.cmu.edu/cgi-bin/cmudict.
The file "XYW_read data.pkl" in the folder "data for training and testing" is the processed data using the speech audio and text from https://www.openslr.org/12 -> train-clean-100.tar.gz

