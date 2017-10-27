# voice-gender-recognition
Train a LSTM neural networks on Vox Forge public audio data set to recognize speaker's gender

## Summary
Inspired by [LSTM Networks for Sentiment Analysis](http://deeplearning.net/tutorial/lstm.html). Here is an implementation repo for training a LSTM neurtal networks for recogonizing audio data's speaker gender. The audio data used is from  [Vox Forge](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/)

## Scraping down tgz audio file
Run [scrap.py](https://github.com/JinScientist/voice-gender-recognition/blob/master/scrap.py) will download every tgz file and save to local directory ./rawdata.

## Parse the README file in each package
In README file, the 5th line contains the speak gender information of all audio .wav files in the directory. The 'labeling' function in [vocal_gender_lstm.py](https://github.com/JinScientist/voice-gender-recognition/blob/master/vocal_gender_lstm.py) parse the REAMFILE and return the labeled data in the format of numpy array. 

##
