# Voice Gender Recognition
Notic: This is the very first version of repo. Future work is required to push the prediction performance to higher level. 

### Summary
Inspired by [LSTM Networks for Sentiment Analysis](http://deeplearning.net/tutorial/lstm.html). Here is an implementation repo for training a LSTM neurtal networks for recogonizing audio data's speaker gender. The audio data used is from  [Vox Forge](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/)

### Scraping down tgz audio file
Run [scrap.py](https://github.com/JinScientist/voice-gender-recognition/blob/master/scrap.py) will download every tgz file and save to local directory ./rawdata.

### Parse the README file in each package
In README file, the 5th line contains the speak gender information of all audio .wav files in the directory. The 'labeling' function in [vocal_gender_lstm.py](https://github.com/JinScientist/voice-gender-recognition/blob/master/vocal_gender_lstm.py) parse the REAMFILE and return the data label and wave raw data in the format of numpy array. 

### Neural Nets Graph
Use fixed number of LSTM cells to take input from squential wave raw data. The hidden state of each cells are concated nated to 2-D matrix as output. The output data dimension is reduced by takeing average pooling in large strides. Then the output layer is stardard softmax on pooling results. The cost function is constructed by caculating the cross entroy between data label and softmax output from the networks.

### Mini Batch training
The training process takes each tgz file as one mini batch.All 10 audio files are taken for one epoch of opitimizing process. Every 10 mini batch, the network prediction performance is validated by run 100 out-of-sample validation samples. The classification accuracy is printed by percentage. 


