Goal
====

The goal for this project is to leverage representative power of transformer to estimate the parameters and 
order of guitar effects. 

Background
==========

Music Information Retrieval (MIR) is the field in which computational methods are used to analyze 
audio, particularly recordings of music, to retrieve musical information, such as tempo, pitch, chords, etc.
With recent explosion of hardware capabilities, MIR community have adapted machine learning methods,
leveraging state of the art models in their research. 

To explore MIR, I have set out to use transformer based model to estimate guitar effects parameters.   
Parameter estimation poses several challenges. First of all, the problem lies in the input data, which is 
inherently high dimensional. A typical audio recording, recorded at 44.1 kHz, contains 44100 samples in 
1 second. There have been methods to derive audio tokens from audio samples. While I have chosen discrete
audio codec (DAC), which uses yet another transformer based model underneath, my understanding of how it 
works is minimal. I plan to investigate more in the following term. Another problem is the continuous nature
of parameters. I have used naive binning method, where I grouped set of parameters in equally sized intervals.
However, considering humans perceive changes in audio logarithmically, it may be worth exploring better 
binning schemes. The last, perhaps the most challenging problem, is somehow associating audio with set of 
audio effect parameters. Tone of electric guitars are tuned by a chain of guitar effects called pedals. 
Thus, it is not enough to guess some set of parameters, but also the ordering of the effects. 

I chose encoder-decoder transformer to approach the problem. In some sense, I am trying to translate 
audio data into sequence of parameters. This task is very reminiscent of machine translation. Thus, I have
chosen encoder-decoder model, which is widely used in machine translation. I have also seen some empirical 
success with encoder-decoder model in Natural Language Processing class, where I use OpenNMT's framework
to train a model to translate English into Chinese. The hope is that I can somehow mimic this behavior, but
with audio data.

Environment
===========

I used `python3.8.20` because that was one that was compatible with DAC code.

For python packages used, see [`requirements.txt`](requirements.txt)


Accomplishments
===============

- I set up a data pipeline, which augments dry audio with Spotify's pedalboard effects in a random fashion.
- I was able to deconstruct the transformer model by implementing its layers in PyTorch. 
- I was able to set up a crude training loop to train my implementation of the transformer and see decrease in loss. See [`src/train.ipynb`](src/train.ipynb)

Plans
=====

- Work on [`DL_Homework_5_EC.ipynb`](src/DL_Homework_5_EC.ipynb) to further ground my understanding of machine learning techniques. This exercise, which is used in machine learning class in Northwestern, was provided by a Carleton alum.
- Ensure that the implementation of the transformer is correct. While I was able to train the model, the model struggled to make sensible inferences. I can test my transformer on more traditional tasks, such as compression of images, categorizing MNIST datasets, and etc. Once I build confidence that my model behaves as it should, I will then try to train it on dummy data, where I just feed it series of pedal tokens.    
- Ensure that the distribution of the dataset is not skewed towards one particular kind of effects chain. If the data is skewed, the model may learn particular chains better than others. Thus, it is important to ensure that I'm not just training the model on, say, dry audio.
- More literature review. It is apparent that I lack significant domain knowledge to know what I am currently doing.

File Descriptions
=================

Datasets
--------

- [`src/pedalboard_utils.py`](src/pedalboard_utils.py): code for working with Spotify's pedalboard library; fixed parameters
- [`src/EGFxSetData.py`](src/EGFxSetData.py): load EGFxSet data with data augmentation; fixed parameters
- [`src/ICMTSMTGuitarData.py`](src/ICMTSMTGuitarData.py): load ICMTSMTGuitar data with data augmentation; fixed parameters
- [`src/pedalboard_param_utils.py`](src/pedalboard_param_utils.py): code for working with Spotify's pedalboard library; allows for tuning parameters of effects
- [`src/EGFxSetDataParam.py`](src/EGFxSetData.py): load EGFxSet data with data augmentation; allows for tuning parameters of effects
- [`src/ICMTSMTGuitarData.py`](src/ICMTSMTGuitarData.py): load ICMTSMTGuitar data with data augmentation; allows for tuning parameters of effects

Model
-----

- [`src/myTransformer.py`](src/myTransformer.py): my attempt at implementing a transformer in PyTorch

Notebooks
---------

- [`src/train.ipynb`](src/train.ipynb): crude train loop to train and test model
- [`src/DL_Homework_5_EC.ipynb`](src/DL_Homework_5_EC.ipynb): homework


Relevant Links and Readings
===========================

Datasets Used
-------------

- EGFxSet - https://zenodo.org/records/7044411#.YzRx2XbMKUl 
- IDMT-SMT-Guitar - https://zenodo.org/records/7544110 

Resources
---------

- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) used heavily during implementation of the transformer in PyTorch
- [`MaxHilsdorf/pedalboard`](https://github.com/MaxHilsdorf/pedalboard_audio_augmentation/blob/main/code) [`pedalboard_utils.py`](src/pedalboard_utils.py) took inspiration from this code
- [DAC](https://github.com/descriptinc/descript-audio-codec) 
