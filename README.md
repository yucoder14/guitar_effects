The Problem
===========

Applying machine translation to estimate the order of pedals in an effects chain given some audio sample. 
Guitarist who are starting out often struggle to figure out how to order the effects in order to get the desired sound. 
I thought it would be interesting if I can use machine learning to do the "heavy lifting," so that the time spent on ordering the effects is greatly reduced. 

TODO
====

Data pipeline
-------------

- [x] figure out how to use pedalboard 
- [x] figure out how to use dataset
- [x] combine pedalboard with dataset to augment data 
- [ ] adding noise to the data; also shifting onset of the notes, tempo
- [ ] figure out how to discrete audio codec (DAC) to derive audio tokens 
- [ ] incorporate DAC into the pipeline 
- [ ] make the dataset compatible with dataloader
- [ ] figure out how to implement dataset for data with varying lengths

Model
-----

- [ ] decide which architecture to use (autoencoder, but what to use for encoder & decoder)
- [ ] implement the model (doesn't look terribly hard given torch abstractions) 
- [ ] train the model  
- [ ] tensorboard logging (loss and other relevant stuff, so I can see model progress) 

Evaluation
----------

- [ ] how do you evaluate model fit in machine translation tasks?  
- [ ] human evaluation because humans can hear sound and make a judgment 

Moonshots
---------

- [ ] Vary the parameters of each effects and to effects estimations on top of order estimation
- [ ] Multi-modal model (people have done this)
- [ ] zeroshot tests with with Logic's stock pedal board
- [ ] VST or AU using the trained model? (JUCE?)

Questions
=========

- How does machine translation work in detail? As of now, I know that you train the autoencoder to map source data to target data.
- Which loss function to use? Teacher forcing?  
- If I want to try training a multi-modal model, what kinds of datasets are out in the world, and how can I modify/augment it to fit my specific needs? 
- What kinds of MIR methodologies will I need during training and/or evaluation?
- Are some ordering of effects symmetric (different ordering produce the "same" sound)? 
- Is training a model on single note/chord enough for the model to extract relevant patterns from a full instrumental sample? 

Notes
=====

As of now, I'm taking clean audio sampe and then using spotify's pedalboard to apply some random effects chain to the audio. I have not considered whether the order of the effects is "realistic", i.e., something that people will consider using. 
- Because I'm just using python's random library, I do not have much control over the distribution of the kinds of pedalboards. 

As of now, my implementation of EGFxSet is not compatible with DataLoader due to varying sizes in the lengths of the audio sample and the label that goes along with the audio.

Relevant Resources
==================

EGFxSet - https://zenodo.org/records/7044411#.YzRx2XbMKUl 
IDMT-SMT-Guitar - https://zenodo.org/records/7544110
