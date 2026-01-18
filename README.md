The Problem
===========


TODO
====

Data pipeline
-------------

- [ ] figure out how to use pedalboard 
- [ ] figure out how to use dataloader 
- [ ] figure out how to discrete audio codec (DAC) to derive audio tokens 
- [ ] combine pedalboard with dataloader to augment data 
- [ ] incorporate DAC into the pipeline 

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

- [ ] Multi-modal model (people have done this)
- [ ] zeroshot tests with with Logic's stock pedal board
- [ ] VST or AU using the trained model? (JUCE?)

Questions
=========

- How does machine translation work in detail? As of now, I know that you train the autoencoder to map source data to target data, which involves  
- Which loss function to use? Teacher forcing?  
- If I want to try training a multi-modal model, what kinds of datasets are out in the world, and how can I modify/augment it to fit my specific needs? 
- What kinds of MIR methodologies will I need during training and/or evaluation? 

Relevant Resources
==================

EGFxSet - https://zenodo.org/records/7044411#.YzRx2XbMKUl 

