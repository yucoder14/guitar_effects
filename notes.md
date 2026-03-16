The Problem
===========

Applying machine translation to estimate the order of pedals in an effects chain given some audio sample. 
Guitarist who are starting out often struggle to figure out how to order the effects in order to get the desired sound. 
I thought it would be interesting if I can use machine learning to do the "heavy lifting," so that the time spent on ordering the effects is greatly reduced. 


TODO
====

Data pipeline (tentatively done)
-------------

- [x] figure out how to use pedalboard 
- [x] figure out how to use dataset
- [x] combine pedalboard with dataset to augment data 
- [x] figure out how to discrete audio codec (DAC) to derive audio tokens 
- [x] make the dataset compatible with dataloader; did this by making sure that data i'm working with have same length
- [x] shifting onsets? i am randomly sampling 2 seconds from given audio sample
- [x] ~~figure out how to implement dataset for data with varying lengths~~ collating function to pad uneven lengths
- [x] ~~adding noise to the data; also shifting onset of the notes, tempo~~ not going to worry about this for now
- [x] tokenize pedal information; the challenge lies in encoding the parameter information somehow; created PedalVocab class
- [x] normalize augmented audio; should I do this after or before running audio through dac?; i do this in `__getitem__` in Dataset classes 
- [x] Vary the parameters of each effects and to effects estimations on top of order estimation; well i don't have a trained model to evaluate, but i used binning method for pedal parameters

Model
-----

- [x] decide which architecture to use (autoencoder, but what to use for encoder & decoder) --> decided to implement transformer just for kicks
- [ ] incorporate DAC into the pipeline 
- [ ] batch norm!; this is implemented as layernorm? 
- [ ] Vocab an embedding/un-embedding matrices for pedal string tokens
- [ ] implement the model (doesn't look terribly hard given torch abstractions) --> a bit challenging because of broadcasting stuff
- [ ] train the model --> getting closer!  
- [ ] tensorboard logging (loss and other relevant stuff, so I can see model progress) 

Evaluation
----------

- [ ] how do you evaluate model fit in machine translation tasks? --> i can extract information from the output sequence and actually apply the settings on spotify's pedalboard; the question is whether the output sequence would be a valid string...
- [ ] human evaluation because humans can hear sound and make a judgment 

Moonshots
---------

- [ ] implement KV-caching
- [ ] try to see if i can implement the optimization that deepseek had made (https://www.youtube.com/watch?v=0VLAoVGf_74) 
- [ ] Is it possible to create embeddings for pedals with their parameters encoded? How would I go about doing this? 
- [ ] Multi-modal model (people have done this)
- [ ] zeroshot tests with with Logic's stock pedal board
- [ ] VST or AU using the trained model? (JUCE?)
- [ ] As part of data augmentation, it is possible to splice the audiofiles together to form phrases

Questions
=========

- If I want to try training a multi-modal model, what kinds of datasets are out in the world, and how can I modify/augment it to fit my specific needs? 
- What kinds of MIR methodologies will I need during training and/or evaluation?
- Are some ordering of effects symmetric (different ordering produce the "same" sound)? 
- Is training a model on single note/chord enough for the model to extract relevant patterns from a full instrumental sample? 
- If pursuing estimating f(dry sample, pedalboard) = wet sample idea, how should i go about even training such a model? (https://arxiv.org/html/2407.10646v1 this seems relevant); can i do the inverse g(wet sample) = (dry sample, pedalboard)?  
- can I just take a weighed sum of the parameters and treat that as if it were a "word" representing the entire pedal? After the model is somehow trained, will the embeddings embody the effects of the order and the parameters of the effects? 

Notes
=====

- what is stft (short-time fourier transform)? essentially, because fourier transform treats any sequence as a one continuous wave, you lose time sensitive information like onset and duration of notes played. in stft, you first divided the original signal into chunks of predetermined lengths. then you do fft on each of those chunks. then, given some time interval, you have a better sense of what frequencies were present at that specific interval. this is essentially what spectogram shows you. however, spectrograms are in Hz and ignores that human perception is not linear but logarithmic. this is where mel-spectrogram becomes useful. although i don't have all the details, it is useful in machine learning, too, because it compresses information while not losing too much information. think about it this way: the note difference between 100 Hz and 200 Hz is much greater than that between 10000 Hz and 10100 Hz.  

- How does machine translation work in detail? As of now, I know that you train the autoencoder to map source data to target data. Because I have decided to work with transformer based autoencoder, the key is in the cross attention layer. You first pass the source data through a encoder (bi-directional, so no masking of the attention scores) to get the latent representation of the source data. Similarly, you pass the target data through the encoder, but, unlike the encoder, the decoder has cross-attention layer, which also uses the latent representation of the source data to do calculate attention score against the transformed target data. 

-  Teacher forcing? Teacher forcing is when you force the model to train on corrected answer rather than its previous wrong answer. For instance, if the correct sequence is 'I have a yellow banana', and model predicted 'I have a yellow monkey', then I for the model to do the next inference on the former, correct sequence. Paralleling teacher forcing involves shifting `y_in` one to the right to create `y_out`, and training the decoder to predict `y_out` (assuming that the decoder is properly masked so it does not see future tokens inside the attention layers)

- I am aware that the parameters are tweaked through back propagation algorithm to decrease loss, but it still blows me away this process results in "intelligent" behavior

- Which loss function to use? Most likely cross-entropy loss. I need to delve deeper into what it calculates, however. 


Brainstorming 
==============

So the current plan for the pipeline is as follows: 
- (On CPU) Dataset (EGFxSet or IDMT) -> pedalboard ~~Augmentation (Time shifts) ->~~  
- Dataloader will load data and apply pedalboard effects using CPU
- descript audio codec to derive the tokens (this is me trying to get audio embeddings); on CPU or GPU? I would have to see if GPU's VRAM will run out of memory if i try to have the dac model be part of the training 
- then train

I'm taking clean audio sample and then using spotify's pedalboard to apply some random effects chain to the audio. I have not considered whether the order of the effects is "realistic", i.e., something that people will consider using. 
- Because I'm just using python's random library, I do not have much control over the distribution of the kinds of pedalboards.
- I modified the code now to disable shuffling, which will yield pedals of following order: Compression --> Distortion --> Chorus --> Phaser --> Reverb 
    - would be nice to have either chorus or phaser in the signal chain not both 

Think about how word embeddings are created and how might I extend them to creating embeddings for guitar pedal effects to somehow encode the type of an effect and the parameters of the effect...
- First of all, there are large classes of effects (e.g. Chorus, Distortion, etc.) that pedals can be categorized into; regardless of specific implementation of the pedal, it's effect on the audio signal is largely the same within the same class (i don't know if this is true in general)
- however the implementation of each pedal can vary as well as number of parameters to tune the effects

To know how the effects impact the dry sample, i do need to work with both the dry and wet samples. However, figuring out how an effect influences a dry sample is the job of the bigger model (associating how a pedal embedding is related to the wet audio); so i can just think about how i get embedding representation of pedals 

- in transformer based model, the embeddings for words are learned during training time, but the size of the vocabulary is fixed; in some ways the vocabulary is the parameter themselves
- treating parameters of each effect as if it were a word or token that could be one-hot encoded
- maybe the order of the pedal could be encoded like how word orders are encoded? i guess one difference is that the order of the pedal matters but the order of parameters do not matter 

- As of now, the approach is to take a wet audio sample and effects order information to see if the model can learn how the ordering of the effects influences audio. While this approach has the potential to learn how orders of effects influence dry samples without having to toy with dry samples, I am not very confident that the model will learn transferable parameters. --> still sticking with this plan; other plans sound interesting though
- I can still try to encode pedal board information as some kind of labeling scheme --> I did this

Another approach is to take dry sample, effects information and wet sample to see if I can get the model to learn to take dry sample with effects information to produce the correct wet sample. This approach, while more complex, has the potential to yield effects embedding, which will contain more rich information of each effect. I am currently unsure how you would "reverse" the inference, so that the model takes a wet sample and yields dry sample with pedal board information. 
- treat parameters of the effects as "words"; while the some parameters are continuous, because the goal is rough estimation, I could arbitrarily divide the range of the parameter into discrete chunks
- then, pedal board will be sentences with words like 'dist\_25' (for distortion with 25db gain) to encode pedal specific information, 'start\_dist/end\_dist' tokens to mark beginning and the end of pedals, and 'start\_pedal' and 'end\_pedal' tokens to mark the beginning and end of effect chain
    - considering the attention mechanism, which I'm hoping to leverage, the transformer based model may learn that 25 in the context of distortion pedal is different than 25 in the context of reverb?
- unlike natural language i will probably have far fewer words in the vocabulary 

Yet another approach seems to be training a encoder decoder model with google's ddsp (differentiable digital signal processing) library to take encoder's predictions and apply it on dry audio to reconstruct the audio; once the training finishes, the decoder should have critiqued the encoder well enough that encoder will produce meaningful prediction of the parameters given wet audio. (https://arxiv.org/pdf/2408.11405) 
- switching matrix to learn order of the pedals

- ~~My implementation of EGFxSet is not compatible with DataLoader due to varying sizes in the lengths of the audio sample and the label that goes along with the audio. I circumvented this problem by cropping the tensors of set lengths.~~ no longer the case


- ~~Dependency hell. Figure out how to configure conda environment correctly so that both torch, tensorflow and discrete audio codec runs on the GPU. there's stuff about NUMA nodes, which I don't fully comprehend yet...~~ I don't need to worry about this for now

- ~~Now that I think of it, it may be difficult for me to run both the tf-based model by dac and torch stuff in the same pipeline because of GPU allocation problems...~~ This might not be of concern since it appears that dac uses tensorflow for tensorboard, not for their models. It is this case. I was able to run dac model on GPU

- ~~also my current approach completely disregards the parameters of the effects~~ I used binning method to create pedal vocabulary 

Concerns
========

- Will I have enough computing power to train a transformer based model?
- Am i trying to be too ambitious with this project?
- Be mindful of the VRAM usage. I don't know if it's because i'm using jupyter notebook, which causes variables to persists in the VRAM, but it's getting filled up pretty fast... No wonder people go crazy for VRAMS 


Relevant Resources
==================

- EGFxSet - https://zenodo.org/records/7044411#.YzRx2XbMKUl 
- IDMT-SMT-Guitar - https://zenodo.org/records/7544110 
- https://github.com/MaxHilsdorf/pedalboard_audio_augmentation/blob/main/code/audio_augmentation.py took inspiration from this code
