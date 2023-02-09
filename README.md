# SLAIN - Audio Inpainting With Conditional GAN

PAPER(unpublished): https://drive.google.com/file/d/1sxTwtQeLqkPX__x7kAilFrQ_JKgehHcF/view?usp=share_link

## Intro

This is my master research in NTU.
It can inpaint the missing part within a segment of music or sound on time axis.

The works has been rejected due to lack of comparison,
with the overlook of my rebuttal.
Then I just leave it along...

The results are good and the method is innovative at that time,
That's all for me to be satisfied.

To reproduce the results, the instruction below lack of details to build env,
and it might not be fixed.

## Instruction

1. git clone repo
2. cd audio_inpainting
3. git submodule update --init

## Description

1. The repo is not well organized right now, but should be available for training and inference.
2. All the hyperparameters are written in the train.py, just for easily developement in personal project.
3. If the hyperparameters are set correctly, should be able to run "python train.py". The details to run the project might be available if need.
4. Those scripts to deal with data are in directory scripts, plz follow the usage in codes.
5. I provide the checkpoints for Esc-50 and the Maestro below.

## Attachments
1. The checkpoints of Esc-50, the Maestro and the pretrained Vocoder MBmelGAN: https://drive.google.com/drive/folders/1IF2HtKee5cmffmm-CfJYrtFzJddAiVRD?usp=sharing

## Results with baselines.
1. Samples results with Esc-50 dataset: https://drive.google.com/drive/folders/1q3ZByD4zqgPn5ou5qs-Ycw8eI0xgNfOv?usp=sharing
2. Samples results with the Maestro dataset: https://drive.google.com/drive/folders/1xQLRO3KRJd74SrEvTjziIWLOHLP05FLE?usp=sharing

## Supplementary
https://drive.google.com/file/d/1CkcF3oDRconF4PsYTXyda8KriSfX1jRN/view?usp=sharing
