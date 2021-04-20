

**Problem Statement:**

Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote . Each gesture corresponds to a specific command: • Thumbs up: Increase the volume • Thumbs down: Decrease the volume • Left swipe: &#39;Jump&#39; backwards 10 seconds • Right swipe: &#39;Jump&#39; forward 10 seconds • Stop: Pause the movie .

**Data Provided:**

Each video is a sequence of 30 frames (or images). There are 666 videos provided as training data and 100 videos provided as validation data all images in a particular video subfolder have the same dimensions different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 There are two csv(one for train, one for validation) files having path of videos Our task is to train a model on the &#39;train&#39; folder which performs well on the &#39;val&#39; folder as well Thus, there are two types of architecture commonly used for analysing videos, both explained below. Convolutions + RNN 3D Convolutional Network, or Conv3D .

**Approach**

Clearly it&#39;s a problem that would require a deep learning solution . Ideally this task can be performed in 2 ways:

- Using the Convolutional 3d approach .
- Using the CNN + LSTM approach

We picked up the Conv3D approach as it is more lightweight since it uses a comparatively lesser amount of parameters which ensures that the model does not take too long to train .

**Summary**

| **#Exp** | **Model** | **Params** | **Result** | **Decision + Explanation** |
| --- | --- | --- | --- | --- |
| **1** | **Conv3D+ ADAM** | **Batch\_size** :32 **Image\_size** :84 x 84 , **Frames** : 15, **Epochs** : 5 | loss: 1.1783 categorical\_accuracy: 0.4587 val\_loss: 46.6730 val\_categorical\_accuracy: 0.2400 | **A small ablation experiment to see our architecture is able to learn from the training data (overfits)** |
| **2.1** | **Conv3D+ADAM** | **Batch\_size** :32 **Image\_size** :84 x 84 , **Frames** : 15, **Epochs** : 5 | **1 epoch takes 40-43 sec** | **We found out that length of sequence and image size has more impact on the training times as compared to the batch size Also we need to keep an eye out for out of memory error if we try to load in too much data (large batch size and large input dimensions) into the RAM . (even though we have written generator for the same)** |
| **2.2** | **Conv3D+ADAM** | **Batch\_size** :32 **Image\_size** :84 x 84 , **Frames** : 21, **Epochs** : 5 | **1 epoch takes 54-57 sec** |
| **2.3** | **Conv3D+ADAM** | **Batch\_size** :64 **Image\_size** :120 x 120 , **Frames** : 21, **Epochs** : 5 | **1 epoch takes 59-61 sec** |
| **2.4** | **Conv3D+ADAM** | **Batch\_size** :128 **Image\_size** :120x120 , **Frames** : 21, **Epochs** : 5 | **OOM when allocating tensor with shape [128,16,21,120,120]** |
| **3** | **Conv3D+ADAM** | **Batch\_size** :32 **Image\_size** :120x120 , **Frames** : 21, **Epochs** : 35 | loss: 0.4328 categorical\_accuracy: 0.8313val\_loss:0.5943 val\_categorical\_accuracy: 0.7500 | **As we can see , the 1st convolution layer has 16 kernels , the second layer has 32 and the subsequent layers has kernels that increase by a factor of x2 . It is kept in such a way as we move deeper into the network more complex features need to be derived from the output of previous layers . Also we chose the filter size of 3 x 3 x 3 as it enables the neuron to look at the neighboring pixels in 8 directions .** |
| **4** | **Conv3D+ADAM** | **Batch\_size** :32 **Image\_size** :120x120 , **Frames** : 30, **Epochs** : 35 | loss: 0.0832 categorical\_accuracy: 0.9798val\_loss: 0.4992 val\_categorical\_accuracy: 0.8300 | **If we consider all the frames in the video , then we observe the learning process/ training becomes slower . Also we get amazing accuracy figures for training and validation datasets .** |
| **5 (final model)** | **Conv3D + SGD** | **Batch\_size** :32 **Image\_size** :120x120 , **Frames** : 30, **Epochs** : 35 | loss:0.1674 - categorical\_accuracy:0.9552 val\_loss:0.4339 val\_categorical\_accuracy: 0.8400 | **SGD, converges quickly as compared to ADAM optimizer alos we got higher accuracy with sgd and we choose this as our final model .** |

