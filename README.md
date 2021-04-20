# gesture-recon
gesture recongnition using tensorflow (conv3d)

# Problem Statement:

Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote .  Each gesture corresponds to a specific command: • Thumbs up: Increase the volume • Thumbs down: Decrease the volume • Left swipe: 'Jump' backwards 10 seconds • Right swipe: 'Jump' forward 10 seconds • Stop: Pause the movie .

# Data Provided:

Each video is a sequence of 30 frames (or images). There are 666 videos provided as training data and 100 videos provided as validation data all images in a particular video subfolder have the same dimensions different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 There are two csv(one for train, one for validation) files having path of videos Our task is to train a model on the 'train' folder which performs well on the 'val' folder as well Thus, there are two types of architecture commonly used for analysing videos, both explained below. Convolutions + RNN 3D Convolutional Network, or Conv3D .

# Approach

Clearly it’s a problem that would require a deep learning solution .  Ideally this task can be performed in 2 ways:
Using the Convolutional 3d approach .
Using the CNN + LSTM approach

We picked up the Conv3D approach as it is more lightweight since it  uses a comparatively lesser amount of parameters which ensures that the model does not take too long to train .

Experimentation Detailed :

Experiment #1 - Ablation Experiment

Objective : This experiment was sort of a trial experiment , to validate if we are headed in the right direction .

Initially we picked up 15 sequences/images from the video , with a batch size of 32 .
Each image was resized to  84 x 84
Since this was sort of an ablation experiment we picked a very simple model whose layers are as follows :
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv3d_33 (Conv3D)           (None, 15, 84, 84, 16)    1312      
_________________________________________________________________
activation_30 (Activation)   (None, 15, 84, 84, 16)    0         
_________________________________________________________________
batch_normalization_52 (Batc (None, 15, 84, 84, 16)    64        
_________________________________________________________________
max_pooling3d_30 (MaxPooling (None, 7, 42, 42, 16)     0         
_________________________________________________________________
conv3d_34 (Conv3D)           (None, 7, 42, 42, 16)     2064      
_________________________________________________________________
activation_31 (Activation)   (None, 7, 42, 42, 16)     0         
_________________________________________________________________
batch_normalization_53 (Batc (None, 7, 42, 42, 16)     64        
_________________________________________________________________
max_pooling3d_31 (MaxPooling (None, 3, 21, 21, 16)     0         
_________________________________________________________________
flatten_12 (Flatten)         (None, 21168)             0         
_________________________________________________________________
dense_35 (Dense)             (None, 64)                1354816   
_________________________________________________________________
batch_normalization_54 (Batc (None, 64)                256       
_________________________________________________________________
dropout_22 (Dropout)         (None, 64)                0         
_________________________________________________________________
dense_36 (Dense)             (None, 5)                 325       
=================================================================
Total params: 1,358,901
Trainable params: 1,358,709
Non-trainable params: 192
_________________________________________________________________

Observation/Result :
1. We ran this model for 5 epochs and everything seems to work fine . We noticed an increase in training accuracy and decrease in loss which provided enough evidence that we are headed into the right direction .

2. Upon running this model for longer time / mode epochs  , we noticed that model begins to overfit . (categorical_accuracy accuracy increases , val_categorical_accuracy remains constant)

3.After 5 epochs , we got the following observation

loss: 1.1783 
categorical_accuracy: 0.4587 
val_loss: 46.6730 
val_categorical_accuracy: 0.2400



Experiment #2

Objective: Keeping the same architecture we want to  find out how the training times are affected by changing the value of batch_size  ,  length of image sequence and image size .
For batch_size=32 , len_sequence=15 , img_size = 84 x 84 , 1 epoch takes about 40-43 seconds .
For batch_size=32 , len_sequence=21 , img_size = 84 x 84 , 1 epoch takes about 54-57 seconds .
For batch_size=32 , len_sequence=21 , img_size = 120 x 120 , 1 epoch takes about 57-60 seconds .
For batch_size=64 , len_sequence=21 , img_size = 120 x 120 , 1 epoch takes about 59-61 seconds .
For batch_size=128 , len_sequence=21 , img_size = 120 x 120 , we run into error OOM when allocating tensor with shape[128,16,21,120,120] 

Observation : 
We find out that length of sequence and image size has more impact on the training times as compared to the batch size . 
As the length of training sequences increases or the image size increases the training time shoots up as well . The main reason is , as we modify the length/shape of the input sequence the trainable parameters changes accordingly.
Also we need to keep an eye out for out of memory error if we try to load in too much data (large batch size and large input dimensions) into the RAM . (even though we have written generator for the same)

Experiment , #3

Objective : Let’s make the architecture more complex by adding more layers .


Here is the model summary that we used :

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv3d (Conv3D)              (None, 21, 120, 120, 16)  1312      
_________________________________________________________________
activation (Activation)      (None, 21, 120, 120, 16)  0         
_________________________________________________________________
batch_normalization (BatchNo (None, 21, 120, 120, 16)  64        
_________________________________________________________________
max_pooling3d (MaxPooling3D) (None, 10, 60, 60, 16)    0         
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 10, 60, 60, 32)    13856     
_________________________________________________________________
activation_1 (Activation)    (None, 10, 60, 60, 32)    0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 10, 60, 60, 32)    128       
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 5, 30, 30, 32)     0         
_________________________________________________________________
conv3d_2 (Conv3D)            (None, 5, 30, 30, 64)     55360     
_________________________________________________________________
activation_2 (Activation)    (None, 5, 30, 30, 64)     0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 5, 30, 30, 64)     256       
_________________________________________________________________
max_pooling3d_2 (MaxPooling3 (None, 2, 15, 15, 64)     0         
_________________________________________________________________
conv3d_3 (Conv3D)            (None, 2, 15, 15, 128)    221312    
_________________________________________________________________
activation_3 (Activation)    (None, 2, 15, 15, 128)    0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 2, 15, 15, 128)    512       
_________________________________________________________________
max_pooling3d_3 (MaxPooling3 (None, 1, 7, 7, 128)      0         
_________________________________________________________________
flatten (Flatten)            (None, 6272)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               802944    
_________________________________________________________________
batch_normalization_4 (Batch (None, 128)               512       
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
batch_normalization_5 (Batch (None, 64)                256       
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 325       
=================================================================
Total params: 1,105,093
Trainable params: 1,104,229
Non-trainable params: 864
_________________________________________________________________


Interesting thing to note here is that , although this model has more layers than the one used in ablation experiments . But the trainable params is around 1.1 million as compared to 1.3 million  of the previous.

Conv3d filter sizes :


We modified the patience parameter in the ReduceLROnPlateau to 2 and set the min_lr = 0.001 , factor=0.2  and a starting learning rate of 0.1 . By setting patience to 2 , it basically means the  training process would wait 2 epochs  and if the val_loss does not decrease it would then reduce the learning rate by a factor of 2 , capping the minimum learning rate to 0.001

Observations/Result :

1. As we can see , the 1st convolution layer has 16 kernels , the second layer has 32 and the subsequent layers has kernels that increase by a factor of  x2 . It is kept in such a way as we move deeper into the network more complex features need to be derived from the output of previous layers .

2. Also we chose the filter size of 3 x 3 x 3 as it enables the neuron to look at the neighboring pixels in 8 directions .

3. Keeping the batch size =32 , image size=120 x 120 , number of frame =21 we train the model for 35 epochs using the Adam Optimizer .

4. The Initial learning rate of  0.1 was reduced to 0.02 after epoch 9 , which further reduced to 0.003 after epoch 16.

4.The best iteration was around epoch 18 with  following metrics :

loss: 0.4328 
categorical_accuracy: 0.8313
val_loss:0.5943 
val_categorical_accuracy: 0.7500 

After this the learning process becomes stagnant .


Experiment  #4

Objective:  Let’s increase the number of frames to 30 (full video) , keeping the above parameters and model exactly the same .

Observation/Result :

1. If we consider all the frames in the video , then we observe the learning process/ training becomes slower .

2. We also observed a considerable improvement in performance 

loss: 0.0832 
categorical_accuracy: 0.9798
val_loss: 0.4992 
val_categorical_accuracy: 0.8300 .

Experiment  #5

Objective : Modifying the ReduceLROnPlateau params and trying it out with different optimizers (ADAM & SGD)  , keeping the model the same as above.
.


We found out that SGD converges quickly  , just under 10 epochs sgd is able to achieve a training accuracy of 80%+  and validation accuracy of about 55% where as Adam’s progresses slowly with training accuracy  around 65% and validation accuracy less than 40% .

Also one more interesting thing to note is that  when we are training with the SGD optimiser the learning rate is reduced to 0.02 , just after 9 epochs .

				
Whereas in the case of Adam optimizer , the learning rate reduces to 0.02 after about 18 epochs .

This means that for this use case , SGD converges quickly (lesser epochs) as compared to Adams. Also with SGD we got higher accuracy about 84% on validation set .

Hence we make this as our final model

Model-Configurations:
Optimiser : SGD
Model : Conv3D
Frames : 30
Batch Size : 32
Image Size : 120 x 120
Epochs : 35

Metrics:
loss:0.1674  
categorical_accuracy:0.9552 
val_loss:0.4339 
val_categorical_accuracy: 0.8400
