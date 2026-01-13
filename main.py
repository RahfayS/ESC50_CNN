import streamlit as st

st.title("ESC50 Classification: Exploring Data Augmentation Strategies using ResNet Models")
st.markdown("""

## Intro to The Project

This is a personal project designed to explore the **ESC-50** dataset and take a look at the implementation of the most popular convolutional neural network archeitures, **ResNet** and take a look at audio form augmentation techniques

## About the ESC50 Dataset

The Dataset contains 2000 samples of 50 evenly split classes of environmental audio recordings (40 samples per class), with each recording is 5 seconds long

For access to the dataset, you can clone this git repo [here](https://github.com/karolpiczak/ESC-50)


| Animals | Natural soundscapes & water sounds | Human, non-speech sounds | Interior/domestic sounds | Exterior/urban noises |
|---|---|---|---|---|
| Dog | Rain | Crying baby | Door knock | Helicopter |
| Rooster | Sea waves | Sneezing | Mouse click | Chainsaw |
| Pig | Crackling fire | Sneezing | Keyboard typing | Siren |
| Cow | Crickets | Clapping | Door, wood creaks | Car horn |
| Frog | Chirping birds | Coughing | Can opening | Engine |
| Cat | Water drops | Footsteps | Washing machine | Train |
| Hen | Wind | Laughing | Vacuum cleaner | Church bells |
| Insects (flying) | Pouring water | Brushing teeth | Clock alarm | Airplane |
| Sheep | Toilet flush | Snoring | Clock tick | Fireworks |
| Crow | Thunderstorm | Drinking, sipping | Glass breaking | Hand saw |


## Data Augmentation
            
The ESC50 dataset is notorious for overfitting in its baseline models. This comes as no surprise as the dataset is only 2000 samples and for 50 classes this only comes out to 40 samples per class.

So my idea to tackle this issue came in the form of 2 strategies
            
    1. Using a deep CNN model (ResNet)
    2. Employ data augmentation techniques to artificially increase the number of training samples

In the notebook (and last page of the app) you can see the 2 techniques I used to generate artificial samples


## The Model (ResNet34)
            

The ResNet models are CNN's designed to handle the common issue all deep CNN's have, the vanishing gradient problem. The ResNet model is composed of:

1. **Residual Blocks**  
   - ResNet introduces *skip connections* or *residual connections*, which allow the input of a layer to bypass one or more layers and be added to the output.  
   - Mathematically:  
        ```
        y = F(x) + x
        ```
     where \(x\) is the input, \(F(x)\) is the output of the convolutional layers, and \(y\) is the final output of the block.  
   - This helps the network learn *residual mappings* instead of trying to learn the full transformation, which makes training deeper networks much easier.

2. **Depth and Structure**  
   - ResNet 34 is deeper than ResNet 18, with 34 layers consisting of convolutional, batch normalization, and ReLU activation layers arranged into residual blocks.  
   - This depth allows the model to capture more complex features from the input data while maintaining efficient gradient flow.

3. **Global Average Pooling and Fully Connected Layer**  
   - After passing through all the convolutional layers, ResNet applies global average pooling to reduce the spatial dimensions, followed by a fully connected layer that outputs class probabilities.

4. **Advantages for Our Task**  
   - Robust feature extraction from audio spectrograms or augmented data treated as images.  
   - Reduces the risk of overfitting due to residual connections and batch normalization.  
   - Supports transfer learning, meaning we can use pretrained weights on ImageNet to speed up training and improve performance on our dataset.


In short, ResNet 34 is a deep CNN with skip connections that allows efficient training of very deep networks. By using it on our preprocessed and augmented data, we can extract meaningful features and improve classification performance while leveraging the benefits of transfer learning.











### References
            
ESC-50 Dataset
Piczak, K. ESC-50: Environment Sound Classification. GitHub repository.
https://github.com/karolpiczak/ESC-50

Understanding the Mel Spectrogram
Analytics Vidhya. Understanding the Mel Spectrogram. Medium.
https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53

Guide to Transfer Learning in Deep Learning
Fagbemi, David. Guide to Transfer Learning in Deep Learning. Medium.
https://medium.com/@davidfagb/guide-to-transfer-learning-in-deep-learning-1f685db1fc94

Signal Framing and Windowing
SuperKogito. Signal Framing and Windowing Explained. SuperKogito Blog.
https://superkogito.github.io/blog/2020/01/25/signal_framing.html

ResNet34 Intuition (Video)
https://www.youtube.com/watch?v=KLYfwigQPuY

""")
