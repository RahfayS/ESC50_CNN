# ESC50_CNN

Convolutional Neural Network (CNN) for environmental sound classification using the **ESC-50 dataset**.  
This project trains a deep learning model to classify 50 types of environmental sounds from 5-second audio clips.

---

## Project Overview

Environmental sound classification assigns semantic labels (e.g., *dog bark*, *rain*, *siren*) to short audio recordings.  
The ESC-50 dataset contains **2,000 labeled audio clips in 50 classes** and is a common benchmark in audio classification research.

This repository includes:
- Preprocessing and feature extraction pipelines
- CNN model definition and training
- Evaluation and metrics visualization
- Notebooks for experimentation and analysis

---

## Data Augmentation Strategy

To improve model robustness and reduce overfitting on the relatively small ESC-50 dataset, multiple audio-domain augmentation strategies were explored.

### Motivation

The ESC-50 dataset contains only **40 samples per class**, making CNN-based models prone to:
- Overfitting to background noise patterns
- Memorization of short-term temporal structures
- Confusion between acoustically similar classes (e.g., *rain vs. wind*, *clock tick vs. keyboard typing*)

Data augmentation was therefore used to artificially increase data diversity while preserving class semantics.

---

###  Acoustic Augmentations

The first strategy focused on **physically motivated waveform-level transformations** applied prior to feature extraction:

- **Time Stretching**  
  Slightly speeds up or slows down the signal without altering pitch.

- **Pitch Shifting**  
  Shifts frequency content to simulate variations in sound sources.

- **Additive Noise Injection**  
  Adds low-amplitude Gaussian noise to improve robustness to background interference.

- **Random Gain Scaling**  
  Simulates changes in recording volume and microphone distance.

Augmentations were applied stochastically during training to avoid generating deterministic duplicates.

**Observation:**  
While acoustic augmentations improved general robustness, the model continued to struggle with specific confusable class pairs. Performance gains plateaued, indicating that global acoustic transformations alone were insufficient.

---

### Balanced (Class-Aware) Augmentation

A second strategy introduced **class-aware augmentation**, where transformations were applied selectively based on class performance.

Key principles:
- More aggressive augmentation for underperforming or frequently misclassified classes
- Reduced augmentation for already well-classified classes
- Preservation of class balance across the training set

This approach led to improved recall for difficult classes and reduced bias toward dominant acoustic patterns.

---

### Key Findings

- Acoustic augmentation alone was insufficient to resolve class confusion.
- Balanced, class-aware augmentation produced more consistent improvements.
- Some ESC-50 classes remain inherently ambiguous using CNN-only approaches.

---

### Conclusion

Despite extensive augmentation, performance improvements eventually saturated. This highlights known limitations of CNN-based models on small audio datasets and suggests future work may benefit from:
- Deeper or attention-based architectures
- Pretrained audio representations (e.g., YAMNet, PANNs)
- Temporal modeling beyond fixed-size spectrograms



### How it Works

![Demo](test_video/output.gif)

### Check it out

![Click here to check it out on streamlit]()