import streamlit as st
from pathlib import Path


# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(layout="wide")

st.title("Model Training Overview")
st.caption(
    "This page documents the training methodology, augmentation strategies, "
    "loss behavior, and evaluation results for the ESC-50 audio classification models."
)

# ==============================================
# MODEL ARCHITECTURE
# ==================================================
st.header("Model Architecture")

st.markdown(
    """
---
Both models use the **same CNN architecture** and differ *only* in how training
data is augmented.

### Input Representation
- Log-mel spectrograms
- 128 Mel frequency bands
- Fixed-length time dimension

### Network Design
- Convolutional feature extractor
- Batch normalization + ReLU
- Global pooling
- Fully connected classifier (50 classes)

This ensures that any performance differences arise from **augmentation strategy**
rather than architectural changes.

---
"""
)


# ==================================================
# AUGMENTATION STRATEGIES
# ==================================================
st.header("Training Strategies")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Balanced Augmentation")
    st.markdown(
        """
A **class-agnostic augmentation strategy** applied uniformly across all ESC-50 samples.

**Techniques include:**
- Time shifting
- Time stretching
- Pitch shifting
- Volume scaling

**Goal:**  
Improve general robustness while preserving sound identity.

This approach encourages the model to learn invariant features across mild-to-moderate
audio distortions.

"""
    )

with col2:
    st.subheader("Acoustic (Context-Aware) Augmentation")
    st.markdown(
        """
A **category-aware augmentation strategy** that simulates realistic acoustic conditions.

Each sound class is mapped to a broader acoustic group:
- Animals
- Natural soundscapes
- Human non-speech
- Interior domestic
- Urban mechanical

**Augmentations simulate:**
- Environmental ambience
- Distance and muffling
- Room acoustics and reverb
- Urban and indoor noise

**Goal:**  
Encourage invariance to real-world recording conditions.
"""
    )


# ==================================================
# TRAINING CONFIGURATION
# ==================================================
st.header("Training Configuration")

st.markdown(
    """
---

- Optimizer: **Adam**
- Loss Function: **Cross-Entropy Loss**
- Batch Size: 32
- Identical learning schedules for both models
- Validation on held-out data

Training was conducted under controlled conditions to ensure a fair comparison.
"""
)


# ==================================================
# LOSS CURVES
# ==================================================
st.header("Training Loss Curves")

img_dir = Path("images")

balanced_loss = img_dir / "balanced_loss.png"
acoustic_loss = img_dir / "acoustic_loss.png"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Balanced Augmentation Loss")
    if balanced_loss.exists():
        st.image(
            str(balanced_loss),
            caption="Training & Validation Loss — Balanced Augmentation",
            use_container_width=True,
        )
    else:
        st.warning("Balanced loss curve image not found.")

with col2:
    st.subheader("Acoustic Augmentation Loss")
    if acoustic_loss.exists():
        st.image(
            str(acoustic_loss),
            caption="Training & Validation Loss — Acoustic Augmentation",
            use_container_width=True,
        )
    else:
        st.warning("Acoustic loss curve image not found.")


st.markdown(
    """
---
### Loss Curve Observations
- Both models converge smoothly without instability
- The acoustic-augmented model shows more stable validation loss
- Reduced oscillation suggests better generalization under realistic conditions
"""
)


# ==================================================
# CONFUSION MATRIX
# ==================================================
st.header("Confusion Matrix Analysis")

col1, col2 = st.columns(2)

balanced_cm = img_dir / "balanced_confusion_matrix.png"
acoustic_cm = img_dir / "acoustic_confusion_matrix.png"

with col1:
    st.subheader("Balanced Augmentation")
    if balanced_cm.exists():
        st.image(
            str(balanced_cm),
            caption="Confusion Matrix – Balanced Augmentation",
            use_container_width=True,
        )
    else:
        st.warning("Balanced confusion matrix image not found.")

with col2:
    st.subheader("Acoustic Augmentation")
    if acoustic_cm.exists():
        st.image(
            str(acoustic_cm),
            caption="Confusion Matrix – Acoustic Augmentation",
            use_container_width=True,
        )
    else:
        st.warning("Acoustic confusion matrix image not found.")

st.markdown(
    """
---

### Error Analysis

- Most misclassifications occur between **acoustically similar classes**
- **Natural soundscapes** (e.g., rain vs. sea waves) remain particularly challenging
- **Human non-speech sounds** exhibit strong class separability across both models
- **Acoustic augmentation** reduces confusion in **urban** and **indoor** environments
- Both augmentation strategies show signs of **overfitting**, as indicated by the train–test performance gap
"""
)


# ==================================================
# Model Performance
# ==================================================

st.header("Augmentations Performance")

st.markdown(
    """

---

### Balanced Augmentation Performance

The **balanced** augmentation strategy yielded (50 epochs)
```
- Train Accuracy = 99.98%
- Validation Accuracy = 80.25%
- Test Accuracy = 71.0%
```

---

### Acoustic Augmentation Performance

The **acoustic** augmentation strategy yielded (50 epochs)
```
- Train Accuracy = 100.00%
- Validation Accuracy = 79.00%
- Test Accuracy = 70.0%
```


---

### Baseline Model Comparison

The baseline results are taken from the official ESC-50 repository:  
https://github.com/karolpiczak/ESC-50?tab=readme-ov-file

The baseline model employs an **18-layer CNN** trained directly on **raw waveform data**, achieving a reported test accuracy of **68.70%**.

---

"""
)


# ==================================================
# KEY TAKEAWAYS
# ==================================================
st.header("Key Takeaways")

st.markdown(
    """
### Balanced Augmentation Strategy Takeaways


- **Training performance:** The model reaches a peak training accuracy of **99.8%**, indicating that it has memorized the training set effectively. This high training performance shows that the network capacity is sufficient to capture the patterns in the dataset, but it also suggests potential overfitting, as the model may be relying on fine-grained details of the training spectrograms rather than learning robust, invariant features.

- **Validation performance:** Validation accuracy peaks at **81.25%**, which is substantially lower than training accuracy. This gap suggests that while the model can generalize to data similar to what it has seen during training, it struggles with variations or distortions not present in the training set. The moderate generalization indicates that the model has learned some features that are transferable, but may also be sensitive to augmentation artifacts or minor spectral variations.

- **Test performance:** The test accuracy of **71%** reflects a further drop in generalization to completely unseen data. This decline highlights that the model is particularly challenged by certain classes, likely due to spectral similarities between classes, short-duration events, or sensitivity to waveform augmentation. It indicates that the learned features are not fully invariant to these variations.

- **Classification report insights:** The report shows a mix of high and low F1-scores across classes. Classes with high F1-scores (>0.85) correspond to sounds with strong, distinct spectral signatures, which the model captures reliably. Low F1-scores (0.27–0.40) are seen in classes with transient, overlapping, or easily distorted sounds, indicating the model struggles with these. This pattern suggests that the network prioritizes dominant or long-duration spectral patterns and fails to generalize well on subtle or ambiguous classes.

- **Potential insights:**
  - Over-augmentation during training may have distorted semantic features for some classes, hurting test performance.
  - The gap between validation and test accuracy may indicate fold-specific differences or insufficient coverage of rare or challenging sound events.
  - High training accuracy combined with moderate test accuracy indicates memorization rather than robust feature learning.
  - Weak classes could benefit from targeted augmentation, improved preprocessing, or leveraging pretrained audio representations that capture invariant spectral and temporal patterns.

- **Overall conclusion:** While the CNN pipeline performs strongly on easily recognizable classes, it exhibits clear limitations in generalization to more challenging sounds. Improvements such as moderated augmentation, careful fold management, or using pretrained audio models (e.g., CNN14, PANNs) are likely necessary to increase overall test performance and stabilize predictions across all classes.

---

### Acoustic Augmentations Conclusions


Despite extensive experimentation with acoustic-based augmentations, the model was ultimately unable to reliably distinguish several of the more challenging ESC-50 classes. While certain classes benefited from the added variability, the overall impact of these augmentations was limited. Persistent confusion remained among acoustically similar categories, indicating that simply increasing intra-class diversity through handcrafted acoustic transformations was not sufficient to resolve the model’s weaknesses.

The classification report highlights this imbalance clearly. Some classes achieved near-perfect precision and recall, while others exhibited very low recall and F1-scores, suggesting that the model continued to collapse predictions toward more dominant or acoustically salient patterns. This uneven performance indicates that the augmentation strategy amplified existing biases rather than encouraging more discriminative feature learning across all classes.

This experiment represents the practical limit of my custom acoustic augmentation approach. While conceptually sound, augmenting strictly based on acoustic class groupings did not consistently translate into improved generalization. In contrast, proven ESC-50 augmentation strategies—such as balanced or class-aware augmentation pipelines—demonstrate more stable performance across classes, achieving an overall accuracy of 75% with significantly reduced variance in class-level metrics.

---

### Conclusion

In conclusion, these results suggest that augmentation design is more critical than augmentation quantity. Effective ESC-50 performance appears to rely on carefully balanced, empirically validated strategies rather than purely intuitive acoustic transformations. Future improvements would likely require either stronger feature representations, more expressive models, or augmentation policies that explicitly target class imbalance and inter-class confusion rather than acoustic similarity alone.


"""
)


st.caption(
    "This training overview complements the interactive inference demo by "
    "providing transparency into model development and evaluation."
)
