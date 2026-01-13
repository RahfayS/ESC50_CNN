import streamlit as st
import torch
import numpy as np
import librosa
import plotly.express as px
from utils.model_loader import load_model
from utils.balanced_augmentations import Augmentations_Balanced
from utils.acoustic_augmentations import Add_Augmentations_Acoustic
from utils.audio_preprocessing import audio_to_mel
from utils.plotting import plot_mel


# ==================================================
# CONFIG
# ==================================================
st.set_page_config(layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 22050


# ==================================================
# ESC-50 LABELS
# ==================================================
ESC50_CLASSES = {
    0: "dog", 1: "rooster", 2: "pig", 3: "cow", 4: "frog",
    5: "cat", 6: "hen", 7: "insects", 8: "sheep", 9: "crow",
    10: "rain", 11: "sea_waves", 12: "crackling_fire", 13: "crickets",
    14: "chirping_birds", 15: "water_drops", 16: "wind", 17: "pouring_water",
    18: "toilet_flush", 19: "thunderstorm", 20: "crying_baby", 21: "sneezing",
    22: "clapping", 23: "breathing", 24: "coughing", 25: "footsteps",
    26: "laughing", 27: "brushing_teeth", 28: "snoring", 29: "drinking_sipping",
    30: "door_wood_knock", 31: "mouse_click", 32: "keyboard_typing",
    33: "door_wood_creaks", 34: "can_opening", 35: "washing_machine",
    36: "vacuum_cleaner", 37: "clock_alarm", 38: "clock_tick",
    39: "glass_breaking", 40: "helicopter", 41: "chainsaw", 42: "siren",
    43: "car_horn", 44: "engine", 45: "train", 46: "church_bells",
    47: "airplane", 48: "fireworks", 49: "hand_saw"
}


# ==================================================
# MODEL LOADING
# ==================================================
@st.cache_resource
def get_models():
    balanced_model, acoustic_model = load_model(DEVICE)
    return balanced_model, acoustic_model


# ==================================================
# INFERENCE
# ==================================================
def predict(model, mel):
    with torch.no_grad():
        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits = model(x.float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def top_k(probs, k=5):
    idx = np.argsort(probs)[::-1][:k]
    return [(ESC50_CLASSES[i], probs[i]) for i in idx]


# ==================================================
# PAGE UI
# ==================================================
st.title("ESC-50 Audio Classification — Augmentation Comparison")
st.caption(
    "Compare Balanced vs Acoustic (context-aware) data augmentation strategies "
    "using ESC-50 environmental sound classification."
)

balanced_model, acoustic_model = get_models()

uploaded = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded is None:
    st.info(
        "Supported sounds include animals, natural soundscapes, human non-speech, "
        "interior domestic, and urban mechanical noises (ESC-50)."
    )
    st.stop()


# ==================================================
# AUDIO PROCESSING
# ==================================================
y, sr = librosa.load(uploaded, sr=SR)
st.audio(uploaded)

mel = audio_to_mel(y)

# Predictions
probs_acoustic = predict(acoustic_model, mel)
probs_balanced = predict(balanced_model, mel)

pred_acoustic = int(np.argmax(probs_acoustic))
pred_balanced = int(np.argmax(probs_balanced))

top5_acoustic = top_k(probs_acoustic)
top5_balanced = top_k(probs_balanced)


# ==================================================
# TOP-5 RESULTS
# ==================================================
st.subheader("Top-5 Predicted Sounds")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Acoustic-Augmented Model")
    for label, p in top5_acoustic:
        st.write(f"- **{label}** — `{p:.3f}`")

with col2:
    st.markdown("### Balanced-Augmented Model")
    for label, p in top5_balanced:
        st.write(f"- **{label}** — `{p:.3f}`")


# ==================================================
# PROBABILITY DISTRIBUTIONS
# ==================================================
st.subheader("Prediction Distributions")

col1, col2 = st.columns(2)

with col1:
    st.success(f"Prediction: **{ESC50_CLASSES[pred_acoustic]}**")
    st.plotly_chart(
        px.bar(
            x=list(ESC50_CLASSES.values()),
            y=probs_acoustic,
            labels={"x": "Class", "y": "Probability"}
        ),
        use_container_width=True
    )

with col2:
    st.success(f"Prediction: **{ESC50_CLASSES[pred_balanced]}**")
    st.plotly_chart(
        px.bar(
            x=list(ESC50_CLASSES.values()),
            y=probs_balanced,
            labels={"x": "Class", "y": "Probability"}
        ),
        use_container_width=True
    )


# ==================================================
# AUGMENTATIONS
# ==================================================
predicted_category = ESC50_CLASSES[pred_acoustic]

y_balanced = Augmentations_Balanced(y, sr)
y_acoustic = Add_Augmentations_Acoustic(y, sr, predicted_category)

mel_balanced = audio_to_mel(y_balanced)
mel_acoustic = audio_to_mel(y_acoustic)


# ==================================================
# MEL SPECTROGRAM COMPARISON
# ==================================================
st.divider()
st.header("Augmentation Comparison — Mel Spectrograms")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original")
    st.audio(y, sample_rate=sr)
    st.plotly_chart(
        plot_mel(mel, "Original Mel Spectrogram"),
        use_container_width=True
    )

with col2:
    st.subheader("Balanced Augmentation")
    st.audio(y_balanced, sample_rate=sr)
    st.plotly_chart(
        plot_mel(mel_balanced, "Balanced Augmentation Mel"),
        use_container_width=True
    )

with col3:
    st.subheader(f"Acoustic Augmentation ({predicted_category})")
    st.audio(y_acoustic, sample_rate=sr)
    st.plotly_chart(
        plot_mel(mel_acoustic, "Acoustic Augmentation Mel"),
        use_container_width=True
    )


st.caption(
    "Augmented samples are generated dynamically to visualize how different "
    "training strategies transform the same input audio."
)
