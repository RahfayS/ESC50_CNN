import streamlit as st
import torch
import numpy as np
import librosa
import plotly.express as px
from utils.model_loader import load_model
from utils.balanced_augmentations import Augmentations_Balanced
from utils.acoustic_augmentations import Add_Augmentations_Acoustic

# ------------------ CONFIG ------------------
SR = 22050
N_MELS = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ ESC-50 LABELS ------------------
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

# ------------------ HELPERS ------------------
def audio_to_mel(y, sr=SR):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def predict(model, mel):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits = model(x.float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def plot_mel(mel, title):
    fig = px.imshow(
        mel,
        origin="lower",
        aspect="auto",
        title=title,
        color_continuous_scale="magma"
    )
    fig.update_layout(height=300)
    return fig


@st.cache_resource
def get_models():
    acoustic = load_model("models/ResNet34_acoustic_augmentation.pth")
    balanced = load_model("models/ResNet34_balanced_augmentation.pth")
    return acoustic, balanced


# ------------------ STREAMLIT APP ------------------
def main():
    st.set_page_config(layout="wide")
    st.title("ESC-50 Audio Classification - Augmentation Comparison")

    acoustic_model, balanced_model = get_models()

    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded is None:
        st.info("Upload an audio file to begin")
        return

    # Load audio
    y, sr = librosa.load(uploaded, sr=SR)
    st.audio(uploaded)

    # Original mel
    mel_orig = audio_to_mel(y)

    # Predictions
    probs_acoustic = predict(acoustic_model, mel_orig)
    probs_balanced = predict(balanced_model, mel_orig)

    pred_a = int(np.argmax(probs_acoustic))
    pred_b = int(np.argmax(probs_balanced))

    # ------------------ PREDICTIONS ------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Acoustic-Augmented Model")
        st.success(f"Prediction: **{ESC50_CLASSES[pred_a]}**")
        fig = px.bar(
            x=[ESC50_CLASSES[i] for i in range(50)],
            y=probs_acoustic,
            labels={"x": "Class", "y": "Probability"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Balanced-Augmented Model")
        st.success(f"Prediction: **{ESC50_CLASSES[pred_b]}**")
        fig = px.bar(
            x=[ESC50_CLASSES[i] for i in range(50)],
            y=probs_balanced,
            labels={"x": "Class", "y": "Probability"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.header("Augmentation Examples")

    category = ESC50_CLASSES[pred_a]

    # Apply augmentations
    y_balanced = Augmentations_Balanced(y, sr)
    y_acoustic = Add_Augmentations_Acoustic(y, sr, category)

    mel_balanced = audio_to_mel(y_balanced)
    mel_acoustic = audio_to_mel(y_acoustic)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.audio(y, sample_rate=sr)
        st.plotly_chart(plot_mel(mel_orig, "Original Mel"), use_container_width=True)

    with col2:
        st.subheader("Balanced Augmentation")
        st.audio(y_balanced, sample_rate=sr)
        st.plotly_chart(plot_mel(mel_balanced, "Balanced Augmentation"), use_container_width=True)

    with col3:
        st.subheader("Acoustic Augmentation")
        st.audio(y_acoustic, sample_rate=sr)
        st.plotly_chart(plot_mel(mel_acoustic, f"Acoustic ({category})"), use_container_width=True)


if __name__ == "__main__":
    main()
