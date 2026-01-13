import librosa
import numpy as np

# --- Config ---
SR = 22050
N_MELS = 128

def audio_to_mel(y, sr=SR):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db