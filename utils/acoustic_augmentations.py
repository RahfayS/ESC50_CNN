import numpy as np
import random
import librosa
from scipy.signal import butter, filtfilt
from .helper_augmentations import *

# --- Animal Augmentations ---
def augment_animals(y,sr):
  y_aug = y.copy()

  # Moderate pitch variation
  if random.random() < 0.7:
    y_aug = librosa.effects.pitch_shift(y_aug,sr=sr,n_steps=random.randint(-1,1))

  if random.random() < 0.6:
    y_aug = add_outdoor_ambience(y_aug)

  if random.random() < 0.5:
    y_aug = change_volume(y_aug,factor=random.uniform(0.6,1.4)) # 40% louder or quieter

  if random.random() < 0.3:
    y_aug = time_stretch(y_aug, rate=random.uniform(0.9,1.1)) # Slowed down or speed up at most 10%

  return y_aug


def add_outdoor_ambience(y):
  """
    Simulates outdoor environment
  """
  # Gentle wind + distant bird chirps
  ambience = np.random.randn(len(y)) * 0.002
  # Low-pass filter for wind-like sound
  b, a = butter(2, 0.05, btype='low')
  ambience = filtfilt(b, a, ambience)
  return y + ambience * random.uniform(0.05, 0.15)


def augment_natural_soundscapes(y,sr):
  y_aug = y.copy()

  if random.random() < 0.7:
    y_aug = time_stretch(y_aug,rate=random.uniform(0.75,1.3))

  if random.random() < 0.7:
    y_aug = change_volume(y_aug, factor=random.uniform(0.5, 1.5))

  # Add environmental depth (reverb simulation)
  if random.random() < 0.5:
    y_aug = add_reverb_simple(y_aug)

  # Light time shift (continuous sounds tolerate this well)
  if random.random() < 0.4:
    y_aug = time_shift(y_aug, scale=random.uniform(0.1, 0.2))

  return y_aug


def add_reverb_simple(y, room_size='medium'):
    """Simple reverb simulation using convolution"""
    room_sizes = {
        'small': 0.3,
        'medium': 0.5,
        'large': 0.8
    }

    decay = room_sizes.get(room_size, 0.5)
    delay_samples = int(0.05 * 22050)  # 50ms delay

    # Create simple impulse response
    impulse = np.zeros(delay_samples)
    impulse[0] = 1.0
    impulse[delay_samples//2] = decay * 0.5
    impulse[-1] = decay * 0.3

    # Convolve
    y_reverb = np.convolve(y, impulse, mode='same')

    # Mix with dry signal
    wet_mix = random.uniform(0.15, 0.4)
    return y * (1 - wet_mix) + y_reverb * wet_mix

def augment_human_non_speech(y, sr):
    """
    Human sounds: Pitch variation (adults vs children), room acoustics
    - Baby cries are high-pitched, adult coughs are lower
    - Indoor reverb is common
    - Age/gender differences = pitch changes
    """
    y_aug = y.copy()

    # Wide pitch variation (children vs adults, male vs female)
    if random.random() < 0.7:
        n_steps = random.randint(-4, 4)
        y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=n_steps)

    # Room acoustics / reverb
    if random.random() < 0.6:
        y_aug = add_reverb_simple(y_aug)

    # Volume (distance from microphone)
    if random.random() < 0.6:
        y_aug = change_volume(y_aug, factor=random.uniform(0.6, 1.4))

    # Moderate time stretch (people do things at different speeds)
    if random.random() < 0.5:
      y_aug = time_stretch(y_aug,rate=random.uniform(0.85,1.2))

    # Indoor background noise
    if random.random() < 0.4:
        y_aug = add_indoor_ambience(y_aug)

    return y_aug


def add_indoor_ambience(y):
    """Simulate indoor environment"""
    # AC hum (60 Hz) + room tone
    t = np.linspace(0, len(y)/22050, len(y))
    hum = np.sin(2 * np.pi * 60 * t) * 0.001
    room_tone = np.random.randn(len(y)) * 0.001
    ambience = hum + room_tone
    return y + ambience * random.uniform(0.03, 0.1)

def augment_interior_domestic(y, sr):
  """
  Interior sounds: Room acoustics, muffled versions, background hum
    - Mechanical sounds from different rooms (muffled)
    - Indoor reverb patterns
    - Electrical hum from appliances
  """
  y_aug = y.copy()

  # Strong room acoustics
  if random.random() < 0.7:
    y_aug = add_reverb_simple(y_aug)

  # Volume variation (near/far in room, through walls)
  if random.random() < 0.7:
    y_aug = change_volume(y_aug, factor=random.uniform(0.4, 1.3))

  # Indoor background (AC, fridge hum, etc.)
  if random.random() < 0.6:
    y_aug = add_indoor_ambience(y_aug)

  # Light time shift (mechanical events start at different times)
  if random.random() < 0.5:
    y_aug = time_shift(y_aug, scale=random.uniform(0.1, 0.25))

  # Minimal pitch shift (mechanical sounds have fixed frequencies)
  if random.random() < 0.3:
    n_steps = random.choice([-1, 1])  # Very subtle
    y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=n_steps)

  return y_aug

def augment_urban(y, sr):
    """
    Urban/mechanical: Distance effects, urban background, minimal pitch
    - Cars/sirens at different distances
    - Urban noise pollution
    - Doppler effect for moving vehicles
    - Mechanical sounds have fixed pitch (engines don't change tone much)
    """
    y_aug = y.copy()

    # Strong volume variation (distance from source)
    if random.random() < 0.8:
        y_aug = change_volume(y_aug, factor=random.uniform(0.3, 1.5))

    # Urban background noise
    if random.random() < 0.7:
        y_aug = add_urban_ambience(y_aug)

    # Distance-based filtering (far away = muffled)
    if random.random() < 0.6:
        y_aug = add_distance_filter(y_aug, sr)

    # Time shift (vehicles pass at different times)
    if random.random() < 0.5:
        y_aug = time_shift(y_aug, scale=random.uniform(0.15, 0.3))

    # Minimal pitch shift (engines have fixed RPM)
    if random.random() < 0.3:
        n_steps = random.choice([-1, 0, 1])
        y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=n_steps)

    return y_aug


def add_urban_ambience(y):
    """Simulate urban environment"""
    # Traffic rumble + general noise
    ambience = np.random.randn(len(y)) * 0.005
    # Emphasize low frequencies (traffic)
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 0.1, btype='low')
    ambience = filtfilt(b, a, ambience)
    return y + ambience * random.uniform(0.1, 0.25)

def add_distance_filter(y, sr):
    """Simulate distance by removing high frequencies"""

    # Far away = more high-frequency loss
    cutoff_freq = random.uniform(2000, 6000)  # Hz
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    b, a = butter(4, normalized_cutoff, btype='low')
    y_filtered = filtfilt(b, a, y)

    # Mix based on "distance"
    distance_factor = random.uniform(0.3, 0.8)
    return y * (1 - distance_factor) + y_filtered * distance_factor

def Add_Augmentations_Acoustic(y, sr, category):
    """
    Apply context-aware augmentation based on acoustic category
    """
    acoustic_categories = {
        'animals': [
            'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen',
            'insects', 'sheep', 'crow'
        ],
        'natural_soundscapes': [
            'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
            'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm'
        ],
        'human_non_speech': [
            'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
            'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping'
        ],
        'interior_domestic': [
            'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks',
            'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm',
            'clock_tick', 'glass_breaking'
        ],
        'urban_mechanical': [
            'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train',
            'church_bells', 'airplane', 'fireworks', 'hand_saw'
        ]
    }
    if category in acoustic_categories['animals']:
        return augment_animals(y, sr)

    elif category in acoustic_categories['natural_soundscapes']:
        return augment_natural_soundscapes(y, sr)

    elif category in acoustic_categories['human_non_speech']:
        return augment_human_non_speech(y, sr)

    elif category in acoustic_categories['interior_domestic']:
        return augment_interior_domestic(y, sr)

    elif category in acoustic_categories['urban_mechanical']:
        return augment_urban(y, sr)