import librosa
import random
import numpy as np
def time_stretch(y,rate = None):
  if rate is None:
    rate = random.uniform(0.75,1.35)
  return librosa.effects.time_stretch(y,rate=rate)

def time_shift(y,scale = None):
  if scale is None:
    scale = random.uniform(0.1,0.4) # Up to 40% shift
  shift = np.random.randint(-int(scale * len(y)), int(scale * len(y))) # forward or backward shift
  y_shifted = np.roll(y, shift)
  if shift > 0:
    y_shifted[:shift] = 0
  elif shift < 0:
    y_shifted[shift:] = 0
  return y_shifted

def shift_pitch(y,sr,n_steps=None):
  """
  Takes audio and changes the frequency of the waveform
  """
  if n_steps is None:
    n_steps = random.randint(-1,1)
  return librosa.effects.pitch_shift(y=y,sr=sr,n_steps=n_steps)

def change_volume(y,factor = None):
  if factor is None:
    factor = random.uniform(0.5,1.5) # Add/remove 50% of the volume
  return y * factor