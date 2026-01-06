import numpy as np
import random
import librosa
from librosa.feature import melspectrogram
from .helper_augmentations import *

def Augmentations_Balanced(y, sr):
    """
    Balanced augmentation: diverse but recognizable
    - 70% moderate augmentation
    - 30% aggressive augmentation
    """

    # Choose intensity
    if random.random() < 0.7:
        # Moderate: 2-3 augmentations, conservative parameters
        num_augs = random.randint(2, 3)
        intensity = 'moderate'
    else:
        # Aggressive: 3-4 augmentations, stronger parameters
        num_augs = random.randint(3, 4)
        intensity = 'aggressive'

    y_aug = y.copy()

    if intensity == 'moderate':
        # Conservative parameters
        augmentations = [
            ('shift', lambda: time_shift(y_aug)),
            ('stretch', lambda: librosa.effects.time_stretch(y_aug, rate=random.uniform(0.85, 1.15))),
            ('pitch', lambda: librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=random.randint(-2, 2))),
            ('volume', lambda: change_volume(y_aug))
        ]
    else:  # aggressive
        # Stronger parameters
        augmentations = [
            ('shift', lambda: time_shift(y_aug)),
            ('stretch', lambda: librosa.effects.time_stretch(y_aug, rate=random.uniform(0.75, 1.35))),
            ('pitch', lambda: librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=random.randint(-4, 4))),
            ('volume', lambda: change_volume(y_aug))
        ]

    # Apply random selection
    selected = random.sample(augmentations, k=num_augs)
    for name, aug_func in selected:
        y_aug = aug_func()

    return y_aug