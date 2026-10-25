"""Shared settings for the tree species generalization study.

All values follow Section 5 (Methods and Materials) of the article, with the
deviations listed in Section 6.1 (Results): the independent test data are
public iNaturalist observations from Central Asia instead of field photographs
from Astana.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"
OUT_DIR = ROOT / "outputs"

# Candidate species: (scientific name, genus, family, leaf habit)
SPECIES = [
    ("Acer negundo", "Acer", "Sapindaceae", "deciduous"),
    ("Acer platanoides", "Acer", "Sapindaceae", "deciduous"),
    ("Betula pendula", "Betula", "Betulaceae", "deciduous"),
    ("Fraxinus pennsylvanica", "Fraxinus", "Oleaceae", "deciduous"),
    ("Larix sibirica", "Larix", "Pinaceae", "deciduous conifer"),
    ("Picea pungens", "Picea", "Pinaceae", "evergreen"),
    ("Pinus sylvestris", "Pinus", "Pinaceae", "evergreen"),
    ("Populus nigra", "Populus", "Salicaceae", "deciduous"),
    ("Populus tremula", "Populus", "Salicaceae", "deciduous"),
    ("Prunus padus", "Prunus", "Rosaceae", "deciduous"),
    ("Prunus virginiana", "Prunus", "Rosaceae", "deciduous"),
    ("Salix alba", "Salix", "Salicaceae", "deciduous"),
    ("Sorbus aucuparia", "Sorbus", "Rosaceae", "deciduous"),
    ("Ulmus laevis", "Ulmus", "Ulmaceae", "deciduous"),
    ("Ulmus pumila", "Ulmus", "Ulmaceae", "deciduous"),
]
SPECIES_NAMES = [s[0] for s in SPECIES]
GENUS = {s[0]: s[1] for s in SPECIES}
FAMILY = {s[0]: s[2] for s in SPECIES}
HABIT = {s[0]: s[3] for s in SPECIES}

# GBIF dataset key of "iNaturalist Research-grade Observations"
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
SOURCE_CONTINENTS = ["EUROPE", "NORTH_AMERICA"]
CENTRAL_ASIA = ["KZ", "KG", "UZ", "TJ", "TM"]  # ISO country codes
OPEN_LICENSES = ("CC0", "CC_BY", "CC_BY_NC", "creativecommons.org")

MAX_OBS_PER_SPECIES = 2000
MAX_IMAGES_PER_OBS = 3
SHORT_SIDE = 512

# Season windows by observation month (shifted test sets)
LEAF_ON_MONTHS = {6, 7, 8, 9}
LEAF_OFF_MONTHS = {12, 1, 2}
MIN_TEST_OBS_PER_SPECIES = 20

# Splitting
SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SAME_TREE_RADIUS_M = 100

# Training
TRAIN_SEEDS = [0, 1, 2]
IMG_SIZE = 224
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 64
MODELS = {
    "M1": "tf_efficientnetv2_s.in21k_ft_in1k",   # EfficientNetV2-S
    "M2": "vit_base_patch16_224.augreg_in21k",   # ViT-B/16, ImageNet-21k
}

# Evaluation
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 2026
ALPHA = 0.05
