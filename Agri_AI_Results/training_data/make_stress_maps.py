import os, numpy as np

DATA_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "Agri_AI_Results", "training_data")
IMG_DIR = os.path.join(DATA_DIR, "images")
TARGET_DIR = os.path.join(DATA_DIR, "stress_maps")
os.makedirs(TARGET_DIR, exist_ok=True)

for fname in os.listdir(IMG_DIR):
    if not fname.endswith(".npy"): continue
    img = np.load(os.path.join(IMG_DIR, fname)).astype(np.float32)  # (6,H,W)

    B2, B3, B4, B8, B11, B12 = img
    NDVI = (B8 - B4) / (B8 + B4 + 1e-8)
    NDMI = (B8 - B11) / (B8 + B11 + 1e-8)
    NDRE = (B8 - B3) / (B8 + B3 + 1e-8)

    stress_stack = np.stack([NDVI, NDMI, NDRE], axis=0)  # (3,H,W)
    np.save(os.path.join(TARGET_DIR, fname), stress_stack)

print("Stress maps saved to:", TARGET_DIR)
