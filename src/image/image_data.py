import os
from pathlib import Path

import cv2
import numpy as np
import kagglehub


def download_dataset():
    """
    Faz download do dataset do Kaggle e retorna o caminho local.
    """
    path = kagglehub.dataset_download(
        "tawsifurrahman/covid19-radiography-database"
    )
    return Path(path)


def load_images(image_size=(224, 224), max_images_per_class=None):
    """
    Baixa o dataset e carrega as imagens em memória.
    """

    dataset_path = download_dataset()

    # Estrutura do dataset (IMPORTANTE)
    base_path = dataset_path / "COVID-19_Radiography_Dataset"

    class_names = [
        "COVID",
        "Normal",
        "Lung_Opacity",
        "Viral Pneumonia"
    ]

    X = []
    y = []

    for label, class_name in enumerate(class_names):
        class_path = base_path / class_name / "images"

        image_files = list(class_path.glob("*.png"))

        if max_images_per_class is not None:
            image_files = image_files[:max_images_per_class]

        for img_path in image_files:
            img = cv2.imread(str(img_path))

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)

            X.append(img)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, class_names