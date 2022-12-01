import numpy as np
import cv2
from numpy.random import randint, seed

from utils.plots import plot_one_box

class BboxVisualizer(object):
    """Constantes"""
    MIN_PRED_SIZE = 5
    MAX_PRED_SIZE = 6
    seed(420)
    def __init__(self) -> None:
         self.colors = []

    def go(self,
            frame: np.ndarray,
            predictions: np.ndarray,
            path: str = "",
            show: bool = False
            ) -> None:
        """Affiche les bboxes sur une image.

        Args:
            frame (np.ndarray): L'image
            predictions (np.ndarray): Les prédictions. Les 4 premier éléments sont les coordonnées de la bbox, le 5ème est le score et le 6ème est l'id du rat.
            path (str, optional): Le chemin de l'image de sortie. Defaults to "".
        """
        if not len(predictions):
            return

        for prediction in predictions:
            if len(prediction) < self.MIN_PRED_SIZE:
                raise ValueError(f"La prédiction doit avoir au moins {self.MIN_PRED_SIZE} éléments.")
            elif len(prediction) < self.MAX_PRED_SIZE:
                *xyxy, cls = prediction
                label = f"Rat {int(cls)}"
            elif len(prediction) == self.MAX_PRED_SIZE:
                *xyxy, conf, cls = prediction
                label = f"Rat {int(cls)}: {conf:.2f}"
                
            while int(cls) + 1 > len(self.colors):
                self.colors.append([randint(0, 255) for _ in range(3)])

            plot_one_box(xyxy, frame, label=label, color=self.colors[int(cls)], line_thickness=1)

        if path:
            cv2.imwrite(path, frame) # On save l'image
        if show:   
            cv2.imshow("prédictions", frame) # Ou on l'affiche
            cv2.waitKey(5000)
        