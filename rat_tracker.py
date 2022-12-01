import torch
import cv2
import numpy as np
from tqdm import tqdm

from sort import Sort
from detector import FrameDetector
from utils.torch_utils import time_synchronized
from bbox_visualizer import BboxVisualizer


class FrameTracker(object):
    """ Utilise une détection d'objet de format YOLO et l'algorithme sort pour tracker les rats.
    """
    def __init__(self,
                 detector: FrameDetector
                 ) -> None:
        self.detector = detector
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.visualizer = BboxVisualizer()

    def track(self, 
              frame: np.ndarray,
              verbose: bool = False,
              output_path: str = ""
              ) -> np.ndarray:
        """Recoit un frame et retourne les bboxes des rats en plus de leurs id.

        Args:
            frame (np.ndarray): L'image
            verbose (bool, optional): Affiche la prédiction. Defaults to False.
            output_path (str, optional): Le chemin de l'image de sortie. Defaults to "".

        Returns:
            np.ndarray: Un array 2d dont chaque row est une bbox avec l'id du rat.
        """
        # Détection des rats
        t1 = time_synchronized()
        detections = self.detector.detect(frame)
        
        # Tracking des rats
        t2 = time_synchronized()
        tracked_objects = self.tracker.update(detections)
        
        t3 = time_synchronized()
        if verbose or output_path:
            if verbose:
                print(f"Temps total: {t3-t1:.3f}s\nTemps de détection: {t2-t1:.3f}s\nTemps de tracking: {t3-t2:.3f}s")
            self.visualizer.go(frame, tracked_objects, path=output_path, show=verbose)

        return tracked_objects

if __name__ == '__main__':
    WEIGHTS = r"G:\My Drive\rat_detection\yolov7_weights\best.pt"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = 640

    single_frame_detector = FrameDetector(DEVICE, WEIGHTS, IMAGE_SIZE)
    single_frame_tracker = FrameTracker(single_frame_detector)

    save_path = "D:/cours/Session Automne 2022/Vision/Projet/problematic_frames/"
    frames = [cv2.imread(f"G:/My Drive/rat_detection/test_frames/{i}.jpg") for i in range(445, 455)]
    for i, frame in tqdm(enumerate(frames)):
        predictions = single_frame_tracker.track(frame, verbose=False, output_path=f"{save_path}tracking{i}.jpg")