import cv2
import torch
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import time_synchronized
from bbox_visualizer import BboxVisualizer

class FrameDetector():
    """ Utilise un modèle YOLO pour détecter les rats dans une image."""
    def __init__(self,
                 device: str,
                 weights: str,
                 img_size: int = 640
                 ) -> None:
        self.device = device
        self.half = self.device.type != 'cpu'  # On utilise la moitié des poids si on utilise le GPU (plus rapide)

        # Charger le modèle
        self.model = attempt_load(weights, map_location=device)  # modèle FP32
        self.stride = int(self.model.stride.max())  # Le stride du modèle
        self.img_size = check_img_size(img_size, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # vers FP16 (plus vite)
            
        # Setter le visualizer
        self.visualizer = BboxVisualizer()
            
    def format_frame(self) -> np.ndarray:
        # Padded resize (pour fitter avec l'architecture du modèle)
        img = letterbox(self.frame, self.img_size, stride=self.stride)[0]

        # Formatter en tensor
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3ximg_sizeximg_size
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 vers fp16/32 (si half ou non)
        img /= 255.0  # normaliser le RGB entre 0 et 1
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  
        
        return img   

    def detect(self,
                frame: np.ndarray,
                conf_thres: int = 0.6,
                iou_thres: int = 0.5,
                verbose=False,
                output_path: str = ""
                ) -> np.ndarray:
        """Détecte les rats dans une image.

        Args:
            frame (np.ndarray): L'image
            conf_thres (int, optional): Le niveau de confiance du NMS. Defaults to 0.6.
            iou_thres (int, optional): Le iou minimum du NMS . Defaults to 0.4.
            verbose (bool, optional): Si on affiche les résultats ou non. Defaults to False.
            output_path (str, optional): Le chemin de l'image de sortie. Defaults to "".

        Returns:
            np.ndarray: Un array 2D avec les bbox, le niveau de confiance et la classe (0) des rats.
        """
        # Formatter le frame
        t1 = time_synchronized()
        self.frame = frame
        self.img = self.format_frame()
        
        # Effectuer l'inférence
        t2 = time_synchronized()
        with torch.no_grad():   # Ne pas claculer le gradient (plus rapide). Le model est en mode eval.
            pred = self.model(self.img, augment=True)[0]
        
        # Appliquer le NMS (classes=None parce qu'on veut l'appliquer à toutes les classes.. on a une classe seulement)
        t3 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=True)[0]
        
        # Réajuster les bbox aux dimensions de l'image d'origine
        t4 = time_synchronized()
        pred[:, :4] = scale_coords(self.img.shape[2:], pred[:, :4], self.frame.shape).round()
        
        t5 = time_synchronized()
        if verbose or output_path:
            if verbose:
                print(f"Frame formatted in {t2 - t1:.3f}s\nInference done in {t3 - t2:.3f}s\nNMS done in {t4 - t3:.3f}s\nRescaling done in {t5 - t4:.3f}s")
            self.visualizer.go(frame, pred, path=output_path, show=verbose)
            
        preds = pred.cpu().numpy()
        if len(preds) != 4:
            stop = True
        return pred.cpu().numpy()


if __name__ == '__main__':
    WEIGHTS = r"G:\My Drive\rat_detection\yolov7_weights\best.pt"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = 640
    frame = cv2.imread(r"G:\My Drive\rat_detection\test_frames\719.jpg")
    single_frame_detector = FrameDetector(DEVICE, WEIGHTS, IMAGE_SIZE)
    predictions = single_frame_detector.detect(frame, verbose=True)
    