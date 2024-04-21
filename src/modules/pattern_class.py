from abc import abstractmethod
from typing import Tuple

import numpy as np


class DetectingPattern:
    def __init__(self, standard_paths: list) -> None:
        self.standard_paths = standard_paths
    
    @abstractmethod
    def add_kps(self, query_img: np.ndarray,
                train_img: np.ndarray) -> Tuple[list, np.ndarray, list, np.ndarray]:
        pass
    
    @abstractmethod
    def match_kps(self, query_des: np.ndarray, train_des: np.ndarray, train_img, query_kps, train_kps) -> np.ndarray:
        pass
    
    @abstractmethod
    def cluster_pts(self, good_matches: np.ndarray, query_kps: list,
                    train_kps: list) -> Tuple[dict, dict]:
        pass
    
    @abstractmethod
    def homography_clusters(self, cluster_pts_q: dict, cluster_pts_t: dict, query_img: np.ndarray,
                            train_img: np.ndarray, sign_name: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def detect_on_image(self, query_img) -> np.ndarray:
        pass
    
    @abstractmethod
    def detect_on_video(self, input_video_path: str, output_video_path: str) -> None:
        pass
