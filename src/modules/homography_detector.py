import copy as cp
from typing import Tuple
from pathlib import Path

import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN

from src.modules.pattern_class import DetectingPattern


class HomographyDetector(DetectingPattern):

    def __init__(self, standard_paths: list) -> None:
        super().__init__(standard_paths)
        self.standard_signs = {}
        for standard_path in self.standard_paths:
            standard_path_name = Path(standard_path).name.split('.')[0]
            self.standard_signs[standard_path_name] = cv.imread(str(standard_path))

    def add_kps(self, query_img: np.ndarray,
                train_img: np.ndarray) -> Tuple[list, np.ndarray, list, np.ndarray]:
        sift = cv.SIFT_create()

        query_kps, query_des = sift.detectAndCompute(query_img, None)
        train_kps, train_des = sift.detectAndCompute(train_img, None)

        return query_kps, query_des, train_kps, train_des

    def match_kps(self, query_des: np.ndarray, train_des: np.ndarray,
                  train_img: np.ndarray, query_kps: list, train_kps: list) -> np.ndarray:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(query_des, train_des, k=2)
        matchesMask = [[0, 0] for _ in range(len(matches))]

        good_matches = []
        for i, (m, n) in enumerate(matches):
            # For Duckietown's test was: if m.distance < .8 * n.distance:
            if m.distance < .65 * n.distance:
                matchesMask[i] = [1, 0]
                good_matches.append(m)
        good_matches = np.asarray(good_matches)

        # draw_params = dict(matchColor=(0, 255, 0),
        #                    matchesMask=matchesMask,
        #                    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # matched_img = cv.drawMatchesKnn(self.query_img, query_kps, train_img, train_kps, matches, None, **draw_params)
        # cv.imshow('', matched_img)
        # cv.waitKey(0)
        return good_matches

    def cluster_pts(self, good_matches: np.ndarray,
                    query_kps: list, train_kps: list) -> Tuple[dict, dict]:
        ptQuery_ptTrain = {}
        for DMatch in good_matches:
            pt_q = query_kps[DMatch.queryIdx].pt
            pt_t = train_kps[DMatch.trainIdx].pt
            ptQuery_ptTrain[pt_q] = pt_t

        clusterized = DBSCAN(eps=50, min_samples=3).fit_predict(list(ptQuery_ptTrain.keys()))

        cluster_pts_q = {}
        for gp, pt in zip(clusterized, list(ptQuery_ptTrain.keys())):
            if gp == -1:
                continue
            else:
                if gp not in cluster_pts_q:
                    cluster_pts_q[gp] = [pt]
                else:
                    cluster_pts_q[gp].append(pt)

        cluster_pts_t = cp.deepcopy(cluster_pts_q)
        for cluster in cluster_pts_t:
            for i, pt in enumerate(cluster_pts_t[cluster]):
                cluster_pts_t[cluster][i] = ptQuery_ptTrain[pt]

        return cluster_pts_q, cluster_pts_t

    def homography_clusters(self, cluster_pts_q: dict, cluster_pts_t: dict,
                            query_img: np.ndarray, train_img: np.ndarray, sign_name: str) -> np.ndarray:
        res_img = query_img
        
        for cluster in cluster_pts_q:
            src = np.float32(cluster_pts_t[cluster]).reshape(-1, 1, 2)
            dst = np.float32(cluster_pts_q[cluster]).reshape(-1, 1, 2)

            M, _ = cv.findHomography(src, dst, cv.RANSAC, 5.)
            h, w, d = train_img.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            try:
                dst = cv.perspectiveTransform(pts, M)
                dst = [np.int32(dst)]
            except Exception:
                print('NO PERSPECTIVE TRANSFORM!')
                continue

            res_img = cv.polylines(query_img, dst, True, 255, 3, cv.LINE_AA)
            res_img = cv.putText(res_img, sign_name, tuple(dst[0][0][0]), cv.FONT_HERSHEY_DUPLEX,
                                 1, (255, 0, 255), 1, cv.LINE_AA)

        return res_img

    def detect_on_image(self, query_img: np.ndarray) -> np.ndarray:
        for sign_name in self.standard_signs:
            train_img = self.standard_signs[sign_name]

            query_kps, query_des, train_kps, train_des = self.add_kps(query_img, train_img)
            good_matches = self.match_kps(query_des, train_des, train_img, query_kps, train_kps)

            if len(good_matches):
                cluster_pts_q, cluster_pts_t = self.cluster_pts(good_matches, query_kps, train_kps)
                query_img = self.homography_clusters(cluster_pts_q, cluster_pts_t, query_img, train_img, sign_name)

        return query_img

    def detect_on_video(self, input_video_path: str, output_video_path: str) -> None:
        cap = cv.VideoCapture(input_video_path)
        w = 2 * int(cap.get(3))
        h = 2 * int(cap.get(4))
        fps = cap.get(5)
        out = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        i = 1
        while cap.isOpened():
            ret, query_img = cap.read()
            self.query_img = cv.resize(query_img, None, fx=2., fy=2.)
            if ret:
                print(f'NEW FRAME: #{i}')
                out.write(self.detect_on_image(self.query_img))
                print(f'END OF FRAME: #{i}\n\n')
                i += 1
            else:
                break


        cap.release()
        out.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    query_img = cv.imread('/Users/savinovddgmail.com/PycharmProjects/road-signs-detector/query_images/query_4.jpg')
    train_img = cv.imread('/Users/savinovddgmail.com/PycharmProjects/road-signs-detector/train_images/30-speed-limit.jpg')
    standards = ['/Users/savinovddgmail.com/PycharmProjects/road-signs-detector/train_images/30-speed-limit.jpg']

    Detector = HomographyDetector(standards)
    query_kps, query_des, train_kps, train_des = Detector.add_kps(query_img, train_img)
    good_matches = Detector.match_kps(query_des, train_des, train_img, query_kps, train_kps)
    print(good_matches, '\n\n\n\n\n')
    cluster_pts_q, cluster_pts_t = Detector.cluster_pts(good_matches, query_kps, train_kps)
    print(cluster_pts_q, '\n\n\n')
    print(cluster_pts_t)
