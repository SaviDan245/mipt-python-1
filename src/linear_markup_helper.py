import json
import os
import re
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
from termcolor import cprint

from src.modules.custom_detector import CustomDetector


class LinearMarkup:

    def __init__(self, signs_path: str, output_folder_path: str, output_img_format: str) -> None:
        """
        signs_path (string):
            Absolute path to the output folder (signs).
        output_folder_path (string, optional):
            Absolute path to the output folder ('linear_markup_results' by default).
        output_img_format (string, optional):
            Format in which images will be saved ('png' format by default).
        """
        IMAGE_FORMATS = ('jpeg', 'jpg', 'png', 'PNG')
        IMAGE_REGEX = re.compile(r'(\.jpeg)|(\.jpg)|(\.png)|(\.PNG)$')

        self.output_folder_path = Path(output_folder_path)
        self.output_images_folder_path = self.output_folder_path/'images'
        if not self.output_images_folder_path.exists():
            os.makedirs(str(self.output_images_folder_path))

        self.output_img_format = output_img_format
        if self.output_img_format not in IMAGE_FORMATS:
            raise NameError('Invalid output images extension')

        self.markup_dict = {}
        self.standards = [
            Path(filepath).absolute() for filepath in Path(signs_path).iterdir()
            if IMAGE_REGEX.search(str(filepath))
        ]
        if not self.standards:
            raise FileExistsError('No standard images were found')

        self.detector = CustomDetector(self.standards)

    def img_markup(self, query_img: np.ndarray, i: int) -> None:
        """
        Append useful images and its markup to dataset, and print the log.
        Args:
            query_img (numpy.ndarray): Read image.
            i (integer): Index of current iteration loop.
        """

        image_to_save = query_img.copy()

        res_img, markup = None, None
        cur_time = '-'.join(re.split(r'[ :.]', str(datetime.now())))
        img_file_name = f'img-{i + 1}__{cur_time}.{self.output_img_format}'
        try:
            res_img, markup = self.detector.detect_image(query_img)
        except Exception:
            if not res_img:
                raise NameError('No result image was returned')
            raise ProcessLookupError('Detector cannot detect the image')
        else:
            if markup['regions']:
                cv.imshow(img_file_name, res_img)
                key = cv.waitKey(0)
                if key == 32:  # spacebar key -- append to the dataset
                    cv.imwrite(
                        str(self.output_images_folder_path/img_file_name), image_to_save
                    )
                    with open(self.output_folder_path/'markup.json', 'w') as out_json:
                        markup['filename'] = img_file_name
                        self.markup_dict.update(
                            {img_file_name: markup}
                        )
                        json.dump(self.markup_dict, out_json, indent=4)
                    cprint(f'Image {img_file_name} (#{i + 1}) and markup were successfully writen', 'green')  # "2" key
                else:
                    cprint(f'Image {img_file_name} (#{i + 1}) was skipped', 'yellow')  # "1" key
                cv.destroyAllWindows()
            else:
                cprint(f'In the image {img_file_name} (#{i + 1}) were no signs detected', 'red')  # no signs
        finally:
            print(f'End processing #{i + 1}: {img_file_name}')
