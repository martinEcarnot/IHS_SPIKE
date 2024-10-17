"""
File: lernels.py
Authors: Clement Plessis - Martin Ecarnot
Date: 11 Oct. 2024
Description: This script is part of the IHS SPIKE project 
and is designed to segment and get the morphology of
wheat kernels after the
hyperspectral data acquistion.

"""

# ===================================
#       IMPORT
import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, modeling
from skimage.measure import regionprops

# ===================================
#       CLASS
class Kernels():
    def __init__(
        self,
        image_rgb: np.ndarray,
        sam_model: modeling.sam.Sam,
        crop_x_left: int, crop_x_right: int
    ):
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model, box_nms_thresh=0.25
        )
        self.image_rgb = image_rgb[:,crop_x_left:crop_x_right]
        self.masks = mask_generator.generate(self.image_rgb)

    def filter_masks(self, area_min: int=4000, area_max: int=15000) -> None:
        """
        Filter masks by their surface.

        Arguments:
            masks (list): List of masks from 'mask_generator.generate'.

        Returns:
            list : List of masks from 'mask_generator.generate'.
        """
        new_masks = []
        for m in self.masks:
            if m['area'] > area_min and m['area'] < area_max:
                new_masks.append(m)

        new_masks = sorted(new_masks, key=(lambda x: x['bbox']), reverse=True)
        self.masks = new_masks

    def save_masks(self, output_path: str, sample: str, date: str, hour: str) -> None:
        img = self.image_rgb
        if len(self.masks) == 0:
            return
        plt.figure(figsize=(22,16))
        plt.imshow(img)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        img = np.ones((self.masks[0]['segmentation'].shape[0], self.masks[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        n = 0
        for ann in self.masks:
            n+=1
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.45]])
            img[m] = color_mask
            bbox = ann['bbox']
            coord=( (bbox[0]*2+bbox[2]) / 2, (bbox[1]*2 + bbox[3]) / 2 )
            plt.text(x=int(coord[0]), y=int(coord[1]), s=str(n), fontsize=10, color= (1,1,1))
        ax.imshow(img)
        plt.axis('off')
        plt.savefig(f"{output_path}/{date}_{hour}_{sample}_masks.jpg",bbox_inches='tight')
        plt.close()

    def add_regionprops(self):
        props = list()
        for i in self.masks:
            p = regionprops(i["segmentation"]*255)[0]
            props.append(p)
        self.rprops = props

    def save_rpops(self, output_path: str, sample: str, date: str, hour: str) -> None:
        attrok = (
            'area', 'bbox_area', 'convex_area', 'eccentricity',
            'equivalent_diameter', 'euler_number', 'extent',
            'feret_diameter_max', 'filled_area', 'label',
            'major_axis_length', 'minor_axis_length', 'orientation',
            'perimeter', 'perimeter_crofton', 'solidity'
        )
        res_list = list()
        for i, props in enumerate(self.rprops):
            i_dict = dict(
                date = date,
                hour = hour,
                sample = sample,
                kernel = i+1,
            )
            for key in attrok:
                i_dict[key] = props[key]
            res_list.append(i_dict)
        
        df = pd.DataFrame(res_list)
        df.to_csv(f"{output_path}/{date}_{hour}_{sample}_props.csv",index=False)
    
    def save_kernels(self, output_path: str, sample: str, date: str, hour: str,
        size: int=320,resize: bool=True) -> None:
        if not os.path.isdir(f"{output_path}/kernels"):
            os.mkdir(f"{output_path}/kernels")

        for i, m in enumerate(self.masks):
            # Float to int
            img = (self.image_rgb * 255).astype(np.uint8)

            # Mask bool to int
            mask_int = m['segmentation'].astype(np.uint8)

            # Étendre le masque pour qu'il ait 3 canaux, car l'image est en RGB (3 canaux)
            mask_rgb = np.repeat(mask_int[:, :, np.newaxis], 3, axis=2)

            # Appliquer le masque sur l'image RGB pour extraire la zone correspondante
            masked_image = cv2.bitwise_and(img, img, mask=mask_int)

            # Définition de la bounding box (x_min, y_min, largeur, hauteur)
            x_min = m['bbox'][0]   # Coordonnée x du coin supérieur gauche
            y_min = m['bbox'][1]   # Coordonnée y du coin supérieur gauche
            largeur = m['bbox'][2] # Largeur de la boîte
            hauteur = m['bbox'][3] # Hauteur de la boîte

            add_width = int(round((size-largeur)/2,0))
            add_height = int(round((size-hauteur)/2,0))

            y1 = y_min-add_height
            y2 = y_min+hauteur+add_height
            x1=x_min-add_width
            x2=x_min+largeur+add_width

            if y1 < 0:
                y2 = y2+(0-y1)
                y1 = 0

            im_height = masked_image.shape[0]
            if y2 > im_height:
                y1 = y1-(y2-im_height)

            if x1 < 0 :
                x2 = x2+(0-x1)
                x1 = 0

            im_length = masked_image.shape[1]
            if x2 > im_length:
                x1 = x1-(x2-im_length)

            # Resize for model
            if resize:
                res = cv2.resize(masked_image[y1:y2,x1:x2], dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
                plt.imsave(f'{output_path}/kernels/{date}_{hour}_{sample}_k{i+1}.jpg', res)
            else:
                plt.imsave(f'{output_path}/kernels/{date}_{hour}_{sample}_k{i+1}.jpg', masked_image[y1:y2,x1:x2])
            plt.close()


