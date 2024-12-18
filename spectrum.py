"""
File: main.py
Authors: Clement Plessis - Martin Ecarnot
Date: 11 Oct. 2024
Description: This script is part of the IHS SPIKE project 
and is designed to segment and get the morphology of
wheat kernels after the
hyperspectral data acquistione.

"""

import os
import cv2 as cv
import spectral as sp
import spectral.io.envi as envi
import numpy as np
import pandas as pd
import matplotlib, specdal
import matplotlib.pyplot as plt

class SpectrumCamera():
    def __init__(
        self,
        hdr_file_path: str,
        id_spectralon: int = 300,
        id_im_grains: tuple = (710,3460),
        thresh_lum: float = 0.11,  # 0.07 for wheat #1800 # threshold of reflectance (or light intensity) to remove background
        band: int = 100,  # spectral bands to extract (#100 : 681 nm)
        bands: tuple = (82, 150, 200),  # spectral bands for segmentation
        rgb: tuple = (82, 54, 14)
        ):
        
        #-------------------
        # File Names
        self.hdr_file_path = hdr_file_path
        self.hyspex_fle_path = hdr_file_path.replace(".hdr", ".hyspex")
        
        #-------------------
        # Init Param
        self.id_spectralon = id_spectralon
        self.id_im_grains = id_im_grains
        self.thresh_lum = thresh_lum
        self.band = band
        self.bands = bands
        self.rgb = rgb

        #-------------------
        # Init Process
        img = envi.open(self.hdr_file_path, self.hyspex_fle_path)

        img = np.array(img.load(),dtype=np.int16)
        img = np.transpose(img, (1, 0, 2))
        imrgb = img[:,:,rgb].astype('float')

        # Detect and extract spectralon
        im0 = img[:, 1:id_spectralon, :]
        ret0, binaryImage0 = cv.threshold(im0[:,:,band], thresh_lum*2, im0.max(), cv.THRESH_BINARY)

        # update rgb and get reference
        ref = np.zeros((img.shape[0],img.shape[2]),img.dtype)

        for x in range(0,img.shape[0]):
             nz=binaryImage0[x,:] != 0
             if sum(nz) > 50:
                ref[x,:] = np.mean(im0[x,nz,:],0)
                imrgb[x, :, :] = imrgb[x, :,:] / np.tile(ref[x,rgb], (img.shape[1], 1)).astype('float')

        imrgb=np.clip(imrgb, a_min=None, a_max=1)
        
        self.reference = ref
        self.image_rgb = imrgb
        self.img = img

    def save_rgb(self, output_path: str, sample: str, date: str, hour: str) -> None:
        plt.imsave(
            f"{output_path}/{date}_{hour}_{sample}_rgb.jpg",
            self.image_rgb
        )
        plt.close()

class SpectrumASD():
    def __init__(
        self,
        asd_file_path: str
    ):
        s = specdal.Spectrum(filepath=asd_file_path)
        df = pd.DataFrame(s.measurement)
        df.index.names = ['index']
        df.index = [int(i) for i in df.index]
        df.rename({'pct_reflect': 'value'}, axis='columns', inplace=True)
        self.spectrum = df

    def save_spectrum(self, output_path: str, sample: str, date: str, hour: str) -> None:
        file = f"{output_path}/asd_all_samples.csv"
        info = pd.DataFrame(
            dict(value=[date, hour, sample]),
            index=["date", "hour", "sample"]
        )
        df = pd.concat([info,self.spectrum])
        df = df.transpose()
        if os.path.exists(file):
            df.to_csv(f"{output_path}/asd_all_samples.csv", mode='a', index=False, header=False)
        else:
            df.to_csv(f"{output_path}/asd_all_samples.csv", index=False, header=True)

    

