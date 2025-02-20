#!/usr/bin/env python3

# Author(s): T. Flutre - M. Ecarnot
# to be shared only with members of the PerfoMix project

# References:
# http://www.plantphysiol.org/lookup/doi/10.1104/pp.112.205120
# https://codegolf.stackexchange.com/questions/40831/counting-grains-of-rice

import sys
# sys.path.insert(0, "/home/ecarnot/Documents/INRA/Projets/perfomix/perfomixspectro/src/")
sys.path.append(r"D:\Mes Donnees\perfomixspectro\src")

#sys.path.append("D:\PycharmProjects\Tasks")

# dependencies
import os
# os.chdir("C:/Users/seedmeister/PycharmProjects/perfomix")
# print(sys.path)
# print("Current working directory: {0}".format(os.getcwd()))
import numpy as np
from numpy import matlib as mb
import scipy as sci
import matplotlib.pyplot as plt
import cv2 as cv
import spectral as sp
import spectral.io.envi as envi
from skimage.measure import label, regionprops, regionprops_table
import gzip
import time
from skimage import measure, util
import pandas
import gzip

from gala import iterprogress
from gala import morpho
#import pysftp


## PATH of hyperspectral images
PATH = "F:/carob/pas_passes/" #"D:/Mes Donnees/IHS_output/images_a_refaire/"
    #'//svgap10/ble/EPO_Montans_Semis_Janv2024/'
PATH_OUT = 'D:/Mes Donnees/IHS_output/'

# cnopts = pysftp.CnOpts(knownhosts=os.path.expanduser(os.path.join('~', '.ssh', 'fake_known_hosts')))
# cnopts.hostkeys = None
# sftp = pysftp.Connection('147.99.78.235', username='data-ecarnot', password='t10vvWwjw!ae', cnopts=cnopts)
# sftp.cwd('/data-ge2pop/data-perfomix/Q3-serie2/CHS/')
# filelist = sftp.listdir()
#
# # Read file of image to read
# with open(PATH+'perfomix_apasser_R2020.csv','r') as f:
#     lines = f.readlines()
# f.close()
# for x in range(0,lines.__len__()):
#     lines[x]=lines[x].replace("\n", "")

filelist = os.listdir(PATH)
#indices = [i for i, filelist in enumerate(filelist) if "BE4" in filelist]

for filename in filelist[0:filelist.__len__()]: #os.listdir(PATH)[27:187]:
    print(filename)
    # res = [ele for ele in lines if (ele in filename)]
    if filename.endswith('.hyspex'): #filename.endswith('.hyspex') : #res.__len__()>0 and filename.endswith('.hyspex'):  # When targetting pure var in Q3-PR1
        print(filename)
        sImg = filename[0:filename.find('.')]  # 'x30y21-var1_11000_us_2x_2020-12-02T101757_corr'  #
        # sftp.get(sImg + '.hyspex', PATH + sImg + '.hyspex')
        # sftp.get(sImg + '.hdr', PATH + sImg + '.hdr')

        t = time.time()

        # # input parameters
        id_spectralon = 300
        id_pap_milli = (700,900)
        id_debut_grain = 1050
        thresh_lum = 0.11# 0.07 for wheat #1800 # threshold of reflectance (or light intensity) to remove background
        areaRange = (10000,40000)  # range of grain area in number of pixels wheat:(1000,20000); carob ((10000,40000)
        band = 100  # spectral band to extract (#100 : 681 nm)
        rgb = (82,54,14)

        img = envi.open(PATH + sImg + '.hdr', PATH + sImg + '.hyspex')

        img = np.array(img.load(),dtype=np.int16)
        img = np.transpose(img, (1, 0, 2))
        imrgb = img[:,:,rgb].astype('float')
        imseuil_grain = img[:, :, band].astype('float')

# Detect and extract spectralon
        im0 = img[:, 1:id_spectralon, :]
        ret0, binaryImage0 = cv.threshold(im0[:,:,band], thresh_lum*2, im0.max(), cv.THRESH_BINARY)
        # plt.figure()
        # plt.imshow(binaryImage0)
        # Save spectralon mean reflectance
        ref = np.zeros((img.shape[0],img.shape[2]),img.dtype)
        for x in range(0,img.shape[0]):
             nz=binaryImage0[x,:] != 0
             if sum(nz) > 50:
                ref[x,:] = np.mean(im0[x,nz,:],0)
                imrgb[x, :, :] = imrgb[x, :,:] / np.tile(ref[x,rgb], (img.shape[1], 1)).astype('float')
                imseuil_grain[x, :] = imseuil_grain[x, :] / np.tile(ref[x,band], (img.shape[1])).astype('float')

        f = gzip.GzipFile(PATH_OUT + sImg + "_ref.gz", "wb")
        np.save(f, ref)
        f.close()
        imrgb=np.clip(imrgb, a_min=None, a_max=1)
        plt.figure()
        plt.imsave(PATH_OUT + sImg + "_RGB.jpg", imrgb)
        del imrgb,im0

        # Save papier milli image
        im_pm= im1 = img[:, id_pap_milli[0]:id_pap_milli[1], 50]
        plt.figure()
        plt.imsave(PATH_OUT + sImg + "_papier_milli.jpg", im_pm, cmap='gray')

        # Grain detection and split close grains
        im1 = imseuil_grain[:, id_debut_grain:img.shape[1]] #img[:, cropIdxDim1:img.shape[1], band]
        ret, binaryImage = cv.threshold(im1, thresh_lum, 1, cv.THRESH_BINARY)
        # plt.figure()
        # plt.imshow(binaryImage)
        D = -cv.distanceTransform(binaryImage.astype(np.uint8), cv.DIST_L2, 3)

        # insipred from https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/
        D2 = morpho.hminima(D, 3)
        L = morpho.watershed(D2, dams=True)
        bw3 = binaryImage;
        bw3[L == 0] = 0;
        # plt.figure()
        # plt.imshow(bw3)

        labeled_array = label(bw3, connectivity=1)
        regions = regionprops(labeled_array)

        # Filtrer les régions en fonction de l'aire
        filtered = [region.label for region in regions if areaRange[0] <= region.area and region.area <= areaRange[1] and region.solidity > 0.8 and region.eccentricity < 0.98] #  & (props_df['major_axis_length'] < 300

        # Créer un masque binaire pour les régions à conserver
        mask = np.isin(labeled_array, filtered)

        # Filtrer labeled_array en utilisant le masque
        filtered_labeled_array = labeled_array * mask

        # Afficher l'image filtrée avec des étiquettes pour chaque région
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(filtered_labeled_array, cmap='nipy_spectral')

        for index, region in enumerate(regionprops(filtered_labeled_array)):
            y, x = region.centroid
            ax.text(x, y, str(index+1), color='white', fontsize=8, ha='center', va='center')

        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
        fig.savefig(PATH_OUT + sImg + "_labels.jpg", dpi=300)


        # Save spectra and morph into file
        attrok = ('area','bbox_area','convex_area','eccentricity','equivalent_diameter','euler_number','extent','feret_diameter_max','filled_area','label','major_axis_length','minor_axis_length','orientation','perimeter','perimeter_crofton','solidity')
        sp = np.empty((0,img.shape[2]+3)).astype(np.int16)  # np.empty((len(o),img.shape[2]))
        morph = np.empty((len(filtered),len(attrok)))
        imr = img[:, id_debut_grain:img.shape[1], :]
        dep = np.reshape(imr,(imr.shape[0]*imr.shape[1],imr.shape[2])) # unfolded image

        for i in range(0,len(filtered)) :
            id = np.ravel_multi_index(np.transpose(regions[filtered[i]-1].coords), (imr.shape[0],imr.shape[1]))  # coord of grains pixels in unfolded image
            sp1 = np.array([dep[j,:] for j in id]).astype(np.int16)
            spcoord=np.concatenate((mb.repmat(i+1,len(id),1),regions[filtered[i]-1].coords,sp1),axis=1).astype(np.int16)
            sp = np.concatenate((sp, spcoord))

            for j in range(0,len(attrok)) :
                morph[i, j] = getattr(regions[filtered[i]-1], attrok[j])

        # Save Spectra file
        # f = gzip.GzipFile(PATH_OUT + sImg + "_sp.gz", "wb")
        # np.save(f, sp)
        # f.close()

        with gzip.GzipFile(PATH_OUT + sImg + "_sp.gz", "wb") as f:
            np.save(f, sp)


        # df = DataFrame(sp)
        # spr = pandas2ri.py2rpy(df)
        # r.assign("foo", spr)
        # r("save(foo, file='here.gzip', compress=TRUE)")



        morph = morph.astype(np.float32)
        fmorph = gzip.GzipFile(PATH_OUT + sImg + "_morph.gz", "wb")
        np.save(fmorph, morph)
        fmorph.close()



        elapsed = time.time() - t



# Commands used to determine existing attributes from regionprops
# attr=dir(o[0])
# for x in range(39,len(attr)) :
#     if hasattr(o[0], attr[x]):
#         print(x)
#         print(attr[x])
#         print(type(getattr(o[0], attr[x])))

# Image NEO HYPSEX 2048
# Papier milimetré:
#     en y: y75 à y1900, = 1825 pixels pour 80 mm (8 carreaux). Soit 0.044 mm/pixel
#     en x: x512 à x833, = 318 pixels pour 15 mm . Soit 0.047 mm/pixel
#     moyenne= 0.0455 mm/pixel
#       Surface d'un pixel = 0.00207025 mm2
#
#Avec binning 2x (1024 pixels/ligne):
#     en y: 0.088 mm/pixel
#     en x: 0.094 mm/pixel
#     moyenne= 0.091 mm/pixel
#       Surface d'un pixel = 0.008281 mm2
