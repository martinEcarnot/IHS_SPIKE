
from ultralytics import SAM
from ultralytics import FastSAM
import cv2 as cv
import spectral as sp
import spectral.io.envi as envi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from skimage.measure import label, regionprops, regionprops_table

# Load a model SAM
model = SAM("models\sam_b.pt")

# Load a model FastSAM
 #model = FastSAM("models\FastSAM-x.pt")

# Display model information (optional)
# model.info()

PATH = "D:/Mes Donnees/BBSoCoul/imagesHS_brut/"
fHS="2024-09-07_105245_740-6_HyperS.hyspex"
PATH_OUT = "images/out/"
# # input parameters
id_spectralon = 300
id_pap_milli = (700, 900)
id_debut_grain = 1050
id_im_grains = (710,3460)
thresh_lum = 0.11  # 0.07 for wheat #1800 # threshold of reflectance (or light intensity) to remove background
areaRange = (1000, 20000)  # range of grain area in number of pixels wheat:(1000,20000); carob ((10000,40000)
band = 100  # spectral band to extract (#100 : 681 nm)
rgb = (82, 54, 14)
attrok = ('area', 'bbox_area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number', 'extent',
          'feret_diameter_max', 'filled_area', 'label', 'major_axis_length', 'minor_axis_length', 'orientation',
          'perimeter', 'perimeter_crofton', 'solidity')

sImg = fHS[0:fHS.find('.')]
img = envi.open(PATH + sImg + '.hdr', PATH + sImg + '.hyspex')
img = np.array(img.load(),dtype=np.int16)
img = np.transpose(img, (1, 0, 2))
imrgb = img[:,:,rgb].astype('float')

## Detect and extract spectralon
im0 = img[:, 1:id_spectralon, :]
ret0, binaryImage0 = cv.threshold(im0[:,:,band], thresh_lum*2, im0.max(), cv.THRESH_BINARY)
# Save spectralon mean reflectance
ref = np.zeros((img.shape[0],img.shape[2]),img.dtype)
for x in range(0,img.shape[0]):
     nz=binaryImage0[x,:] != 0
     if sum(nz) > 50:
        ref[x,:] = np.mean(im0[x,nz,:],0)
        imrgb[x, :, :] = imrgb[x, :,:] / np.tile(ref[x,rgb], (img.shape[1], 1)).astype('float')
        #imseuil_grain[x, :] = imseuil_grain[x, :] / np.tile(ref[x,band], (img.shape[1])).astype('float')

f = gzip.GzipFile(PATH_OUT + sImg + "_ref.gz", "wb")
np.save(f, ref)
f.close()
imrgb=np.clip(imrgb, a_min=None, a_max=1)
plt.imsave(PATH_OUT + sImg + '.jpg', imrgb)

# Run inference
imrgb=np.round(imrgb[:,id_im_grains[0]:id_im_grains[1],:]*255)

# Model SAM
res = model(imrgb, device="cpu")

# Model FastSAM
#res = model(imrgb, device="cpu", retina_masks=True, imgsz=2048)

masks=[i.masks for i in res[0]]

# Save spectra and morph into file
sp = np.empty((0, img.shape[2] + 3)).astype(np.int16)  # np.empty((len(o),img.shape[2]))
morph = np.empty((len(masks), len(attrok)))
imr = img[:, id_im_grains[0]:id_im_grains[1], :]
dep = np.reshape(imr, (imr.shape[0] * imr.shape[1], imr.shape[2]))  # unfolded image

m0 = masks[1].data.numpy()
for i in masks[1:len(masks)] :

    m=np.array(masks[1].data.numpy()[0]).astype(int)
    regions = regionprops(m)

    id = np.ravel_multi_index(np.transpose(regions[0].coords),
                              (imr.shape[0], imr.shape[1]))  # coord of grains pixels in unfolded image
    sp1 = np.array([dep[j, :] for j in id]).astype(np.int16)
    spcoord = np.concatenate((mb.repmat(i + 1, len(id), 1), regions[filtered[i] - 1].coords, sp1), axis=1).astype(
        np.int16)
    sp = np.concatenate((sp, spcoord))

    for j in range(0, len(attrok)):
        morph[i, j] = getattr(regions[0], attrok[j])


with gzip.GzipFile(PATH_OUT + sImg + "_sp.gz", "wb") as f:
    np.save(f, sp)

morph = morph.astype(np.float32)
fmorph = gzip.GzipFile(PATH_OUT + sImg + "_morph.gz", "wb")
np.save(fmorph, morph)
fmorph.close()


