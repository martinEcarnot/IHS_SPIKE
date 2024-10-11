
from ultralytics import SAM
from ultralytics import FastSAM
import cv2 as cv
import spectral as sp
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt



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
thresh_lum = 0.11  # 0.07 for wheat #1800 # threshold of reflectance (or light intensity) to remove background
areaRange = (10000, 40000)  # range of grain area in number of pixels wheat:(1000,20000); carob ((10000,40000)
band = 100  # spectral band to extract (#100 : 681 nm)
rgb = (82, 54, 14)

sImg = fHS[0:fHS.find('.')]
img = envi.open(PATH + sImg + '.hdr', PATH + sImg + '.hyspex')
img = np.array(img.load(),dtype=np.int16)
img = np.transpose(img, (1, 0, 2))
imrgb = img[:,:,rgb].astype('float')

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
                #imseuil_grain[x, :] = imseuil_grain[x, :] / np.tile(ref[x,band], (img.shape[1])).astype('float')

        # f = gzip.GzipFile(PATH_OUT + sImg + "_ref.gz", "wb")
        # np.save(f, ref)
        # f.close()
        imrgb=np.clip(imrgb, a_min=None, a_max=1)
        plt.imsave(PATH_OUT + sImg + '.jpg', imrgb)
        # plt.imsave(PATH_OUT + sImg + "_RGB.jpg", imrgb)
# Run inference
imrgb=np.round(imrgb[:,710:3460,:]*255)

# Model SAM
res = model(imrgb, device="cpu")

# Model FastSAM
#res = model(imrgb, device="cpu", retina_masks=True, imgsz=2048)

masks=[i.masks for i in res[0]]

# Mettre tous les masques sur la meme image
m0 = masks[1].data.numpy()
for i in masks[1:len(masks)] : # Pour chaque canal (R, G, B)
        m0 += i.data.numpy()
        #output_image[:, :, i] += (masks[i].data.numpy()).astype(np.uint8)
plt.imsave(PATH_OUT + 'mtot2.jpg', m0[0]*255)



m1=masks[40].data.numpy()
plt.imsave(PATH_OUT + 'm1.jpg', m1[0]*255)



print(len(masks))
attrok = ('area', 'bbox_area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number', 'extent',
          'feret_diameter_max', 'filled_area', 'label', 'major_axis_length', 'minor_axis_length', 'orientation',
          'perimeter', 'perimeter_crofton', 'solidity')
morph = np.empty((len(masks),len(attrok)))
        for i in range(0,len(masks)) :
            regions = regionprops(masks[i])
            for j in range(0, len(attrok)):
                morph[i, j] = getattr(regions[filtered[i] - 1], attrok[j])





# Solution Chat_GPT pour une image en couleur

# Dimensions des masques
h, w = imrgb.shape[0:2]
nmasks = len(masks)

# Générer des couleurs uniques en utilisant matplotlib
cmap = plt.get_cmap('tab20')  # Utilisation d'une palette avec 20 couleurs différentes
colors = [cmap(i % 20)[:3] for i in range(nmasks)]  # Prendre les RGB des couleurs, en bouclant si nécessaire
colors = np.array(colors) * 255  # Convertir en valeurs de 0 à 255


# Créer une image vide pour superposer les masques avec des couleurs différentes
output_image = np.zeros((h, w, 3), dtype=np.uint8)

# Appliquer chaque masque avec une couleur différente
for mask, color in zip(masks, colors):
    for i in range(3):  # Pour chaque canal (R, G, B)
        output_image[:, :, i] += (mask.data.numpy()[0] * color[i]).astype(np.uint8)

plt.imsave(PATH_OUT + 'mtot_gpt.jpg', output_image)
