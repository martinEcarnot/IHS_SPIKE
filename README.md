# Projet IHS_SPIKE

Amélioration du post process de chaine de passage d'échantillon SPIKE.

Plateau de phénotypage ARCAD

# Usage

## Install

**1)** Clone the repository:
   
`git clone ...`


**2)** Create a virtual env and activate it:

`python -m venv ihs_spike_env`

**Windows** : `./ihs_spike_env/Scripts/activate`

**Linx/mac** : `source ihs_spike_env/bin/activate`


**3)** Install requirements:

`pip install -r requirements`

## Download models

Download sam models from teh following link
and add it too your "models" directory.

- model : [sam_b.pt]([https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_b.pt))


## Run

...




# Outputs

## Kernels properties (*_props.csv) 

**area**: Area of the region i.e. number of pixels of the region scaled by pixel-area.

**bbox_area**: Area of the bounding box i.e. number of pixels of bounding box scaled by pixel-area.

**convex_area**: Area of the convex hull image, which is the smallest convex polygon that encloses the region.

**eccentricity**: Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.

**equivalent_diameter**: The diameter of a circle with the same area as the region.

**euler_number**: Euler characteristic of the set of non-zero pixels. Computed as number of connected components subtracted by number of holes (input.ndim connectivity). In 3D, number of connected components plus number of holes subtracted by number of tunnels.

**extent**: Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)

**feret_diameter_max**: Maximum Feret’s diameter computed as the longest distance between points around a region’s convex hull contour as determined by find_contours.

**filled_area**: Area of the region with all the holes filled in.

**label**: The label in the labeled input image.

**major_axis_length**: The length of the major axis of the ellipse that has the same normalized second central moments as the region.

**minor_axis_length**: The length of the minor axis of the ellipse that has the same normalized second central moments as the region.

**orientation**: Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.

**perimeter**: Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.

**perimeter_crofton**: Perimeter of object approximated by the Crofton formula in 4 directions.

**solidity**: Ratio of pixels in the region to pixels of the convex hull image.


