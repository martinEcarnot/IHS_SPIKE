"""
File: main.py
Authors: Clement Plessis - Martin Ecarnot
Date: 11 Oct. 2024
Description: This script is part of the IHS SPIKE project 
and is designed to segment and get the morphology of
wheat kernels after the
hyperspectral data acquistione.

"""

import yaml, time, os
from pathlib import Path
from segment_anything import sam_model_registry
from kernels import Kernels
from spectrum import SpectrumCamera, SpectrumASD


# ================================
#       PARAMETERS
config = yaml.load(open('config.yml', 'r'), Loader=yaml.SafeLoader)

# ================================
#       LOAD MODEL
model_type = config["segment_kernels"]["model_type"]
model_path = config["segment_kernels"]["model_path"]
sam = sam_model_registry[model_type](checkpoint=model_path)
sam.to(device=config["segment_kernels"]["device"])

# ================================
#       PROCESS FILES
print(f"\nProcessing files from {config['data']['input_path']}\n")
files = [i for i in Path(config["data"]["input_path"]).glob("*.hdr")]

t0 = time.time()
n=0
for hdr_file in files:
    # ================================
    #           INIT
    n+=1
    t0samp = time.time()
    print(f"progress {n}/{len(files)}")
    date = hdr_file.stem.split("_")[0]
    hour = hdr_file.stem.split("_")[1]
    sample = "_".join(hdr_file.stem.split("_")[2:-1])

    print(f"    > file name = {hdr_file}")
    print(f"    > sample = {sample}")
    print(f"    > date = {date}")
    print(f"    > hour = {hour}")

    # check files exsits and complete
    asd_file = str(hdr_file).replace("HyperS.hdr", "Spectre01.asd")
    hyspex_file = str(hdr_file).replace(".hdr", ".hyspex")
    if any([
        not os.path.exists(asd_file),os.stat(asd_file).st_size < 3e4,
        not os.path.exists(hyspex_file),os.stat(hyspex_file).st_size < 3e9
    ]):
        print("One of ASD or HYSPEX are missing or too small.")
        print("Starting next file ...")
        continue

    # ================================
    #       Spectrum
    
    # Load spectrum
    spectrum = SpectrumCamera(str(hdr_file))
    
    # Save rgb image
    spectrum.save_rgb(
        output_path=config["data"]["output_path"],
        sample=sample, date=date, hour=hour
    )
    
    # ================================
    #      Kernels Segmentation
    
    # Segmentation
    print("segmentation...")
    t0seg = time.time()
    kernels = Kernels(
        image_rgb=spectrum.image_rgb,
        sam_model=sam,
        crop_x_left = config["segment_kernels"]["crop_x_left"],
        crop_x_right = config["segment_kernels"]["crop_x_right"] 
    )
    t1seg = time.time()
    print(f"    > seg time = {round(t1seg-t0seg, 0)}s")    

    # Filter masks
    print("number of masks BEFORE filtering : ", len(kernels.masks))
    kernels.filter_masks(
        area_min=config["segment_kernels"]["area_min"],
        area_max=config["segment_kernels"]["area_max"]
    )
    print("number of masks AFTER filtering : ", len(kernels.masks))

    # Save masks
    kernels.save_masks(
        output_path=config["data"]["output_path"],
        sample=sample, date=date, hour=hour
    )

    # Save regionprops
    kernels.add_regionprops()
    kernels.save_rpops(
        output_path=config["data"]["output_path"],
        sample=sample, date=date, hour=hour
    )

    # ASD
    specasd = SpectrumASD(asd_file)
    specasd.save_spectrum(
        output_path=config["data"]["output_path"],
        sample=sample, date=date, hour=hour
    )

    t1samp = time.time()
    print(f"    > sample time = {round(t1samp-t0samp,0)}s")
    print("\n")

t1 = time.time()
print(f"All process done in {t1-t0} seconds.")