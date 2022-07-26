import os, sys
import traceback as tb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

from ray.geometry import ParallelGeo, ConeGeo
from ray.pixel_detector import make_detector_result
from ray.source_detector import get_rays

import sub_funct.constant as cst


def main():
    
    num_of_proj:int = 10
    proj_angles:np.ndarray = np.linspace(
        0, 2*np.pi, num_of_proj,
        dtype=np.float64
    )
    print(proj_angles.shape)

    # geo = ParallelGeo(
    #     nVoxel=np.array((2,2), dtype=np.int64),
    #     nDetec=6
    # )
    geo = ConeGeo(
        nVoxel=(2,2),
        nDetec=6
    )
    # print(geo)

    tex = np.array(
        [
            # [0.21, 0.87, 0.64, 0.32],
            # [0.81, 0.47, 0.34, 0.12],
            # [0.71, 0.37, 0.14, 0.92],
            # [0.11, 0.67, 0.24, 0.52],
            [0.21, 0.87],
            [0.46, 0.12]
        ],
        dtype=np.float64
    )
    detector_pixels = [0 for _ in range(geo.nDetec.u)]


    tot_sino:np.ndarray = None
    for angle_idx, rot_angle in enumerate(proj_angles[:]):

        ray_info = get_rays(
            # mode='parallel',
            mode='cone',
            geo=geo,
            proj_angle=rot_angle
        )
        # print(ray_info)

        sino = make_detector_result(
            geo=geo,
            tex=tex,
            ray_info=ray_info,
            detector_pixels=detector_pixels,
            angle_idx=angle_idx
        )
        if tot_sino is None:
            tot_sino = sino
        else:
            tot_sino = np.vstack((tot_sino, sino))

    print("sinogram shape", tot_sino.shape)
    fig, axes = plt.subplots(1,2, sharex=False, sharey=False)
    axes[0].imshow(tex, cmap=plt.cm.Greys_r)
    axes[1].imshow(tot_sino.T, cmap=plt.cm.Greys_r)
    fig.tight_layout()
    if not os.path.exists(f"{cst.OUTPUT_PATH}/fig"):
        os.makedirs(f"{cst.OUTPUT_PATH}/fig", exist_ok=True)
    fig_fname:str = f"{cst.OUTPUT_PATH}/fig/projection.png"
    plt.savefig(fig_fname)
    # plt.show()
    plt.cla()
    plt.clf()


if __name__ == '__main__':
    start = dt.now()
    main()
    print(f'[ Done ]: {dt.now()-start}')
