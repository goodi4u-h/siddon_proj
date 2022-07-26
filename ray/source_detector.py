import os, sys
import numpy as np

from ray.unit_geo import Point2D, Vector2D
from ray.geometry import ParallelGeo

import sub_funct.constant as cst


def get_rays(
    mode:str,
    geo:ParallelGeo,
    proj_angle:np.float64=0
) -> dict:

    available_mode = ['cone', 'parallel']
    if mode not in available_mode:
        raise TypeError(
            f'Invalid mode; pick one of these {available_mode}'
        )

    du_range:tuple = (
        -geo.sDetec.u/2,
        geo.sDetec.u/2,
    )

    if mode=='cone':
        du_list = np.linspace(
            du_range[0],
            du_range[1],
            geo.nDetec.u*cst.PROJ_PER_PIXEL,
            dtype=np.float64
        )

        detector_coord = [
            Point2D(x=geo.DSO, y=i, axis_rotation_angle=proj_angle)
            for i in du_list
        ]

        src_coord = [
            Point2D(x=-geo.DSO, y=0, axis_rotation_angle=proj_angle)
            for _ in du_list
        ]

        ray_vectors = [
            Vector2D(start, end)
            for start, end in zip(src_coord, detector_coord)
        ]

    else:
        du_list:np.ndarray = np.linspace(
            du_range[0],
            du_range[1],
            geo.nDetec.u*cst.PROJ_PER_PIXEL,
            dtype=np.float64
        )

        detector_coord = [
            Point2D(
                x=geo.DSO-geo.DSD,
                y=i,
                axis_rotation_angle=proj_angle
            )
            for i in du_list
        ]

        src_coord = [
            Point2D(x=-geo.DSO, y=i, axis_rotation_angle=proj_angle)
            for i in du_list
        ]

        ray_vectors:list = [
            Vector2D(start, end)
            for start, end in zip(src_coord, detector_coord)
        ]

    return {
        'du_range': du_range,
        'du': du_list,
        'vectors': ray_vectors,
        'source_coordinates': src_coord,
        'detector_coordinates': detector_coord,
    }
