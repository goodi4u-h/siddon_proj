import os
import traceback as tb
import numpy as np
import pandas as pd

from ray.pixel_detector_sub import *

import sub_funct.constant as cst


def make_detector_result(
    geo,
    tex,
    ray_info,
    detector_pixels,
    angle_idx
) -> np.ndarray:

    numerics1:list = []
    numerics2:list = []
    numerics3:list = []

    ## START: Run loop by ray vector
    for div_i, (src, detector, ray) in enumerate(zip(
        ray_info['source_coordinates'],
        ray_info['detector_coordinates'],
        ray_info['vectors']
    )):

        pixel_idx = div_i//cst.PROJ_PER_PIXEL
        
        ray_x:np.float64 = np.float64(detector.x-src.x)
        ray_y:np.float64 = np.float64(detector.y-src.y)
        # assert ray_x==ray.vector[0] and ray_y==ray.vector[1]

        first_x_plane:np.float64 = np.float64(-geo.sVoxel.x/2)
        first_y_plane:np.float64 = np.float64(-geo.sVoxel.y/2)
        # last_x_plane:np.float64 = np.float64(geo.sVoxel.x/2)
        # last_y_plane:np.float64 = np.float64(geo.sVoxel.y/2)
        # assert (first_x_plane+geo.nVoxel.x*geo.dVoxel.x)==last_x_plane \ #     and (first_y_plane+geo.nVoxel.y*geo.dVoxel.y)==last_y_plane


        # euclidian_dist = np.linalg.norm(
        #     np.array((src.x, src.y)) - \
        #     np.array((detector.x, detector.y))
        # )
        # assert euclidian_dist==ray.size
        euclidian_dist = ray.size
        assert euclidian_dist!=0

        attenuation_sum = np.float64(0)

        ## the first x/y plane of voxels
        ax0 = alpha_of_idx(
            0, first_x_plane, geo.dVoxel.x,
            src.x, ray_x
        )
        ay0 = alpha_of_idx(
            0, first_y_plane, geo.dVoxel.y,
            src.y, ray_y
        )
        
        ## the last x/y plane of voxels
        axN = alpha_of_idx(
            geo.nVoxel.x, first_x_plane, geo.dVoxel.x,
            src.x, ray_x
        )
        ayN = alpha_of_idx(
            geo.nVoxel.y, first_y_plane, geo.dVoxel.y,
            src.y, ray_y
        )
        axm = np.minimum(ax0, axN, dtype=np.float64)
        aym = np.minimum(ay0, ayN, dtype=np.float64)
        axM = np.maximum(ax0, axN, dtype=np.float64)
        ayM = np.maximum(ay0, ayN, dtype=np.float64)

        am = np.maximum(axm, aym, dtype=np.float64)
        aM = np.minimum(axM, ayM, dtype=np.float64)


        ## assign zero if ray is off all voxels
        if am>=aM:
            detector_pixels[pixel_idx] = np.float64(0)


        ## get i,j range based on alpha values
        imax, imin = get_min_max_index(
            src=src.x, detector=detector.x,
            ray=ray_x, alpha_min=am,
            alpha_max=aM, alpha_min_of_axis=axm,
            alpha_max_of_axis=axM, first_plane=first_x_plane,
            dist_per_voxel=geo.dVoxel.x, num_of_voxel=geo.nVoxel.x
        )
        jmax, jmin = get_min_max_index(
            src=src.y, detector=detector.y,
            ray=ray_y, alpha_min=am,
            alpha_max=aM, alpha_min_of_axis=aym,
            alpha_max_of_axis=ayM, first_plane=first_y_plane,
            dist_per_voxel=geo.dVoxel.y, num_of_voxel=geo.nVoxel.y
        )

        ax = get_alpha_of_axis(
            src=src.x, detector=detector.x, ray=ray_y,
            first_plane=first_x_plane, dist_per_voxel=geo.dVoxel.x,
            idx_min=imin, idx_max=imax
        )
        ay = get_alpha_of_axis(
            src=src.y, detector=detector.y, ray=ray_y,
            first_plane=first_y_plane, dist_per_voxel=geo.dVoxel.y,
            idx_min=jmin, idx_max=jmax
        )
        aminc = np.minimum(ax, ay, dtype=np.float64)


        ## get i, j corresponded to the first intersected pixel
        i = np.int64(np.floor(
            rho_of_alpha(
                (aminc+am)/2, src.x, ray_x,
                first_x_plane, geo.dVoxel.x
            )
        ))
        j = np.int64(np.floor(
            rho_of_alpha(
                (aminc+am)/2, src.y, ray_y,
                first_y_plane, geo.dVoxel.y
            )
        ))
        # assert 0<=i<=geo.nVoxel.x and 0<=j<=geo.nVoxel.y

        
        ## Initialize variables that need to be updated every loop
        ac = am
        axu = np.float64(
            geo.dVoxel.x/np.abs(ray_x, dtype=np.float64)
        )
        ayu = np.float64(
            geo.dVoxel.y/np.abs(ray_y, dtype=np.float64)
        )
        iu = get_idx_update_amount(src.x, detector.x)
        ju = get_idx_update_amount(src.y, detector.y)


        ## Number of intersections
        Np = int((imax-imin+1)+(jmax-jmin+1))

        
        ## START: For the numerical analysis
        numerics2.append({
            'div_idx': div_i,
            'intersect': am<aM,
            'am': am,
            'axm': axm, 'aym': aym,
            'aM': aM,
            'axM': axM, 'ayM': ayM,
            'ax0': ax0, 'axN': axN,
            'ay0': ay0, 'ayN': ayN,
        })
        numerics3.append({
            'div_idx': div_i,
            'i': i, 'j': j,
            'iu': iu, 'ju': ju,
            'imin': imin, 'imax': imax,
            'jmin': jmin, 'jmax': jmax,
            'ac': ac, 'aminc': aminc,
            'ax': ax, 'ay': ay,
            'axu': ax, 'ayu': ay,
        })
        ## END: For the numerical analysis

        i += np.float64(0.5)
        j += np.float64(0.5)

        for N_idx in range(Np):

            i, j = np.int64(i), np.int64(j)
            # if ax==aminc:
            if ax<=ay:
                if ax==ay:
                    print(f'ax==ay: {ax}')
                try:
                    attenuation_sum += (ax-ac)*euclidian_dist*tex[i][j]
                    if ax-ac==0:
                        print(
                            f'[ RECHECK {N_idx} / {Np} ] ' + \
                            f'{ax}, {ac}, {i}, {j}'
                        )
                except:
                    print(tb.format_exc())
                    print(i,j,angle_idx,div_i)
                    pass
                i += iu
                ac = ax
                ax += axu

            # elif ay==aminc:
            # elif ax>ay:
            else:
                try:
                    attenuation_sum += (ay-ac)*euclidian_dist*tex[i][j]
                    if ay-ac==0:
                        print(
                            f'[ RECHECK {N_idx} / {Np} ] ' + \
                            f'{ay}, {ac}, {i}, {j}'
                        )
                except:
                    print(tb.format_exc())
                    print(i,j,angle_idx,div_i)
                    pass
                j += ju
                ac = ay
                ay += ayu
            
            aminc = np.minimum(ay,ax, dtype=np.float64)

        detector_pixels[pixel_idx] += euclidian_dist*attenuation_sum

        ## START: For the numerical analysis
        numerics1.append({
            'div_idx': div_i,
            'pixel_idx': pixel_idx, "Np": Np,
            # 'x-plane': (first_x_plane, last_x_plane),
            # 'y-plane': (first_y_plane, last_y_plane),
            'source': tuple(src.__dict__.values()),
            'detector': tuple(detector.__dict__.values()),
            'ray': ray.vector,
            'x/y direction': (src.x<detector.x, src.y<detector.y),
            # 'max_len': maxlength,
            'euclidian': euclidian_dist,
            'attenu': attenuation_sum,
        })
        ## END: For the numerical analysis
        
    ## END: Run loop by ray vector
    

    ## START: Save the numerial data
    if not os.path.exists(cst.OUTPUT_PATH):
        os.makedirs(cst.OUTPUT_PATH, exist_ok=True)
    df = pd.DataFrame(numerics1)
    save_numerics:str = \
        f'{cst.OUTPUT_PATH}/proj_numerics_{angle_idx+1:04}_1.tsv'
    df.to_csv(
        save_numerics,
        sep='\t', index=False
    )
    df = pd.DataFrame(numerics2)
    save_numerics:str = \
        f'{cst.OUTPUT_PATH}/proj_numerics_{angle_idx+1:04}_2.tsv'
    df.to_csv(
        save_numerics,
        sep='\t', index=False
    )
    df = pd.DataFrame(numerics3)
    save_numerics:str = \
        f'{cst.OUTPUT_PATH}/proj_numerics_{angle_idx+1:04}_3.tsv'
    df.to_csv(
        save_numerics,
        sep='\t', index=False
    )
    ## END: Save the numerial data


    ## Normalize detector matrix
    normalized_arr = np.array(detector_pixels)
    normalized_arr -= np.amin(normalized_arr)
    normalized_arr /= np.amax(normalized_arr)
    # print(normalized_arr)

    return normalized_arr
