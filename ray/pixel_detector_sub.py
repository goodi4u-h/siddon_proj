import numpy as np


def alpha_of_idx(idx, first_plane, length_per_voxel, src, ray):
    return np.float64(
        (first_plane+idx*length_per_voxel-src) / \
        ray
    )


def rho_of_alpha(alpha, src, ray, first_plane, length_per_voxel):
    return np.float64(
        (src+alpha*ray-first_plane) / \
        length_per_voxel
    )

    
def get_min_max_index(
    src, detector, ray,
    alpha_min, alpha_max, 
    alpha_min_of_axis, alpha_max_of_axis, 
    first_plane, dist_per_voxel,
    num_of_voxel
) -> tuple:

    if src<detector:
        idx_min = np.int64(1) if alpha_min==alpha_min_of_axis \
            else np.int64(np.ceil(
                rho_of_alpha(
                    alpha_min, src, ray,
                    first_plane, dist_per_voxel
                )
            ))
        idx_max = np.int64(num_of_voxel) if alpha_max==alpha_max_of_axis \
            else np.int64(np.floor(
                rho_of_alpha(
                    alpha_max, src, ray,
                    first_plane, dist_per_voxel
                )
            ))

    else:
        idx_min = np.int64(0) if alpha_max==alpha_max_of_axis \
            else np.int64(np.ceil(
                rho_of_alpha(
                    alpha_max, src, ray,
                    first_plane, dist_per_voxel
                )
            ))
        idx_max = np.int64(num_of_voxel-1) if alpha_min==alpha_min_of_axis \
            else np.int64(np.floor(
                rho_of_alpha(
                    alpha_min, src, ray,
                    first_plane, dist_per_voxel
                )
            ))

    return (idx_max, idx_min)


def get_alpha_of_axis(
    src, detector, ray,
    first_plane, dist_per_voxel,
    idx_min, idx_max,
) -> np.float64:

    if src<detector:
        alpha_of_axis = alpha_of_idx(
            idx_min, first_plane, dist_per_voxel,
            src, ray
        )

    else:
        alpha_of_axis = alpha_of_idx(
            idx_max, first_plane, dist_per_voxel,
            src, ray
        )

    return alpha_of_axis


def get_idx_update_amount(
    src, detector
) -> int:

    if src<detector:
        idx_update = np.int64(1)

    else:
        idx_update = np.int64(-1)

    return idx_update
