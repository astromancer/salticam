import numpy as np
import inspect
from . import get_binning, CHIP_GAP


def hstack(data):
    """
    stack images from all 4 amplifier channels into single image including
    spacing for chip gap of ~102 unbinned pixels
    """

    assert data.shape[0] == 4
    ch0, ch1, ch2, ch3 = data
    _, b = get_binning(ch0)
    gap = np.ma.masked_all((ch0.shape[0], int(round(CHIP_GAP / b))))
    return np.ma.hstack([ch0, ch1, gap, ch2, ch3])


def get_pixel_neighbourhood(ij, image, connectivity=None, ndist=1,
                            left=False, right=False,
                            lower=False, upper=False,
                            upper_left=False, upper_right=False,
                            lower_left=False, lower_right=False):
    """
    Get the indices of the neighbouring pixels to given pixel. Which
    neighbours are returned can be specified by


    Parameters
    ----------
    ij
    image
    connectivity
    left
    right
    lower
    upper
    upper_left
    upper_right
    lower_left
    lower_right

    Returns
    -------

    """

    from scipy import ndimage

    if connectivity is None:
        # check which pixels are on
        frame = inspect.currentframe()
        arg_names, *_, arg_values = inspect.getargvalues(frame)
        # which neighbours turned on?
        on = [arg_values[_] for _ in arg_names[3:]]
        if not np.any(on):  # no pixels turned on by user. go to default
            # neighbourhood with "+" connectivity
            connectivity = 4

    if connectivity is not None:
        if connectivity == 4:
            struct = ndimage.generate_binary_structure(2, 1)
        elif connectivity == 8:
            struct = ndimage.generate_binary_structure(2, 2)
        else:
            raise ValueError('Invalid connectivity={0}.  '
                             'Options are 4 or 8'.format(connectivity))

        #
        z = np.zeros(([2 * ndist + 1] * 2))
        z[ndist, ndist] = True
        struct = ndimage.binary_dilation(z, struct, 2)
        deltas = np.transpose(np.where(struct)) - (ndist, ndist)

    else:
        deltas = []
        for i in range(1, ndist + 1):
            if left:
                deltas.append((0, -i))
            if right:
                deltas.append((0, +i))
            if lower:
                deltas.append((-i, 0))
            if upper:
                deltas.append((+i, 0))
            if upper_left:
                deltas.append((+i, -i))
            if upper_right:
                deltas.append((+i, +i))
            if lower_left:
                deltas.append((-i, +i))
            if lower_right:
                deltas.append((-i, +i))

    indices = np.add(ij, deltas)
    ignore = np.any((indices < 0) | (indices >= image.shape[-2:]), 1)

    if np.ma.is_masked(image):
        ji = np.transpose(np.where(image.mask))
        ignore |= (ji[..., None] == indices[..., None].T).all(1).any(0)

    return image[(...,) + tuple(indices[~ignore].T)]


def find_bad_pixels(image):
    # needs work
    neigh = view_neighbours(image)

    mim = np.nanmedian(neigh, (-1, -2), keepdims=True)
    mmad = mad(neigh, mim, (-1, -2))

    return np.abs(image - mim.squeeze()) > 7.5 * mmad


def neighbourhood_median(cube, mask, nframes=1000, connectivity=2):
    """
    Median

    Parameters
    ----------
    image
    mask
    connectivity

    Returns
    -------

    """

    from scipy import ndimage

    # select a 1000 frames randomly throughout cube
    n = max(len(cube), int(nframes))
    use = np.random.randint(0, len(cube), n)
    sub = cube[use]

    w = np.array(np.where(mask))
    pm = np.mgrid[-1:2, -1:2]  # index deltas
    # connectivity for neighbourhood
    n_use = ndimage.generate_binary_structure(2, connectivity)
    n_use[1, 1] = False  # ignore self
    # get neighbour indices
    nix = w[:, None, None] + pm[..., None]
    unix = nix[:, n_use, :]
    # limit to within array
    up = np.array(mask.shape, ndmin=3).T
    inside = np.all((unix >= 0) & (unix < up), 0)

    # flag neighbours that are also bad pixels
    bad = (unix[..., None, :] == w[:, None, :, None]).all(0).any(-1)
    good = inside & ~bad  # good neighbours mask
    # since we'll mask the bad neighbours below, we set the bad indices to zero
    # and keep the array dimensionality for convenience
    unix[:, ~good] = 0

    yu, xu = tuple(unix)
    neighvals = np.ma.array(sub[:, yu, xu])
    neighvals[:, ~good] = np.ma.masked
    yw, xw = tuple(w)
    ff3 = sub[:, yw, xw] / np.ma.median(neighvals, 1)
    ffpix = np.ma.median(ff3, 0)

    flat = np.ones(cube.shape[1:])
    flat[mask] = ffpix
    return flat


# from motley import profiler
#
# @profiler.histogram()


# from scipy import ndimage

# def make_flat(cube):
#     from scipy import ndimage
#
#     bp = get_bad_pixel_mask(cube[0])
#     struct = np.ones((3, 3))
#     flat = np.ones(cube.shape[1:])
#     for yi, xi in zip(*np.where(bp)):
#         z = np.zeros_like(bp)
#         z[yi, xi] = True
#         s = ndimage.binary_dilation(z, struct) & ~bp
#         f = np.median(cube[:, z] / np.median(cube[:, s], 1))
#         flat[yi, xi] = f
#     return flat
