import numpy as np
from recipes.containers.dicts import AttrReadItem
from salticam.slotmode import get_binning, CHANNEL_SIZE, \
    ALL_CHANNELS, check_channels

import numbers

# TODO: load external list
#  tools to find these automatically

BAD_PIXELS = {
    0:  # channel 1
        AttrReadItem(ranges=(),
                     cols=()
                     ),
    1:  # channel 2
        AttrReadItem(ranges=((84, [732, 816]),
                             (90, [800, 840]),
                             (90, [732, 738])),

                     cols=(596,)
                     ),
    2:  # channel 3
        AttrReadItem(ranges=(((92, 144), 174),
                             ((0, 24), 288)),
                     cols=(882,)
                     ),

    3:  # channel 4
        AttrReadItem(ranges=((132, 762),),
                     cols=())
}


def mask_dark_edges(data, threshold=0.2):
    """

    Parameters
    ----------
    data
    threshold

    Returns
    -------

    """
    nch, r, c = data.shape
    cross_col = np.ma.median(data, 1) / np.ma.median(data, (1, 2))[:, None]
    mask = np.array(cross_col < threshold)
    mask = mask[:, None] * np.ones((1, r, 1), bool)
    return np.ma.MaskedArray(data, mask)


def mask_bad_pixels(image, ch=None):
    """

    Parameters
    ----------
    image
    ch

    Returns
    -------

    """
    return np.ma.MaskedArray(image, get_bad_pixel_mask(image, ch))


def get_bad_pixel_mask(data, channels=None):
    """
    Create a mask (boolean array) for SALTICAM image.

    Parameters
    ----------
    data: array (2d or 3d)
    channels: int or tuple

    Returns
    -------

    """

    if channels is None:
        if data.ndim == 2:
            raise ValueError('Please provide an amplifier channel number that '
                             'this image corresponds to.')
        channels = ALL_CHANNELS

    if data.ndim == 2:
        data = data.reshape((1,) + data.shape)

    # check channels ok
    channels = check_channels(channels, data)
    binning = get_binning(data[0])
    mask = np.zeros(data.shape, bool)
    for i, image in zip(channels, data):
        mask[i] = _get_bad_pixel_mask(binning, i)

    return mask


def _get_bad_pixel_mask(binning, ch):
    # For some binning schemes, the full CCD is not used if the number of
    # pixels in the CCD does not divide evenly by the binning. This will
    # cause the first column to be dead
    extra_pixels = np.remainder(CHANNEL_SIZE, binning).astype(bool)
    bad_cols = [0] * int(extra_pixels[1])
    # bad_rows, bad_pix = [], []

    # if ch in BAD_PIXELS:
    bad_pix_ch = BAD_PIXELS[ch]
    bad_cols.extend(bad_pix_ch.cols)

    # create boolean array
    bad_pixel_mask = _mask_from_indices(binning, (), bad_cols,
                                        (), bad_pix_ch.ranges)
    return bad_pixel_mask


def ceil_divide(a, binning):
    return np.ceil(np.divide(a, binning)).astype(int)


def _mask_from_indices(binning, bad_pix=None, bad_cols=None, bad_rows=None,
                       bad_ranges=None):
    """

    Parameters
    ----------
    binning
    bad_pix
    bad_cols
    bad_rows
    bad_ranges

    Returns
    -------

    """

    def null(thing):
        return (thing is None) or (len(thing) == 0)

    # init mask
    shape = ceil_divide(CHANNEL_SIZE, binning)
    br, bc = binning
    mask = np.zeros(shape, bool)

    # fill bad pixels
    if not null(bad_pix):
        mask[ceil_divide(np.array(bad_pix).T, binning)] = True

    if not null(bad_cols):
        mask[:, ceil_divide(bad_cols, bc)] = True

    if not null(bad_rows):
        mask[ceil_divide(bad_rows, br), :] = True

    if not null(bad_ranges):
        for rows, cols in bad_ranges:
            rows = ceil_divide(rows, br)
            cols = ceil_divide(cols, bc)

            if not isinstance(rows, numbers.Integral):
                rows = slice(*rows)

            elif not isinstance(cols, numbers.Integral):
                cols = slice(*cols)

            # mask them
            mask[rows, cols] = True

    return mask
