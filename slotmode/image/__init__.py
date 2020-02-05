import numpy as np

from .. import CHIP_GAP, get_binning


def single_image_slot(istack):
    """
    Horizontally stack amplifier channel images including chip gap region.
    This is for display purposes. Pixel locations in the stacked image will be
    slightly offset from the actual on-sky location of the channel images
    since we do not account for the gap left when binning does not evenly
    divide the ccd and there are some left-over pixels.  Difference should be
    less than one pixel of the binned image, so if you are ok with that go
    ahead and use this.


    Parameters
    ----------
    istack

    Returns
    -------

    """
    binning = get_binning(istack[0])
    g = int(CHIP_GAP / binning[0])
    ch1, ch2, ch3, ch4 = istack
    gap = np.ma.masked_all((istack.shape[1], g))
    return np.ma.hstack([ch1, ch2, gap, ch3, ch4])
