import logging

import numpy as np

from recipes.introspection.utils import get_module_name

from astropy.io.fits.hdu.base import register_hdu

# from .utils import *
from .core import SaltiCamHDU

register_hdu(SaltiCamHDU)

# setup logger
logger = logging.getLogger(get_module_name(__file__))

# Number of amplifier channels
N_CHANNELS = 4
# default value for channels argument in functions
ALL_CHANNELS = (0, 1, 2, 3)
# Slot dimension on CCD
CHANNEL_SIZE = SLOT_HEIGHT, CHANNEL_WIDTH = (144, 1024)  # pixels per channel
# pixel size:
# "Each unbinned pixel samples 0.142 arcseconds of sky"
#  - SALTICAM.pdf, L. Balona, 2006

# My more recent measurements indicate that the pixel size is closer to
# 0.138 arc-seconds.  This is also what is says in the fits headers...
PIXEL_SCALE_ARCMIN = 0.138 / 60  # 0.00236667
CHANNEL_SIZE_ARCMIN = np.multiply(CHANNEL_SIZE, PIXEL_SCALE_ARCMIN)
# PRESCAN = OVERSCAN = 50  # pixels
# separation between channels / CCDs
CHIP_GAP = 102.08  # pixels

SLOT_WIDTH = CHANNEL_WIDTH * N_CHANNELS + CHIP_GAP
SLOT_WIDTH_ARCMIN = SLOT_WIDTH * PIXEL_SCALE_ARCMIN
SLOT_HEIGHT_ARCMIN = SLOT_HEIGHT * PIXEL_SCALE_ARCMIN
FOV = (SLOT_HEIGHT_ARCMIN, SLOT_WIDTH_ARCMIN)

UNITS = {'arcmin': PIXEL_SCALE_ARCMIN,
         'pixels': 1}


def check_channel(channel):
    channel = int(channel)
    if channel < 0:
        channel += N_CHANNELS
    assert 0 <= channel <= 3, 'Invalid amplifier channel %i' % channel
    return channel


def check_channels(channels, data=None, axis=0):
    # resolve valid amplifier channels.  Optionally check against data for same
    # number of elements in leading dimension.
    channels = np.atleast_1d(channels).astype(int)
    if np.any((0 > channels) | (channels > N_CHANNELS)):
        raise ValueError('Invalid amplifier(s) in %s' % channels)

    # check if data has same number of channels
    if (data is not None) and (len(data) != len(channels)):
        raise ValueError(f'Incorrect number of channels {len(channels)} for '
                         f'data with {len(data)} elements along axis {axis}.')

    return channels


def get_binning(image):
    """Calculate binning from image size and CCD chip size"""
    return tuple(np.divide(CHANNEL_SIZE, image.shape).round().astype(int))

# def get_channel_offset(ch, binning):
#     ghost_pixels = np.remainder(CHANNEL_SIZE, binning)
#
#     CHANNEL_WIDTH * ch + CHIP_GAP * (ch > 1)


def to_instr_coords(coords, binning, channels=ALL_CHANNELS, unit='arcmin'):
    # convert channel coordinates to instrument coordinates
    # assume yx (row column) coordinates

    channels = check_channels(channels, coords)
    scale = UNITS.get(unit)

    # since the binning may not evenly divide the size of the full chip,
    # we may have a few pixels left over.  SALTICAM it seems reads out these
    # remaining pixels as a full (binned) pixel. The image will therefore have
    # size that is one pixel larger than the number of pixels that would
    # divide the chip equally.  This means that the first column in the image
    # does not have the same physical size as all the other pixels.  We need to
    # correct for this when transforming from image to instrument coordinates.
    # Eg: pixel (0, 0) in 6x6 binning corresponds to (0, 2) in un-binned pixel
    # coordinates

    # number of un-binned pixels left over
    ghost_pixels = np.remainder(CHANNEL_SIZE, binning)

    new_coords = []
    for ch, cx in zip(channels, coords):
        # make relative to un-binned pixels in channel
        cx = (cx * binning) + ghost_pixels
        # add offset for each channel
        cx[:, 1] += CHANNEL_WIDTH * ch + CHIP_GAP * (ch > 1)
        new_coords.append(cx * scale)
    return np.vstack(new_coords)
