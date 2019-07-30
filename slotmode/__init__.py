import logging

from recipes.dict import AttrReadItem
from recipes.introspection.utils import get_module_name

from astropy.io.fits.header import Header

from .utils import *

logger = logging.getLogger(get_module_name(__file__))

# Number of amplifier channels
N_CHANNELS = 4
# default value for channels argument in functions
ALL_CHANNELS = (0, 1, 2, 3)
# Slot dimension on CCD
CHANNEL_SIZE_PIXELS = (144, 1024)  # pixels per channel
# "Each unbinned pixel samples 0.142 arcsecs of sky"
CHANNEL_SIZE_ARCMIN = np.multiply(CHANNEL_SIZE_PIXELS, 0.00236667)  # 0.142 / 60


def _check_channel(channel):
    channel = int(channel)
    if channel < 0:
        channel += N_CHANNELS
    assert 0 <= channel <= 3, 'Invalid amplifier channel %i' % channel
    return channel

def _check_channels(channels):
    # resolve valid amplifier channels
    channels = np.atleast_1d(channels).astype(int)
    if np.any((0 > channels) | (channels > N_CHANNELS)):
        raise ValueError('Invalid amplifier(s) in %s' % channels)
    return channels

# _shapes = {(4, 4): (36, 254),
#           (6, 6): np.floor_divide(CHANNEL_SIZE_PIXELS, (6, 6))}

# Badpixel mask
# TODO: generalize for all binnings.
# TODO: from pySALT.slotmode import bad_pixels, bad_pixel_mask
# bad_pixel_mask[2] for amp2 etc

# def get_bad_pixels(binning=None):
#     """"""
#     # pix = get_bad_pixels(binning)
#     pass


# POTENTIAL BAD PIXEL INDICES
# channel 2
# -------------

# channel 3
# -------------
# [(72, 786),
#
#  (0, 288),
#  (1, 288),
#  (2, 288),
#  (3, 288),
#  (4, 288),
#  (5, 288),
#  (6, 288),
#  (7, 288),
#  (8, 288), ]


def get_binning(image):
    return tuple(
        np.divide(CHANNEL_SIZE_PIXELS, image.shape).round().astype(int))


# TODO: load external list
#  tools to find these automatically
BADPIXELS = {
    1:  # channel 2
        AttrReadItem(ranges={(1, 1): ((84, [732, 816]),
                                      (90, [800, 840]),
                                      (90, [732, 738])),
                             (6, 6): ((5, [22, 23]),
                                      (14, [123, 137]),
                                      (15, [136, 142]),
                                      (15, [123, 124]))
                             },
                     cols={(6, 6): [100],
                           (4, 4): [148]}),
    2:  # channel 3
        AttrReadItem(ranges={(3, 3): (((0, 3), 96),
                                      (24, 262))},
                     cols={(3, 3): [59, 295], }
                     )
}


def get_bad_pixel_mask(image, ch, detect_edge_col=True):
    """
    Create a mask (boolean array) for SALTICAM image.

    Parameters
    ----------
    image
    ch
    detect_edge_col

    Returns
    -------

    """

    bad_cols, bad_rows, bad_pix, bad_ranges = [], [], [], []
    if ch in BADPIXELS:
        bad_pix_ch = BADPIXELS[ch]
        binning = get_binning(image)
        if binning in bad_pix_ch.ranges:  # FIXME THIS SHOULD BE DONE BETTER!!!!
            bad_ranges = bad_pix_ch.ranges[binning]
            bad_cols = bad_pix_ch.cols[binning]
        else:
            logger.warning('No bad pixel map available for channel: %i at '
                           'binning (%i, %i)', ch, *binning)
    else:
        logger.warning('No bad pixel map available for channel: %i', ch)

    if detect_edge_col:
        # Some SLOTMODE cubes have all 0s in 0th column for some obscure reason
        if np.median(image[:, 0]) / np.median(image) < 0.2:
            bad_cols.append(0)

    # create boolean array
    bad_pixel_mask = _mask_from_indices(image.shape, bad_pix, bad_cols,
                                        bad_rows, bad_ranges)
    # bad_cols = [59, 295]
    # bad_pix = [(22, 262), (0, 96), (1, 96), (2, 96), (3, 96)]
    return bad_pixel_mask

    # shape = image.shape
    # # by, bx = binning = tuple(np.floor_divide(CHANNEL_SIZE_PIXELS, shape))
    # by, bx = binning = get_binning(image)
    # bad_pixel_mask = np.zeros(shape, bool)
    # bad_pixel_mask[:, _badcols[binning]] = True
    #
    # for row, xrng in _badpixmap:
    #     xrng = np.floor_divide(xrng, bx)
    #     sx = slice(*xrng)
    #     # print(row // by, xrng)
    #     bad_pixel_mask[row // by, sx] = True
    #
    # #  Some SLOTMODE cubes have all 0s in first column for some obscure reason
    # if image[:, 0].mean() < 10:
    #     bad_pixel_mask[:, 0] = True

    # return bad_pixel_mask


# def get_bad_pixel_mask(filename):

def _mask_from_indices(shape, bad_pix=None, bad_cols=None, bad_rows=None,
                       bad_ranges=None):
    """

    Parameters
    ----------
    shape
    bad_pix
    bad_cols
    bad_rows
    bad_ranges

    Returns
    -------

    """

    def null(thing):
        return (thing is None) or (len(thing) == 0)

    mask = np.zeros(shape, bool)

    if not null(bad_pix):
        br, bc = np.array(bad_pix).T
        mask[br, bc] = True

    if not null(bad_cols):  # or np.size(.) == 0
        mask[:, bad_cols] = True
    if not null(bad_rows):
        mask[bad_rows, :] = True

    if not null(bad_ranges):
        for row, col in bad_ranges:
            # print(row, col)
            if not isinstance(row, int):
                row = slice(*row)
            elif not isinstance(col, int):
                col = slice(*col)
            # else: # both are int
            #     raise ValueError('Either row or column in `bad_ranges` must '
            #                      'be int')
            # mask them
            mask[row, col] = True

    return mask


#
# def get_edge_cutoffs(image):
#     # TODO: find these automatically!!!!
#     # FIXME: this depends on the amplifier
#
#     # mask = np.zeros(image.shape, bool)
#     binning = get_binning(image)  # by, bx =
#
#     if np.all(binning == (6, 6)):
#         return (0, 162, 3, 18)  # this is for amp 3!!
#
#     elif np.all(binning == (4, 4)):
#         return (0, 240, 8, 29)  # this is for amp 3!!
#
#     elif np.all(binning == (3, 3)):
#         return (10, 342, 10, 37)  # this is for amp 2!!
#     else:
#         raise NotImplementedError
#     # return mask

def pprint_header(filename, n):
    from salticam.slotmode.extract import parse_header
    _pprint_header(*parse_header(filename), n=n)


def _convert(header):
    if isinstance(header, (str, bytes)):
        return Header.fromstring(header)
    if isinstance(header, Header):
        return header
    raise TypeError('Invalid type for header')


def _pprint_header(*headers, n):
    """"""
    # TODO:  moon phase

    import motley
    from motley.table import Table
    from recipes import pprint

    nh = len(headers)
    if nh == 1:
        header0 = header1 = _convert(headers[0])
    elif nh == 2:
        header0, header1 = map(_convert, headers)
    else:
        raise ValueError('This function accepts either 1 or 2 fits headers as '
                         'arguments, received: %i' % nh)

    t_dead = 43e-3  # median dead time 43 ms.  just for duration estimate
    exp_time = header0['EXPTIME']
    duration = n * (exp_time + t_dead)
    rows, cols = map(header1.get, ('NAXIS2', 'NAXIS1'))
    shape = (n, rows, cols)

    txt_props = ('bold', 'k', 'underline')
    obj_info = Table({'Object': motley.cyan(motley.bold(header0['OBJECT'])),
                      'α': ' {}ʰ{}ᵐ{}ˢ'.format(*header0['RA'].split(':')),
                      'δ': '{}°{}′{}″'.format(*header0['DEC'].split(':'))},
                     title='Object info',
                     title_props=dict(bg='g', txt=txt_props))

    obs_info = Table({'Date': header0['DATE-OBS'],
                      'Start Time': f"{header0['TIME-OBS']} UT",
                      'Duration': pprint.hms(duration, unicode=True),
                      'Exp time': f"{exp_time} s",
                      'Airmass': header0['AIRMASS'],
                      'Filter': header0['FILTER']},
                     title='Observation info',
                     title_props=dict(bg='b', txt=txt_props))
    ccd_info = Table({'binning': header0['CCDSUM'].replace(' ', 'x'),
                      'Dimensions': shape,
                      'Gain setting': header0['GAINSET'],
                      'Readout speed': header0['ROSPEED']},
                     title='CCD info',
                     title_props=dict(bg='b', txt=txt_props))

    prop_info = Table({'Proposal ID': header0['PROPID'],
                       'PI': header0['PROPOSER']},
                      title='Proposal info',
                      title_props=dict(bg='y', txt=txt_props))

    s = motley.table.hstack((obj_info,
                             motley.table.vstack((obs_info, ccd_info)),
                             prop_info))
    logger.info(f'\n{s}\n')
    return s, header0
