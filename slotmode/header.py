from astropy.io.fits import Header

from . import logger


def pprint_header(filename, n):
    from salticam.slotmode.extract import parse_header

    s = pformat_header(*parse_header(filename), n=n)
    logger.info(f'{filename}\n{s}\n')


def _convert(header):
    if isinstance(header, (str, bytes)):
        return Header.fromstring(header)
    if isinstance(header, Header):
        return header
    raise TypeError('Invalid type for header')


def pformat_header(*headers, n):
    """"""
    # TODO:  moon phase

    import motley
    from motley.table import Table
    from motley.utils import hstack, vstack
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

    # need to convert tbl to str so the headers align properly
    s = hstack((str(obj_info),
                vstack((obs_info, ccd_info)),
                prop_info))

    return s  #, header0
