from . import CHANNEL_SIZE_ARCMIN
from .extract import parse_header

import numpy as np
from astropy.io.fits.header import Header
from astropy.io.fits.hdu import PrimaryHDU
from astropy.io.fits.hdu import register_hdu
from obstools.phot.utils import ImageSamplerHDUMixin


class SaltiCamHDU(PrimaryHDU, ImageSamplerHDUMixin):
    @classmethod
    def match_header(cls, header):
        return header['INSTRUME'] == 'SALTICAM'

    def get_fov(self):
        return CHANNEL_SIZE_ARCMIN  # yx


register_hdu(SaltiCamHDU)


class SaltiCamObservation(object):
    # emulate pySHOC.shocObs

    @classmethod
    def load(cls, filename, **kws):
        return cls(filename)

    def __init__(self, filename):
        self._filename = str(filename)
        self.data = np.lib.format.open_memmap(filename)[:, 1]  # HACK!!

        filename0 = next(filename.parent.glob('*.fits'))
        header0_bytes, header1_bytes = parse_header(filename0)
        self.header = Header.fromstring(header0_bytes)

    def filename(self):
        return self._filename
