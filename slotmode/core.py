from . import CHANNEL_SIZE_ARCMIN
from .extract import parse_header

import numpy as np
from astropy.io.fits.header import Header
from astropy.io.fits.hdu import PrimaryHDU
from astropy.io.fits.hdu import register_hdu
from obstools.phot.utils import ImageSamplerHDUMixin
from recipes.logging import LoggingMixin


class SaltiCamHDU(PrimaryHDU, ImageSamplerHDUMixin, LoggingMixin):
    def __init__(self, data=None, header=None, do_not_scale_image_data=False,
                 ignore_blank=False, uint=True, scale_back=None):
        super().__init__(data, header, do_not_scale_image_data, ignore_blank,
                         uint, scale_back)

        # TODO: Mixin class that converts header info to attributes?
        # add some attributes for convenience
        self.telescope = 'SALT'
        self.target = header.get('OBJECT')
        self.coords = self.get_coords()

    @classmethod
    def match_header(cls, header):
        return header.get('INSTRUME') == 'SALTICAM'

    def get_fov(self):
        return CHANNEL_SIZE_ARCMIN  # yx

    def get_rotation(self):
        """image rotation in radians"""
        return np.radians(self.header['telpa'])

    def get_coords(self):
        from pySHOC.utils import retrieve_coords, convert_skycoords

        header = self.header
        ra, dec = header.get('ra'), header.get('dec')
        coords = convert_skycoords(ra, dec)
        if coords:
            return coords

        if self.target:
            # from pySHOC.utils import retrieve_skycoords

            # No / bad coordinates in header, but object name available - try
            # resolve
            coords = retrieve_coords(self.target)

        if coords:
            return coords

        # No header coordinates / Name resolve failed / No object name available
        # LAST resort use TELRA, TELDEC. This will only work for newer SHOC
        # data for which these keywords are available in the header
        # note: These will lead to slightly less accurate timing solutions,
        #  so emit warning
        ra, dec = header.get('telra'), header.get('teldec')
        coords = convert_skycoords(ra, dec)

        # TODO: optionally query for named sources in this location
        if coords:
            self.logger.warning('Using telescope pointing coordinates. This '
                                'may lead to barycentric timing correction / '
                                'image registration being less accurate.')

        return coords

    # def get_telescope(self):
    #     return 'SALT'


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
