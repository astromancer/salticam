import numpy as np

from obstools.phot.campaign import HDUExtra
from recipes.logging import LoggingMixin


class SaltiCamHDU(HDUExtra, LoggingMixin):
    def __init__(self, data=None, header=None, do_not_scale_image_data=False,
                 ignore_blank=False, uint=True, scale_back=None):
        super().__init__(data, header, do_not_scale_image_data, ignore_blank,
                         uint, scale_back)

        # TODO: Mixin class / method that converts header info to attributes?
        # add some attributes for convenience
        self.telescope = 'SALT'
        self.target = header.get('OBJECT')
        self.coords = self.get_coords()

    @classmethod
    def match_header(cls, header):
        return header.get('INSTRUME') == 'SALTICAM'

    def get_fov(self):
        from salticam.slotmode import CHANNEL_SIZE_ARCMIN
        return CHANNEL_SIZE_ARCMIN  # yx

    def get_rotation(self):
        """image rotation in radians"""
        return np.radians(self.header['telpa'])

    def get_coords(self):
        from pySHOC.utils import get_coords_named, convert_skycoords

        header = self.header
        ra, dec = header.get('ra'), header.get('dec')
        coords = convert_skycoords(ra, dec)
        if coords:
            return coords

        if self.target:
            # No / bad coordinates in header, but object name available - try
            # resolve
            coords = get_coords_named(self.target)

        if coords:
            return coords

        # No header coordinates / Name resolve failed / No object name available
        # LAST resort use TELRA, TELDEC.
        ra, dec = header.get('telra'), header.get('teldec')
        coords = convert_skycoords(ra, dec)

        # TODO: maybe optionally query for named sources in this location and
        #  list them
        if coords:
            self.logger.warning('Using telescope pointing coordinates. This '
                                'may lead to barycentric timing correction / '
                                'image registration being less accurate.')

        return coords

    # def get_telescope(self):
    #     return 'SALT'

    # # plotting
    # def display(self, **kws):
    #     """Display the data"""
    #     n_dim = len(self.shape)
    #     if n_dim == 2:
    #         from graphing.imagine import ImageDisplay
    #         im = ImageDisplay(self.data, **kws)
    #
    #     elif n_dim == 3:
    #         from graphing.imagine import VideoDisplay
    #         # FIXME: this will load entire data array which might be a tarpit
    #         #  trap
    #         im = VideoDisplay(self.section, **kws)
    #
    #     else:
    #         raise TypeError('Data is not image or video.')
    #
    #     im.figure.canvas.set_window_title(self.filepath.name)
    #     return im


# class SaltiCamObservation(object):
#     # emulate pySHOC.shocObs
#
#     @classmethod
#     def load(cls, filename, **kws):
#         return cls(filename)
#
#     def __init__(self, filename):
#         self._filename = str(filename)
#         self.data = np.lib.format.open_memmap(filename)[:, 1]  # HACK!!
#
#         filename0 = next(filename.parent.glob('*.fits'))
#         header0_bytes, header1_bytes = parse_header(filename0)
#         self.header = Header.fromstring(header0_bytes)
#
#     def filename(self):
#         return self._filename
