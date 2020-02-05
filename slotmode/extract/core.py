"""
This module houses the machinery that enables super fast extraction of
SALTICAM slotmode data from a list of multi-extension FITS files to one big
memory mapped 4D numpy array (or optionally FITS) persisting on disk.
Typical extraction rates are ~100 frames/s/cpu.

Automated detection of noise frames prior to science data as well as trailing
blank frames is done.

Any desired info from the file headers can also be extracted at minimal
overhead.
"""

import io
import os
import re
import mmap
import logging

from pathlib import Path
import multiprocessing as mp

# scientific libs
import numpy as np
from numpy.lib.format import open_memmap

from astropy.io import fits
import itertools as itt
from astropy.io.fits import BITPIX2DTYPE, DTYPE2BITPIX
from astropy.io.fits.header import BLOCK_SIZE

# local libs
import motley
from recipes.logging import LoggingMixin, all_logging_disabled
from recipes import pprint
from recipes.decor.memoize import memoize

# profiling & decorators
from motley.profiling.timers import timer

from .. import N_CHANNELS, ALL_CHANNELS, check_channels
from ..header import _pprint_header
from .utils import regex_maker, prepare_path, truncate_memmap

from recipes.introspection.utils import get_module_name

import tempfile

from multiprocessing.managers import SyncManager

try:
    from tqdm import tqdm
    # use tqdm progress bar ftw!
except ImportError:
    # null object pattern for progressbar
    # FIXME: null object here else fail with multiprocessing and no tqdm ...
    '''def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)'''

# proxy for tqdm progressbar
SyncManager.register("tqdm", tqdm, exposed=('update', 'display', 'close'))
manager = SyncManager()
manager.start()

# module level logger
logger = logging.getLogger(get_module_name(__file__))

# Regex pattern matcher for identifying header blocks
SRE_END = re.compile(rb'END {77}\s*')
SRE_NAXIS = regex_maker(r'naxis(\d)')
SRE_BITPIX = regex_maker('bitpix', capture_keywords=False)
SRE_NEXTEND = regex_maker('nextend', capture_keywords=False)
SRE_DATASEC = regex_maker('datasec')

# Regex matcher for the SALTICAM slot mode filename convention:
# eg.:  'bxgpS201504240003.fits'
SRE_FILENAME = re.compile(r"""
                    (?P<prefix>[^\d]+)
                    (?P<date>(?P<year>20\d\d)(?P<month>[01]\d)(?P<day>[0-3]\d))
                    (?P<nr>[0-9]{4}).fits""",
                          re.VERBOSE)

ex_doc_template = \
    """
    Convenience method for extracting to %s {n, n_ch, n_rows, n_cols}
    
    Parameters
    ----------
    channels
    start
    stop
    data_file
    head_file
    
    Returns
    -------
    
    """


def parse_header(filename, return_map=False):
    """
    Extract primary header and first extension header.
    """
    file_size = os.path.getsize(filename)
    with open(filename, 'rb') as file_obj:
        mm = mmap.mmap(
                file_obj.fileno(), file_size, access=mmap.ACCESS_READ
        )

    # Find the index position of the first extension and data start
    mo = SRE_END.search(mm)
    ext1_start = mo.end()  # extension 1 starts here
    # master header is before ext1_start

    # search forward for end of 1st extension
    mo = SRE_END.search(mm, ext1_start)
    # data_start = mo.end()  # data for extension 1 starts here
    # header of first extension is between *ext1_start* and *data_start*
    headers = mm[:ext1_start], mm[ext1_start:mo.end()]
    if return_map:
        return headers, mm
    return headers


def get_indexing_info(header0, header1):
    # Extract necessary info from primary header and first extension header

    # header0, header1 = parse_header(filename)
    ext_head_size = len(header1)
    ext_data_start = ext_head_size + len(header0)
    n_ext = get_nextend_re(header0)  # nr of fits extensions
    # NB NB!! Get BITPIX value from the first extension header not main header!
    bitpix = int(_get_card(SRE_BITPIX, header1))
    # image dimensions: order of axes on an numpy array are opposite of
    # that of the FITS file.
    return (ext_data_start, ext_head_size, n_ext, bitpix,
            get_image_shape(header1))


def get_header_dtype(headers, header_keys, mm, max_depth=3):
    dtype_dict, missing = check_header(headers[1], header_keys)

    if len(missing):
        # could not find all the requested header keys in the first
        # extension header. check the next extension. This will not
        # happen very often.
        ex = _ExtractorBase.from_headers(*headers)
        # generate indices starting from frame zero which might not be the
        # same frame as the extraction will start from, but should be OK
        # since we are only determining the dtype for the output header info
        # array
        itr = ex.gen_indices(0, 0, 0, 0)
        while len(missing):
            (k, l), byte_pos = next(itr)
            if k == max_depth:
                logger.warning('Could not find keywords %s in the headers of '
                               'the first %i extensions. Ignoring!', missing,
                               max_depth)
                break

            hstart = byte_pos - ex.ext_head_size
            new_dtype_dict, missing = check_header(mm[hstart:byte_pos], missing)
            dtype_dict.update(new_dtype_dict)

    # create dtype for structured array
    head_dtype = np.dtype(list(dtype_dict.items()))
    return head_dtype, tuple(dtype_dict.keys())


def check_header(header, keys):
    # check that these keywords are all in the first extension header so
    # we don't fail in the parallel loop
    if (keys is None) or (len(keys) == 0):
        return {}, ()
    else:
        # convert to upper case
        keys = [_.upper() for _ in keys]  # list(map(str.upper, keys))

    # create regex matcher
    # if dealing with bytes, we have to encode the regex
    key_matcher = regex_maker(keys, encode=isinstance(header, bytes))
    dtype_dict = _get_header_dtype(header, key_matcher)

    # check which keys could not be found:
    # it may be that the requested keyword could not be found in this header
    # (eg. DEADTIME missing from some headers) in which case we do not know
    # the expected byte size for the field. In this case use generic byte
    # size of 15 - this should be enough for most header keyword data,
    # but if your entries are longer than this they will be silently
    # truncated.

    # FIXME: or check next extension header
    # If this key is missing from all headers, corresponding  entry is
    # structured array will be entirely empty. # TODO: fix in cleanup
    missing = set(keys) - set(dtype_dict.keys())
    return dtype_dict, missing


def _get_header_dtype(header, key_matcher):
    #
    output_kind = 'U'
    dtype_fmt = output_kind + '%i'
    # NOTE: encoding as U type requires 4 times the amount of memory that
    #  S type requires. U type more convenient though since less cleanup req
    # FIXME: more space efficient to use bytes array and convert during
    #  cleanup

    # parse the header keys
    dtype_dict = {}
    for mo in key_matcher.finditer(header):
        key, val = mo.groups()
        # numpy dtype does not understand field names that are bytes arrays!
        dtype_dict[key.decode()] = dtype_fmt % len(val)

    return dtype_dict


# todo: may as well memoize!
def get_file_nr(filename):
    # regex matcher for filenames
    filename = Path(filename).name
    match = SRE_FILENAME.match(filename)
    if match is None:
        raise ValueError('Could not extract file sequence nr for filename %r' %
                         filename)
    return int(match.group(3))


def is_blank(image, pre_check_indices):
    """
    Fast check for blank (all zero) frames by checking the last column in the
    image

    Parameters
    ----------
    image
    pre_check_indices

    Returns
    -------

    """
    if np.allclose(image[pre_check_indices], 0):
        # this will almost always be False
        # if True check full frame
        return np.allclose(image, 0)
    return False


def _get_card(sre, header_bytes):
    mo = sre.search(header_bytes)
    if mo is None:
        from recipes.string import matchBrackets
        name = matchBrackets(SRE_NEXTEND.pattern.decode())[0].strip('?:')
        raise ValueError('%s card not found!' % name)
    return mo.group(1)


def get_nextend_re(header_bytes):
    return int(_get_card(SRE_NEXTEND, header_bytes))


def get_image_shape(header):
    *_, shape = zip(*SRE_NAXIS.findall(header))
    return tuple(map(int, shape[::-1]))


def is_acquisition_image(mheader, ext1header):
    # There are probably numerous equivalent ways of identifying if a fits
    # file is an acquisition image.  Here we do the following:
    # acquisition images have 4 extensions. possible (but unlikely)
    # that science frames also have 4 ext, so check for DATASEC in header of
    # the first extension

    if get_nextend_re(mheader) == 4:
        #
        mo = SRE_DATASEC.search(ext1header)
        if mo is None:
            # 'DATASEC' not in header -- most likely an acquisition image
            return True

        # # dsec = ext1header['datasec']
        # y_extent = int(dsec.strip('[]').split(',')[1].split(':')[1])
        # return y_extent >= CHANNEL_SIZE[1]  # acquisition image!
    return False


@memoize
def get_mmap(filename):
    # create memory map for input file
    file_size = os.path.getsize(filename)
    with open(filename, 'rb') as file_obj:
        # since the image data is non-contiguous. Use `mmap.mmap` instead
        # of `np.memmap`
        return mmap.mmap(file_obj.fileno(), file_size,
                         access=mmap.ACCESS_READ)


def first_science_stack(file_list):
    # The first image in the list may be an acquisition image, which we
    # don't want to extract. Check whether this is an acquisition image
    for i, filename in enumerate(file_list):
        headers, mm = parse_header(filename, return_map=True)
        if is_acquisition_image(*headers):
            continue  # the next one should be the first science frame

        # at this point we have the first science file
        return i, Path(filename), headers, mm


def get_date(filename):
    """
    Regex search for date in fits header of `filename`

    Parameters
    ----------
    filename

    Returns
    -------

    """
    return SRE_FILENAME.search(Path(filename).name).group(2)


class SlotModeExtract(LoggingMixin):
    """
    High level interface for SaltiCam slotmode image (video) extraction and
    conversion.

    Salticam data typically comes off the SALT pipeline in a list of
    multi-extension fits files. This format is not particularly amenable to
    computational work since it is not contiguous (image data is interspersed
    with header information), and is not very efficient (header information
    is duplicated unnecessarily for each image / amplifier channel.

    This class supports fast (multiprocessed) flexible extraction of data from
    said egregious list of input fits files to multiple user specified output
    formats.

    Typically output consists of 2 (memory mapped) arrays:
        The first contains the actual image data as a multidimensional array.
        The second contains the header keywords extracted from each image
        extension.

    Assuming all the files in the input list are fits compliant and have the
    same basic structure, data extraction can be done ~10**6 times faster
    than it can with pyfits (even without multiprocessing!!) by skipping
    unnecessary checks etc and mapping fits image data directly into one big
    contiguous numpy array.
    """

    outputFilenameTemplate = 's{date:s}'
    fileExtensionHeaderInfo = '.info'

    def __init__(self, file_list, loc=None, header_keys=(), overwrite=True):

        # parse the input list
        file_list = sorted(file_list)
        if not len(file_list):
            raise ValueError('Input file list is empty.')

        # show files
        self.logger.info('Received list of %i files.' % len(file_list))

        # advance to first science observation
        i, path0, headers, mm = first_science_stack(file_list)

        # keyword extraction setup
        self.head_dtype, self.header_keys = get_header_dtype(
                headers, header_keys, mm)

        # key_matcher = regex_maker(header_keys)
        if len(self.head_dtype):
            logger.info('Values for %i keywords: %s to be extracted',
                        len(self.head_dtype), tuple(self.header_keys))
        else:
            logger.info('No header keywords to be extracted')

        # noinspection PyUnboundLocalVariable
        self.file_list = file_list[i:]
        self.overwrite = bool(overwrite)

        # create output folder if needed
        loc = path0.parent if loc is None else Path(loc)

        # output filename template
        self.loc = loc
        self.date = get_date(path0)
        self.basename = \
            self.loc / self.outputFilenameTemplate.format(date=self.date)

        # need some info from header
        # noinspection PyUnboundLocalVariable
        n_ext = get_nextend_re(headers[0])
        self.n_images_per_amp_file = n_ext // N_CHANNELS
        # self.headers = tuple(map(bytes.decode, headers))
        self.headers = headers
        self.image_shape = get_image_shape(headers[1])
        self.ex = None  # placeholder for extraction class

    @property
    def n_files(self):
        return len(self.file_list)

    @property
    def n_total_frames(self):
        return self.n_files * self.n_images_per_amp_file

    def pprint(self):
        n = self.n_files * self.n_images_per_amp_file
        _pprint_header(*self.headers, n=n)

    def get_file_nrs(self, indices):
        """
        Map filename indices to a file number given in the name.
        For example: Given a list of filenames:
            bxgpS201504240003.fits
            bxgpS201504240004.fits
            ...
            bxgpS201504240403.fits
            bxgpS201504240404.fits

        `get_file_nr(1)`  #returns 4
        `get_file_nr(-1)` #returns 404
        """
        return np.vectorize(self._get_file_nr, 'i')(indices)

    def _get_file_nr(self, index):
        return get_file_nr(self.file_list[index])

    def get_shape(self, start, stop, channels):
        # get data dimensions
        n_tot = self.n_total_frames
        data_pre = info_pre = None
        if start is None:  # detect
            n_dud, data_pre, info_pre = self.detect_noise_frames(channels)
            start = len(data_pre)
            data_pre = data_pre[n_dud:]
            info_pre = info_pre[n_dud:]
            if stop is None:
                stop = n_frames = n_tot
            else:
                # interpret as total number of frames to extract
                n_frames = stop
                stop += n_dud

        else:
            stop = min(n_tot, stop or n_tot)
            n_frames = stop - start

        #
        shape = (n_frames, len(channels)) + self.image_shape
        self.logger.info('Using subset: %s', (start, stop))
        return shape, start, stop, data_pre, info_pre

    def detect_noise_frames(self, channels, n=12, threshold=5):
        # detect initial white noise frames
        _, tmp = tempfile.mkstemp('.npy')
        _, tmp_i = tempfile.mkstemp('.info')
        # load first `n` frames
        self.logger.info('Pre-extracting %i frames from %i amplifier channels '
                         'to %r for white noise detection.', n, len(channels),
                         str(tmp))

        # temporarily disable info logger during pre extraction
        # FIXME: any exception that happens in the next line will not raise!!!
        with all_logging_disabled():
            data, _, info = self.to_array(channels, 0, n, tmp, tmp_i,
                                          False, False)

        # simple threshold test: typically, the standard deviation of image data
        # is 10+ times larger
        std = np.std(data, (2, 3))
        n_dud = np.where(np.all((std / std[0]) > threshold, 1))[0][0]
        if n_dud:
            self.logger.info('Detected %i white noise frames at the start of '
                             'the image stack. Ignoring these.', n_dud)
        return n_dud, data, info

    def _init_mem(self, ex, name, shape, header_name=None):
        #
        header, offset = ex.get_header(name, shape, self.headers[0])

        # create shared memory for output frames / info
        n_frames, n_ch, *_ = shape
        self.logger.info('Extracting %i frames from %i amplifier channel%s to '
                         '%r', n_frames, n_ch, 's' * (n_ch > 1), str(name))

        # check free hd space
        req_bytes_head = self.head_dtype.itemsize * n_frames * n_ch
        req_bytes_data = (ex.image_size_bytes * n_frames * n_ch) + offset
        self.check_free_space(req_bytes_data, req_bytes_head)

        # create memory map for extraction (4D)
        data = np.memmap(name, ex.dtype, 'w+', offset, shape)
        # FIXME: w+ will always overwrite, r+ fails on create

        # header info data
        header_data = None
        if header_name:
            # read the extracted keys to structured memory map
            header_data = open_memmap(str(header_name), 'w+', self.head_dtype,
                                      (n_frames, n_ch))
        return data, header, header_data

    def check_free_space(self, req_bytes_data, req_bytes_head=0):
        statvfs = os.statvfs(self.loc)
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        req_bytes = req_bytes_head + req_bytes_data

        if free_bytes < req_bytes:
            raise Exception('Not enough space on disc: Extraction requires '
                            '%s, you only have %s available at location %r',
                            pprint.eng(req_bytes, unit='b'),
                            pprint.eng(free_bytes, unit='b'), str(self.loc))

        self.logger.info('Expected file size: %s (header: %s  + data: %s)',
                         pprint.eng(req_bytes, unit='b'),
                         pprint.eng(req_bytes_head, unit='b'),
                         pprint.eng(req_bytes_data, unit='b'))

    def run(self, ext, channels=ALL_CHANNELS, start=None, stop=None,
            data_file=None, head_file=None, clean=True, progress_bar=True):
        """
        Main extraction routine.

        Parameters
        ----------
        ext: str, {'fits', 'npy'}
            file extension. Determines the type of output file.
        channels: int or tuple of int
            indices for channels to extract
        start: int
            requested starting frame index for extraction. If not given,
            this will automatically pick the first useful image data (first
            frame that is not just white noise)
        stop: int
            requested final frame index for extraction. If not given,
            the last useful image frame will be used (that is not just all zero)
        data_file:
            output filename for data file. If not given, filename will
            automatically be created according to the `basename` attribute
            (which is created using `outputFilenameTemplate` class attribute)
        head_file:
            output filename for header info file. If not given, it will be
            created using the `basename` attribute.
        clean: bool
            whether or not to run the cleanup routine post extraction.
        progress_bar: bool
            whether or not to display a progressbar during extraction.

        Returns
        -------

        """
        cls = EXTRACTORS[ext]
        channels = check_channels(channels)

        # init output files and folders
        if data_file is None:
            if len(channels) == 1:
                # single channel extraction. put channel number in filename
                data_file = self.basename.with_suffix(f'.ch{channels[0]}.{ext}')
            else:
                data_file = self.basename.with_suffix(f'.{ext}')
        data_file = prepare_path(data_file, self.loc, self.overwrite)

        # if no header file given, and header key extraction requested
        if (head_file is None) and len(self.header_keys):
            head_file = self.basename.with_suffix(self.fileExtensionHeaderInfo)

        # if no header keys provided at init, but filename given, ignore file
        if len(self.header_keys) == 0:
            head_file = None
        else:
            head_file = prepare_path(head_file, self.loc, self.overwrite)

        # init extractor
        self.ex = ex = cls.from_headers(*self.headers,
                                        header_keys=self.header_keys)

        # pre-run to detect noise frame and get final output shape
        detect_start = (start is None)
        shape, start, stop, data_pre, info_pre = self.get_shape(start, stop,
                                                                channels)
        # data_pre contains the first useful image frames only (ignoring
        # white noise frames)

        # allocate memory
        data, header, info = self._init_mem(ex, data_file, shape, head_file)

        # get frame offset for dud frame detections
        if detect_start:
            out_start = len(data_pre)  # index of starting frame for data
            data[:out_start] = data_pre
            info[:out_start] = info_pre
        else:
            out_start = 0

        # do the work! (write data)
        self.logger.info('Extracting data. This may take a few moments...')
        data, info = ex.loop_mp(self.file_list, start, stop, out_start, data,
                                info, channels, progress_bar)

        if clean:
            return self.cleanup(ex, data, header, info, channels)

        return data, header, info

    # @timer  # FIXME: timer empty print statements messing with tqdm
    def cleanup(self, ex, data, header, info, channels):
        """
        Cleanup the extracted data and header files by removing trailing
        blank frames.

        Parameters
        ----------
        ex
        data
        header
        info
        channels

        Returns
        -------

        """
        self.logger.info('Extraction done. Cleaning up.')

        # some slot mode files are have all zeros at the end. Strip those
        # along with corresponding header entries
        ignore = np.zeros(len(data), bool)
        # check single column for all zeros to flag blank frames
        check_col = (slice(None), ex.image_shape[1] // 2)
        for n_blank, image in enumerate(data[::-1, 0]):
            if not is_blank(image, check_col):
                break

        # noinspection PyUnboundLocalVariable
        if n_blank:
            # noinspection PyUnboundLocalVariable
            self.logger.info('Detected %i blank images at the end of stack',
                             n_blank)
            n = len(data)
            ignore[n - n_blank: n] = True

        # Identify and remove bad frames at the end of run
        if 'DATE-OBS' in info.dtype.fields.keys():
            ut_date = info['DATE-OBS'][:, 0]
            l1969 = (ut_date == '1969-12-31')
            # The last ~50 frames have this date for some reason
            ignore[l1969] = True

        # remove bad frames
        idx_rmv, = np.where(ignore)
        data, header = ex.finalise(data, header, idx_rmv, channels)

        if info is not None:
            info = self.cleanup_info(info, idx_rmv)

        return data, header, info

    def cleanup_info(self, info, indices_remove, write=True):
        """
        Cleanup extracted header info by removing duplicate (by channel) data

        Parameters
        ----------
        info: memmap
            structured array with header info
        indices_remove: array-like of int
            Trailing indices that will be removed from array
        write: bool
            whether to rewrite the file to disk after cleanup
        Returns
        -------

        """
        filename = info.filename
        if len(indices_remove):
            info = info[:-len(indices_remove)]

        # header value are most likely duplicate for each amplifier
        duplicate = np.all(info == info[:, 0, None], 1)
        if duplicate.all():
            # no need to return redundant info
            self.logger.info(
                    'Extracted header info is duplicate for %i amplifier '
                    'channels. Removing duplicate channels.',
                    info.shape[1])
            info = info[:, 0]
        else:
            self.logger.info('Extracted header info unique across '
                             'channels. Keeping header data as %iD.',
                             info.ndim)

        # return header data as record array with lower case field titles
        # with '-OBS' stripped for convenient access
        # eg. info.utc gives info['UTC-OBS']
        new_dtype = np.dtype([((n.lower().strip('-obs'), n), v)
                              for n, v in self.head_dtype.descr])

        if write:
            # use file obj instead of str to avoid npy extension. might not
            # be best policy??
            with open(filename, 'wb') as fp:
                np.save(fp, info)

        return np.rec.array(info, new_dtype)

    def to_fits(self, channels=ALL_CHANNELS, start=None, stop=None,
                data_file=None, head_file=None, clean=True, progress_bar=True):
        # don't put a docstring here
        return self.run('fits', channels, start, stop, data_file, head_file,
                        clean, progress_bar)

    to_fits.__doc__ = ex_doc_template % 'FITS file'

    def to_array(self, channels=ALL_CHANNELS, start=None, stop=None,
                 data_file=None, head_file=None, clean=True, progress_bar=True):
        # don't put a docstring here
        return self.run('npy', channels, start, stop, data_file, head_file,
                        clean, progress_bar)

    to_array.__doc__ = ex_doc_template % 'numpy.ndarray'

    # def burst(self):
    #     # split into single frame fits images.  This is really silly,
    #     # why would you do this??!
    #     raise NotImplementedError  # TODO

    # def to_multiext(self):
    #     # Convert to single, multi-extention fits file
    #     raise NotImplementedError  # TODO


class _ExtractorBase(LoggingMixin):
    """
    Internal base class for extraction to FITS or npy
    """

    @classmethod
    def from_file(cls, filename, **kws):
        return cls(*parse_header(filename), **kws)

    @classmethod
    def from_headers(cls, header0, header1, *args, **kws):
        return cls(*get_indexing_info(header0, header1), **kws)

    def __init__(self, data_start_bytes, ext_head_size, n_extensions,
                 bits_per_pixel, image_shape, header_keys=None):
        """

        Parameters
        ----------
        data_start_bytes
        ext_head_size
        n_extensions
        bits_per_pixel
        image_shape
        header_keys
        """
        self.ext_start = int(data_start_bytes)
        self.ext_head_size = ext_head_size
        self.n_ext = n_extensions
        self.n_images_per_amp_file = self.n_ext // N_CHANNELS
        self.image_shape = rows, cols = image_shape
        self.image_size_bytes = abs(bits_per_pixel) * rows * cols // 8
        self.dtype = np.dtype(BITPIX2DTYPE[bits_per_pixel]).newbyteorder('>')

        # frame range
        # self.start = 0
        # self.idx_data_off = 0

        # figure out the size of a data block:
        # NOTE: from here on outward we assume FITS compliance -- i.e. file
        #  is divided in chunks of 2880 bytes. If this is not the case,
        #  you will end up with garbage data!
        n_blocks_per_ext = np.ceil(
                (self.ext_head_size + self.image_size_bytes) / BLOCK_SIZE)
        self.ext_size = int(BLOCK_SIZE * n_blocks_per_ext)

        # finally make pattern matcher for parallel extraction (no keyword
        # capture)
        self.key_matcher = regex_maker(header_keys)

        # progressbar placeholder
        self.bar = None

    def gen_indices(self, start, stop, channels, idx_data_start):
        """
        Generate indices for output file frame number, channel number,
        byte position for data

        Parameters
        ----------
        start:  int
            starting frame index relative to current file
        stop:  int
            stopping frame index relative to current file
        channels: int or tuple of int
            amplifier channels to use
        idx_data_start: int
            Starting frame index for output

        Yields
        ------
        frame index
        channel index
        byte position for start of image
        """
        # map indices from input file number to output (frame, channel, n_byte)
        # index
        # frame_offset is relative to output cube

        # idx_data_start = i * self.n_images_per_amp_file + self.idx_data_off
        # print('\n---->', start, stop, idx_data_start)
        out_frame = itt.count(idx_data_start)
        for j, k in zip(range(start, stop), out_frame):
            # j is frame number relative to this file
            ext_offset = j * N_CHANNELS
            # now we need it relative to extracted array
            for l, ch in enumerate(channels):
                ext_nr = ch + ext_offset
                yield (k, l), self.ext_start + ext_nr * self.ext_size

    # def _file_pos(self, j, k):
    #     ext_nr = k + (j % self.n_images_per_amp_file) * N_CHANNELS
    #     return self.ext_start + ext_nr * self.ext_size

    # class _Extract(_ExtractorBase):
    #     def __init__(self, *args, keys):
    #         super().__init__(*args)

    def iter_files(self, files, start, stop, out_start):
        """
        Generator that for each file in the sequence creates the indices
        needed for the extraction of the data

        Parameters
        ----------
        files
        start
        stop
        out_start

        Yields
        -------
        file number
        file name
        start frame in file
        stop frame in file
        starting index in output for current file

        """
        n = self.n_images_per_amp_file
        #  get list of files corresponding to frame interval (start, stop)
        # make sure we have native int: itt.islice raises uninformative error
        #  with any np.integer
        frame_start = int(start // n)
        frame_stop = int(np.ceil(stop / n))
        n_1 = frame_stop - frame_start - 1
        start_n = start % n
        yield from zip(
                # file number
                range(frame_start, frame_stop),
                # file name
                itt.islice(files, frame_start, frame_stop),
                # start frame in file
                itt.chain(iter((start_n,)), itt.repeat(0, n_1)),
                # stop frame in file
                itt.chain(itt.repeat(n, n_1), iter((((stop % n) or n),))),
                # start index data
                itt.chain(iter((out_start,)),
                          range(n - start_n + out_start, stop, n))
        )

    # @profiler.histogram()
    @timer
    def loop(self, files, start, stop, out_start, data, info, channels,
             progress_bar):
        """
        Loop over the files and extract the data and optional header keywords


        Parameters
        ----------
        files: list
            list of file names
        start: int
            starting frame index
        stop:
            stopping frame index
        out_start:
            starting index for output data
        data
        info
        channels
        progress_bar

        Returns
        -------

        """

        with tqdm(total=len(data), disable=not progress_bar, ncols=120,
                  unit='frames', unit_scale=len(channels)) as self.bar:
            for i, filename, i_start, i_stop, i_start_out in \
                    self.iter_files(files, start, stop, out_start):
                self.loop_file_safe(i, filename, i_start, i_stop, channels,
                                    i_start_out, data, info)

        return data, info

    # @timer
    def loop_mp(self, files, start, stop, out_start, data, info, channels,
                progress_bar):
        """
        Loop over the files and extract the data and optional header keywords
        """
        from joblib import Parallel, delayed

        # fixme: multiprocessing seems to be slower than single process
        #  not sure why
        n_jobs = 1  # mp.cpu_count()

        # noinspection PyUnresolvedReferences
        self.bar = manager.tqdm(total=len(data), disable=not progress_bar,
                                ncols=120, unit=' frames', mininterval=0.2,
                                initial=out_start)
        # the progressbar does above does not work as a context manager with
        # multiprocessing:
        # _pickle.PicklingError: Can't pickle <function <lambda>:
        # attribute lookup <lambda> on jupyter_client.session failed

        with Parallel(n_jobs=n_jobs) as parallel:
            parallel(delayed(self.loop_file_safe)(
                    i, filename, i_start, i_stop, channels, i_out0, data, info)
                     for i, filename, i_start, i_stop, i_out0
                     in self.iter_files(files, start, stop, out_start))
        self.bar.close()
        return data, info

    def loop_file_safe(self, i, filename, i_start, i_stop, channels, i_out0,
                       data, info):
        """
        Loop through the images in a single file and write them to `data`,
        write the header info to the `info` array. Builtin exception trap
        catches and logs any exceptions that might be raised

        Parameters
        ----------
        i: int
            file index
        filename: str
            name of the file
        i_start: int
            starting frame relative to this file
        i_stop: int
            stopping frame relative to this file
        channels: int or tuple of int
            amplifier channels
        i_out0: int
            index position in output array for first frame from this file
        data:
            output array
        info:
            output array for header info

        Returns
        -------

        """
        itr = self.gen_indices(i_start, i_stop, channels, i_out0)

        # inner loop with exception trap
        filename = Path(filename)  # FIXME: should be read in as Paths
        name = filename.name
        logging.debug('Extracting frames from file %i: %r', i, name)
        try:
            self._loop_file(filename, itr, data, info)
            # print(f'\n{i}: {motley.green(name)}', end='')
        except Exception as err:
            self.logger.exception(f'{i}: {motley.red(name)}')
            raise

    def _loop_file(self, filename, itr, data, info):
        # loop through the frames in a single file and write them to `data`
        file_size = os.path.getsize(filename)
        with open(filename, 'rb') as file_obj:
            # since the image data is non-contiguous. Use `mmap.mmap` instead
            # of `np.memmap`
            memmap = mmap.mmap(file_obj.fileno(), file_size,
                               access=mmap.ACCESS_READ)
        for (j, k), byte_pos in itr:
            # extract frame
            self._proc(data, j, k, memmap, byte_pos)

            if info is not None:
                self._proc_info(info, j, k, memmap, byte_pos, filename)

            # update progress bar
            if k == 0:
                self.bar.update()
            self.logger.debug('Done %i', j, filename.name)

    def _proc(self, data, j, k, mm, byte_pos):  # get_image?
        # get image from input file and write to it memmap
        data[j, k] = np.frombuffer(
                mm[slice(byte_pos, byte_pos + self.image_size_bytes)],
                self.dtype).reshape(self.image_shape)

    def _proc_info(self, info, j, k, mm, byte_pos, filename):
        # header for this extension (actually only need to
        # extract the values for one of the channels since they
        # are duplicated)
        hstart = byte_pos - self.ext_head_size
        matched = self.key_matcher.findall(mm[hstart:byte_pos])
        if len(matched) == 0:
            self.logger.warning(
                    'Null match in header of file: %r'
                    'frame = %i; channel = %i; byte position = %i',
                    filename, j, k, byte_pos)
            raise ValueError()
        else:
            # note. below won't work for len(keys) < info.shape[1]
            # keys, vals = zip(*matched)
            # info[list(keys)][j, k] = vals
            # for loop as work around
            for key, val in matched:
                info[key.decode()][j, k] = val

    def finalise(self, data, header, indices_remove, channel):
        raise NotImplementedError


class _ExtractArray(_ExtractorBase):
    """
    Class for extracting salticam data to numpy array using the npy format
    """
    fileExtension = '.npy'

    def get_header(self, data_file, shape, original_header):
        """
        Create header for numpy array

        Parameters
        ----------
        data_file
        shape
        original_header

        Returns
        -------

        """
        # use `npy` format (see `np.lib.format.open_memmap`)
        # save file header with some meta data (shape, dtype etc) - enables
        # easy loading.
        fmt = np.lib.format
        d = dict(descr=fmt.dtype_to_descr(self.dtype),
                 fortran_order=False,
                 shape=shape)

        with io.BytesIO() as s:
            fmt._write_array_header(s, d, (2, 0))  # returns version
            return s, s.tell()

    def update_header(self, data, header, channels):
        # noinspection PyProtectedMember

        # update the header for npy format so the array can easily be read
        # with np.open_memmap

        fmt = np.lib.format
        with open(data.filename, 'rb+') as fp:
            version = fmt.read_magic(fp)
            fmt._check_version(version)

            _, fortran_order, dtype = fmt._read_array_header(fp, version)
            d = dict(descr=fmt.dtype_to_descr(dtype),
                     fortran_order=fortran_order,
                     shape=data.shape)
            fp.seek(0)
            used_ver = fmt._write_array_header(fp, d, version)

    def finalise(self, data, header, idx_rmv, channels):
        # remove trailing frames
        not_in_seq = (np.diff(idx_rmv) != 1)
        # non_sequential = np.any(not_in_seq)
        if np.any(not_in_seq):
            self.logger.warning('Non-sequential ignore frames: %s\nNot '
                                'cleaning.' % idx_rmv[not_in_seq])
            idx_rmv = []
        elif len(idx_rmv):
            # ignore blank frames
            self.logger.info('Stripping %i frames: %i -> %i.',
                             len(idx_rmv), idx_rmv[0], idx_rmv[-1])

        # remove bad frames, re-write header, fits compliance etc
        # self.logger.debug('Finalising')
        new_data = truncate_memmap(data, len(idx_rmv))

        # remove singleton dimensions. note .squeeze() does not return memmap
        new_data = new_data.reshape(
                tuple((d for d in new_data.shape if not d == 1)))
        header = self.update_header(new_data, header, channels)
        return new_data, header


class _ExtractFITS(_ExtractArray):
    """
    Class for extracting salticam data to FITS
    """

    fileExtension = '.fits'

    def get_header(self, data_file, shape, original_header):
        """
        Create the FITS header for output

        Parameters
        ----------
        data_file
        shape
        original_header

        Returns
        -------

        """
        # include master header
        header = fits.Header.fromstring(original_header.decode())
        for key in ('NSCIEXT', 'NEXTEND'):  # EXTEND
            header.pop(key)

        # add data dimensions. just placeholders for now so we can get
        # accurate header size before initializing memory
        ndim = len(shape)
        for i in range(1, ndim + 1):
            header.insert(f'NAXIS{(i - 1) or ""}', f'NAXIS{i}', 0, after=True)

        return header, len(header.tostring())

    def update_header(self, data, header, channels):

        self.logger.debug('Updating header')

        # make FITS compliant
        header['NAXIS'] = data.ndim
        for i, d in enumerate(data.shape[::-1], 1):  # FITS axes order backwards
            header[f'NAXIS{i}'] = d

        # Set dtype to image dtype (original bitpix does not get updated by
        # SALT pipeline when data is edited !?!?)
        header['BITPIX'] = DTYPE2BITPIX[data.dtype.name]

        # For single channel extraction, write channel number to header
        if len(channels) == 1:
            header.set('CHANNEL', channels[0],
                       'Amplifier channel for this data set',
                       after='NCCDS')

        # write header to file
        with open(data.filename, 'rb+') as fp:
            fp.write(header.tostring().encode())

        return header

    def finalise(self, data, header, indices_remove, channels):
        """
        Ensure FITS compliance by padding to multiple of block size 2880 bytes
        """
        data, header = super().finalise(data, header, indices_remove, channels)

        with open(data.filename, 'rb+') as fp:
            # pad whitespace at end of file for fits compliance by padding up to
            # multiple of 2880 bytes
            size = fp.seek(0, 2)
            trailing_bytes = (size % BLOCK_SIZE)
            if trailing_bytes:
                padding = BLOCK_SIZE - trailing_bytes
                fp.write(b' ' * padding)

        return data, header


# extractor classes by file type
EXTRACTORS = dict(npy=_ExtractArray,
                  fits=_ExtractFITS)
