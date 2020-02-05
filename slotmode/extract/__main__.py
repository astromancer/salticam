"""
Main script for extracting SALTICAM slot mode data into single file (fits or
npy) and calculating fixed timestamps.  Run this script by calling python on
the parent directory:
>>> python salticam/slotmode/extract path/to/product/directory
or if you prefer to use the interactive ipython / jupyter shell
>>> %run -m salticam.slotmode.extract path/to/product/directory
"""

import logging
import os

from matplotlib import rc

from recipes.introspection.utils import get_module_name
from recipes.io.utils import WarningTraceback

# module level logger
logger = logging.getLogger(get_module_name(__file__))

# stack trace for warnings
wtb = WarningTraceback()


# @timer
def setup():
    from sys import argv
    import argparse
    from pathlib import Path
    from recipes.io import parse

    # TODO: add verbosity argument

    # TODO: --keys='FOO'.  --keys-only

    parser = argparse.ArgumentParser(
            description='Extraction for SALTICAM slot mode data.')

    # Positional arguments
    help_input = """\
    Science data to be processed. Argument(s) can be one of the following:' 
        1) a explicit list of file names
        2) a directory name (all ".fits" files in the folder will be read)
        3) name of file containing a list of the input items
        4) file glob expression eg: '*.fits'"""
    parser.add_argument(
            'files_or_directory', nargs='*',  # metavar='N',
            help=help_input
    )
    # Optional arguments
    parser.add_argument('-i', '--images', nargs='+', type=str,
                        help=help_input)
    parser.add_argument(
            '-o', '--output-dir', type=str,
            help=('The data directory where the reduced data is to be placed.'
                  'Defaults to input dir.'))
    parser.add_argument(
            '-x', '--overwrite', choices=['y', 'n'], default='y',
            help='Whether or not to overwrite existing files.')
    # channels
    parser.add_argument(
            '-t', '--timing-only', action='store_true',
            help='Switch for exclusively running time stamp fixes on already '
                 'extracted data.')
    # channels
    parser.add_argument(
            '-ch', '--channels', nargs='+', default=ALL_CHANNELS,
            help='Amplifier channel(s)')

    # selection of frame / file subsets
    subset_args = parser.add_mutually_exclusive_group()
    file_sub_help = (
        "File subset. Useful since glob expressions can't really be used to "
        "select range of files in sequence. Also Useful for testing/debugging."
        """\
        Arguments are as follows:
            If not given, entire list of files will be used. 
            If a single integer `k`, first `k` files will be used.
            If 2 integers (k,  l), all files starting at `k` and ending at `l-1`
            will be used.""")
    subset_args.add_argument(
            '-m', '--file-subset', nargs='*', default=[None], type=int,
            help=file_sub_help)

    subset_help = (
        "Data subset to load. Arguments are interpreted as for --file-subset ("
        "-m) save that if a single integer is given, this is taken to be the "
        "total number of desired frames.  Since the first ~6 frames are "
        "usually dud frames (white noise), the actual range of frames extracted"
        " from the raw data will be (n_dud, k + n_dud) where `n_dud`"
        " is the detected number of dud frames.")
    subset_args.add_argument(
            '-n', '--subset', nargs='*', default=[None], type=int,
            help=subset_help)

    # output types
    output_types = parser.add_mutually_exclusive_group()
    output_types.add_argument('--fits', action='store_true', default=False,
                              help='Output data will be in fits format')
    output_types.add_argument('--npy', action='store_true', default=False,
                              help='Output data will be in npy format')
    # TODO: all the other types npz, burst

    # plotting
    plotting_args = parser.add_mutually_exclusive_group()
    plotting_args.add_argument('--plot', action='store_true', default=True,
                               help='Do plots')
    plotting_args.add_argument('--no-plots', dest='plot', action='store_false',
                               help="Don't do plots")

    args = parser.parse_args(argv[1:])

    # Interpret the parsed arguments
    if args.files_or_directory and not args.images:
        args.images = args.files_or_directory

    # read list of files
    files = parse.to_list(args.images,
                          os.path.exists,
                          include='bxgp*.fits',
                          abspaths=True,
                          raise_error=1, sort=True)

    # Check & set input and output paths
    if not len(files):
        raise IOError('Could not resolve file list: %r' % args.images)

    # set IO paths from input file list if not provided
    product_dir = Path(files[0]).parent
    root_folder = product_dir.parent
    if args.output_dir is None:
        args.output_dir = root_folder / 'reduced'

    out_path = Path(args.output_dir).resolve()
    if not out_path.exists():
        out_path.mkdir()

    # setup logging to file
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(out_path / 'extract.log')
    # fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    #
    logger.info('Output folder: %s', args.output_dir)

    # select subset of files
    slice_ = slice(*args.file_subset)
    if slice_.start or slice_.stop:
        logger.info('Requested subset: %s',
                    (slice_.indices(len(files)[:2])))

    files = files[slice_]
    if not len(files):
        raise IOError('selecting subset %r returned empty list' %
                      args.file_subset)

    # Set global plotting parameters
    rc('savefig', directory=str(out_path))

    return root_folder, out_path, files, args


def fix_timing(info, coords, filename, plot=False):
    # Construct time data
    # `info` may be shaped (n, n_amps). Times are simultaneous, so use
    # first row sufficient
    ix_amp = 0 if info.ndim > 1 else ...
    utc, ut_date = info.utc[:, ix_amp], info.date[:, ix_amp]

    # check for date change
    assert np.all(ut_date == ut_date[0]), 'DATE CHANGES'

    # construct time object (raw times as read from header)
    # need a location for barycentrization
    sutherland = EarthLocation(lat=-32.376006, lon=20.810678, height=1771)
    t = Time.isomerge(ut_date, utc, precision=9, location=sutherland)

    # convert to seconds from midnight UTC.  Fix incorrect frame times.
    # Converting to floats instead of TimeDelta for performance..
    ut_sec = t.time_from_local_midnight('s')
    t_fix, *ui = fix_timestamps(ut_sec.value, info.deadtime,
                               plot=plot,
                               save_path=fig_path)
    t = reconstruct_times(t_fix, ut_date)

    # export times
    save_times(filename, t, coords)  # info.exptime,
    return t, ui


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    from astropy.io.fits.header import Header
    from astropy.coordinates import EarthLocation, SkyCoord

    from pySHOC.timing import Time
    from salticam.slotmode.extract.core import (SlotModeExtract, ALL_CHANNELS,
                                                )
    from salticam.slotmode.timing import fix_timestamps, reconstruct_times, \
        save_times

    # TODO: resume - need special handling for info array.
    #
    # TODO: tool that can display entire slot with optional layouts
    # layout=2x2, 1x4, 4x1 etc
    # SlotDisplay

    _, out_path, file_list, args = setup()
    fig_path = out_path / 'figs'

    # default header keys to extract
    keys = 'UTC-OBS', 'DATE-OBS', 'EXPTIME', 'DEADTIME'
    # these keys will be accessible on the `info` array as lower case with
    # -OBS stripped

    # initialize extraction class
    fx = SlotModeExtract(file_list, header_keys=keys, loc=out_path,
                         overwrite=(args.overwrite == 'y'))
    # print info from header. source name date, etc
    fx.pprint()
    # TODO: plot some of the environmental stuff from headers. eg track?

    if args.timing_only:
        # FIXME: does this really fit here??
        # load previously extracted data from file
        ext = SlotModeExtract.fileExtensionHeaderInfo
        try:
            info_file = next(out_path.glob(f'*{ext}'))
        except StopIteration as err:
            raise IOError(
                    'Previously extracted header info file (extension %r) '
                    'could not be found at location: %r' %
                    (str(ext), out_path)) from err

        # noinspection PyTypeChecker
        info = np.rec.array(np.load(info_file, 'r'))
    else:
        # Extract the data
        subset = slice(*args.subset)
        data, header, info = fx.to_fits(args.channels, start=subset.start,
                                        stop=subset.stop)

    # Object coordinates (for barycentrization)
    header = Header.fromstring(fx.headers[0].decode())
    ra, dec, equinox = map(header.get, ('ra', 'dec', 'equinox'))
    coords = SkyCoord(ra, dec, unit=('h', 'deg'))  # equinox=equinox

    # fix timestamps
    time_file = str(fx.basename.with_suffix('.time'))
    t, ui = fix_timing(info, coords, time_file, args.plot)

    if ui:
        ui[0].show()
    plt.show()
