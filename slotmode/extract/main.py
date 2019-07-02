from scipy.stats import mode
from astropy.coordinates import EarthLocation, SkyCoord
from matplotlib import rc
from matplotlib.transforms import blended_transform_factory as btf

from tsa.gaps import get_delta_t  # , get_delta_t_mode
from obstools.airmass import altitude, Young94
from pySHOC.timing import Time, get_updated_iers_table, light_time_corrections

from graphical.ts import TSplotter
from graphical.hist import hist
from graphical.imagine import ImageDisplay

from motley.profiler.timers import timer

import numpy as np
import logging
import warnings

import os
from astropy.io import fits
from recipes.introspection.utils import get_module_name


@timer
def savefig(fig, path):
    fig.savefig(str(path))


@timer
def show_jitter_stats(delta_t, kct, title='Time jitter'):
    """Plots of time jitter statistics."""
    # plot delta_t as TS + histogram
    tsplt = TSplotter()
    plot_props = dict(fmt='ro')
    hist_props = dict(log=True, bins=100)  # 10 millisecond time bins
    fig, pl, *rest = tsplt.plot(delta_t,
                                errorbar=plot_props,
                                hist=hist_props,
                                title=title,
                                axlabels=[r'$N_{frame}$', r'$\Delta t$ (s)'],
                                draggable=False)
    ax, hax = fig.axes

    # Show multiples of KCT
    trans = btf(ax.transAxes, ax.transData)
    n = round(np.nanmax(delta_t).item() / kct) + 1

    for i in range(1, int(n)):
        yoff = 0.001
        ax.axhline(i * kct, color='g', lw=2, alpha=0.5)
        ax.text(0.001, i * kct + yoff, r'$%it_{kct}$' % i, transform=trans)

    ax.set_ylim(0, n * kct)
    # hax.set_xlim( hlim )

    return fig, ax, hax


@timer
def fix_time_jitter(t, tolerances=(3e-2, 5e-4), undead=None, plot=True):
    """
    Fix time jitter by iterative time differencing

    Parameters
    ----------
    t
    tolerances:
        upper and lower tolerance for interval selection
    undead
    plot

    Returns
    -------

    """
    figures = []
    t_fix = t.copy()
    delta_t = get_delta_t(t)  # np.diff(t)  #
    frames = np.arange(len(t))

    # Determine KCT from delta_t distribution
    kct = mode(delta_t).mode.round(6)[0]  # round to microsecond

    if plot:
        fig, ax, _ = show_jitter_stats(delta_t, kct)
    figures.append(fig)

    i, j = 0, 0
    imax = 4  # maximal interval to apply correction to
    while i < imax:
        # For each interval loop until no more points need correcting
        # print( '\tj = ', j )
        _t_fix, flagged = fixer(t_fix, kct, i, tolerances, undead)

        if _t_fix is None:  # No fix needed for this interval
            i += 1  # move on to the next higher interval
            j = 0
            # print( 'i = ', i )
        else:
            # print( '\tl.sum()', flagged.sum() )
            j += 1
            if plot:
                delta_t = get_delta_t(t_fix)  # for the updated sequence
                fig, ax, _ = show_jitter_stats(delta_t, kct)
                plot_flagged(ax, frames, delta_t, flagged, undead)
                figures.append(fig)
            t_fix = _t_fix

    if plot:
        delta_t = get_delta_t(t_fix)
        fig, ax, _ = show_jitter_stats(delta_t, kct)
        figures.append(fig)

    return t_fix, figures


@timer
def fixer(t, kct, i, tols=(5e-2, 5e-4), undead=None):
    """
    Fix the frame arrival times for the interval (i*kct,(i+1)*kct) by
    subtracting the difference in arrival times from the next higher
    multiple of kct.  i.e. For a frame n arriving at time tn, assume the correct
    arrival time is (i+1)*kct. The positive difference (i+1)*kct - tn
    is subtracted from the preceding frame between
    """
    ltol, utol = tols
    t_fix = t.copy()
    delta_t = get_delta_t(t_fix)

    # flagged points in interval of kct for which to correct
    flagged = (i * kct + ltol < delta_t) & (delta_t < (i + 1) * kct - utol)
    # FIXME: RuntimeWarning
    # We only apply the correction if the frame is one following one with
    # missing DEADTIME
    if not undead is None:
        flagged &= np.roll(undead, 1)
    # If no flagged points, no fix needed
    if ~np.any(flagged):
        return None, None  # no fix required for this interval

    diff = (i + 1) * kct - delta_t[flagged]  # discrepancy in kct
    # print( 'diff\n', diff )
    # print( diff.max(), diff.min(), diff.ptp() )

    # Subtract the differnce in arrival times from the points preceding
    # flagged  points
    lfix = np.roll(flagged, -1)
    t_fix[lfix] -= diff
    return t_fix, flagged


@timer
def plot_flagged(ax, frames, delta_t, flagged, undead):
    # Mark flagged data points on figure
    ax.plot(frames[flagged], delta_t[flagged], 'o',
            mfc='none', mec='g',
            mew=1.5, ms=7.5,
            label='flagged')

    # Mark undead frames
    ax.plot(frames[undead], delta_t[undead], 'o',
            mfc='none', mec='r',
            ms=10, mew=1.5,
            label=r'$t_{i}$  :  $t_{dead}=0$')

    # Frames following the undead ones
    lr1 = np.roll(undead, 1)
    ax.plot(frames[lr1], delta_t[lr1], 'o',
            mfc='none', mec='orange',
            ms=12.5, mew=1.5,
            label=r'$t_{i-1}$')

    # show selection limits
    # y0, y1 = i*kct+ltol, (i+1)*kct-utol
    # ax.axhline( y0 );             ax.axhline( y1 )
    ax.legend()


# ====================================================================================================
# @timer
# def extraction(filelist, save=True, filename=None):
#     # read dead time from saved multi-ext fits
#
#     nduds = 0  # Number of initial dud frames (usually just noise)
#     #  TODO: find a way of identifying the number of initial dud frames
#     # start = nduds * N_CHANNELS
#     keys = 'deadtime', 'utc-obs', 'date-obs', 'exptime'
#     fx = SlotModeExtract(filelist, header_keys=keys)
#
#     data, head_data = fx._4d(start=nduds * N_CHANNELS)
#
#     # utc = np.char.array(head_data['UTC-OBS']).decode()
#     # utdate = np.char.array(head_data['DATE-OBS']).decode()
#     # texp = head_data['EXPTIME'].astype(float)
#     # tdead = np.char.array(head_data['DEADTIME']).decode()
#
#     return data, (tdead, utc, utdate, texp)

# if save:
#     # Update the header info with info needed when doing aperture_photometry
#     firstfile = filelist[0]
#     readnoise = fits.getval(firstfile, 'rdnoise', xamp + 1)
#     gain = fits.getval(firstfile, 'gain', xamp + 1)
#     hdu.header['rdnoise'] = readnoise, 'Nominal readout noise in e'
#     hdu.header['gain'] = gain, 'Nominal CCD gain (e/ADU)'
#     #
#     # print('EXTRA:', hdu.header['naxis1'], hdu.header['naxis2'], 'data', data.shape)
#
#     # Save extracted cube
#     hdu.writeto(filename, overwrite=True)

# make_iraf_slicemap( filename, len(data), 'slices.txt' )


def make_times(utc, utdate, save=False):
    # Create timing array from header time-date strings
    sutherland = EarthLocation(lat=-32.376006, lon=20.810678, height=1771)
    t = Time.isomerge(utdate, utc, precision=9, location=sutherland)
    # ts = map('T'.join, zip(utdate, utc))
    # t = Time(list(ts), format='isot', scale='utc', precision=9,
    #          location=sutherland)

    if save:
        # Save extracted utc times
        np.savetxt(str(save), t.value, '%s')

    return t


def to_isot(d, h, m, s):
    return '{}T{:d}:{:d}:{:f}'.format(d, h, m, s)


def reconstruct_times(utsec, utdate):
    # FIXME: this function would not be necessary if you can do the
    # corrections with Time objects

    # reconstruct the frame times from the fixed utc seconds since midnight
    sutherland = EarthLocation(lat=-32.376006, lon=20.810678, height=1771)
    #
    th = utsec / 3600.
    h = np.floor(th).astype(int)
    tm = (th - h) * 60
    m = np.floor(tm).astype(int)
    s = (tm - m) * 60

    tmap = map(to_isot, utdate, h, m, s)
    t = Time(list(tmap), format='isot', scale='utc', precision=9,
             location=sutherland)

    # do stuff
    t_test = t[np.array((0, -1))]
    status = t_test.check_iers_table()

    # cache = np.any(status) and not np.all(status)
    # WARNING!  THIS WILL ALWAYS DOWNLOAD THE FILE, EVEN IF IT IS IS THE CACHE!!
    # print( 'CACHE', cache )
    try:
        # update the IERS table and set the leap-second offset
        iers_a = get_updated_iers_table(cache=True)

    except Exception as err:
        logging.warning('Unable to update IERS table due to:\n', err)
        iers_a = None

    # set leap second offset from most recent IERS table
    delta, status = t.get_delta_ut1_utc(iers_a, return_status=True)
    if np.any(status == -2):
        warnings.warn('Using predicted leap-second values from IERS.')

    t.delta_ut1_utc = delta

    return t


def save_times(filename, t, texp, coords):
    # , with_slices=False, with_obspar=False):

    def make_header_line(info, fmt, delimiter):
        import re
        matcher = re.compile('%-?(\d{1,2})')
        padwidths = [int(matcher.match(f).groups()[0]) for f in fmt]
        padwidths[0] -= 2
        colheads = [s.ljust(p) for s, p in zip(info, padwidths)]
        return delimiter.join(colheads)

    # Extract UT-DATE
    utdate, utcstr = t.isosplit()

    # UTC
    utsec = t.time_from_local_midnight('s')
    uth = utsec.to('h')

    # Sidereal time
    lmst = t.sidereal_time('mean')
    lmst_r = lmst.radian
    lat_r = t.location.lat.radian

    # Airmass
    ra_r, dec_r = coords.ra.radian, coords.dec.radian
    alt = altitude(ra_r, dec_r, lmst_r, lat_r)
    z = np.pi / 2 - alt
    airmass = Young94(z)

    # Geocentric and Barycentric JD
    gjd = t.tcg.jd
    bjd = light_time_corrections(t, coords, precess='first', abcorr=None)

    # Save the tims to ascii
    keys = ['utdate', 'uth', 'utsec', 'lmst', 'airmass', 'jd', 'gjd', 'bjd']
    timeData = np.empty((len(keys), len(t)), dtype=object)
    for i, tx in enumerate([utdate, uth, utsec, lmst, airmass, t.jd, gjd, bjd]):
        timeData[i] = tx[:]

    fmt = ('%-10s', '%-12.9f', '%-12.6f', '%-12.9f', '%-12.9f', '%-18.9f',
           '%-18.9f', '%-18.9f')  # %-10s  ==> left justify

    # if with_slices:
    #     # from irafhacks imp
    #     slicefile = cubename.with_suffix('.slice')
    #     slices = make_iraf_slicemap(cubename.name, len(t), slicefile)
    #     keys = ['filename'] + keys
    #     timeData = np.c_[slices, timeData.T]
    #     fmt = ('%-35s',) + fmt  # max(map(len,slices))
    #
    # if with_obspar:
    #     # cubename = 'extr.cube.fits'
    #     # filename = str(outpath/'obsparams.txt')
    #     obsparfile = str(cubename.with_suffix('.obspar'))
    #     z = make_obsparams_file(obsparfile, cubename.name, utcstr, texp, 'WL',
    #                             airmass)

    # header = ''.join(fmt).replace('f','s') % tuple(map(str.upper,keys))
    delimiter = ' '
    header = make_header_line(keys, fmt, delimiter)
    np.savetxt(str(filename), timeData.T, fmt=fmt, header=header,
               delimiter=delimiter)

    return timeData


def fix_timing_errors(utsec, tdead, plot=True, save=False, outpath=None):
    # Flag frames with missing DEADTIME     exx, exy = exclude
    undead = (tdead == '')  # np.equal(tdead, '')  # No dead time given in

    # header
    badfrac = undead.sum() / len(undead)
    logging.info('DEADTIME missing in {:3.2%} of headers'.format(badfrac))

    # Set deadtime to zero where missing
    tdead[undead] = 0  # Assume deadtime = 0 when not given
    tdead = tdead.astype(int)

    # TODO: only plot this if necessary. i.e if there are timing issues...
    if plot:
        # dead time histogram
        h, ax = hist(tdead, range=(0, 50), bins=50, log=True,
                     axlabels=[r'$t_{dead}$ (ms)'],
                     title='Dead time stats')
        ylim = 1e-1, 10 ** np.ceil(np.log10(h[0].max()))
        ax.set_ylim(ylim)

        if not outpath.exists():
            outpath.mkdir(parents=True)

        savefig(ax.figure, outpath / 'jitter.dead.hist.png')

    if undead.any():
        # Fix time jitter
        logging.info('Fixing timing data.')
        tolerances = (3e-2, 5e-4)
        t_fix, figures = fix_time_jitter(utsec, tolerances, undead, plot=plot)

        # Show the fix for each iteration as multi-tab plot
        if len(figures):
            # app = qt.QApplication(sys.argv)
            # ui = MplMultiTab(figures=figures)
            for i, fig in enumerate(figures):
                filename = outpath / 'jitter.fix.{}.png'.format(i)
                savefig(fig, filename)
                # ui.show()
        # app.exec_()
        if save:
            np.savetxt(str(outpath / 'utc.sec.fix'), t_fix, '%s')

        return t_fix

    else:
        logging.info('No timing fix required!')
        # Determine KCT from delta_t distribution
        delta_t = get_delta_t(utsec)
        kct = mode(delta_t).mode  # .round(6)          # round to microsecond
        fig, ax, _ = show_jitter_stats(delta_t, kct, title='Time jitter')

        if save:
            savefig(fig, outpath / 'jitter.fix.png')

        return utsec


def graphics(data, save=None, outpath=None):
    # TODO: nice plots with all 4 amp channels

    # Save envelope/variance/mad/mad_std image (can help to determine data quality)
    # TODO entropy image??
    if save is None:
        save = []
    env = data.max(0)
    var = data.var(0)
    # TODO: These ops are slow --> do in bg thread
    # np.apply_along_axis(mad, 0, data)              useful???
    # np.apply_along_axis(mad_std, 0, data)
    figsize = (18, 8)
    for image, title in zip([env, var], ['envelope', 'variance']):
        basename = str(outpath / '{}.{}')
        # save as FITS
        if 'fits' in save:
            fits.writeto(basename.format(title, 'fits'), image,
                         overwrite=True)
            save.pop(save.index('fits'))

        if len(save):
            # save as png (or whatever) for quick ref
            imd = ImageDisplay(image, origin='lower left',
                               colorscale='z', title=title)
            fmt = save.pop(0)
            imd.figure.savefig(basename.format(title, fmt))

            # TODO:  Save data as mp4 ---> Slow! ==> thread??


# ====================================================================================================
# @profile( follow=(extraction, fix_timing_errors, graphics) )
# def main():
#     args, filelist, path, outpath = setup()
#
#     # Extract the data
#     cubename = outpath / 's.fits'
#     data, (tdead, utc, utdate, texp) = extraction(filelist, save=True, filename=cubename)
#
#     rawtimes = outpath / 's.raw.time'
#     t = make_times(utc, utdate, save=rawtimes)
#
#     # raise SystemExit
#
#     t_fix = fix_timing_errors(t.sec, tdead, plot=True, save=True, outpath=outpath)
#     t = reconstruct_times(t_fix, utdate)
#
#     # Object coordinates (for baricentrization)
#     header = fits.getheader(filelist[0], 0)
#     ra, dec = header['ra'], header['dec']
#     equinox = header['equinox']
#     coords = SkyCoord(ra, dec, unit=('h', 'deg'), equinox=equinox)  # obstime
#
#     timefile = str(cubename.with_suffix('.time'))
#     # obsparfile = str(cubename.with_name('obspar.txt'))
#     # print( timefile )
#     save_times(timefile, cubename, t, texp, coords, with_slices=True, with_obspar=True)
#
#     graphics(data, save=['fits', 'png'], outpath=outpath)


# def resolve_subset(subset, default_start=0, default_end=None):
#     # subset of frames to use
#     if subset is None:
#         subset = (default_start, default_end)
#     elif len(subset) == 1:
#         subset = (default_start, min(subset[0], default_end))
#         logging.info('Using subset: %s', subset)
#     elif len(subset) == 2:
#         logging.info('Using subset: %s', subset)
#     else:
#         raise ValueError('Invalid subset: %s' % subset)
#
#     return subset
#
#
# def get_subset(items, subset):
#     if subset is None:
#         return items
#
#     # subset = resolve_subset(subset, 0, len(items))
#     # subsize = np.ptp(subset)
#     # if subsize == 0:
#     #     raise IOError('selecting subset %r returned empty list' %
#     #                   subset)
#
#     return items[slice(*subset)]


# @timer
def setup():
    from sys import argv
    import argparse
    from pathlib import Path
    from recipes.io import parse

    # TODO: add verbosity argument

    parser = argparse.ArgumentParser(
            description='Extraction for SALTICAM slotmode data.')

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
            '-m', '--file-subset', nargs='*',  default=[None], type=int,
            help=file_sub_help)

    subset_help = (
        "Data subset to load. Arguments are interpreted as for --file-subset ("
        "-m) save that if a single integer is given, this is taken to be the "
        "total number of desired frames.  Since the first ~6 frames are "
        "usually dud frames (white noise), the actual range of frames extracted"
        " from the raw data will be (n_dud, k + n_dud) where `n_dud`"
        " is the detected number of dud frames.")
    subset_args.add_argument(
            '-n', '--subset', nargs='*',  default=[None], type=int,
            help=subset_help)

    # channels
    parser.add_argument(
            '-ch', '--channels', nargs='+', default=ALL_CHANNELS,
            help='Amplifier channel(s)')


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
    # TODO: remove in favour of @list.txt?
    filelist = parse.to_list(args.images,
                             os.path.exists,
                             include='bxgp*.fits',
                             abspaths=True,
                             raise_error=1)

    # Check & set input and output paths
    if not len(filelist):
        raise IOError('Could not resolve file list: %r' % args.images)

    # select subset of files
    slice_ = slice(*args.file_subset)
    if slice_.start or slice_.stop:
        logging.info('Requested subset: %s',
                     (slice_.indices(len(filelist)[:2])))

    filelist = filelist[slice_]
    if not len(filelist):
        raise IOError('selecting subset %r returned empty list' %
                      args.file_subset)

    # set IO paths from input file list if not provided
    product_dir = Path(filelist[0]).parent
    root = product_dir.parent
    if args.output_dir is None:
        args.output_dir = root / 'reduced'

    outpath = Path(args.output_dir).resolve()
    if not outpath.exists():
        outpath.mkdir()

    # Set global plotting parameters
    rc('savefig', directory=str(outpath))

    return args, filelist, root, outpath


if __name__ == '__main__':
    from salticam.slotmode.extract.core import SlotModeExtract, ALL_CHANNELS
    from recipes.io.utils import WarningTraceback

    # module level logger
    root = logging.getLogger()
    logger = logging.getLogger(get_module_name(__file__))
    root.setLevel(logging.INFO)
    # stack trace for warnings
    wtb = WarningTraceback()

    # TODO: log to file NBNBNBNB!!!!!!!!!!!
    # TODO: resume / clobber option
    # TODO: tool that can display entire slot with optional layouts
    # layout=2x2, 1x4, 4x1 etc
    # SlotDisplay

    args, filelist, _, outpath = setup()
    figpath = outpath / 'figs'

    logger.info('Output folder: %s', args.output_dir)

    # default header keys to extract
    keys = 'UTC-OBS', 'DATE-OBS', 'EXPTIME', 'DEADTIME'
    # initialize extraction class
    fx = SlotModeExtract(filelist, header_keys=keys, loc=outpath)
    # print info from header. source name date, etc
    fx.pprint()
    # TODO: plot some of the environmental stuff from headers. eg track?

    # Extract the data
    subset = slice(*args.subset)
    data, header, info = fx.to_fits(args.channels, start=subset.start,
                                    stop=subset.stop)

    # Construct time data
    # `info` may be shaped (n, namps). Times are simultaneous, so use
    # first row sufficient
    ixamp = 0 if info.ndim > 1 else ...
    utc, utdate = info.utc[:, ixamp], info.date[:, ixamp]
    # check for date change
    assert np.all(utdate == utdate[0]), 'DATE CHANGES'

    # construct time object
    # need a location for barycentrization
    sutherland = EarthLocation(lat=-32.376006, lon=20.810678, height=1771)
    t = Time.isomerge(utdate, utc, precision=9, location=sutherland)

    # convert to seconds from midnight UTC.  Fix incorrect frame times.
    utsec = t.time_from_local_midnight('s')
    t_fix = fix_timing_errors(utsec.value, info.deadtime,
                              plot=args.plot,
                              outpath=figpath)
    t = reconstruct_times(t_fix, utdate)  # FIXME. remove the need for this

    # Object coordinates (for barycentrization)
    header = fits.Header.fromstring(fx.headers[0].decode())
    ra, dec, equinox = map(header.get, ('ra', 'dec', 'equinox'))
    coords = SkyCoord(ra, dec, unit=('h', 'deg'))  # equinox=equinox

    # export times
    timefile = str(fx.basename.with_suffix('.time'))
    save_times(timefile, t, info.exptime, coords)

    # graphics(data, save=['png'], outpath=figpath)
