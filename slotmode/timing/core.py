import logging
import warnings

import numpy as np
from astropy.coordinates import EarthLocation

from graphing.hist import hist

from motley.profiling.timers import timer
from obstools.airmass import altitude, Young94
from pySHOC import Time
from pySHOC.timing import get_updated_iers_table, light_time_corrections
from scipy.stats import mode
from tsa.gaps import get_delta_t

# any frame following a frame with DEADTIME less than this will be corrected
UPPER_LIMIT_DEADTIME_FLAG = 1  # in milliseconds


def fix_timestamps(ut_sec, t_dead, plot=True, save_path=None):
    """

    Parameters
    ----------
    ut_sec
    t_dead
    plot
    save_path

    Returns
    -------

    """
    # Flag frames with missing DEADTIME
    undead = (t_dead == '')  # np.equal(t_dead, '')

    # header
    logging.info('DEADTIME missing in {:3.2%} of headers.'.format(
            undead.sum() / len(undead)))

    # Set dead time to zero where missing
    t_dead[undead] = 0  # Assume deadtime = 0 when not given
    t_dead = t_dead.astype(int)
    # sometimes dead time is actually recorded as 0 or 1 (ms) which also
    # means the timestamp on the subsequent frame will be wrong
    undead = (t_dead <= UPPER_LIMIT_DEADTIME_FLAG)

    # plot the dead time histogram
    if plot:
        save = (save_path is not None)
        if save and not save_path.exists():
            save_path.mkdir(parents=True)

        # dead time histogram
        h, ax = hist(t_dead, range=(0, 50), bins=50, log=True,
                     axes_labels=[r'$t_{dead}$ (ms)'],
                     title='Dead time stats',
                     show_stats=('mode',))
        ylim = (1e-1, 10 ** np.ceil(np.log10(h[0].max())))
        ax.set_ylim(ylim)

        if save:
            # from .diagnostics import save_figure
            # save_figure(ax.figure, save_path / 'deadtime.hist.png')
            ax.figure.savefig(save_path / 'deadtime.hist.png')

    # Fix time jitter
    # ----------------
    logging.info('Fixing timing data.')
    tolerances = (3e-2, 5e-4)
    t_fix, figures = fix_time_jitter(ut_sec, tolerances, undead, plot=plot)
    # ----------------

    # embed plots in tabbed window
    if plot:
        from graphing.multitab import MplMultiTab

        ui = MplMultiTab(figures=figures)
        # noinspection PyUnboundLocalVariable
        if save:
            ui.save_figures('stamps.fix{:d}.png', path=save_path)

        return t_fix, ui

    return t_fix

    # else:
    #     logging.info('No timing fix required!')
    #     #
    #     # if plot:
    #     #     from .diagnostics import plot_timestamp_stats
    #     #
    #     #     # Determine cycle time (exposure + dead time) from delta_t
    #     #     # distribution
    #     #     delta_t = get_delta_t(ut_sec)
    #     #     # noinspection PyUnresolvedReferences
    #     #     t_cyc = mode(delta_t).mode.item()  # .round(6) # round to μs
    #     #     tsp = plot_timestamp_stats(delta_t, t_cyc)

    # if save:
    #     # noinspection PyUnboundLocalVariable
    #     save_figure(tsp.fig, out_path / 'jitter.fix.png')

    # return ut_sec, []


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
    delta_t = get_delta_t(t)
    frames = np.arange(len(t))

    # Determine cycle time (exposure + dead time) from delta_t distribution
    # noinspection PyUnresolvedReferences
    t_cyc = mode(delta_t).mode.round(6).item()  # round to microsecond

    if plot:
        from .diagnostics import plot_timestamp_stats, plot_flagged
        kws = dict(tolerances=tolerances)
        tsp = plot_timestamp_stats(delta_t, t_cyc, **kws)
        figures.append(tsp.fig)

    # detect maximum time gap to determine final loop value
    i_max = np.round(delta_t.max() / t_cyc)
    i = 0
    while i < i_max:
        # For each interval loop until no more points need correcting
        _t_fix, flagged = fixer(t_fix, t_cyc, i, tolerances, undead)

        if ~np.any(flagged):  # No fix needed for this interval
            i += 1  # move on to the next higher interval
        else:
            if plot:
                # time stamp differences for the updated sequence
                delta_t = get_delta_t(t_fix)
                # noinspection PyUnboundLocalVariable
                tsp = plot_timestamp_stats(delta_t, t_cyc, **kws)
                # noinspection PyUnboundLocalVariable
                plot_flagged(tsp.ax, frames, delta_t, flagged, undead)
                figures.append(tsp.fig)
            t_fix = _t_fix

    if plot:
        delta_t = get_delta_t(t_fix)
        tsp = plot_timestamp_stats(delta_t, t_cyc, **kws)
        figures.append(tsp.fig)

    return t_fix, figures


def fixer(t, t_cyc, i, tols=(5e-2, 5e-4), undead=None):
    """
    Fix the frame time stamps for the interval (i * t_cyc, (i + 1) * t_cyc) by
    subtracting the difference in time stamps from the next higher
    multiple of t_cyc.  i.e. For a frame `n` stamped at time `tn`, assume the
    correct time stamp is (i + 1) * t_cyc. The positive difference
    `(i + 1) * t_cyc - tn` is subtracted from the preceding frame
    """
    ltol, utol = tols
    t_fix = t.copy()
    delta_t = get_delta_t(t)

    # flagged points in interval of t_cyc for which to correct
    flagged = (i * t_cyc + ltol < delta_t) & (delta_t < (i + 1) * t_cyc - utol)

    # We only apply the correction if the frame is one following one with
    # missing DEADTIME
    if undead is not None:
        flagged &= np.roll(undead, 1)

    # discrepancy in t_cyc
    diff = (i + 1) * t_cyc - delta_t[flagged]

    # Subtract the difference in time stamp from the points preceding flagged
    # points
    t_fix[np.roll(flagged, -1)] -= diff
    return t_fix, flagged


def make_times(utc, ut_date, save=False):
    # Create timing array from header time-date strings
    sutherland = EarthLocation(lat=-32.376006, lon=20.810678, height=1771)
    t = Time.isomerge(ut_date, utc, precision=9, location=sutherland)

    if save:
        # Save extracted utc times
        np.savetxt(str(save), t.value, '%s')

    return t


def to_isot(d, h, m, s):
    return '{}T{:d}:{:d}:{:f}'.format(d, h, m, s)


def reconstruct_times(utsec, utdate):
    # FIXME: this function would not be necessary if you can do the
    #  corrections with Time objects

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

    #
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
        logging.warning('Using predicted leap-second values from IERS.')

    t.delta_ut1_utc = delta

    return t


def make_header_line(info, fmt, delimiter):
    import re
    matcher = re.compile(r'%-?(\d{1,2})')
    pad_widths = [int(matcher.match(f).groups()[0]) for f in fmt]
    pad_widths[0] -= 2
    col_heads = [s.ljust(p) for s, p in zip(info, pad_widths)]
    return delimiter.join(col_heads)


def save_times(filename, t, coords, delimiter=' '):
    """write to ascii"""
    # UTC /  Sidereal time
    utsec = t.time_from_local_midnight('s')
    lmst = t.sidereal_time('mean')

    # Airmass
    # ra_r, dec_r = coords.ra.radian, coords.dec.radian
    alt = altitude(coords.ra.radian, coords.dec.radian,
                   lmst.radian, t.location.lat.radian)
    z = np.pi / 2 - alt
    airmass = Young94(z)

    # Geocentric and Barycentric JD
    gjd = t.tcg.jd
    bjd = light_time_corrections(t, coords, precess='first', abcorr=None)

    # Save the times to ascii
    keys = ['utc', 'utsec', 'lmst', 'airmass', 'jd', 'gjd', 'bjd']
    time_data = np.empty((len(keys), len(t)), dtype=object)
    for i, tx in enumerate([t.isot, utsec, lmst, airmass, t.jd, gjd, bjd]):
        time_data[i] = tx[:]

    # column formats     %-10s  ==> left justify
    fmt = ('%-25s', '%-12.6f', '%-12.9f', '%-12.9f', '%-18.9f', '%-18.9f',
           '%-18.9f')
    header = make_header_line(keys, fmt, delimiter)
    np.savetxt(str(filename), time_data.T, fmt=fmt, header=header,
               delimiter=delimiter)

    return time_data
