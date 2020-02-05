import numpy as np

from motley.profiling.timers import timer

from graphing import ts

from .core import UPPER_LIMIT_DEADTIME_FLAG


@timer
def save_figure(fig, path):
    fig.savefig(str(path))


def plot_flagged(ax, frames, delta_t, flagged, undead):
    # Mark flagged data points on figure
    ax.plot(frames[flagged], delta_t[flagged], 'o',
            mfc='none', mec='forestgreen',
            mew=1.5, ms=7.5,
            label='flagged')

    # Mark undead frames
    ax.plot(frames[undead], delta_t[undead], 'o',
            mfc='none', mec='0.75',
            ms=10, mew=1.5,
            label=r'$t_{i}$  : $t_{dead} < %i$ (ms)' %
                  UPPER_LIMIT_DEADTIME_FLAG)

    # Frames following the undead ones
    lr1 = np.roll(undead, 1)
    ax.plot(frames[lr1], delta_t[lr1], 'o',
            mfc='none', mec='goldenrod',
            ms=12.5, mew=1.5,
            label=r'$t_{i-1}$')

    ax.legend(loc=1)


def plot_timestamp_stats(delta_t, t_cyc, title='Timestamp differences',
                         tolerances=None):
    """
    Plots of timestamp differences to check for incorrectly timestamped
    data points. Plot shows delta_t vs frame number with histogram panel on the
    right. Integer multiples of the cycle time are also shown, as well as
    optionally, the interval limits for flagging points.
    """

    from matplotlib.transforms import blended_transform_factory as btf

    # plot delta_t as TS + histogram
    plot_props = dict(fmt='ro')
    hist_props = dict(log=True, bins=100)  # 10 millisecond time bins
    tsp = ts.plot(delta_t,
                  errorbar=plot_props,
                  hist=hist_props,
                  title=title,
                  axes_labels=[r'$N_{frame}$', r'$\Delta t$ (s)'],
                  draggable=False)
    ax, hax = tsp.fig.axes

    # Show multiples of t_cyc
    trans = btf(ax.transAxes, ax.transData)
    n = round(delta_t.max().item() / t_cyc) + 1

    y_off = 0.01
    for i in range(1, int(n)):
        ax.axhline(i * t_cyc, color='g', lw=2, alpha=0.5)
        ax.text(0.001, i * t_cyc + y_off, r'$%it_{cyc}$' % i, transform=trans)

        if tolerances is not None:
            # show selection limits
            ltol, utol = tolerances
            y0, y1 = i * t_cyc + ltol, (i + 1) * t_cyc - utol
            ax.axhline(y0, ls=':', color='royalblue')
            ax.axhline(y1, ls=':', color='royalblue')

    ax.set_ylim(0, n * t_cyc)
    return tsp
