
import os
import itertools as itt
from time import time

#scientfic libs
import numpy as np
import scipy as sp

#astronomy libs
from astropy.io import fits as pyfits
from astropy.coordinates import EarthLocation
from astropy.time import Time
#from astropy.stats import median_absolute_deviation as mad
#from astropy.stats import mad_std

#from skimage.feature import blob_log as LoG
 #blob_dog, , blob_doh
#from photutils import daofind, irafstarfind
# from photutils import aperture_photometry, CircularAperture

#Plotting libs
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.transforms import  blended_transform_factory as btf
# from mplMultiTab import MplMultiTab
# from PyQt4 import QtGui as qt

#my rad libs
from recipes.list import flatten#, neighbours, grid_like, shape2grid
#from outliers import WindowOutlierDetection, generalizedESD
from tsa.gaps import get_deltat

from grafico.lc import lcplot, hist
from grafico.imagine import supershow
from fastfits import FastExtractor
# from ansi.string import ProgressBar, Table

#profiling
#from profilehooks import profile
from decor import profile



#====================================================================================================
#Decor
def savefig(fig, path):
    fig.savefig(str(path))

#====================================================================================================
def show_jitter_stats(deltat, kct, title='Time jitter'):
    '''Plots of time jitter statistics.'''
    #plot deltat as TS + histogram
    plot_props = dict( fmt='ro' )
    hist_props = dict( log=True, bins=1000 )                            #milisecond time bins
    fig, pl, *rest = lcplot( deltat,
                                errorbar=plot_props,
                                hist=hist_props,
                                title=title,
                                axlabels=[r'$N_{frame}$',r'$\Delta t$ (s)'],
                                draggable=False )
    ax, hax = fig.axes

    #Show multiples of KCT
    trans = btf( ax.transAxes, ax.transData )
    N = round( deltat[1:].max() / kct ) + 1
    for i in range(1,int(N)):
        yoff = 0.001
        ax.axhline( i*kct, color='g', lw=2, alpha=0.5 )
        ax.text( 0.001, i*kct+yoff, r'$%it_{kct}$'%i, transform=trans )

    ax.set_ylim( 0, N*kct )
    #hax.set_xlim( hlim )

    return fig, ax, hax

#====================================================================================================
def fix_time_jitter(t, tolerances=(3e-2, 5e-4), undead=None, plot=True):
    '''Fix time jitter by iterative time differencing
    #upper and lower tolerance for interval selection'''

    tfix = t.copy()
    deltat = get_deltat( tfix )
    frames = np.arange(len(t))

    #Determine KCT from deltat distribution
    kct, Nocc = flatten( sp.stats.mode(deltat) )
    kct = kct.round(6)                                      #round to microsecond

    fig, ax, _ = show_jitter_stats( deltat, kct )

    figures = [fig]
    i, j = 0, 0
    imax = 4                                #maximal interval to apply correction to
    while i < imax:
        #For each interval loop until no more points need correcting
        #print( '\tj = ', j )
        _tfix, flagged = fixer( tfix, kct, i, tolerances, undead )

        if _tfix is None:                   #No fix needed for this interval
            i += 1                          #move on to the next higher interval
            j = 0
            #print( 'i = ', i )
        else:
            #print( '\tl.sum()', flagged.sum() )
            j += 1
            if plot:
                deltat = get_deltat( tfix )     #for the updated sequence
                fig, ax, _ = show_jitter_stats( deltat, kct )
                plot_flagged(ax, frames, deltat, flagged, undead)
                figures.append( fig )
            tfix = _tfix
    if plot:
        deltat = get_deltat( tfix )
        fig, ax, _ = show_jitter_stats( deltat, kct )
        figures.append( fig )

    return tfix, figures

#====================================================================================================
def fixer(t, kct, i, tols=(5e-2,5e-4), undead=None):
    '''Fix the frame arrival times for the interval (i*kct,(i+1)*kct) by subtracting the
    difference in arrival times from the next higher multiple of kct.  i.e. For a frame n arriving
    at time tn, assume the correct arrival time is (i+1)*kct. The positive difference (i+1)*kct-tn
    is subtracted from the preceding frame
    between '''
    ltol, utol = tols
    tfix = t.copy()
    deltat = get_deltat( tfix )

    #flagged points in interval of kct for which to correct
    flagged = (i*kct+ltol < deltat) & (deltat < (i+1)*kct-utol) #FIXME: RuntimeWarning
    #We only apply the correction if the frame is one following one with missing DEADTIME
    if not undead is None:
        flagged &= np.roll( undead, 1 )
    #If no flagged points, no fix needed
    if ~np.any(flagged):
        return None, None#no fix required for this interval

    diff = (i+1)*kct - deltat[flagged]  #discrepancy in kct
    #print( 'diff\n', diff )
    #print( diff.max(), diff.min(), diff.ptp() )

    #Subtract the differnce in arrival times from the points preceding flagged points
    lfix = np.roll(flagged,-1)
    tfix[lfix] -= diff
    return tfix, flagged

#====================================================================================================
def plot_flagged(ax, frames, deltat, flagged, undead):
    # Mark flagged datapoints on figure
    ax.plot(frames[flagged], deltat[flagged], 'o',
            mfc='none', mec='g',
            mew=1.5, ms=7.5,
            label='flagged')
    # Mark undead frames
    ax.plot(frames[undead], deltat[undead], 'o',
            mfc='none', mec='r',
            ms=10, mew=1.5,
            label=r'$t_{i}$  :  $t_{dead}=0$')
    # Frames following the undead ones
    lr1 = np.roll(undead, 1)
    ax.plot(frames[lr1], deltat[lr1], 'o',
            mfc='none', mec='orange',
            ms=12.5, mew=1.5,
            label=r'$t_{i-1}$')

    # show selection limits
    # y0, y1 = i*kct+ltol, (i+1)*kct-utol
    # ax.axhline( y0 );             ax.axhline( y1 )
    ax.legend()

#====================================================================================================
def setup():
    #if __name__ == '__main__':
    from sys import argv
    import argparse
    from pathlib import Path
    from myio import iocheck, parsetolist

    main_parser = argparse.ArgumentParser(description='Animation for SALTICAM slotmode data.')
    main_parser.add_argument('-i', '--images', nargs='+', type=str,
                             help='''Science data cubes to be processed.  Requires at least one argument.'
                                'Argument can be explicit list of files, a glob expression, a txt list,
                                or a directory.''')
    main_parser.add_argument('-d', '--dir', dest='dir',
                             default=os.getcwd(), type=str,
                             help='The data directory. Defaults to current working directory.')
    main_parser.add_argument('-o', '--output-dir', type=str,
                             help=('The data directory where the reduced data is to be placed.'
                                   'Defaults to input dir.'))
    main_parser.add_argument('-n', '--subset', type=int,
                             help='Data subset to load.  Useful for testing.')
    args = main_parser.parse_args(argv[1:])

    # Check & set input and output paths
    path = iocheck(args.dir, os.path.exists, 1)
    ppath = Path(path).resolve()
    if args.output_dir is None:
        args.output_dir = outpath = path
    else:
        outpath = iocheck(args.output_dir, os.path.exists, 1)
    poutpath = Path(outpath).resolve()

    if args.images is None:
        args.images = args.dir  # no cubes explicitly provided will use list of all files in input directory

    filelist = parsetolist(args.images, path=args.dir)
    filelist = list(itt.islice(filelist, args.subset))
    if args.subset:
        print('Using subset', args.subset)

    # Set global plotting parameters
    rc('savefig', directory=outpath)

    return args, filelist, ppath, poutpath


#====================================================================================================
def extraction(filelist, save=True, filename=None ):
    # read dead time from saved multi-ext fits
    # TODO Logging with decor for verbosity
    pre = 'Extracting data.  This may take a few moments...'
    print(pre)

    Namp = 4  # Number of amplifier channels
    Nduds = 6  # Number of initial dud frames #(usually just noise) TODO: find a way of identifying the number of initial dud frames
    Xamp = 2  # Which amp to extract frames from (0-3)
    start = Nduds * Namp + Xamp
    keys = 'deadtime', 'utc-obs', 'date-obs'
    fx = FastExtractor(filelist, start=start, step=Namp, keygrab=keys)
    hdu = fx.cube()
    data = hdu.data  # dead time in ms

    post = '\nDONE!  {} frames extracted from {} files.'.format(len(data), len(filelist))
    print(post)

    # Identify and remove bad frames at the end of run
    header_data = tdead, utc, utdate = fx.get_keys(return_type='array')
    l = utdate == '1969-12-31'  # The last ~50 frames has this date...?
    l_ = data.sum((1, 2)) == 0
    assert np.all(l == l_)  # These frames can also be identified by having all 0 pixel values.

    data = data[~l]  # Chuck out the bad frames at the end
    times = tdead, utc, utdate = header_data.T[~l].T

    if save:
        # Save extracted cube
        hdu.writeto(str(filename), clobber=True)

    return data, times

#====================================================================================================
def make_times(utc, utdate, save=True, filename=None):
    # Create timing array from header time-date strings
    sutherland = EarthLocation(lat=-32.376006, lon=20.810678, height=1771)
    ts = map('T'.join, zip(utdate, utc))
    t = Time(list(ts), format='isot', scale='utc', precision=9, location=sutherland)

    if save:
        # Save extracted utc times
        np.savetxt(str(filename), t.value, '%s')

    return t


#====================================================================================================
def great_timing_fix(utsec, tdead, plot=True, save=True, outpath=None):
    # Flag frames with missing DEADTIME
    # exx, exy = exclude
    undead = np.equal(tdead, None)  # No dead time given in header
    badfrac = undead.sum() / len(undead)
    print('DEADTIME missing in {:3.2%} of headers'.format(badfrac))

    # Set deadtime to zero where missing
    tdead[undead] = 0  # Assume deadtime = 0 when not given
    tdead = tdead.astype(int)

    if plot:
        # dead time histogram
        h, ax = hist(tdead, range=(0, 50), bins=50, log=True,
                     axlabels=[r'$t_{dead}$ (ms)'],
                     title='Dead time stats')
        ylim = 1e-1, 10 ** np.ceil(np.log10(h[0].max()))
        ax.set_ylim(ylim)

        savefig(ax.figure, outpath / 'jitter.hist.fix.png')

    if any(undead):
        # Fix time jitter
        print('Fixing timing data.')
        t0 = time()
        tolerances = (3e-2, 5e-4)
        tfix, figures = fix_time_jitter(utsec, tolerances, undead, plot=True)
        print('DONE in {} sec'.format(time() - t0))

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
            np.savetxt(str(outpath / 'utc.sec.fix'), tfix, '%s')

    else:
        print('No timing fix required!')
        deltat = get_deltat(utsec)
        kct, Nocc = flatten(sp.stats.mode(deltat))
        fig, ax, _ = show_jitter_stats(deltat, kct, title='Time jitter')

        if save:
            savefig(fig, outpath / 'jitter.fix.png')

    return tfix

#====================================================================================================
def graphics(data, save=[], outpath=None):
    #Save envelope/variance/mad/mad_std image (can help to determine data quality)
    #TODO entropy image??
    env = data.max(0)
    var = data.var(0)
    # TODO: These ops are slow --> do in bg thread
    # np.apply_along_axis(mad, 0, data)              useful???
    # np.apply_along_axis(mad_std, 0, data)
    figsize = (18, 6)
    for image, title in zip([env, var], ['envelope', 'variance']):
        basename = str(outpath / '{}.{}')
        # save as FITS
        if 'fits' in save:
            pyfits.writeto(basename.format(title, 'fits'), image, clobber=True)
            save.pop(save.index('fits'))
        if len(save):
            # save as png (or whatever) for quick ref
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
            supershow(ax, image, origin='lower left', autoscale='z', title=title)
            fmt = save.pop(0)
            fig.savefig(basename.format(title, fmt))

    #TODO:  Save data as mp4 ---> Slow! ==> thread??

#====================================================================================================
@profile(follow=(extraction, great_timing_fix, graphics))
def main():
    #setup for work
    args, filelist, path, outpath = setup()

    # Extract the data
    data, (tdead, utc, utdate) = extraction(filelist, save=True, filename=outpath / 'extr.cube.fits')

    t = make_times(utc, utdate, save=True, filename=outpath / 'extr.utc')
    great_timing_fix(t.sec, tdead, plot=True, save=True, outpath=outpath)

    graphics(data, save=['fits', 'png'], outpath=outpath)


if __name__ == '__main__':
    main()
