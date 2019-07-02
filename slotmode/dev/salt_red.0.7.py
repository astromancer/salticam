import os
import sys
import functools

from time import time

import scipy as sp
import numpy as np
import pyfits
from astropy.coordinates import EarthLocation

import pylab as plt
from matplotlib.transforms import  blended_transform_factory as btf
from mplMultiTab import MplMultiTab
from PyQt4 import QtGui as qt

from misc import flatten 
from myio import parsetolist, iocheck
from outliers import WindowOutlierDetection, generalizedESD
from tsa import get_deltat, detrend, fill_gaps
from chronos import Time

from superplot import lcplot, hist
from superstring import ProgressBar

from superfits import superheadhunter

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
    flagged = (i*kct+ltol < deltat) & (deltat < (i+1)*kct-utol)
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
    #Mark flagged datapoints on figure
    ax.plot( frames[flagged], deltat[flagged], 'o', mfc='none', mec='g',
                mew=1.5, ms=7.5,
                label='flagged' )
    #Mark undead frames
    ax.plot( frames[undead], deltat[undead], 'o', mfc='none', mec='r',
                ms=10, mew=1.5, 
                label=r'$t_{i}$  :  $t_{dead}=0$' )
    #Frames following the undead ones
    lr1 = np.roll( undead, 1 )
    ax.plot( frames[lr1], deltat[lr1], 'o',  mfc='none', mec='orange',
                ms=12.5, mew=1.5, 
                label=r'$t_{i-1}$' )
    
    #show selection limits
    #y0, y1 = i*kct+ltol, (i+1)*kct-utol
    #ax.axhline( y0 );             ax.axhline( y1 )
    ax.legend()

#====================================================================================================

#if __name__ == '__main__':
from sys import argv
import argparse
from pathlib import Path
from myio import iocheck, parsetolist

main_parser = argparse.ArgumentParser( description='Animation for SALTICAM slotmode data.' )
main_parser.add_argument( '-i', '--images', nargs='+', type=str, 
                    help = '''Science data cubes to be processed.  Requires at least one argument.'  
                            'Argument can be explicit list of files, a glob expression, a txt list,
                            or a directory.''' )
main_parser.add_argument( '-d', '--dir', nargs=1, dest='dir', 
                            default = os.getcwd(), type=str,
                            help = 'The data directory. Defaults to current working directory.' )
main_parser.add_argument('-o', '--output-dir', nargs=1, default=os.getcwd(), type=str, 
                            help = ('The data directory where the reduced data is to be placed.' 
                                    'Defaults to $PWD/ReducedData.'))
args = main_parser.parse_args( argv[1:] )

path = iocheck(args.dir, os.path.exists, 1)
path = Path(path).resolve()
outpath = iocheck(args.dir, os.path.exists, 1)
outpath = Path(outpath).resolve()

if args.images is None:
    args.images = args.dir       #no cubes explicitly provided will use list of all files in input directory

filelist = parsetolist( args.images )


        
sutherland = EarthLocation( lat=-32.376006, lon=20.810678, height=1771 )
    
#saltpath = '/media/Oceanus/UCT/Observing/SALT/EX_Hya/20140611/product/'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/NY_Lup/20140606/product'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/BL_Hyi/product'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/2014-1-RSA_OTH-020.20140611/product'
saltpath = '/media/Oceanus/UCT/Observing/SALT/V2400_Oph/20140921/product'
#imfn = saltpath + '/images.txt'

#saltpath = '/media/Oceanus/UCT/Observing/SALT/2014-1-RSA_OTH-020.20140802/product'
#filelist = parselist( saltpath+'/images.last50' )
filelist = parselist( saltpath+'/bxgp*.fits' )

#fnlc = os.path.join( saltpath, 'phot2.lc' )
#idx, utsec, flux_r, flux_r_err, xt, yt, flux_t, flux_t_err, xc, yc, flux_c, flux_c_err = np.loadtxt( fnlc, unpack=1 )


#read dead time from saved multi-ext fits
print( 'Retrieving header data.  This may take a few moments...' )
t0 = time()
keys = 'deadtime', 'utc-obs', 'date-obs'
data = superheadhunter( filelist[:10], keys, return_type='array', Nchunks=10 )            #dead time in ms
rdata = data[:,2::4]
print( 'DONE in {} sec'.format(time()-t0) )

#Identify and remove bad frames at the end of run
utdate, utc, tdead = rdata
l = utdate == '1969-12-31'                              #The last ~50 frames has this date...?  These frames can also be identified by having all 0 pixel values.
rdata = rdata[:,~l]                                     #Chuck these out
utdate, utc, tdead = rdata

#Create timing array from header time-date strings
ts = map( 'T'.join, zip(utdate, utc) )
t = Time( list(ts), format='isot', scale='utc', precision=9, location=sutherland )
utsec = t.sec

#Flag frames with missing DEADTIME
undead = np.equal(tdead, None)                               #No dead time given in header
badfrac = undead.sum() / len(undead)
print( 'DEADTIME missing in {:3.2%} of headers'.format(badfrac) )

#Set deadtime to zero where missing
tdead[undead] = 0                                            #Assume deadtime = 0 when not given
tdead = tdead.astype(int)

#Fix time jitter
print( 'Fixing timing data.' )
t0 = time()
tolerances=(3e-2, 5e-4)
tfix, figures = fix_time_jitter(utsec, tolerances, undead, plot=True)
print( 'DONE in {} sec'.format(time()-t0) )

#Show the fix for each iteration as multi-tab plot
if len(figures):
    #app = qt.QApplication(sys.argv)
    ui = MplMultiTab( figures=figures )
    #ui.show()
    #app.exec_()

    
#dead time histogram
h, ax = hist( tdead, range=(0,50), bins=50, log=1, 
                    axlabels=[r'$t_{dead}$ (ms)'], 
                    title='Dead time stats' )
ylim = 1e-1, 10**np.ceil(np.log10( h[0].max() ))
ax.set_ylim( ylim )


#fig.savefig( saltpath+'/jitter.ts.fix.png' )
#figh.savefig( saltpath+'/jitter.hist.fix.png' )

from embedded_qtconsole import qtshell
qtshell( vars() )


#light curves
name = 'V2400 Oph'#'V834 Cen' #
F = np.c_[flux_t, flux_c].T
E = np.c_[flux_t_err, flux_c_err].T

fig, plots, *rest = lcplot( (tfix, F, E), labels=[name, 'C1'], axlabels=['t (s)', 'Flux'] )
ax = fig.axes[0]
nwindow = 50
noverlap = 25
out = []
for i,f in enumerate(F):
    idx = WindowOutlierDetection( f, nwindow, noverlap, generalizedESD, maxOLs=2, alpha=0.05 ) #(i+1)*
    ax.plot( tfix[idx], f[idx], 'rx' )    
    out.append( idx )    
out = np.unique( flatten(out) )

lout = np.ones(F.shape[-1], bool)
lout[out] = False

Fc = F[:,lout]
Ec = E[:,lout]
t = t[lout]
 
coof = np.polyfit(t, F[1,lout], 5)
trend = np.polyval(coof, t)
fig, plots, *rest = lcplot( (t, Fc, Ec), labels=[name, 'C1'], axlabels=['t (s)', 'Flux'] )

ax = fig.axes[0]
ax.plot( t, trend, 'r-' )

Fcm = Fc[1].mean()
Fcd = Fc - trend + Fcm
#tf, Fcdf, gidx = fill_gaps( t, Fcd[0], kct, ret_idx=True )

fig, plots, *rest = lcplot( (t, Fcd, Ec), labels=[name, 'C1'], axlabels=['t (s)', 'Flux'] )


#fig, (ax1, ax2) = plt.subplots( 2,1, tight_layout=1, figsize=(18,8) )
#ax1.plot( ohm[1:], Ps, 'b', label=name )
#ax2.plot( ohm[1:], Pc, 'g', label='C1' )

#ax1.set_title( 'LS Periodogram V2400 Oph' )
#ax2.set_xlabel( 'f (Hz)' )
#for ax in fig.axes:    
    #ax.set_ylabel( 'LS power' )
    #ax.set_xlim( 0, f[-1] )
    #ax.set_yscale( 'log' )
    #ax.grid()
    #ax.legend()

#plt.show()

