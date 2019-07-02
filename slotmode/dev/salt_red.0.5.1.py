import os
import functools

import scipy as sp
import numpy as np
import pyfits
from astropy.coordinates import EarthLocation

import pylab as plt
from matplotlib.transforms import  blended_transform_factory as btf
from mplMultiTab import MplMultiTab

from misc import flatten 
from myio import *
from outliers import *
from tsa import get_deltat, detrend, fill_gaps
from chronos import Time

from superplot import lcplot, hist
from superstring import ProgressBar

from superfits import superheadhunter

        
def show_jitter_stats(t, title='Time jitter'):
    '''Plots of time jitter statistics.'''
    #plot deltat as TS + histogram
    deltat = get_deltat( t )
    kct, Nocc = flatten( sp.stats.mode(deltat) )

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

    return fig#, figh


def fixer(t, i, tols=(5e-2,5e-4)):
    '''Fix the frame arrival times for the interval (i*kct,(i+1)*kct) by subtracting the 
    difference in arrival times from the next higher multiple of kct.  i.e. For a frame n arriving 
    at time tn, assume the correct arrival time is (i+1)*kct. The positive difference (i+1)*kct-tn
    is subtracted from the preceding frame
    between '''
    ltol, utol = tols
    tfix = t.copy()
    deltat = get_deltat( tfix )
    
    #flagged points in interval of kct for which to correct
    l0 = (i*kct+ltol < deltat) & (deltat < (i+1)*kct-utol)
    #We only apply the correction if the frame is one following one with missing DEADTIME
    l = l0 & np.roll( lbad, 1 )
    #If no flagged points, no fix needed
    if ~np.any(l):
        return None, None#no fix required for this interval

    diff = (i+1)*kct - deltat[l]  #discrepancy in kct
    #print( 'diff\n', diff )
    #print( diff.max(), diff.min(), diff.ptp() )
    lrb1 = np.roll(l,-1)
    
    tfix[lrb1] -= diff
    return tfix, l

    
def plot_deltat():
    #plot deltat as TS
    fig = show_jitter_stats( tfix )
    ax = fig.axes[0]
    #Mark flagged values
    ax.plot( frames[l], deltat[l], 'o', ms=7.5, mfc='none', mew=1.5, mec='g', label='flagged' )
    ax.plot( frames[lbad], deltat[lbad], 'o', ms=10, mfc='none', mew=1.5, mec='r', label=r'$t_{i}$  :  $t_{dead}=0$' )
    ax.plot( frames[lb2], deltat[lb2], 'o', ms=12.5, mfc='none', mew=1, mec='orange', label=r'$t_{i-1}$' )            
    
    #show selection limits
    y0, y1 = i*kct+ltol, (i+1)*kct-utol
    ax.axhline( y0 );             ax.axhline( y1 )
    ax.legend()
    figures.append( fig )


        
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


#read dead time from saved cube
keys = 'deadtime', 'utc-obs', 'date-obs'
data = superheadhunter( filelist[:100], keys, return_type='array' )            #dead time in ms
rdata = data[:,2::4]

utdate, utc, tdead = rdata
l = utdate == '1969-12-31'                              #The last ~50 frames has this date...?  These frames can also be identified by having all 0 pixel values.
rdata = rdata[:,~l]                                     #Chuck these out
utdate, utc, tdead = rdata


ts = map( 'T'.join, zip(utdate, utc) )
t = Time( list(ts), format='isot', scale='utc', precision=9, location=sutherland )
utsec = t.sec
    
lbad = np.equal(tdead, None)                               #No dead time given in header
#lr1 = np.roll(lbad,1)                                      #Frames following those without deadtime
badfrac = lbad.sum() / len(lbad)
print( 'DEADTIME missing in {:3.2%} of headers'.format(badfrac) )

#
tdead[lbad] = 0                                            #Assume deadtime = 0 when not given
tdead = tdead.astype(int)

#
tolerances = ltol, utol = (3e-2, 5e-4)
tfix = utsec.copy()
deltat = get_deltat( tfix )
frames = np.arange(len(deltat))
kct, Nocc = flatten( sp.stats.mode(deltat) )
kct = kct.round(6)

lb2 = np.roll( lbad, 1 )

figures = []
i, j = 0, 0
print( 'enter loop' )
while i < 3:
    #input( '..' )
    print( '\tj = ', j )
    _tfix, l = fixer( tfix, i, tolerances )
    
    if _tfix is None:
        i += 1
        j = 0
        print( 'i = ', i )
  
    else:
        print( '\tl.sum()', l.sum() )
        j += 1
        deltat = get_deltat( tfix )
        plot_deltat()
        tfix = _tfix

fig = show_jitter_stats( tfix )
figures.append( fig )
#plt.show()


from embedded_qtconsole import qtshell
qtshell( vars() )




#Time jitter stats
fig = show_jitter_stats( utsec ) #, figh 
#fig.savefig( saltpath+'/jitter.ts.png' )
#figh.savefig( saltpath+'/jitter.hist.png' )


#Show flagged datapoints on figure
#deltat = get_deltat( utsec )

        
ax = fig.axes[0]
ax.plot( frames[lbad], deltat[lbad], 'o', ms=5, mfc='None', mec='r', label=r'$t_{i}$  :  $t_{dead}=0$' )
ax.plot( frames[lr1], deltat[lr1], 'o', mfc='None', mec='b', label=r'$t_{i+1}$' )
ax.figure.canvas.draw()
ax.legend()
#fig.savefig( saltpath+'/jitter.ts.flagged.png' )




#dead time histogram
h, ax = hist( tdead, range=(0,50), bins=50, log=1, 
                    axlabels=[r'$t_{dead}$ (ms)'], 
                    title='Dead time stats' )
ylim = 1e-1, 10**np.ceil(np.log10( h[0].max() ))
ax.set_ylim( ylim )

#Fix time jitter
tfix = utsec.copy()
kct, Nocc = flatten( sp.stats.mode(deltat) )
kct = kct.round(6)

tol = 1e-6
lsm = kct-deltat > tol

dd = kct - deltat[lsm]
tfix[lsm] += dd

fig = show_jitter_stats( tfix, 'Time jitter (fixed)' )#, figh
fig.savefig( saltpath+'/jitter.ts.fix.png' )
#figh.savefig( saltpath+'/jitter.hist.fix.png' )



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

plt.show()


