import os
import functools

import scipy as sp
import numpy as np
import pyfits

import pylab as plt
from matplotlib.transforms import  blended_transform_factory as btf

from misc import amap, flatten 
from myio import *
from outliers import *
from tsa import get_deltat, detrend, fill_gaps

from suzaku import histnice, lcplot
from superstring import ProgressBar

from superfits import superheadhunter

        
def show_jitter_stats(t, title='Time jitter'):
    '''Plots of time jitter statistics.'''
    deltat = get_deltat( t )
    kct, Nocc = flatten( sp.stats.mode(deltat) )

    #h, ax = histnice( deltat, range=(0,1), bins=1000,              #milisecond time bins
                        #log=True, 
                        #title=title,
                        #axlabels= [r'$\Delta$t (s)', 'Counts'] )
    #ax.axvline( kct, color='g', ls='--', lw=2 )
    #hlim = 1e-1, 10**np.ceil(np.log10( h[0].max() ))
    #ax.set_ylim( hlim )
    #figh = ax.figure

    fig, pl, *rest = lcplot( deltat, fmt='ro',
                                hist='log', bins=1000,
                                title=title ,
                                axlabels=[r'$N_{frame}$',r'$\Delta t$ (s)'], 
                                draggable=False )
    ax, hax = fig.axes
    trans = btf( ax.transAxes, ax.transData )
    N = round( deltat[1:].max() / kct ) + 1
    for i in range(1,int(N)):
        yoff = 0.001
        ax.axhline( i*kct, color='g', lw=2, alpha=0.5 )
        ax.text( 0.001, i*kct+yoff, r'$%it_{kct}$'%i, transform=trans )

    ax.set_ylim( 0, N*kct )
    #hax.set_xlim( hlim )        

    return fig#, figh



        

    
#saltpath = '/media/Oceanus/UCT/Observing/SALT/EX_Hya/20140611/product/'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/NY_Lup/20140606/product'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/BL_Hyi/product'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/2014-1-RSA_OTH-020.20140611/product'
saltpath = '/media/Oceanus/UCT/Observing/SALT/V2400_Oph/20140921/product'
#imfn = saltpath + '/images.txt'

#saltpath = '/media/Oceanus/UCT/Observing/SALT/2014-1-RSA_OTH-020.20140802/product'
filelist = parselist( saltpath+'/images.last50' )
#filelist = parselist( saltpath+'/bxgp*.fits' )
fnlc = os.path.join( saltpath, 'phot2.lc' )

idx, utsec, flux_r, flux_r_err, xt, yt, flux_t, flux_t_err, xc, yc, flux_c, flux_c_err = np.loadtxt( fnlc, unpack=1 )

#Time jitter stats
fig = show_jitter_stats( utsec ) #, figh 
fig.savefig( saltpath+'/jitter.ts.png' )
#figh.savefig( saltpath+'/jitter.hist.png' )

#read dead time from saved cube
keys = 'deadtime', 'utc-obs', 'date-obs'
data = superheadhunter( filelist, keys, return_type='raw' )            #dead time in ms
#rdata = data[:,2::4]

#utdate, utc, tdead = rdata
#l = utdate == '1969-12-31'                              #The last ~50 frames has this date...?  These frames can also be identified by having all 0 pixel values.
#rdata = rdata[:,~l]                                     #Chuck these out


from embedded_qtconsole import qtshell
qtshell( vars() )

#from misc import make_ipshell
#ipshell = make_ipshell()
#ipshell()


td = info['deadtime'.upper()]
lbad = np.equal(td, None)                               #No dead time given in header
td[lbad] = 0                                            #Assume deadtime = 0 when not given



l = lbad[:len(utsec)]                                   #HACK!!
lr1 = np.roll(l,1)                                      #Frames following those without deadtime

#Show flagged datapoints on figure
deltat = get_deltat( utsec )
frames = np.arange(len(deltat))
        
ax = fig.axes[0]
ax.plot( frames[l], deltat[l], 'o', ms=5, mfc='None', mec='r', label=r'$t_{i}$  :  $t_{dead}=0$' )
ax.plot( frames[lr1], deltat[lr1], 'o', mfc='None', mec='b', label=r'$t_{i+1}$' )
ax.figure.canvas.draw()
ax.legend()
fig.savefig( saltpath+'/jitter.ts.flagged.png' )

#dead time histogram
h, ax = histnice( td, range=(0,50), bins=50, log=1, 
                    axlabels=[r'$t_{dead}$ (ms)'], 
                    title='Dead time stats' )
ylim = 1e-1, 10**np.ceil(np.log10( h[0].max() ))
ax.set_ylim( ylim )

#Fix time jitter
t = utsec.copy()
kct, Nocc = flatten( sp.stats.mode(deltat) )
kct = kct.round(6)
dd = kct - deltat[l]
t[l] += dd

fig = show_jitter_stats( t, 'Time jitter (fixed)' )#, figh
fig.savefig( saltpath+'/jitter.ts.fix.png' )
#figh.savefig( saltpath+'/jitter.hist.fix.png' )



#light curves
name = 'V2400 Oph'#'V834 Cen' #
F = np.c_[flux_t, flux_c].T
E = np.c_[flux_t_err, flux_c_err].T

fig, plots, *rest = lcplot( (t, F, E), labels=[name, 'C1'], axlabels=['t (s)', 'Flux'] )
ax = fig.axes[0]
nwindow = 50
noverlap = 25
out = []
for i,f in enumerate(F):
    idx = WindowOutlierDetection( f, nwindow, noverlap, generalizedESD, maxOLs=2, alpha=0.05 ) #(i+1)*
    ax.plot( t[idx], f[idx], 'rx' )    
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


