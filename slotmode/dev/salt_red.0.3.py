
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
    
    
class Extractor(object):
    def __init__(self, filelist, outfilename, **kw):
        #TODO:
        '''
        Extract frames from list of fits files.
        '''
        self.start      = start         = kw.setdefault( 'start',       0               )
        self.step       = step          = kw.setdefault( 'step',        1               )
        self.clobber                    = kw.setdefault( 'clobber',     True            )
        
        self.keygrab                    = kw.setdefault( 'keygrab',     None            )
        self.headkeys                   = []
        
        self.filelist                   = parselist( filelist )
        self.outfilename                = outfilename
        
        self.data_element  = lambda hdulist, i:  ( hdulist[i+1].header, hdulist[i+1].data )                     #return header and data for extension i
        
        #read firts file
        first = pyfits.open( filelist[0], memmap=True )
        pheader = first[0].header

        #Assume all files have same number of extensions and calculate total number of frames.
        Next = pheader.get( 'nextend' )
        self.Nfiles     = Nfiles        = len(filelist)
        self.Ntot       = Ntot          = (Nfiles*Next - start) // step
        self.stop       = stop          = kw.setdefault( 'stop',        Ntot            )
        self.naming                     = kw.setdefault( 'naming',      'sci{}.fits'    )
        self.padwidth   = len(str(Ntot))                #for numerical file naming
        self.count      = 0
        
        self.bar        = kw.pop( 'progressbar', ProgressBar()    )     #initialise progress bar unless False
        
        #create master output file (HduList with header of first file)
        primary = pyfits.PrimaryHDU( header=pheader  )
        self.master = pyfits.HDUList( primary )                    #Use this as the master header for the output file
        #pheader = master.header
        
        #TODO:update master header

    def loop(self, func):
        '''
        loop through the files and extract selected frames.
        '''
        start, stop, step = self.start, self.stop, self.step
        if self.bar:    self.bar.create( stop-1 )
        for i,f in enumerate(self.filelist):
            s = start if i==0 else start%step  #first file start at start, thereafter continue sequence of steps
            with pyfits.open( f, memmap=True ) as hdulist:
                end = len(hdulist)-1
                datamap = map( functools.partial( self.data_element, hdulist ),
                                range(s,end,step) )
                for j, (header, data) in enumerate(datamap):
                    
                    func( data, header )
                    
                    if not self.keygrab is None:
                        self.headkeys.append( [header.get(k) for k in self.keygrab] )
                    
                    if self.bar:        
                        self.bar.progress( self.count )
                    
                    self.count += 1
                    #exit clause
                    if self.count > stop:
                        return

    def _burst(self, data, header):
        numstr = '{1:0>{0}}'.format(self.padwidth, self.count )
        fn = self.naming.format( numstr )
        pyfits.writeto( fn, data, header )

    def _multiext(self, data, header):
        self.master.append( pyfits.ImageHDU( data, header ) )
            
    def _cube(self):
        hdu = self.master[0]
        if hdu.data is None:
            hdu.data = data
        else:
            hdu.data = np.dstack( [hdu.data, data] )
        #pyfits.HeaderDiff

    def burst(self):
        '''Save files individaully.  This is probably quite inefficient (?)'''
        self.loop( self._burst )
    
    def multiext(self):
        '''Save file as one big multi-extension FITS file.'''
        self.loop( self._multiext )
        self.finalize()

    def cube(self):
        '''Save file as one big 3D FITS data cube.'''
        self.loop( self._cube )
        self.finalize()
    
    def finalize(self):
        #optional to update NCCDS, NSCIEXT here
        master = self.master
        header = master[0].header
        header['nextend'] = len(master)
        master.verify()
        master.writeto( self.outfilename, clobber=self.clobber )
        master.close()

        
        
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
filelist = parselist( saltpath+'/bxgp*.fits' )
fnlc = os.path.join( saltpath, 'phot2.lc' )

idx, utsec, flux_r, flux_r_err, xt, yt, flux_t, flux_t_err, xc, yc, flux_c, flux_c_err = np.loadtxt( fnlc, unpack=1 )

#Time jitter stats
fig = show_jitter_stats( utsec ) #, figh 
fig.savefig( saltpath+'/jitter.ts.png' )
#figh.savefig( saltpath+'/jitter.hist.png' )

#read dead time from saved cube
#hdu = pyfits.open( saltpath+'/sci.fits', memmap=1 )
info = superheadhunter( filelist, 'deadtime', 'utc-obs', 'date-obs' )            #dead time in ms
td = info['deadtime'.upper()]
lbad = np.equal(td, None)                               #No dead time given
td[lbad] = 0                                            #Assume deadtime = 0 when not given

raise Exception

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


