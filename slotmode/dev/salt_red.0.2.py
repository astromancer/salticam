import numpy as np
import pylab as plt

import os
import functools

from suzaku import lcplot
#from lightcurve import Cube

from misc import amap, flatten 
from myio import *

def quickheader( filename ):
    with open(filename,'rb') as fp:
        return pyfits.Header.fromfile( fp )

def quickheadkey(filename, key, default=None):
    '''Quick method to grad header keyword from fits file.'''
    return quickheader( filename ).get(key, default)
    
def headhunter( hdu, key, default=None ):
    '''Return an array containing the header keywords for multi-extension data cube.'''
    grab = lambda h : h.header.get(key, default)
    return amap( grab, hdu[1:] )

class Extractor(object):
    def __init__(self, filelist, outfilename, **kw):
        #TODO:
        '''
        Extract frames from list of fits files.
        '''
        self.start      = start         = kw.setdefault( 'start',       0               )
        self.step       = step          = kw.setdefault( 'step',        1               )
        self.clobber                    = kw.setdefault( 'clobber',     True            )

        self.filelist                   = parselist( filelist )
        self.outfilename                = outfilename
        
        self.data_element = lambda hdulist, i:  (hdulist[i+1].header, hdulist[i+1].data)                     #return header and data for extension i

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
        
        self.progressbar                = kw.setdefault( 'progressbar',      True    )
        
        #create master output file (HduList with header of first file)
        primary = pyfits.PrimaryHDU( header=pheader  )
        self.master = pyfits.HDUList( primary )                    #Use this as the master header for the output file
        #pheader = master.header
        
        #TODO:update master header

    def loop(self, func):
        '''loop through the files and extract selected frames.'''
        start, stop, step = self.start, self.stop, self.step
        for i,f in enumerate(self.filelist):
            s = start if i==0 else start%step  #first file start at start, thereafter continue sequence of steps
            with pyfits.open( f, memmap=True ) as hdulist:
                end = len(hdulist)-1
                datamap = map( functools.partial( self.data_element, hdulist ),
                                range(s,end,step) )
                for j, (header, data) in enumerate(datamap):
                    
                    func( data, header )
                    
                    self.count += 1
                    #exit clause
                    if self.count > stop:
                        return

    def _burst(self, data, header):
        numstr = '{1:0>{0}}'.format(self.padwidth, self.count )
        fn = self.naming.format( numstr )
        pyfits.writeto( fn, data, header )

    def _multiext(self, data, header):
        self.master.append( pyfits.PrimaryHDU( data, header ) )

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

        


        
from suzaku import histnice, lcplot
from tsa import get_deltat
import scipy as sp

import pyfits
from matplotlib.transforms import  blended_transform_factory as btf
    
#saltpath = '/media/Oceanus/UCT/Observing/SALT/EX_Hya/20140611/product/'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/NY_Lup/20140606/product'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/BL_Hyi/product'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/2014-1-RSA_OTH-020.20140611/product'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/V2400_Oph/20140921/product'
#imfn = saltpath + '/images.txt'

saltpath = '/media/Oceanus/UCT/Observing/SALT/2014-1-RSA_OTH-020.20140802/product'
fnlc = os.path.join( saltpath, 'phot2.lc' )

#light curves
idx, utsec, flux_r, flux_r_err, xt, yt, flux_t, flux_t_err, xc, yc, flux_c, flux_c_err = np.loadtxt( fnlc, unpack=1 )

F = np.c_[flux_t, flux_c].T
E = np.c_[flux_t_err, flux_c_err].T

#fig, plots, *rest = lcplot( (utsec, F, E), labels=['V834 Cen', 'C1'], axlabels=['t (s)', 'Flux'] )
#plots.connect()

#Time jitter stats
deltat = get_deltat( utsec )
kct, Nocc = flatten( sp.stats.mode(deltat) )

h, ax = histnice( deltat, range=(0,1), bins=1000,              #milisecond time bins
                    log=True, 
                    title='Time jitter histogram',
                    axlabels= [r'$\Delta$t (s)', 'Counts'] )
ax.axvline( kct, color='g', ls='--', lw=2 )
hlim = 1e-1, 10**np.ceil(np.log10( h[0].max() ))
ax.set_ylim( hlim )

fig, pl, *rest = lcplot( deltat, fmt='ro',
                            hist='log', bins=1000,
                            title='Time jitter' ,
                            axlabels=[r'$N_{frame}$',r'$\Delta t$ (s)'], 
                            draggable=False )
ax, hax = fig.axes
trans = btf( ax.transAxes, ax.transData )
for i in range(1,6):
    yoff = 0.001
    ax.axhline( i*kct, color='g', ls='--', lw=2 )
    ax.text( 0.001, i*kct+yoff, r'$%it_{kct}$'%i, transform=trans )

ax.set_ylim( 0, 5.5*kct )
hax.set_xlim( hlim )
#hax.set_xscale('log')


#dead time histogram
#td = headhunter( hdu, 'deadtime' )  #dead time in ms
#lbad = np.equal(td, None)           #No dead time given
#h = histnice( td[~lbad], log=1, axlabels=[r'$t_{dead}$ (ms)'] )



plt.show()



