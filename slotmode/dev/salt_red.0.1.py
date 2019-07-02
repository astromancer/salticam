import numpy as np
import pylab as plt
from lightcurve import Cube
import os



def extract_frames(filelist, start, stop=None, step=4):
    
    struct = pyfits.open( filelist[0] )
    pheader = struct[0].header
    nextend = pheader.get( 'nextend' )
    stop = nextend if stop is None else stop
    data_element_map = lambda i:  (struct[i+1].header, struct[i+1].data)
    datamap = map(data_element_map, range(start,stop,step))
    
    count = 0
    for header, data in datamap:
        count += 1
        hdu = pyfits.PrimaryHDU()
        hdu.header = pheader
        hdu.header.update(header)
        hdu.data = data
        hduList = pyfits.HDUList(hdu)
        hduList.writeto( 'BL_Hyi_{}.fits'.format(count) )
     #= 
    #hdu = pyfits.PrimaryHDU()
    #hdu.header = struct[0].header
    #hdu.header['NCCDS']=1
    #hdu.header['NSCIEXT']=ntotal-ignorexp
    #hdu.header['NEXTEND']=ntotal-ignorexp
    #hduList=pyfits.HDUList(hdu)
    #hduList.verify()
    #hduList.writeto(writenewfits)

#saltpath = '/media/Oceanus/UCT/Observing/SALT/EX_Hya/20140611/product/'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/NY_Lup/20140606/product'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/BL_Hyi/product'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/2014-1-RSA_OTH-020.20140611/product'
saltpath = '/media/Oceanus/UCT/Observing/SALT/V2400_Oph/20140921/product'
#imfn = saltpath + '/images.txt'

fnlc = os.path.join( saltpath, 'phot2.lc' )

idx, utsec, flux_r, flux_r_err, xt, yt, flux_t, flux_t_err, xc, yc, flux_c, flux_c_err = np.loadtxt( fnlc, unpack=1 )

target, ref = 0,1
cube = Cube(Nstars=2, target=0, ref=1)
cube.utsec = utsec
cube.date = ''

cube['target'].flux = flux_t = np.ma.array( flux_t )
cube['target'].data = 2.5*np.log10(flux_t) 
cube['target'].err = np.ma.array( flux_t_err )
cube['target'].coo = list(zip(xt, yt))


cube['ref'].flux = flux_c =  np.ma.array( flux_c )
cube['ref'].data = 2.5*np.log10(flux_c) 
cube['ref'].err = np.ma.array( flux_c_err )
cube['ref'].coo = list(zip(xc, yc))

cube.stars.set_names(target, ref, 'NY Lup')
cube.stars.set_colours(target, ref)

plt.close('all')
fig, ax = plt.subplots()
cube.plot_lc(ax, tkw='utsec')

plt.grid()
plt.show()
