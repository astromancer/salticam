import numpy as np
import pylab as plt
from lightcurve import Cube
import os

saltpath = '/media/Oceanus/UCT/Observing/SALT/2014-2-SCI-069.20141006/product'
#saltpath = '/media/Oceanus/UCT/Observing/SALT/2014-1-RSA_OTH-020.20140611/product'
#imfn = saltpath + '/images.txt'
fnlc = os.path.join( saltpath, 'opt.lc' )

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

cube.stars.set_names(target, ref, 'BL Hyi')
cube.stars.set_colours(target, ref)

plt.close('all')
fig, ax = plt.subplots()
cube.plot_lc(ax, tkw='utsec')

plt.grid()
plt.show()
