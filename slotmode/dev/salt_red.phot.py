import itertools as itt

from astropy.stats import sigma_clipped_stats
from photutils.detection import detect_sources
#from photutils.background import Background
from scipy.ndimage import binary_dilation
from scipy.spatial import KDTree




#Find stars
bg = np.median( data )
bg_sig = mad_std( data[0] )
sig_thresh = 2.5
threshold = sig_thresh * bg_sig

image = data[0] / data[0].max()         #normalized image
min_sigma, max_sigma, num_sigma = 3, 10, 10
threshold = sig_thresh * bg_sig / data[0].max()
LoGcoo = LoG(image, min_sigma, max_sigma, num_sigma, threshold)
#blobs_dog = blob_dog(image, min_sigma, max_sigma, threshold)
#blobs_doh = blob_doh(image, min_sigma, max_sigma, num_sigma, threshold)

#TODO:  Cross check found stars for consistency! DoG, DoH, daofind.....

#Do a quick psf fit to the stars in the initial frame to get some basic properties
from psffit import GaussianPSF, StellarFit, PSFPlot, Snapper
psf = GaussianPSF()
fitter = StellarFit( psf )
plotter = PSFPlot()
snap = Snapper(data[0], window=25, snap='peak', edge='edge')

starcoo = []
starrad = []
grid = Y, X = shape2grid( snap.window )
for i,(y,x) in enumerate(LoGcoo[:,:2]):
    image, offset = snap.zoom( *snap(x,y) )
    yoff, xoff = offset
    
    plsq = fitter(grid, image)
    Z = psf( plsq, X, Y )
    plotter.update( X+xoff, Y+yoff, Z, image )
    savefig( plotter.fig, ppath/'star{}.png'.format(i) )

    info = psf.get_description( plsq, offset )
    table = Table( info, 
                   title='Stars', title_props={'text':'bold', 'bg':'light green'} )
    print( table )
    
    starcoo.append( info['coo'][::-1] )
    starrad.append( 3*info['fwhm'] )
starcoo = np.array(starcoo)
staricoo = np.round(starcoo[:,::-1]).astype(int)


#Plot the position of the fit stars on the first frame
fig, ax = plt.subplots( figsize=figsize, tight_layout=True )
supershow( ax, data[0], origin='lower', autoscale='z' )

aps = ApertureCollection( radii=starrad, coords=starcoo )
aps.axadd(ax)
savefig( fig, ppath/'firstframe.png' )






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
