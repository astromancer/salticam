import itertools as itt

from astropy.stats import sigma_clipped_stats
from photutils.detection import detect_sources
#from photutils.background import Background
from scipy.ndimage import binary_dilation
from scipy.spatial import KDTree


def neighbour_fill(data, mask=None, method='median', k=5, **kw ):
    '''Fill masked values with median of k nearest neighbours '''
    grid = grid_like(data)
    mask = data.mask if mask is None else mask
    good = yg, xg =  grid[:,~mask]
    bad = grid[:,mask].T
    filled = data.copy()
    #exclusion region where neighbours cannot be picked from
    exclude = kw.get('exclude')
    if not exclude is None:
        exx, exy = exclude
        lex = (exx[0] < xg) & (xg < exx[1]) &                 (exy[0] < yg) & (yg < exy[1])
        good = good[:, lex]
    
    tree = KDTree( good.T )
    _dist, _ix = tree.query(bad, k) #intermediate indeces
    ix = tree.data[_ix] #image pixel coordinates of nearest neighbours
    nn = data[ ix[...,0],ix[...,1] ] #nearest neighbour pixel value
    nn = np.atleast_2d(nn)
    
    if method == 'mean':
        fillvals = np.mean(nn, axis=1)
    if method=='median':
        fillvals = np.median(nn, axis=1)    
    if method=='mode':
        fillvals = np.squeeze( scipy.stats.mode(nn, axis=1)[0] )
    if method=='weighted':
        weights = kw['weights']
        w = weights[ ix[...,0],ix[...,1] ]
        fillvals = np.average( nn, axis=1, weights=w )
    #if 'clip' in method:

    filled[bad[...,0], bad[...,1]] = fillvals
    return filled

def get_threshold(image, sigma_threshold, sigma_clip):
    #mean, median, std = np.ma.mean( image ), np.ma.median( image ), np.ma.std( image )
    mean, median, std = sigma_clipped_stats(image, sigma=sigma_clip)
    return median + (std * sigma_threshold)

def pull_mask(image, threshold=None, sigma_threshold=3.0, sigma_clip=2.5, badpixels=None):
  
    threshold = threshold or get_threshold(image, sigma_threshold, sigma_clip)

    segm_img = detect_sources(image, threshold, npixels=3)
    det_mask = segm_img.astype(bool)
    if not badpixels is None:
        det_mask &= ~badpixels
    
    #To ensure completely masking of extended detected sources, dilate the source mask
    #dilate_box0 = np.ones((5, 5))# dilate using a 8x8 box
    _masks = [binary_dilation(det_mask, dilate) for dilate in boxes]
    mask = np.any(_masks, axis=0)
    return mask



path = '/media/Oceanus/UCT/Observing/SALT/MASTER_J0614-2725/20150424/product/'
ppath = Path(path).resolve()
data = pyfits.getdata( str(ppath/'extr.cube.fits') )

image = data[0]
grid = xg,yg = grid_like( image )
#mean, median, std = sigma_clipped_stats(image, sigma=3.0)
#threshold = median + (std * 2.0)

badrows = 59, 295
badpixels = np.any([xg == br for br in badrows], axis=0)


#ix = np.random.choice( np.arange(len(data)), 100 )
bg_subset = data[:100,...]
boxes = np.ones((13, 15)), np.ones((20, 3))

mask = np.array( [pull_mask(im, sigma_threshold=3.0, badpixels=badpixels) 
                      for im in bg_subset] )

bg_subset = np.ma.masked_where(mask, bg_subset)
bg_med = np.ma.median(bg_subset, axis=0)

#Now do it again, this time detect fainter sources!
mask2 = pull_mask( bg_med, sigma_threshold=3., sigma_clip=2., badpixels=badpixels )
bg_med.mask |= mask2

#fill the bad pixels
filled = neighbour_fill(bg_med, badpixels, k=5, method='median' )

#filled masked stars
exclusion_bounds = (8, image.shape[1]), (8, 39)
filled = neighbour_fill(filled, k=12, method='median', exclude=exclusion_bounds )



#Do a quick psf fit to the stars in the initial frame to get some basic properties
#from psffit import GaussianPSF, StellarFit, PSFPlot, Snapper
#psf = GaussianPSF()
#fitter = StellarFit( psf )
#plotter = PSFPlot()
#snap = Snapper(data[0], window=25, snap='peak', edge='edge')

#starcoo = []
#starrad = []
#grid = Y, X = shape2grid( snap.window )
#for i,(y,x) in enumerate(LoGcoo[:,:2]):
    #image, offset = snap.zoom( *snap(x,y) )
    #yoff, xoff = offset
    
    #plsq = fitter(grid, image)
    #Z = psf( plsq, X, Y )
    #plotter.update( X+xoff, Y+yoff, Z, image )
    #savefig( plotter.fig, ppath/'star{}.png'.format(i) )

    #info = psf.get_description( plsq, offset )
    #table = Table( info, 
                   #title='Stars', title_props={'text':'bold', 'bg':'light green'} )
    #print( table )
    
    #starcoo.append( info['coo'][::-1] )
    #starrad.append( 3*info['fwhm'] )
#starcoo = np.array(starcoo)
#staricoo = np.round(starcoo[:,::-1]).astype(int)


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
