import numpy as np
from obstools.modelling import load_memmap


def make_flat(cube, mask, use=None):
    """
    Construct flat field for SLOTMODE data from on-sky images

    Parameters
    ----------
    cube
    mask

    Returns
    -------

    """
    # TODO: SPEED UP!!

    from scipy import ndimage

    if use is None:
        # select a 1000 frames randomly throughout cube
        n = max(len(cube), 1000)
        use = np.random.randint(0, len(cube), n)

    struct = np.ones((3, 3))
    flat = np.ones(cube.shape[1:])
    for yi, xi in zip(*np.where(mask)):
        # for each pixel in the mask, get the (non-masked) neighbours
        z = np.zeros_like(mask)
        # there are probably far more efficient ways
        z[yi, xi] = True
        neighbours = ndimage.binary_dilation(z, struct) & ~mask
        # ratio of neighbourhood median to pixel median
        f = np.median(cube[use, z] / np.median(cube[use][:, neighbours], 1))

        flat[yi, xi] = f
    return flat


class FlatFieldEstimator(object):
    def __init__(self, folder, pixel_mask, lock, stat=np.ma.median,
                 clobber=True):
        #
        self.use = pixel_mask
        self.shape = pixel_mask.shape
        size = pixel_mask.sum()
        self.stat = stat

        # persistent shared mem
        self.sum = load_memmap(folder / 'sum.dat', size, float, 0,
                               clobber=clobber)
        self.count = load_memmap(folder / 'count.dat', size, int, 0,
                                 clobber=clobber)
        # sync lock
        self.lock = lock

    def aggregate(self, data, ignore_pix, neighbourhood=7):
        # update flat field data
        mdat = np.ma.MaskedArray(data)
        mdat[..., ignore_pix] = np.ma.masked

        neighbours = slotmode.view_neighbours(mdat, neighbourhood)
        leading_dim = (slice(None),) * (data.ndim - 2)
        use = leading_dim + (self.use,)
        #
        ratio = data[use] / self.stat(neighbours[use], (-1, -2))
        counts = np.logical_not(ratio.mask).sum(0)

        # update flat field
        with self.lock:
            # these operations not atomic
            self.sum += ratio.sum(0)
            self.count += counts

    def finalize(self):
        ff = np.ones(self.shape)
        ff[self.use] = self.sum / self.count
        return ff

