import numpy as np
from obstools.modelling import load_memmap


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