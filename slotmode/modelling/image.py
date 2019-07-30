# std libs
import logging
import itertools as itt
import multiprocessing as mp

# third-party libs
import numpy as np
import more_itertools as mit
from scipy import ndimage

# local libs
from salticam.slotmode import get_binning, _check_channel
from recipes.introspection.utils import get_module_name
from obstools.stats import mad
from obstools.modelling.core import Model
from obstools.phot.utils import shift_combine
from obstools.modelling.image import SegmentedImageModel
from obstools.phot.segmentation import SegmentationHelper, \
    SegmentationGridHelper
from obstools.modelling.core import UnconvergedOptimization

# relative libs
from .spline2d import Spline2D

PHOTON_BLEED_THRESH = 3e4
PHOTON_BLEED_WIDTH = 10
# TODO YOU CAN PROBABLY FIT THESE PARAMETERS WITH A GOOD MODEL

# module level logger
logger = logging.getLogger(get_module_name(__file__))


def get_order_pairs(yrange, xrange):
    assert len(yrange) == len(xrange) == 2, 'Invalid order range'
    return np.mgrid[slice(*yrange), slice(*xrange)].reshape(2, -1).T + 1


def _gen_order_tuple(low, high):
    assert len(low) == len(high)
    return itt.product(*map(range, low, high))


def _gen_order_tuples(min_orders, max_orders):
    return itt.product(*map(_gen_order_tuple, min_orders, max_orders))


def _try_fit(orders, knots, image, do_knot_search, kws):
    # construct
    mdl = SlotModeBackground(orders, knots)
    # fit
    try:
        if do_knot_search:
            mdl.optimize_knots(image, **kws)
        r = mdl.fit(image, **kws)
    except UnconvergedOptimization as err:
        r, gof = None, None
        # print(i, o, 'fail')
    else:
        gof = mdl.redchi(r, image)
        # print(i, o, 'success')
    return mdl, r, gof


def detect_combine(images, shifts, mask=False, background=None, snr=3.,
                   npixels=7, edge_cutoff=None, deblend=False, dilate=0,
                   n_accept=2):
    images = np.asarray(images)
    seg_data = np.zeros(images.shape, int)
    for i, image in enumerate(images):
        seg = SegmentationHelper.detect(image, mask, background, snr,
                                        npixels, edge_cutoff, deblend, dilate)
        seg_data[i] = seg.data

    #
    eim = shift_combine(seg_data, shifts, 'sum', extend=True)
    return ndimage.label(eim > n_accept)


def flux_estimate(seg, image, labels=None, label_bg=(0,)):
    """

    Parameters
    ----------
    seg
    image
    labels
    label_bg

    Returns
    -------

    """
    if not isinstance(seg, SegmentationHelper):
        seg = SegmentationHelper(seg)

    star_flux = seg.flux(image, labels, label_bg)
    pixel_area = np.product(get_binning(image))
    star_flux /= pixel_area
    # note this is a global bg flux estimate which may not be very
    #  accurate, but should be sufficient to find out which stars are
    #  bright enough to cause bleeding
    return star_flux


def flux_estimate_annuli(image, seg, labels=None):
    """

    Parameters
    ----------
    seg
    image
    labels

    Returns
    -------

    """
    if not isinstance(seg, SegmentationHelper):
        seg = SegmentationHelper(seg)

    star_flux = seg.flux(image, labels, label_bg)
    pixel_area = np.product(get_binning(image))
    star_flux /= pixel_area
    # note this is a global bg flux estimate which may not be very
    #  accurate, but should be sufficient to find out which stars are
    #  bright enough to cause bleeding
    return star_flux


def gauss1d(p, grid):
    x0, z0, a = p
    _, gx = grid
    r = np.square(gx - x0)
    return z0 * np.exp(-a * r)


class VerticalGaussianStreaks(Model):

    def __init__(self, loc, grids):
        self.dof = len(loc) + 1
        self.loc = np.array(loc)
        self.grids2 = np.square(np.array(grids) - self.loc)
        self.ranges = np.transpose([(g.min(), g.max() + 1) for g in grids])

    def get_slice(self, offset):
        return list(map(slice, *(self.ranges + offset)))

    def rss(self, p, data, grid=None):
        self.rs(p, data, grid)

    def __call__(self, p):
        *z0, a = p
        return z0 * np.array(list(map(np.exp, -a * self.grids2)))

        # return gauss1d(p, grid)

    # def p0guess(self, data, grid, std=None):
    #     bg = data[[0, -1]].mean()
    #     data_bg = data - bg
    #     imx = data_bg.argmax()
    #     z0 = data_bg[imx]
    #     x0 = grid[imx]  # .data
    #     # print('p0', (x0, z0, 1))
    #     return (x0, z0, 1)
    #
    # def fit(self, data, grid, stddev=None, p0=None, **kws):
    #     # print(Gaussian1D)
    #     # print(grid)
    #     bounds = [(grid.min(), grid.max()), (0, 1e6), (0, 1e2)]
    #     # TODO:  include in staticGridMixin?
    #     return Model.fit(self, data, grid, stddev, p0, bounds=bounds, **kws)


class MedianEstimator(Model):
    axis = -2

    def fit(self, data, grid=None, stddev=None, p0=None, *args, **kws):
        return np.ma.median(data, self.axis, **kws)

    def residuals(self, p, data, grid=None):
        return data - p

    # def p0guess(self, data, grid=None, stddev=None):
    #     return


# class VerticalGaussianStreak(Gaussian1D):  # SummaryStatFitter
#     # name = 'bleed'
#
#     def __init__(self):
#         SummaryStatFitter.__init__(self, 1, 2)
#
#     def adapt_grid(self, grid):
#         return grid[self.axis, 0]
#         # print(Streak)
#         # print(grid.shape)
#         # return grid[self.axis, 0]  # zeroth row of x-grid


class FrameTransferBleed(SegmentedImageModel):
    @classmethod
    def from_image(cls, image, seg, flux_threshold=20, width=PHOTON_BLEED_WIDTH,
                   labels=None,
                   label_insert=None):
        """

        Parameters
        ----------
        image
        seg
        flux_threshold
        width
        labels
        label_insert

        Returns
        -------

        """
        # todo: default width should be in unbinned pixels?

        if not isinstance(seg, SegmentationHelper):
            seg = SegmentationHelper(seg)

        if labels is None:
            labels = seg.labels_nonzero
        if len(labels) == 0:
            return seg, []

        # Decide based on flux in segments and threshold, which stars
        # need segments for modelling frame transfer bleeding
        star_flux = flux_estimate(seg, image, labels)
        bright = seg.labels[star_flux > flux_threshold]
        if len(bright):
            seg, new_labels = cls.adapt_segments(seg, bright, width,
                                                 label_insert)
            obj = cls()
        else:
            obj = None
        return obj, seg, bright

    @classmethod
    def from_segments(cls, seg, labels=None, loc=None,
                      width=PHOTON_BLEED_WIDTH):
        #
        if loc is None:
            loc = seg.com(labels)

        seg, labels = cls.adapt_segments(seg, labels, loc, width, copy=True)
        return cls(seg, loc)

    @staticmethod
    def adapt_segments(seg, labels=None, loc=None, width=PHOTON_BLEED_WIDTH,
                       label_insert=None, copy=False):
        """

        Parameters
        ----------
        seg
        labels
        loc
        width
        label_insert
        copy

        Returns
        -------

        """

        if not isinstance(seg, SegmentationHelper):
            seg = SegmentationHelper(seg)

        #
        new = np.zeros(seg.shape, int)

        if loc is None and labels is None:
            raise ValueError('Either labels or loc should be given and not '
                             'None')

        if loc is None:
            loc = seg.com(labels=labels)[:, 1]

        # segment regions
        rng = np.atleast_2d(loc) - 0.5 * np.multiply(width, [1, -1])[:, None]
        rng = rng.round().astype(int).clip(0, seg.shape[1]).T
        for i, r in enumerate(rng, 1):
            new[:, slice(*r)] = i

        # non-overlapping segments
        new[seg.to_boolean()] = 0

        if copy:  # return new object with only new segments
            seg = seg.__class__(new)
            return seg, seg.labels
        else:
            return seg.add_segments(new, label_insert)

    def __init__(self, segm, labels, adapt=False, width=PHOTON_BLEED_WIDTH):

        invalid = np.setdiff1d(labels, segm.labels)
        if len(invalid):
            raise ValueError('Invalid labels %s' % invalid)

        if adapt:
            segm, labels = self.adapt_segments(segm, labels)

        #
        SegmentedImageModel.__init__(self, segm)

        me = MedianEstimator()
        me.dof = int(width)
        self.add_model(me, labels)

    def residuals(self, p, data, xy_offset):
        x_slices = list(map(slice, *(self.ranges - xy_offset[1])))
        # modelled = self(p, gx2)

        # if we have a masked image, unmask here to get residuals of full image
        for m, xs in zip(p, x_slices):
            data[..., xs] -= m
        return data

    def set_segments(self, segm, adapt=False, labels=None,
                     width=PHOTON_BLEED_WIDTH):
        if not isinstance(segm, SegmentationGridHelper):
            segm = SegmentationGridHelper(segm)

        if adapt:
            segm, labels = self.adapt_segments(segm, labels, self.centres,
                                               width, copy=True)
        # set segmentation
        self.segm = segm
        # set mask
        self.masks = ~np.array(list(segm.coslice(segm.to_bool())))
        # set grid
        # self.set_centres(self.centres)

    # def set_centres(self, loc):
    #     x = np.array(loc)[:, None, None]
    #     gx = np.array(list(self.segm.coord_grids.values()))[:, 1]
    #     self.gx2 = np.square(gx - x)

    # @property
    # def dof(self):
    #     return len(self.centres) + 1

    # def get_dtype(self, **kws):
    #     return float

    # def p0guess(self, data, grid=None, stddev=None):
    #     p0 = np.empty(self.dof)
    #     p0[0] = 1
    #     p0[1:] = np.ma.median(data, 1).max(1)
    #     return p0

    # def fit(self, data, grid=None, stddev=None, p0=None, *args, **kws):
    #     kws.setdefault('bounds', [(0, None)] * self.dof)
    #     return Model.fit(self, data, grid, stddev, p0, *args, **kws)

    # def fit(self, data, grid=None, stddev=None, p0=None, *args, **kws):
    #     # HACK
    #     off = kws.pop('xy_offset')
    #     data, _ = self._prepare_data(data, off)
    #     return np.ma.median(data, -2)

    def pre_process(self, p0, data, grid=None, stddev=None, *args, **kws):
        off = kws.pop('xy_offset')
        data, gx2 = self._prepare_data(data, off)
        return super().pre_process(p0, data, gx2, stddev, *args, **kws)

    def _prepare_data(self, data, xy_offset):
        # get slices
        yo, xo = xy_offset
        x_slices = list(map(slice, *(self.ranges - xo)))
        y_slice = slice(yo, yo + data.shape[-2])

        # get grid
        # gx2 = self.gx2[:, y_slice]

        # get data
        data = np.ma.array([data[..., _] for _ in x_slices])
        # apply mask
        ix = (slice(None),) + (None,) * (data.ndim - 3) + (y_slice,)
        data.mask |= self.masks[ix]
        return data  # , gx2

    def reduce_image(self, image, p, xy_offset):
        # data = self._prepare_data(image, xy_offset)
        x_slices = list(map(slice, *(self.ranges - xy_offset[1])))
        # modelled = self(p, gx2)

        # if we have a masked image, unmask here to get residuals of full image
        image = np.ma.getdata(image)
        for m, xs in zip(p, x_slices):
            image[..., xs] -= m
        return image

    def reduce_image_stat(self, image, xy_offset, statistic=np.ma.median):
        # get image sections to which model applies
        data, _ = self._prepare_data(image, xy_offset)
        x_slices = list(map(slice, *(self.ranges - xy_offset[1])))

        # compute stat
        modelled = statistic(data, -2)

        # if we have a masked image, unmask here to get residuals of full image
        image = np.ma.getdata(image)
        for m, xs in zip(modelled, x_slices):
            image[..., xs] -= m
        return image, modelled

    # def __init__(self, ):

    # models = {}
    # name_base = 'g'
    # for i, lbl in enumerate(segm.labels):
    #     m = VerticalGaussianStreak()
    #     m.name = name_base + str(i)
    #     models[lbl] = m
    #
    # super().__init__(segm, models)

    def plot_fit_results(self, fig, p, data, grid, std, modRes=50):

        ax1, ax2 = fig.axes

        gm = np.linspace(*grid[[0, -1]], modRes)
        dfit = self(p, gm)
        lines_mdl, = ax1.plot(gm, dfit, 'r-', label='model')


def spline_order_search1(cls, image, channel, report=None, **kws):
    # brute force optimization for polynomial orders
    # TODO: look at scipy.optimize.brute ???

    # assert len(labels) == 2, 'Invalid pair of labels'
    # # todo: additionally, you may want to check that they are adjacent
    # assert len(yrange) == 2, 'Invalid y order range'
    # assert len(xrange) == 2, 'Invalid x order range'

    MIN_SPLINE_ORDERS = (3, 1, 3), (1, 3)
    MAX_SPLINE_ORDERS = (8, 3, 8), (3, 8)

    kws = dict(method='leastsq')
    knots = cls.guess_knots(image, channel)
    model = cls(MAX_SPLINE_ORDERS, knots)
    # primary = model.models[1]

    # r = model.fit_sequential(image, labels=1, **kws)

    # first set all dof to 0
    for m in model.models.values():
        m.free[:] = False

    # systematically increase orders
    p_best = []
    for m, (l,) in model.models_to_labels().items():
        print('-' * 80, l, m, '-' * 80)

        seg = model.segm.slices[l - 1]
        params, gof, i_best = m.order_search(image[seg], **kws)
        p_best.append(params[i_best])
        if l == 2:
            return params, gof, i_best

    return model, p_best


from .spline2d import Spline2DImage


class SlotModeBackground_V2(Spline2DImage, SegmentedImageModel):

    def __init__(self, orders, knots, smooth=True, continuous=True,
                 primary=None):
        Spline2DImage.__init__(self, orders, knots, smooth, continuous, primary)
        SegmentedImageModel.__init__(self, self.get_segmented_image(),
                                     list(self.models))

    @classmethod
    def from_image(cls, image, channel, orders, detection=None,
                   plot=False, **detect_opts):
        #
        """
        Construct a Spline2D instance from an image and polynomial
        multi-order.  Position of the knots will be estimated based on the
        median cross sections of the image. Objects (stars) in the image
        will be identified using threshold detection algorithm.

        Parameters
        ----------
        image
        channel
        orders
        detection


        Returns
        -------

        """

        #
        knots = cls.guess_knots(image, channel, plot=plot)
        mdl = cls(orders, knots)
        # mdl.groups['spline2d'] = mdl.segm.labels

        # Detect objects & segment image
        if detection is True or detect_opts:
            from obstools.phot.segmentation import SegmentationHelper
            detection = SegmentationHelper.detect

        if detection not in (None, False):
            seg = detection(image, False, None, **detect_opts)
        else:
            seg = mdl.segm

        return mdl, seg

    @staticmethod
    def guess_knots(image, channel, δσ=3, n_knots=(1, 2), edges=True,
                    offsets=(0, 0), plot=False):
        """
        A fast estimate of spline knot positions from a SALTICAM image by
        sigma-thresholding the point to point gradient of the cross sectional
        image median.

        Parameters
        ----------
        image:  array
            SALTICAM slotmode image to use for guessing the knots
        channel: int, {0-3}
            amplifier channel number (0-3)
        δσ:
            Malhanobis distance cutoff
        n_knots: tuple
            number of knots in each dimension. Only allow {1, 2} knots in
            each dimension
        edges: bool
            include end points as knots
        offsets: tuple of int
            y,x values to add to the guessed knot positions to get final result.
            Somewhat of a hack.
        plot: bool or callable
            if bool:
                Whether or not to make plots of the guessed knot positions
            if callable:
                The function used to display the plots in the interactive
                console

        Returns
        -------

        """
        channel = _check_channel(channel)
        assert len(n_knots) == 2

        if plot:
            import matplotlib.pyplot as plt
            from collections import Callable

            if isinstance(plot, Callable):
                display = plot
                plot = True
            else:
                def display(*_):
                    pass

            #
            figsize = (12, 6)
            fig_x, axes_x = plt.subplots(2, 1, figsize=figsize)
            fig_y, axes_y = plt.subplots(2, 1, figsize=figsize)

        #
        logger.info('Guessing knot positions for image background spline '
                    'model via `guess_knots_gradient_threshold`')

        yx_knots = []
        # xo, yo = offsets  # offsets (found empirically)
        for i in range(2):
            #
            oi = int(not bool(i))
            knots, (m, dm, mm, s, w) = guess_knots_gradient_threshold(
                    image, channel, i, n_knots[oi], δσ, edges)
            yx_knots.append(knots)

            if plot:
                xy = 'xy'[oi]
                axes = [axes_x, axes_y][i]

                # plot median cross section
                ax = axes[0]
                ax.plot(m, marker='o', ms=3)
                ax.axhline(np.ma.median(m), color='darkgreen')
                for knot in knots:
                    ax.axvline(knot, color='chocolate')
                # ax.set_xlabel(xy)
                ax.xaxis.set_ticklabels([])
                ax.set_title('Image median cross section')
                ax.set_ylabel('Counts (e⁻)')
                ax.grid()

                # plot gradient & threshold + flagged points
                ax = axes[1]
                ax.plot(dm, marker='o', ms=3)
                ax.axhline(mm, color='darkgreen')
                ax.axhspan(*(mm + np.multiply((-1, 1), (δσ * s))), color='g',
                           alpha=0.3)
                ax.plot(w, dm[w], 'o', color='maroon', mfc='none')
                ax.set_title('Cross section median gradient (threshold)')
                ax.set_xlabel(xy)
                ax.set_ylabel(r'$\displaystyle\frac{\Delta f}{\Delta %s}$' % xy,
                              usetex=True, rotation='horizontal',
                              va='center', ha='right')
                ax.grid()
                # TODO legend

        if plot:
            display(fig_x)
            display(fig_y)
        return tuple(yx_knots)  # todo OR yxTuple ????????????

    def init_mem(self, folder, shape, fill=np.nan, clobber=False):
        from obstools.modelling.utils import load_memmap
        params = self._init_mem(folder / 'bg.par',
                                shape, fill, clobber)

        dtype = [(yx, int, k) for yx, k in zip('yx', self.n_knots)]
        knots = load_memmap(folder / 'knots.dat',
                            shape, dtype,
                            0,
                            clobber=clobber)
        return params, knots



def guess_knots_gradient_threshold(image, channel, axis, n_knots, δσ=3,
                                   edges=True):
    # median cross section

    m = np.ma.median(image, int(not bool(axis)))

    # detect knot positions by gradient threshold
    dm = np.diff(m)
    mm = np.ma.median(dm)
    s = mad(dm, mm)
    w, = np.where(np.abs(dm - mm) > δσ * s)

    # array one less in size since taking point to point diff
    grp = list(map(list, mit.consecutive_groups(w + 1)))
    n_grp = len(grp)

    knots = [0] * (n_knots + 2 * edges)
    if edges:
        knots[-1] = image.shape[axis]

    if n_knots == 1:
        # odd numbered channels seem to have the same background
        # structure
        if channel in (1, 3):
            # first item from last consecutive group
            *_, (k0, *_) = grp
            knots[edges] = k0  # - xo

        else:
            # last item from first consecutive group
            (*_, k0), *_ = grp
            knots[edges] = k0  # + xo

    elif n_knots == 2:
        if n_grp < 2:
            raise ValueError('Could only find 1 knot')

        # last item from first consecutive group and first item from
        # last consecutive group
        (*_, k0), *_, (k1, *_) = grp
        knots[edges:-1 if edges else None] = (k0, k1 - 2)  # (k0 + yo, k1 - 2)
    else:
        raise NotImplementedError

    return knots, (m, dm, mm, s, w)


class SlotModeBackground(Spline2D):  # TODO Maybe SlotModeImageModel  ??
    """
    Image model for SALTICAM slot-mode background.
    """

    # TODO: implement p0guess??

    # TODO: combine Spline2d and FrameTransferBleed models

    @classmethod
    def spline_order_search0(cls, image, channel, cream=2, **kws):
        """
        Initializer that does a brute force hyperparameter search to
        determine the optimal value for the polynomial orders

        Parameters
        ----------
        image
        channel
        cream
        kws

        Returns
        -------

        """
        knots = cls.guess_knots(image, channel)

        # FIXME: this first implementation is very inefficient.
        # FIXME: MODELS ARE BEING REFITTED ON SAME DATA MULTIPLE TIMES.

        MIN_SPLINE_ORDERS = (3, 1, 3), (1, 3)
        MAX_SPLINE_ORDERS = (8, 3, 8), (3, 8)

        # order search!
        kws.setdefault('method', 'leastsq')
        o_tuples = list(_gen_order_tuples(MIN_SPLINE_ORDERS, MAX_SPLINE_ORDERS))
        gof = np.ma.empty(len(o_tuples))
        gof[:] = np.ma.masked

        cls.logger.info("Testing %i polynomial splines for optimal value of "
                        "hyperparameter 'multi-order'.",
                        len(o_tuples))

        pool = mp.Pool()
        models, params, gof = zip(
                *pool.starmap(_try_fit, ((o, knots, image, False, kws)
                                         for o in o_tuples))
        )

        # aggregate results
        gof = np.array(gof, 'f')
        gof = np.ma.masked_where(np.isnan(gof), gof)

        # run knot search for top 2% of models
        i_best, = np.where(gof < np.percentile(gof.compressed(), cream))
        n_best = len(i_best)
        # i_best = i_best[gof[i_best].argsort()]  # reorder for best first
        cls.logger.info('Running knot search on %i models (top %.1f%%).',
                        n_best, cream)
        o_best = np.take(o_tuples, i_best, 0)
        best_models, best_results, best_gof = zip(
                *pool.starmap(_try_fit, ((o, knots, image, True, kws)
                                         for o in o_best))
        )
        pool.close()
        pool.join()

        best_gof = np.array(best_gof)
        best_models = np.array(best_models, 'O')
        best_results = np.array(best_results, 'O')

        # reorder the top results
        reorder = np.argsort(best_gof)
        return best_models[reorder], best_results[reorder], best_gof[reorder]

    @classmethod
    def from_image(cls, image, channel, orders, detection=None,
                   plot=False, **detect_opts):
        #
        """
        Construct a Spline2D instance from an image and polynomial
        multi-order.  Position of the knots will be estimated based on the
        median cross sections of the image. Objects (stars) in the image
        will be identified using threshold detection algorithm.

        Parameters
        ----------
        image
        channel
        orders
        detection


        Returns
        -------

        """

        #
        knots = cls.guess_knots(image, channel, plot=plot)
        mdl = cls(orders, knots)
        # mdl.groups['spline2d'] = mdl.segm.labels

        # Detect objects & segment image
        if detection is True or detect_opts:
            from obstools.phot.segmentation import SegmentationHelper
            detection = SegmentationHelper.detect

        if detection not in (None, False):
            seg = detection(image, False, None, **detect_opts)
        else:
            seg = mdl.segm

        return mdl, seg

    @staticmethod
    def guess_knots(image, channel, δσ=3, n_knots=(1, 2), edges=True,
                    offsets=(0, 0), plot=False):
        """
        A fast estimate of spline knot positions from a SALTICAM image by
        sigma-thresholding the point to point gradient of the cross sectional
        image median.

        Parameters
        ----------
        image:  array
            SALTICAM slotmode image to use for guessing the knots
        channel: int, {0-3}
            amplifier channel number (0-3)
        δσ:
            Malhanobis distance cutoff
        n_knots: tuple
            number of knots in each dimension. Only allow {1, 2} knots in
            each dimension
        edges: bool
            include end points as knots
        offsets: tuple of int
            y,x values to add to the guessed knot positions to get final result.
            Somewhat of a hack.
        plot: bool or callable
            if bool:
                Whether or not to make plots of the guessed knot positions
            if callable:
                The function used to display the plots in the interactive
                console

        Returns
        -------

        """
        channel = _check_channel(channel)
        assert len(n_knots) == 2

        if plot:
            import matplotlib.pyplot as plt
            from collections import Callable

            if isinstance(plot, Callable):
                display = plot
                plot = True
            else:
                def display(*_):
                    pass

            #
            figsize = (12, 6)
            fig_x, axes_x = plt.subplots(2, 1, figsize=figsize)
            fig_y, axes_y = plt.subplots(2, 1, figsize=figsize)

        #
        logger.info('Guessing knot positions for image background spline '
                    'model via `guess_knots_gradient_threshold`')

        yx_knots = []
        # xo, yo = offsets  # offsets (found empirically)
        for i in range(2):
            #
            oi = int(not bool(i))
            knots, (m, dm, mm, s, w) = guess_knots_gradient_threshold(
                    image, channel, i, n_knots[oi], δσ, edges)
            yx_knots.append(knots)

            if plot:
                xy = 'xy'[oi]
                axes = [axes_x, axes_y][i]

                # plot median cross section
                ax = axes[0]
                ax.plot(m, marker='o', ms=3)
                ax.axhline(np.ma.median(m), color='darkgreen')
                for knot in knots:
                    ax.axvline(knot, color='chocolate')
                # ax.set_xlabel(xy)
                ax.xaxis.set_ticklabels([])
                ax.set_title('Image median cross section')
                ax.set_ylabel('Counts (e⁻)')
                ax.grid()

                # plot gradient & threshold + flagged points
                ax = axes[1]
                ax.plot(dm, marker='o', ms=3)
                ax.axhline(mm, color='darkgreen')
                ax.axhspan(*(mm + np.multiply((-1, 1), (δσ * s))), color='g',
                           alpha=0.3)
                ax.plot(w, dm[w], 'o', color='maroon', mfc='none')
                ax.set_title('Cross section median gradient (threshold)')
                ax.set_xlabel(xy)
                ax.set_ylabel(r'$\displaystyle\frac{\Delta f}{\Delta %s}$' % xy,
                              usetex=True, rotation='horizontal',
                              va='center', ha='right')
                ax.grid()
                # TODO legend

        if plot:
            display(fig_x)
            display(fig_y)
        return tuple(yx_knots)  # todo OR yxTuple ????????????

    def init_mem(self, folder, shape, fill=np.nan, clobber=False):
        from obstools.modelling.utils import load_memmap
        params = self._init_mem(folder / 'bg.par',
                                shape, fill, clobber)

        dtype = [(yx, int, k) for yx, k in zip('yx', self.n_knots)]
        knots = load_memmap(folder / 'knots.dat',
                            shape, dtype,
                            0,
                            clobber=clobber)
        return params, knots

    def detect(self, image, mask=False, background=None, snr=3., npixels=7,
               edge_cutoff=None, deblend=False, dilate=0,
               flux_thresh_ft_bleed=PHOTON_BLEED_THRESH):
        """
        Run detection, add labels, add group

        Parameters
        ----------
        image
        mask
        background
        snr
        npixels
        edge_cutoff
        deblend
        dilate
        flux_thresh_ft_bleed

        Returns
        -------

        """
        # detect stars
        new_seg = self.segm.detect(image, mask, background, snr, npixels,
                                   edge_cutoff, deblend, dilate)

        if new_seg.data.any():
            # aggregate
            _, new_labels = self.segm.add_segments(new_seg)

            # group info
            self.groups.append(new_labels)
            self.groups.info.append(
                    dict(snr=snr, npixels=npixels, dilate=dilate,
                         deblend=deblend)
            )

            # TODO: check for ft bleeding

        return new_seg

    def get_edge_cutoffs(self):
        # FIXME: channel dependant
        return np.hstack([self.knots.x[0:2], self.knots.y[1:3]])


# class SlotBackground(SegmentedImageModel, ModellingResultsMixin):  #
#     """
#     Image model for SALTICAM slot-mode background.
#
#     combine Vignette and FrameTransferBleed sequentially
#     """
#
#     @classmethod
#     def from_image(cls, image, mask=None, bg_model=None, snr=(10, 7, 5, 3),
#                    npixels=(7, 5, 3), edge_cutoff=None, deblend=(True, False),
#                    dilate=(4, 1)):
#         """Multi-threshold blob detection"""
#         # TODO: automatically determine orders, breaks
#
#         from obstools.phot.tracking import SegmentationHelper, LabelGroups, \
#             Record
#
#         #
#         cls.logger.info('Running detection loop')
#
#         # make iterables
#         variters = tuple(map(iter_repeat_last, (snr, npixels, dilate, deblend)))
#         vargen = zip(*variters)
#
#         # segmentation data
#         data = np.zeros(image.shape, int)
#         original_mask = mask
#         if mask is None:
#             mask = np.zeros(image.shape, bool)
#
#         # first round detection without background model
#         residual = image
#         results = None
#         # keep track of group info + detection meta data
#         groups = LabelGroups(bg=[0])
#         groups.info = Record()
#         groups._auto_name_fmt = groups.info._auto_name_fmt = 'stars%i'
#         counter = itt.count(0)
#         j = 0
#         while True:
#             # keep track of iteration number
#             count = next(counter)
#             group_name = 'stars%i' % count
#
#             # detect stars
#             _snr, _npix, _dil, _debl = next(vargen)
#             sh = SegmentationHelper.detect(residual, mask, None, _snr, _npix,
#                                            edge_cutoff, _debl, _dil)
#
#             # update mask, get new labels
#             new_mask = sh.to_bool()
#             new_data = new_mask & np.logical_not(mask)
#
#             # since we dilated the detection masks, we may now be overlapping
#             # with previous detections. Remove overlapping pixels here
#             if dilate:
#                 overlap = data.astype(bool) & new_mask
#                 # print('overlap', overlap.sum())
#                 sh.data[overlap] = 0
#                 new_data[overlap] = False
#
#             if not new_data.any():
#                 break
#
#             # aggregate
#             new_labelled = sh.data[new_data]
#             new_labels = np.unique(new_labelled)
#             data[new_data] += new_labelled + j
#             mask = mask | new_mask
#             # group
#             group = new_labels + j
#             groups[group_name] = group
#             groups.info[group_name] = \
#                 dict(snr=_snr, npixels=_npix, dilate=_dil, deblend=_debl)
#             # update
#             j += new_labels.max()
#
#             #
#             # logger.info('detect_loop: round nr %i: %i new detections: %s',
#             #       count, len(group), tuple(group))
#             cls.logger.info(
#                     'detect_loop: round nr %i: %i new detections: %s',
#                     count, len(group), seq_repr_trunc(tuple(group)))
#
#             if count == 0:
#                 # initialise the background model
#                 # decide based on estimated star flux whether or not to add ft
#                 # streak model.
#                 threshold = 500
#                 ft, segm, strkLbl = FrameTransferBleed.from_image(
#                         image, sh, threshold)
#                 # The frame transfer bleed model might be None if we only have
#                 # faint stars in the image
#                 if ft:
#                     groups['streaks'] = strkLbl
#
#                 # mdlr = cls(segm, (bg_model, ft), groups)
#                 # note both models might be None
#             else:
#                 # add segments to ignore
#                 cls.logger.info('adding segments')
#                 _, labels = mdlr.segm.add_segments(sh, replace=True)
#
#                 mdlr.groups.append(labels)
#
#             return mdlr, results, data, mask, groups
#
#             # fit background
#             mimage = np.ma.MaskedArray(image, original_mask)
#             # return mdlr, mimage, segm
#             results, residual = mdlr.minimized_residual(mimage)
#
#             # print(results)
#
#         # TODO: log what you found
#         return mdlr, results, residual, data, mask, groups
#
#     # @classmethod
#     # def from_cube(cls, cube, n_init=10, n_sample=25, mask=None, bg_model=None,
#     #               snr=3, npixels=3, edge_cutoff=None, deblend=(True, False),
#     #               dilate=(4, 1)):
#     #     # loop sample median images across cube to get relative star positions
#     #
#     #     from obstools.phot.utils import ImageSampler
#     #     from obstools.phot.tracking.core import SegmentationHelper
#     #     import more_itertools as mit
#     #
#     #     sampler = ImageSampler(cube)
#     #     coms = []
#     #
#     #     for interval in mit.pairwise(range(0, len(cube), len(cube) // n_init)):
#     #         image = sampler.median(n_sample, interval)
#     #         model, p0bg, seg_data, sky_mask, star_groups = cls.from_image(
#     #                 image, mask, bg_model, snr, npixels, edge_cutoff, deblend,
#     #                 dilate)
#     #
#     #         # segm = SegmentationHelper.detect(image, mask, bg, snr,
#     #         #                                  npix)
#     #         com = segm.com_bg(image)
#     #         coms.append(com)
#     #
#     #         im = segm.display()
#     #
#     #     #     ui.add_tab(im.figure)
#     #     # ui.show()
#     #
#     #     ndetected = list(map(len, coms))
#     #     max_nstars = max(ndetected)
#     #     comsRef = coms[np.argmax(ndetected)]
#     #     rxx = np.full((len(coms), max_nstars, 2), np.nan)
#     #     for i, com in enumerate(coms):
#     #         # print('i', i)
#     #         if len(com) == max_nstars:
#     #             rxx[i] = np.atleast_2d(com) - com[0]
#     #         else:
#     #             for cx in com:
#     #                 j = np.square((comsRef - cx)).sum(1).argmin()
#     #                 rxx[i, j] = cx - com[0]
#     #
#     #     #
#     #     rvec = np.nanmean(rxx, 0)
#
#     def __init__(self, segm, models, label_groups=None, save_residual=True):
#         #
#         SegmentedImageModel.__init__(self, segm, models, label_groups)
#         ModellingResultsMixin.__init__(self, save_residual)
#
#     def __call__(self, p, grid=None, out=None):
#         # labels = self.resolve_labels(labels)
#         # assert len(p) == len(labels)
#
#         if out is None:
#             out = np.zeros(self.segm.shape)
#
#         # loop over models
#         for i, model in enumerate(self.models):
#             labels = self.groups[i]
#             for j, lbl in enumerate(labels):
#                 grid = model.adapt_grid(self.segm.coord_grids[lbl])
#                 # print(self, '__call__', grid)
#                 r = model(p[model.name][j], grid)
#                 # print('ri', r.shape, ' slice', self.segm.slices[lbl],
#                 #       out[self.segm.slices[lbl]].shape)
#                 sy, sx = self.segm.slices[lbl]
#                 out[:, sx] += r  # HACK
#
#         return out
#
#     def __reduce__(self):
#         # helper method for unpickling.
#         return self.__class__, \
#                (self.segm, list(self.models), self.groups), \
#                dict(data=self.data)
#
#     def residuals(self, p, data, grid=None):
#         return super().residuals(p, data, grid)
#
#     def plot_fit_results(self, image, params, modRes=500):
#
#         import matplotlib.pyplot as plt
#         import more_itertools as mit
#
#         # imm = self.segm.mask_foreground(image)
#         # figs = self.vignette.plot_fit_results(imm, params.vignette, modRes)
#         #
#         # segmented = self.segm.coslice(image, self.segm.grid,
#         #                               labels=self.groups.streaks, mask=True)
#         # for j, (sub, g) in enumerate(segmented):
#         #     for i, fig in enumerate(figs):
#         #         ax1, ax2 = fig.axes
#         #         if i == 0:
#         #             g = g[i]
#         #         else:
#         #             g = g[:, i].T
#         #             sub = sub.T
#         #
#         #         ax1.plot(g, sub, 'b')
#         #
#         #         p = params.bleed[j]
#         #         ax1.plot(g, self.bleed(p, g), color='orange')
#
#         modelled = self(params)
#         vignette = self.vignette(params.vignette)
#         # bleed = vignette - modelled
#
#         new_groups = self.groups.copy()
#         new_groups.pop('streaks', None)
#         # segm = self.segm
#         segm = self.segm.__class__(
#                 self.segm.select(list(mit.collapse(new_groups.values()))))
#         segm.allow_zero = True
#         # ss = self.segm
#         pixels, gplot = next(segm.coslice(image, segm.grid,
#                                           labels=0, mask_bg=True))
#         resi = pixels - modelled
#
#         # figs = []
#
#         fig, axes = plt.subplots(2, 2, figsize=(20, 8),
#                                  gridspec_kw=dict(hspace=0,
#                                                   height_ratios=(3, 1)))
#
#         for i, yx in enumerate('yx'):
#             # share x axes between data and residual axes
#             ax1, ax2 = axes[:, i]
#             ax1.get_shared_x_axes().join(ax1, ax2)
#
#             #
#             g = gplot[i]
#             if i == 1:
#                 pixels = pixels.T
#                 g = g.T
#                 modelled = modelled.T
#                 resi = resi.T
#
#             # plot model
#             model_lines = ax1.plot(g, modelled, 'mediumslateblue', ls='-',
#                                    alpha=0.55, label='Model', )
#             # model median
#             model_median = ax1.plot(g[:, 0], np.median(modelled, 1), 'maroon',
#                                     ls='-', alpha=0.55, label='Model median', )
#             model_mean = ax1.plot(g[:, 0], np.mean(modelled, 1), 'purple',
#                                   ls='-', alpha=0.55, label='Model median', )
#
#             dataCol = 'palegreen'
#             dataMedCol = 'darkgreen'
#             data_lines = ax1.plot(g, pixels, dataCol, alpha=0.35,
#                                   label='Data',
#                                   zorder=-1)  # plot below everything else
#
#             # extract the data
#             # other_axis = int(not bool(i))
#             median = np.ma.median(pixels, 1)
#             mean = np.ma.mean(pixels, 1)
#             std = np.ma.std(pixels, 1)
#             # WARNING. not actually the data used for the modelling!!
#             uncertainty = mad(resi, axis=1)
#             # std = np.ma.std(image, other_axis)
#
#             # plot fitted data
#             ebc = ax1.errorbar(g[:, 0], median, uncertainty, fmt='o',
#                                color=dataMedCol,
#                                zorder=10,
#                                label='Median $\pm$ MAD')
#             pxlmn, = ax1.plot(g[:, 0], mean, 'o',
#                               color='lime',
#                               zorder=10,
#                               label='Mean')
#
#             # breakpoints
#             # print(axs, 'breaks', db.model[axs].breakpoints)
#             breakpoints = self.vignette[yx].breakpoints * len(pixels)  # *
#             # g.max()
#             for ax in (ax2, ax1):
#                 lines_bp = ax.vlines(breakpoints, 0, 1,
#                                      linestyle=':', color='0.2',
#                                      transform=ax.get_xaxis_transform(),
#                                      label='Break points')
#
#             # model
#             # p = params.squeeze()
#             # gm = np.linspace(0, 1, modRes)  # bg.shape[not bool(i)]
#             # dfit = self(p, gm) * np.sqrt(np.ma.median(data))
#             # lines_mdl, = ax1.plot(gm * len(data), dfit, 'r-', label='model')
#
#             # plot model residuals
#             # np.ma.median(pixels, 1)
#
#             resim = np.ma.median(resi, 1)
#             resiu = mad(resi, axis=1)
#
#             ax2.errorbar(g[:, 0], resim, resiu, fmt='o', color=dataMedCol)
#             ax2.plot(g, resi, dataCol, alpha=0.35, zorder=-1)
#
#             # percentile limits on errorbars (more informative display for
#             #  large errorbars)
#             # ax2.set_ylim(np.percentile(resi - resiu[:, None], 25),
#             #              np.percentile(resi + resiu[:, None], 75))
#             #
#
#             ax2.set_ylim(resim.min() - 5 * resiu.mean(),
#                          resim.max() + 5 * resiu.mean())
#
#             ax1.set_ylim(median.min() - 5 * uncertainty.mean(),
#                          median.max() + 5 * uncertainty.mean())
#
#             ax1.set_title(('%s cross section fit' % yx).title())
#             ax1.set_ylabel('Counts')  # Normalized
#             ax1.set_xticklabels([])
#             legArt = [ebc, pxlmn, data_lines[0], model_lines[0], lines_bp]
#             legLbl = [_.get_label() for _ in legArt]
#             ax1.legend(legArt, legLbl, loc=1)  # upper right
#             ax1.grid()
#
#             ax2.set_ylabel('Residuals')
#             ax2.grid()
#
#             fig.tight_layout()
#         return fig

# for i in range(2):

# get data

# i = 'yx'.index(self.name)
# axs = 'yx'[i]
# grid, data, uncertainty = g, d, e = get_cross_section_data(image, i)

# extract the data
#     other_axis = int(not bool(i))
#
#     # data np.ma.mean(image, other_axis)
#     gplot = g * len(data)
#     scale = 1  # np.ma.median(data)
#
#     # plot fitted data
#     ebc = ax.errorbar(gplot, d, e, fmt='mo', zorder=10,
#                        label='Median $\pm$ (MAD)')
#
#     # plot rows / column data
#     pixels = image / scale
#     if i == 1:
#         pixels = pixels.T
#
#     lines = ax1.plot(gplot, pixels, 'g', alpha=0.35, label='Data')
#
#     # breakpoints
#     # print(axs, 'breaks', db.model[axs].breakpoints)
#     breakpoints = self.breakpoints * len(data)  # * g.max()
#     lines_bp = ax1.vlines(breakpoints, 0, 1,
#                           linestyle=':', color='0.2',
#                           transform=ax1.get_xaxis_transform(),
#                           label='Break points')
#
#     # model
#     p = params.squeeze()
#     gm = np.linspace(0, 1, modRes)  # bg.shape[not bool(i)]
#     dfit = self(p, gm) * np.sqrt(np.ma.median(data))
#     lines_mdl, = ax1.plot(gm * len(data), dfit, 'r-', label='model')
#
#     # plot model residuals
#     res = data - self(p, grid) * np.sqrt(np.ma.median(data))
#     ax2.errorbar(gplot, res, e, fmt='mo')
#     # percentile limits on errorbars (more informative display for large errorbars)
#     lims = np.percentile(res - e, 25), np.percentile(res + e, 75)
#     ax2.set_ylim(*lims)
#
#     ax1.set_title(('%s cross section fit' % self.name).title())
#     ax1.set_ylabel('Normalized Counts')
#     legArt = [ebc, lines[0], lines_mdl, lines_bp]
#     legLbl = [_.get_label() for _ in legArt]
#     ax1.legend(legArt, legLbl)
#     ax1.grid()
#
#     ax2.set_ylabel('Residuals')
#     ax2.grid()
#
#     fig.tight_layout()
# return fig

# def _fit_reduce(self, data, stddev=None, **kws):
#     # loop over models and fit
#     results = self._get_output()
#     for i, model in enumerate(self.models):
#         labels = self.groups[i]
#         r = results[model.name]
#         self._fit_model(model, data, stddev, labels, r, **kws)
#         # todo: check success
#         data = model.residuals(r, data)
#         # todo: update stddev

# def fit(self, image, p0=None, grid=None, **kws):
#     # p = vignette.fit(image)
#
#     # if p0 is None:
#     #     p0 = self.p0guess(image, grid)
#
#     # since we are modelling in separate segments, we can do it sequentially
#     v, bled = self.models
#     vmask = .segm.to_boolean(v.segm.labels[1:])
#     bmask = bled.segm.to_boolean()
#
#     pv = v.fit(np.ma.MaskedArray(image, bmask | vmask), **kws)
#     resi = v.residuals(pv, image)
#
#     ps = bled.fit(resi)
#     # resi - self.streaks(ps, streak_regions)
#
#     return pv, ps

# class ImagePSFModeller(SegmentedImageModel):
#     def __init__(self, segm, models, label_groups=None, rmax=10):
#         super().__iit__(segm, models, label_groups)

# segm.slices.enla

# class FrameTransferBleed(ImageSegmentsModel, OptionallyNamed):
#     name = 'bleed'
#
#     def __init__(self, segm, use_labels=None):
#         segm = self.adapt_segments(segm)
#         ImageSegmentsModel.__init__(self, Streak(), segm, use_labels)
#
#     # def adapt_grid(self, grid):
#     #     return grid[self.model.axis, 0]  # zeroth row of x-grid
#
#     @staticmethod
#     def adapt_segments(segm, labels=None, width=10):
#
#         if labels is None:
#             labels = segm.labels_nonzero
#
#         new = np.zeros(segm.shape, int)
#         for lbl, (sly, slx) in segm.enum_slices(labels):
#             start, stop = slx.stop, slx.start
#             ptp = stop - start
#             d = (ptp - width) / 2  # or use segm.com(segm.data, labels)
#             ss = np.round([start + d, stop - d]).clip(0, segm.shape[1])
#             new[:, slice(*ss.astype(int))] = lbl
#
#         #
#         new[segm.to_boolean(labels)] = 0
#         segm.add_segments(new, 1)
#         return segm

# @classmethod
# def from_image(cls, image, mask=False, snr=10., npixels=7,
#                edge_cutoff=None, deblend=False, dilate=2, buffer=2):
#     # from scipy import ndimage
#     from obstools.phot.tracking import SegmentationHelper
#     #
#     segm = SegmentationHelper.detect(image, mask, None, snr, npixels,
#                                      edge_cutoff, deblend, dilate)
#     return cls(segm)

# def adapt_segments(self, segm, buffer=3):
#     z = segm.data.astype(bool)
#     # zd = ndimage.binary_dilation(z, iterations=buffer)
#     labelled = (z.any(0) & ~z) * segm.data.max(0)
#     return labelled

# @staticmethod
# def adapt_segments(segm, labels=None, width=10):
#
#     if labels is None:
#         labels = segm.labels_nonzero
#
#     new = np.zeros(segm.shape, int)
#     for lbl, (sly, slx) in segm.enum_slices(labels):
#         start, stop = slx.stop, slx.start
#         ptp = stop - start
#         d = (ptp - width) / 2  # or use segm.com(segm.data)
#         ss = np.round([start + d, stop - d]).clip(0, segm.shape[1])
#         new[:, slice(*ss.astype(int))] = lbl
#
#     #
#     new[segm.to_boolean(labels)] = 0
#     segm.add_segments(new, 1)
#     return segm

# class FrameTransferBleed(SegmentedImageModel, OptionallyNamed):
#     name = 'bleed'
#
#     def __init__(self, segm, models=Streak(), use_labels=None,
#                  metrics=None):
#         segm = self.adapt_segments(segm)
#         SegmentedImageModel.__init__(self, segm, models, metrics, use_labels)
#
#     def __call__(self):
#
#
#     @classmethod
#     def from_image(cls, image, mask=False, snr=10., npixels=7,
#                    edge_cutoff=None, deblend=False, dilate=2, buffer=2):
#         from scipy import ndimage
#         from obstools.phot.tracking import SegmentationHelper
#         #
#         segm = SegmentationHelper.detect(image, mask, None, snr, npixels,
#                                          edge_cutoff, deblend, dilate)
#         return cls(segm)
#
#     def adapt_segments(self, segm, buffer=3):
#         z = segm.data.astype(bool)
#         # zd = ndimage.binary_dilation(z, iterations=buffer)
#         labelled = (z.any(0) & ~z) * segm.data.max(0)
#         return segm.__class__(labelled)

# class NamedSegmentGroups(object):
#     groups = {}

# class SlotBackground(SegmentedImageModel):  # SlotBackground
#     # combine Vignette + FrameTransferBleed
#
#     def __init__(self, segm, orders, breaks):
#
#         # adapt segmentation to model frame transfer bleed on bright stars
#         bright_obj = segm.labels
#         FrameTransferBleed.adapt_segments(segm)
#         label_groups = [[0], bright_obj]
#
#         # init models
#         vignette = Vignette2DCross(orders, breaks)
#         vignette.set_grid(segm.data)
#         # bleeding = FrameTransferBleed()
#         models = (vignette, FrameTransferBleed())
#         SegmentedImageModel.__init__(self, segm, models, label_groups)
#
#     # def fit(self, image, p0=None, grid=None, **kws):
#     #     # p = vignette.fit(image)
#     #
#     #     # if p0 is None:
#     #     #     p0 = self.p0guess(image, grid)
#     #
#     #     # since we are modelling in separate segments, we can do it sequentially
#     #     v, bled = self.models
#     #     vmask = .segm.to_boolean(v.segm.labels[1:])
#     #     bmask = bled.segm.to_boolean()
#     #
#     #     pv = v.fit(np.ma.MaskedArray(image, bmask | vmask), **kws)
#     #     resi = v.residuals(pv, image)
#     #
#     #     ps = bled.fit(resi)
#     #     # resi - self.streaks(ps, streak_regions)
#     #
#     #     return pv, ps
#
#     def __call__(self, p, grid=None):
#         # assert self.nmodels
#
#         for mdl in self.models:
#             mdl(p[mdl.name], grid)

#
# def get_dtype(self, base='f'):
#     # compound dtype for compound model
#     return np.dtype([('vignette', 'f'),
#                      ('streaks', 'f', self.streaks.npar)])

# def _init_mem(self, loc, shape, dtype='f', fill=np.nan, clobber=False):
#     # parV = self.vignette._init_mem(loc, shape, dtype, fill, clobber)
#     # parS = self.streaks._init_mem(loc, shape, dtype, fill, clobber)
#     #
#
#     # compound dtype for compound model
#     dtype = np.dtype([('vignette', 'f'),
#                       ('streaks', 'f', self.streaks.npar)])

# fig, axes = plt.subplots(3, 1, figsize=(8,8))
#
# resi = vftb.vignette.residuals(p.vignette, image, None)
# g1 = vftb.bleed
# i = 0
# for lbl, slice_ in vftb.segm.enum_slices(vftb.groups[1]):
#     im = resi[slice_]
#     grid = vftb.segm.coord_grids[lbl]
#     g = g1.adapt_grid(grid)
#     y = np.ma.median(im, 0)
#
#     ax = axes[i]
#     ax.plot(grid[1].ravel(), np.ma.MaskedArray(im, vftb.segm.masks[lbl]).ravel(), 'o')
#     ax.plot(g, y, marker='x', ls='none')
#     r = p[g1.name][i]
#     if r is not None:
#         gm = np.linspace(g[0], g[-1])
#         ax.plot(gm, g1(r, gm), 'r-')
#         ax.set_ylim(top=y.max() * 1.25)
#     i += 1

# if __name__ == '__main__':
# pickling tests
# import pickle

# from recipes.io import load_pickle
# path = '/media/Oceanus/UCT/Observing/data/sources/V2400_Oph/SALT/20140921' \
#        '/product/s20140921.segm.pkl'
# segm = load_pickle(path)


def check_model(cube, bad_pixel_mask, vignette, n=10, n_sample=25, snr=(10, 3),
                npixels=(7, 3),
                dilate=(5, 1)):
    # note: setting initial dilation high prevents spurious detections
    # for pixels from the same star.
    # TODO: integrate psf models...

    import more_itertools as mit
    from obstools.phot.utils import ImageSampler
    from graphical.multitab import MplMultiTab
    from graphical.imagine import ImageDisplay

    sampler = ImageSampler(cube)

    ui1 = MplMultiTab()
    ui2 = MplMultiTab()

    coms = []

    for interval in mit.pairwise(range(0, len(cube), len(cube) // n)):
        image = sampler.median(n_sample, interval)

        mdlr, results, seg_data, resi, mask, groups = \
            SlotBackground.from_image(image, bad_pixel_mask, vignette, snr,
                                      npixels, dilate=dilate)

        im = ImageDisplay(resi)
        ui1.add_tab(im.figure)

        sky = np.ma.MaskedArray(image, mask)
        fig = mdlr.plot_fit_results(sky, results)
        ui2.add_tab(fig)
        # break

    ui1.show()
    ui2.show()
