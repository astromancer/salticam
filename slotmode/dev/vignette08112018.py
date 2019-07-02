"""
Piecewise polynomials with boundary conditions
"""

import logging
import itertools as itt
import numbers
import operator
from collections import namedtuple
import warnings

import lmfit as lm
import more_itertools as mit
import numpy as np
# from recipes.interactive import is_interactive
from IPython import embed
from obstools.modelling.core import CompoundModel, RescaleInternal, \
    UnconvergedOptimization

from recipes.array import fold
from recipes.dict import AttrDict

from .ppoly import PPolyModel, transform_poly, get_ppoly_repr
from obstools.modelling import nd_sampler


def mad(data, data_median=None, axis=None):
    """
    Median absolute deviation

    Parameters
    ----------
    data
    data_median

    Returns
    -------

    """
    nans = np.isnan(data)
    if nans.any():
        data = np.ma.MaskedArray(data, nans)

    if data_median is None:
        data_median = np.ma.median(data, axis, keepdims=True)
    # else:
    # make sure we can broadcast them together
    return np.ma.median(np.abs(data - data_median), axis)


# def get_cross_section_data(image, i, masked_ignore_thresh=0.35):
#     # extract the data
#     other_axis = int(not bool(i))
#     median = np.ma.median(image, other_axis)
#
#     if np.ma.isMA(image):
#         # make mask
#         mask = image.mask.mean(other_axis) > masked_ignore_thresh
#         median = np.ma.masked_array(median, mask=mask)
#
#     # scale data - polynomial fitting better behaved
#     # scale = np.ma.median(image)
#     # data = cross_masked / scale
#
#     grid = np.linspace(0, 1, image.shape[i])
#     # print('!!', i, image.shape[i])
#     # grid = np.mgrid[0:image.shape[i]]
#     uncertainty = mad(image, axis=other_axis)
#     # uncertainty = np.ones_like(data)
#
#     return grid, median, uncertainty  # , scale


# def plist(params):  # can import from lm_compat
#     """convert from lm.Parameters to ordered list of float values"""
#     if isinstance(params, lm.Parameters):
#         params = list(params.valuesdict().values())
#     return np.asarray(params)


# def convert_params(func):
#     """decorator to convert lm.Parameters to a list on the fly"""
#
#     @functools.wraps(func)
#     def wrapper(*args, **kws):
#         obj, p, *rest = args
#         return func(obj, plist(p), *rest, **kws)
#
#     return wrapper


# from numpy.polynomial.polynomial import polyval2d

import itertools as itt


class RegionHelper(object):
    def __init__(self, breakpoints):
        #
        yb, xb = tuple(map(np.array, breakpoints))
        self.breakpoints = tuple(np.round(b).astype(int) for b in (yb, xb))

    def __getitem__(self, key):
        intervals = tuple(self._get(i, jk) for i, jk in enumerate(key))
        return tuple(slice(*i) for i in intervals)

    def _get(self, i, jk):
        #
        bp = self.breakpoints[i]
        n = len(bp) - 1
        if isinstance(jk, numbers.Integral):
            if jk < 0:
                jk += n
            if jk >= n:
                raise IndexError('Only %i regions available for %s coordinate'
                                 % (n, 'yx'[i]))
            return bp[jk:jk + 2]

        if isinstance(jk, slice):
            start = jk.start or 0
            stop = jk.stop or len(bp)
            stop = min(stop, n)
            return bp[[start, stop]]

        raise IndexError('Invalid index: %s (%s)' % (jk, type(jk)))


from slotmode.ppoly import PPoly2D
from obstools.modelling.core import StaticGridMixin

# class ContainerAttrGetterMixin(object):
#     """
#     Mixin class for container type objects that implements ``attrgetter''
#     method - a vectorized attribute getter on the constituents similar to the
#     builtin `operator.attrgetter'.
#     """
#
#     def attrgetter(self, *attrs):
#         getter = operator.attrgetter(*attrs)
#         # values = getter(self.polys[0, 0])
#         # check return type, shape
#         # TODO: better handling output types and shapes? via inspection
#         return np.vectorize(getter, 'O')(self.polys)


# simple container for 2-component objects
yxTuple = namedtuple('yxTuple', ['y', 'x'])


class Vignette(StaticGridMixin, RescaleInternal, CompoundModel):
    # Spline2d,
    # CompoundSequentialFitter
    # Todo: use ImageSegmentsModeller ??????
    """
    Models the 2D vignetting pattern in SALTICAM slot mode images.
    """

    # TODO: This is actually a non-uniform spline!  use proper terms. knots,
    # knot_vector etc...

    # tldr stitch a bunch of polynomials together while keeping overall
    # smoothness and continuity

    # equality and continuity bc are implied by definition of independent
    # domains starting at 0

    # each polynomial has domain in unit square. This is good for numerical
    # stability.

    # primary polynomial has full parameter freedom. Neighbours have some
    # parameters (coefficients) not free, but tied to the value of neighbours
    # coefficient matrix due to continuity and smoothness constraints.

    # mixed coefficient terms can be frozen to 0 by using the `no_mixed_terms`
    # method

    # see : https://en.wikipedia.org/wiki/Spline_(mathematics)

    # FIXME: check smoothness constraints

    def __init__(self, orders, knots, pri=(0, 0), smooth=True, continuous=True):
        """
        Non-uniform 2D spline model

        Parameters
        ----------
        orders:     2-tuple of int, 2-tuple of lists of int
            x, y sequences of orders for polynomials, or x, y multi-order pair.
            If sequences of orders, the combinatorial product of multi-order
            pairs will be used to create a grid of 2D polynomials.  If
            2-tuple of int, all polynomials will be of the same multi-order.
            The number of constituent polynomials will be determined from the
            knot vectors in `knots`.
        knots:      2-tuple of lists of floats
            Coordinate position of the knots (including the domain endpoints)
        pri: 2-tuple
            Index position of the primary 2D polynomial
        smooth:     bool or sequence of bool
            Smoothness condition between neighbouring polynomials
        continuous: bool or sequence of bool
            Continuity condition between neighbouring polynomials

        """

        # TODO smoothness / continuity at each knot

        # checks
        assert len(orders) == len(knots) == 2
        # reverse the order of input arguments: images are yx (row, column)
        checker = (self._check_orders_knots(*ok)
                   for ok in zip(orders[::-1], knots[::-1]))

        # wrap namedtuple for easy attribute access
        self.orders, self.knots = itt.starmap(yxTuple, zip(*checker))

        # static image dimensions
        self.ishape = (self.knots.y[-1], self.knots.x[-1])

        # regions
        self.primary = ip, jp = pri
        self.regions = RegionHelper(self.knots)
        self.n_polys = npolys = tuple(map(len, self.orders))
        self.n_knots = np.subtract(self.n_polys, 1)
        self._ix = np.indices(self.n_polys)  # index grid for polys

        # domains
        # NOTE: ImageSegmentsModeller will handle some of these automagically
        domains = itt.product(*map(mit.pairwise, self.knots))
        self.domains = np.reshape(list(domains), npolys + (2, 2))
        self.slices = np.vectorize(slice)(*self.domains.T).T
        self.sizes = self.domains.ptp(-1)

        # re-order so we iterate starting from primary region
        yi = sorted(range(npolys[0]), key=lambda i: np.abs(i - pri[0]))
        xi = sorted(range(npolys[1]), key=lambda j: np.abs(j - pri[1]))
        self.iorder = list((i[::-1] for i in itt.product(xi, yi)))
        reorder = tuple(np.transpose(self.iorder))
        self.rorder = np.empty(self.n_polys, int)
        self.rorder[reorder] = np.arange(np.product(npolys))

        # create grid of 2D polynomials
        mdict = {}
        for i, j in self.iorder:
            # create 2d polynomial
            o = self.orders.y[i], self.orders.x[j]  # 2d multi-order
            poly = PPoly2D(o, smooth, continuous)

            # get domain ranges
            y0, y1 = np.subtract((0, 1), i < ip)
            x0, x1 = np.subtract((0, 1), j < jp)
            szy, szx = self.sizes[i, j] * 1j
            # static grid
            poly.static_grid = np.mgrid[y0:y1:szy, x0:x1:szx]

            # add to container
            mdict['poly2d_%i%i' % (i, j)] = poly

        # initialize parent container
        CompoundModel.__init__(self, **mdict)

        # get neighbours
        for ij, poly in zip(self.iorder, self.models):
            # print(ij, n, '-' * 30, sep='\n')
            n = self.get_neighbours(*ij)
            poly.set_neighbours(**n)

    def __call__(self, p, grid=None):
        # TODO: with an actual grid ??

        # dof consistency check
        self._check_params(p)

        # grid check
        grid = self._check_grid(grid)

        # broadcast output
        yg, xg = grid
        b = np.broadcast(yg, xg)
        out = np.zeros(b.shape)  # result

        # update coefficient matrices
        self.update_coeff(self[0])

        # compute and fill in regions for each poly
        self.eval_loop(p, self.regions, out)
        return out

    def eval_loop(self, p, slices, out):
        # compute and fill in regions for each poly
        ps = self.split_params(p)
        for poly, (i, j), pp in zip(self.models, self.iorder, ps):
            out[slices[i, j]] += poly(pp, poly.static_grid)

    def p0guess(self, data, grid=None, stddev=None):
        return np.zeros(self.dof)

    def get_param_vector(self):
        """
        Construct a parameter vector from the coefficient matrices of the
        constituent polynomials considering only the free parameters.  This
        is useful when initially fitting a model with restricted parameter
        freedoms, then freeing some parameters and re-fitting starting from
        the previous optimum.
        """
        p = np.zeros(self.dof)
        splix = np.zeros(self.nmodels + 1, int)
        splix[1:] = np.cumsum(self.dofs)
        slices = itt.starmap(slice, mit.pairwise(splix))
        for poly, sl in zip(self.models, slices):
            p[sl] = poly.coeff[poly.free]
        return p

    def fit(self, data, grid=None, stddev=None, p0=None, *args, **kws):
        # sequential fitting. large performance gain when partitioning the
        # parameter space
        results = []
        for i, poly in enumerate(self.models):
            if poly.free.any():
                reg = self.regions[self.iorder[i]]
                r = poly.fit(data[reg])
                if r is None:
                    raise ValueError
                results.append(r)

        return np.hstack(results)

    def set_grid(self, data):
        sy, sx = data.shape
        self.static_grid = np.mgrid[:1:sy * 1j, :1:sx * 1j]

    def adapt_grid(self, grid):
        return None  # HMMMMMM...

    def split_params(self, p):
        """
        Split parameter vector into `n` parts, where `n` is the number of
        polynomials in the model, and each part has size `m` corresponding to
        the number of free coefficients in the component polynomials.

        Parameters
        ----------
        p

        Returns
        -------

        """

        return np.split(p, np.cumsum(self.dofs))

    def update_coeff(self, poly):
        # set coefficients given by constraints (in neighbours)
        poly.apply_bc()
        for _, poly in poly.get_neighbours().items():
            self.update_coeff(poly)

    def no_mixed_terms(self):
        for poly in self.models:
            i = poly.smooth + poly.continuous
            poly.free[i:, i:] = False

        for poly in self.models:
            poly.set_freedoms()

    def full_freedom(self):
        for poly in self.models:
            i = poly.smooth + poly.continuous
            poly.free[i:, i:] = True

        for poly in self.models:
            poly.set_freedoms()

    def diagonal_freedom(self):
        for poly in self.models:
            i = poly.smooth + poly.continuous
            r, c = np.diag_indices(poly.shape.min())
            rm, cm = poly.shape
            poly.free[r[i:rm], c[i:cm]] = True

        for poly in self.models:
            poly.set_freedoms()

    def get_neighbours(self, i, j):
        # iterate through segments clockwise (origin lower left)
        n = {}
        ix = np.array(self.get_dependants(i, j))
        for ii, jj in zip(*ix):
            poly = self[self.rorder[ii, jj]]
            # print('neighbours to ', (i, j), ':', (ii, jj), repr(poly))
            if (ii - i) == 1:
                n['top'] = poly
            if (ii - i) == -1:
                n['bottom'] = poly
            if (jj - j) == -1:
                n['left'] = poly
            if (jj - j) == 1:
                n['right'] = poly

        return AttrDict(n)

    def get_dependants(self, i, j):
        # check which neighbouring polynomials depend on this one
        ip, jp = self.primary
        # taxicab (rectilinear / L1 / city block / Manhattan) distance
        yd, xd = self._ix - np.r_['0,3,0', (i, j)]
        d = np.abs([yd, xd]).sum(0)
        b = (d == 1)
        # dependencies move outward from primary polynomial
        if i < ip:  #
            b[i + 1:] = False
        if i > ip:
            b[:i] = False
        if j < jp:
            b[:, j + 1:] = False
        if j > jp:
            b[:, :j] = False

        return np.where(b)

    def segmented_image(self):
        seg = np.empty((self.yb[-1], self.xb[-1]))

        yro, xro = tuple(np.array(self.iorder).T)
        for k, s in enumerate(self.slices[yro, xro]):
            seg[tuple(s)] = k

        return seg + 1

    @staticmethod
    def _check_orders_knots(orders, knots):
        # check consistence of break points and polynomial orders
        bp = knots[1:]  # no constraint provided by 1st knot (edge)

        if isinstance(orders, float):
            orders = int(orders)

        if isinstance(orders, int):
            orders = [orders] * len(bp)

        if len(orders) != len(bp):
            raise ValueError('Order / knot vector size mismatch: %i, %i. '
                             'Knots should have size `len(orders) + 1`'
                             % (len(orders), len(knots)))

        # Might want to check the orders - if all 1, (straight lines) and
        # continuous = True, this is actually just a straight line, and it's
        # pointless to use a PiecewisePolynomial, but OK!

        return np.array(orders, int), np.asarray(knots)


from obstools.modelling.image import SegmentedImageModel

# from obstools.phot.segmentation import Slices, ModelledSegments


# class FastSlices(Slices):


#
# class DomainAdapterMixin():
#     def __init___(self):
#         super().__init__(self)
#         self.domains = []

#
# def get_coord_grids(self, sizes):
#     grids = []
#     for d in self.domains:
#         grids.append(
#                 np.mgrid[y0:y1:szy, x0:x1:szx])


# class ModelledSegments(GriddedSegments):
#
#     def __init__(self, data, grid=None):
#         super().__init__(data, grid)
#         self.domains = None
#
#     def get_coord_grids(self, labels=None):
#         grids = []
#         for i, (sy, sx) in enumerate(self.iter_slices(labels)):
#             h = sy.stop - sy.start
#             w = sx.stop - sx.start
#             (ylo, yhi), (xlo, xhi) = self.domains[i]
#             grids.append(np.mgrid[ylo:yhi:h * 1j,
#                                   xlo:xhi:w * 1j])
#
#         return grids


from obstools.modelling.parameters import Parameters


class Spline2D(SegmentedImageModel):

    # @classmethod
    # def from_image(cls, image, mask=None):
    #     """"""
    #
    #     knots = cls.guess_knots(image)

    @staticmethod
    def guess_knots(image, δσ=3, n_knots=(1, 2)):
        """
        A fast estimate of spline knot positions from a SALTICAM image

        Parameters
        ----------
        image
        δσ
        n_knots

        Returns
        -------

        """
        assert len(n_knots) == 2

        ishape = image.shape[::-1]
        yx_knots = []
        xo, yo = offset = (3, 3)  # offset (found empirically)
        for i in range(2)[::-1]:
            # median cross section
            m = np.ma.median(image, i)
            mm = np.ma.median(m)
            s = mad(m, mm)
            w = np.where((mm + δσ * s < m) | (m < mm - δσ * s))[0]
            grp = mit.consecutive_groups(w)

            nk = n_knots[i]
            if nk == 1:
                # last item from first consecutive group
                (*_, k0), *_ = map(list, grp)
                knots = (0, k0 + xo, ishape[i])
            elif nk == 2:
                # last item from first consecutive group and first item from
                # last consecutive group
                (*_, k0), *_, (k1, *_) = map(list, grp)
                knots = (0, k0 + yo, k1 - 2, ishape[i])
            else:
                raise NotImplementedError

            #
            yx_knots.append(knots)

        return yx_knots

    def detection_loop(self, image, mask=None, snr=(10, 7, 5, 3),
                       npixels=(7, 5, 3), edge_cutoff=None,
                       deblend=(True, False),
                       dilate=(5, 3)):
        """
        Multi-threshold blob detection on minimized residual background for
        modelled image
        """

        from recipes.string import seq_repr_trunc  # todo move to pprint
        from obstools.phot.utils import iter_repeat_last
        from obstools.phot.segmentation import SegmentationHelper

        #
        self.logger.info('Running detection loop')

        # make iterables
        variters = tuple(map(iter_repeat_last, (snr, npixels, dilate, deblend)))
        vargen = zip(*variters)

        # segmentation data
        data = np.zeros(image.shape, int)  # for new segments only!
        if mask is None:
            mask = np.zeros(image.shape, bool)

        # first round detection on raw data
        residual = image
        results = None
        counter = itt.count(0)
        while True:
            # keep track of iteration number
            count = next(counter)
            group_name = 'stars%i' % count

            # todo: detect method here ??
            # -----------------------------------------------------------------
            _snr, _npix, _dil, _debl = next(vargen)
            # or self.segm.__class__.detect

            edge_cutoff = [kk for k in self.knots for kk in k[1:-1]] + [None]
            sh = SegmentationHelper.detect(residual, mask, None, _snr, _npix,
                                           edge_cutoff, _debl, _dil)

            # update mask, get new labels
            new_mask = sh.to_bool()
            new_data = new_mask & np.logical_not(mask)
            mask = mask | new_mask

            # since we dilated the detection masks, we may now be overlapping
            # with previous detections. Remove overlapping pixels here
            if _dil:
                overlap = data.astype(bool) & new_mask
                sh.data[overlap] = 0
                new_data[overlap] = False

            # aggregate
            _, new_labels = self.segm.add_segments(sh)

            # add new segments
            data[new_data] = sh.data[new_data]

            # -----------------------------------------------------------------

            # log what has been found
            self.logger.info(
                    'detect_loop: round %i: %i new detections: %s',
                    count, len(new_labels), seq_repr_trunc(tuple(new_labels)))

            # break the loop if there are no new detections
            if not len(new_labels):
                break

            # add group info, detection meta data
            self.groups[group_name] = new_labels
            self.groups.info[group_name] = \
                dict(snr=_snr, npixels=_npix, dilate=_dil, deblend=_debl)

            # mask stars and bad pixels
            mimage = np.ma.MaskedArray(image, mask)

            if count == 0:
                self.logger.info('Optimizing knots')
                results = self.optimize_knots(mimage)
                # p0 = None
            # elif count == 1:
            #     self.diagonal_freedom()
            # p0 = self.get_param_vector()  # start from previous optimum
            # else:
            #     self.full_freedom()

            # fit background
            results = self.fit(mimage)  # method='nelder-mead'
            residual = self.residuals(results, image)
            # note: manage bad pixels inside segmentation to avoid passing
            # masked array in here

            return results, residual, data, mask

            # self.segm.display()

        # TODO: log summary of what you found
        return results, residual, data, mask

    def __init__(self, orders, knots, pri=(0, 0), smooth=True, continuous=True):

        # TODO smoothness / continuity at each knot

        # checks
        assert len(orders) == len(knots) == 2
        # reverse the order of input arguments: images are yx (row, column)
        checker = (self._check_orders_knots(*ok)
                   for ok in zip(orders, knots))

        # wrap namedtuple for easy attribute access
        self.orders, self.knots = itt.starmap(yxTuple, zip(*checker))
        self.n_polys = n_polys = tuple(map(len, self.orders))
        self.n_knots = np.subtract(n_polys, 1)
        self._ix = np.indices(self.n_polys)  # index grid for polys

        # static image dimensions
        self.ishape = (self.knots.y[-1], self.knots.x[-1])

        # get polynomial sequence order (iteration starts from primary poly)
        self.primary = ip, jp = pri
        yi = sorted(range(n_polys[0]), key=lambda i: np.abs(i - pri[0]))
        xi = sorted(range(n_polys[1]), key=lambda j: np.abs(j - pri[1]))
        self.iorder = list((i[::-1] for i in itt.product(xi, yi)))
        reorder = yro, xro = tuple(np.transpose(self.iorder))
        self.rorder = np.empty(self.n_polys, int)
        self.rorder[reorder] = np.arange(np.product(n_polys)) + 1

        # domain ranges
        domains = np.zeros((2, 2) + self.n_polys)  #
        # lower/upper, y/x, polys
        domains[1] = 1
        domains[:, self._ix < self.primary] -= 1
        domains = domains[:, :, yro, xro].T

        # segments
        segm = self.get_segmented_image()

        # init parent
        SegmentedImageModel.__init__(self, segm)
        self.segm.domains = dict(zip(self.segm.labels, domains))

        # create grid of 2D polynomials
        for count, ij in enumerate(self.iorder):
            # create 2d polynomial
            o = self.multi_order(*ij)  # 2d multi-order
            poly = PPoly2D(o, smooth, continuous)
            # ensure unique names
            poly.name = 'p%i%i' % ij  # p₀₁ ? poly2d_%i%i
            # add to container
            self.add_model(poly, count + 1)

        # get neighbours
        for ij, poly in zip(self.iorder, self.models.values()):
            n = self.get_neighbours(*ij)
            poly.set_neighbours(**n)

    # def __reduce__(self):

    def __call__(self, p, grid=None):
        # TODO: with an actual grid ??

        # dof consistency check
        self._check_params(p)

        # grid check
        # grid = self._check_grid(grid)

        # broadcast output
        # yg, xg = grid
        # b = np.broadcast(yg, xg)
        # out = np.empty(b.shape)  # result

        #
        out = np.zeros(self.ishape)

        # update coefficient matrices
        self.update_coeff(self.models[1])  # todo why before call

        # compute and fill in regions for each poly
        self.eval_loop(p, out)
        return out

    # def fit(self, data, stddev=None, **kws):
    #     # median rescale
    #     scale = nd_sampler(data, np.median, 100)
    #     results = SegmentedImageModel.fit(self, data / scale, stddev)
    #     results[...] = tuple(r * scale for r in results.tolist())
    #     return results

    def _check_params(self, p):
        if isinstance(p, Parameters):
            npar = p.npar
        else:
            npar = len(p)

        if npar != self.dof:
            raise ValueError('Parameter vector size (%i) does not match '
                             'degrees of freedom (%i) for model %r' %
                             (len(p), self.dof, self))

    def get_param_vector(self, nested=True):
        """
        Construct a parameter vector from the coefficient matrices of the
        constituent polynomials considering only the free parameters. This
        is useful when initially fitting a model with restricted parameter
        space, then freeing some parameters and re-fitting starting from
        the previous optimum.
        """
        if nested:
            p = self._results_container()
            for poly in self.models.values():
                p[poly.name] = poly.coeff[poly.free]
        else:
            p = np.zeros(self.dof)
            splix = np.zeros(self.nmodels + 1, int)
            splix[1:] = np.cumsum(self.models.dofs)
            slices = itt.starmap(slice, mit.pairwise(splix))
            for poly, sl in zip(self.models.values(), slices):
                p[sl] = poly.coeff[poly.free]
        return p

    def eval_loop(self, p, out):
        # compute and fill in regions for each poly
        pp = self.split_params(p)
        grids = self.segm.coord_grids
        for i, ((lbl, slc), mdl) in enumerate(zip(self.segm.enum_slices(),
                                                  self.models.values())):
            ppp = pp[i]
            if np.isnan(ppp).any():
                continue

            out[slc] += mdl(ppp, grids[lbl])

    def _optimize_knots(self, data, labels, range=(-3, 4)):
        # brute force optimization for knot position (boundary) between two
        # adjacent 2d polynomials in the spline
        # note this method overwrites the segmentation image!

        assert len(labels) == 2

        knots = self.knots
        # get polynomial index position in list of models
        pix = np.take(self.iorder, np.subtract(labels, 1), 0)
        i = np.asscalar(np.where(np.all(pix[0] == pix, 0))[0])
        io = int(not bool(i))
        j = pix[:, io].max()

        k_start = knots[io][j]
        k_vals = k_start + np.arange(*range)
        χ2r = np.ma.empty(np.ptp(range))
        χ2r.mask = True
        results = {}
        for l, k in enumerate(k_vals):
            knots[io][j] = k
            self.set_knots(knots)  # update segmentation
            try:
                # optimize
                r = self.fit(data, labels=labels, full_output=True)
            except UnconvergedOptimization:
                pass
                # self.logger.exception('UnconvergedOptimization in'
                #                       ' `_optimize_knots`')
            else:
                χ2r[l] = self.redchi(r, data)
                results[k] = r

        i_best = χ2r.argmin()
        k_best = k_vals[i_best]
        r_best = results[k_best]
        knots[io][j] = k_best
        if r_best is not None:
            self.set_knots(knots)  # update segmentation

        # report
        if self.logger.getEffectiveLevel() <= logging.INFO:
            from motley.table import Table

            title = 'Knot optimizer results: labels=%s; k: %i --> %i' \
                    % (labels, k_start, k_best)
            tbl_data = [k_vals, χ2r]
            states = np.zeros_like(tbl_data)
            states[:, i_best] = 1

            # what precision do we need to see minimum
            z = χ2r - χ2r.min()
            z[i_best] = np.ma.masked
            precision = int(np.ceil(np.abs(np.ma.log10(z)).max())) + 1

            tbl = Table(tbl_data,
                        row_headers=['k', 'χ²ᵣ'],
                        precision=precision, minimalist=True,
                        title=title, title_align='l')
            tbl.colourise(states, 'g')
            self.logger.info('\n' + str(tbl))

        return knots, r_best

    def optimize_knots(self, data, ):
        """
        Brute force hyper parameters optimization for knot positions.
        """

        self.logger.info('Running knot optimizer')

        pri_lbl = 1
        labels_neighbours = self.rorder[self.get_dependants(*self.primary)]
        results = self._results_container()
        for lbl in labels_neighbours:
            knots, r = self._optimize_knots(data, [pri_lbl, lbl])
            # check if the optimization converged
            if r is not None:
                for l in (lbl, pri_lbl):
                    name = self.models[l].name
                    results[name] = r[name]
            else:
                self.logger.warning(
                        'Knot optimization failed for labels: %s', lbl)
        return results



    # def p0guess(self, data, grid=None, stddev=None):
    #     return np.zeros(self.dof)

    def split_params(self, p):
        """
        Split parameter vector into `n` parts, where `n` is the number of
        polynomials in the model, and each part has size `m` corresponding to
        the number of free coefficients in the component polynomials.

        Parameters
        ----------
        p

        Returns
        -------

        """
        if isinstance(p, Parameters):
            return p.tolist()
        return np.split(p, np.cumsum(self.models.dofs))

    def multi_order(self, i, j):
        return self.orders.y[i], self.orders.x[j]  # 2d multi-order

    def get_slices(self, knots):
        domains = itt.product(*map(mit.pairwise, knots))
        domains = np.reshape(list(domains), self.n_polys + (2, 2))
        return np.vectorize(slice)(*domains.round().astype(int).T).T

    def update_coeff(self, poly):
        # set coefficients given by constraints (in neighbours)
        poly.apply_bc()
        for _, poly in poly.get_neighbours().items():
            self.update_coeff(poly)

    # --------------------------------------------------------------------------
    # Parameter space restrictions

    def maximal_degree(self, n):
        # restrict polynomials to have degree smaller equal n
        models = self.models.values()
        for poly in models:
            poly.free[n:, n:] = False

        for poly in models:
            poly.set_freedoms()

    def no_mixed_terms(self):
        models = self.models.values()
        for poly in models:
            poly.free[1:, 1:] = False

        for poly in models:
            poly.set_freedoms()

    def full_freedom(self):
        models = self.models.values()
        for i, poly in enumerate(models):
            if self.iorder[i] == self.primary:
                poly.free[:] = True
            else:
                j = poly.smooth + poly.continuous
                poly.free[j:, j:] = True

        for poly in models:
            poly.set_freedoms()

    def diagonal_freedom(self):
        models = self.models.values()
        for poly in models:
            i = poly.smooth + poly.continuous
            r, c = np.diag_indices(poly.shape.min())
            rm, cm = poly.shape
            poly.free[r[i:rm], c[i:cm]] = True

        for poly in models:
            poly.set_freedoms()

    # --------------------------------------------------------------------------

    def get_neighbours(self, i, j):
        # iterate through segments clockwise (origin lower left)
        n = AttrDict()
        ix = np.array(self.get_dependants(i, j))
        for ii, jj in zip(*ix):
            poly = self.models[self.rorder[ii, jj]]
            # print('neighbours to ', (i, j), ':', (ii, jj), repr(poly))
            if (ii - i) == 1:
                n['top'] = poly
            if (ii - i) == -1:
                n['bottom'] = poly
            if (jj - j) == -1:
                n['left'] = poly
            if (jj - j) == 1:
                n['right'] = poly

        return n

    def get_dependants(self, i, j):
        # check which neighbouring polynomials depend on this one
        ip, jp = self.primary
        # taxicab (rectilinear / L1 / city block / Manhattan) distance
        yd, xd = self._ix - np.r_['0,3,0', (i, j)]
        d = np.abs([yd, xd]).sum(0)
        b = (d == 1)
        # dependencies move outward from primary polynomial
        if i < ip:  #
            b[i + 1:] = False
        if i > ip:
            b[:i] = False
        if j < jp:
            b[:, j + 1:] = False
        if j > jp:
            b[:, :j] = False

        return np.where(b)

    def get_segmented_image(self):
        """Create segmented image data from knots"""
        seg = np.zeros(self.ishape, int)
        slices = self.get_slices(self.knots)
        for k, ij in enumerate(self.iorder):
            seg[tuple(slices[ij])] = k + 1

        return seg

    def set_knots(self, knots):
        assert len(knots) == 2
        # reverse the order of input arguments: images are yx (row, column)
        checker = (self._check_orders_knots(*ok)
                   for ok in zip(self.orders, knots))
        _, self.knots = itt.starmap(yxTuple, zip(*checker))

        # get the updated background segmentation
        seg = self.get_segmented_image()

        # preserve any other segments "on top of" background
        current = self.segm.labels
        keep = current[current > self.nmodels]
        seg_fg = self.segm.select(keep)
        mask = (seg_fg != 0)
        seg[mask] = seg_fg[mask]

        self.segm.data = seg

    @staticmethod
    def _check_orders_knots(orders, knots):
        # check consistence of break points and polynomial orders
        k = knots[1:]  # no constraint provided by 1st knot (edge)

        if isinstance(orders, float):
            orders = int(orders)

        if isinstance(orders, int):
            orders = [orders] * len(k)

        if len(orders) != len(k):
            raise ValueError('Order / knot vector size mismatch: %i, %i. '
                             'Knots should have size `len(orders) + 1`'
                             % (len(orders), len(knots)))

        # Might want to check the orders - if all 1, (straight lines) and
        # continuous = True, this is actually just a straight line, and it's
        # pointless to use a Spline2d, but OK!

        return np.array(orders, int), np.asarray(knots)

    def plot_fit_results(self, image, result, show_knots=True):

        from obstools.modelling.image.diagnostics import plot_modelled_image

        fig = plot_modelled_image(self, image, result)

        if result is not None and show_knots:
            ishape = self.segm.shape
            for ax in fig.axes[2:3]:
                ax.vlines(self.knots.x, 0, ishape[0], 'k')
                ax.hlines(self.knots.y, 0, ishape[1], 'k')
                ax.set(xlim=(-0.5, ishape[1] - 0.5),
                       ylim=(-0.5, ishape[0] - 0.5))
        return fig


class Vignette2(Vignette):

    # non-uniform spline with variable knot positions

    def __init__(self, orders, knots, pri=(0, 0), smooth=True,
                 continuous=True, free_knots=False):
        super().__init__(orders, knots, pri, smooth, continuous)

        self._var_knots = yxTuple(self.knots.y.copy(), self.knots.x.copy())
        self._free_knots = free_knots

    def __call__(self, p, grid=None):
        if self._free_knots:
            p, *knots_internal = np.split(p, (-np.cumsum(self.n_knots)))
            # have to re-calculate the grid sizes for polys
            for i in (0, 1):
                self._var_knots[i][1:-1] = knots_internal[i]

            # regions
            slices = self.get_slices(self._var_knots)

            out = np.empty(self.ishape)
            return self.eval_loop(p, slices, out)
        else:
            return super().__call__(p, grid)

    @property
    def dof(self):
        # number of free parameters. including the knot vector
        dof = super().dof  # free coefficients
        if self._free_knots:
            return dof + self.n_knots.sum()
        return dof

    def get_slices(self, knots):
        domains = itt.product(*map(mit.pairwise, knots))
        domains = np.reshape(list(domains), self.n_polys + (2, 2))
        return np.vectorize(slice)(*domains.round().astype(int).T).T

    def get_intervals(self, knots):
        domains = itt.product(*map(mit.pairwise, knots))
        domains = np.reshape(list(domains), self.n_polys + (2, 2))
        slices = np.vectorize(slice)(*domains.T).T
        sizes = domains.ptp(-1)
        return sizes, slices

    def get_bounds(self, inf=None, knot_freedom=0.35):
        """layperson speak for (uniform) priors"""

        if not self._free_knots:
            return

        if inf is None:
            unbound = (None, None)
        elif isinstance(inf, numbers.Real):
            unbound = (-inf, inf)
        else:
            raise ValueError

        bounds = [unbound] * self.dof
        bpb = self.get_knot_bounds(knot_freedom)
        bounds[-len(bpb):] = bpb
        return bounds

    def get_knot_bounds(self, freedom=0.35):
        # each knot is now allowed to vary on a bounded interval surrounding the
        # initial value so long as it does not cross more than halfway the
        # distance to the neighbouring knot, or fall outside the interval (0, 1)
        assert freedom < 0.5
        bp = self.breakpoints
        v = np.diff(bp) * freedom
        δ = fold.fold(v, 2, 1) * [-1, 1]
        return tuple(bp[None, 1:-1].T + δ)

    def free_knots(self):
        self._free_knots = True


class Vignette2DCross(CompoundModel):  # StaticGridMixin
    """
    Models the 2D vignetting pattern in SALTICAM slot mode images as outer
    product between two smooth piecewise polynomials.

    This is a subset of 2d polynomials, but with fewer free parameters
    """

    name = 'vignette'

    def __init__(self, orders, breaks, smoothness=None):
        # TODO: might be better ito pickling to have this as a constructor
        # classmethod
        # def from_orders():

        """

        Parameters
        ----------
        orders: xorders, yorders
        breaks: xbreaks, ybreaks
        """

        # save init_args for pickling
        self._init_args = (orders, breaks, smoothness)

        if smoothness is None:
            smoothness = (True, True)

        # initialize models
        polyx, polyy = (PPolyModel(o, b, smooth=s)  # PPolyModel
                        for (o, b, s) in zip(orders, breaks, smoothness))
        # init base class
        CompoundModel.__init__(self, y=polyy, x=polyx)

    def __call__(self, p, grid=None):
        if grid is None:
            grid = self.static_grid

        p = p.squeeze()
        zy = self.y(p['y'], grid[0])
        zx = self.x(p['x'], grid[1])
        return np.outer(zy, zx)

    def __reduce__(self):
        # overwrites OrderedDict.__reduce__.
        # since the base class implements __reduce__, simply overwriting
        # __getnewargs__ will not work
        return Vignette2DCross, self._init_args, dict(grid=self.grid)

    def __setstate__(self, state):
        # HACK for static grid on component models
        for m, g in zip(self.models, state['grid']):
            m.set_grid(len(g))

        self.__dict__.update(state)

    def set_grid(self, data):

        # for m, sz in zip(self.models,                         data.shape):
        #     m.static_grid =

        self.static_grid = [m.set_grid(sz) for m, sz in zip(self.models,
                                                            data.shape)]
        return self.static_grid

    def adapt_grid(self, grid):
        return None

    def fwrs(self, p, data, grid, data_stddev=None):
        # HACK since computing via outer product can't directly apply mask
        # remove masked elements here
        wrs = self.wrs(p, data, grid, data_stddev)
        if np.ma.is_masked(wrs):
            return wrs[~wrs.mask].data
        return wrs

    def get_cross_sections(self, image):

        # median scale image
        scale = abs(np.ma.median(image))
        data = image / scale

        dtype = list(zip('yx', 'ff', image.shape))
        media = np.ma.empty((), dtype)
        madness = np.ma.empty((), dtype)
        # m = []
        for i, mdl in enumerate(self.values()):

            oth = int(not bool(i))

            # mdl.name
            # print(i, mdl.name, mdl.grid.shape)

            # fit median cross section
            # (full model often doesn't converge with inferior (but faster)
            # frequentist methods)

            # mask edges so that median doesn't skew due to edge effects
            em = np.zeros(image.shape, bool)
            bppix = self[oth].breakpoints * self[oth]._scale
            not_edge = np.diff(bppix).argmax()
            for (j, bp) in enumerate(mit.pairwise(bppix)):
                ix = [slice(None), slice(None)]
                if j != not_edge:
                    ix[oth] = slice(*np.round(bp).astype(int))
                    em[tuple(ix)] = True

            # media.append(np.ma.MaskedArray(image, em))
            # return np.ma.MaskedArray(image, em)
            # median = np.ma.median(np.ma.MaskedArray(data, em), oth)
            # media.append(median)
            # m.append(np.ma.MaskedArray(data, em) )
            using = np.ma.MaskedArray(data, em)
            media[mdl.name] = np.ma.median(using, oth)
            madness[mdl.name] = mad(np.ma.median(using, axis=oth))
        # return m
        return media, madness, scale

    def fit(self, p0, image, grid=None, std=None, report=False):
        """Fit the segmented background median cross sections."""

        if p0 is None:
            p0 = self.p0guess(image, grid)

        results = np.empty_like(p0)
        media, _, scale = self.get_cross_sections(image)
        for i, mdl in enumerate(self.values()):
            # HACK not using grid
            r = mdl.fit(p0[mdl.name], media[mdl.name], mdl.grid, )
            # post_args=(np.sqrt(scale),))  # does re-scale
            if r is not None:
                results[mdl.name][:mdl.npar] * np.sqrt(scale)  # HACK!

            # if report:
            #     print(' %s \n---\n' % 'YX'[i])
            #     print(lm.fit_report(result))

        return results

    # def post

    def fit_report(self, data, results, _display=True):
        # diagnostic plots
        from IPython.display import display, Math

        figs, reprs = [], []
        for i, r in enumerate(results):
            v = 'yx'[i]
            mdl = self.models[i]
            print(' %s \n---\n' % v.title())
            print(lm.fit_report(r))

            fig = mdl.plot_fit_results(data, r.params)
            figs.append(fig)
            if _display:
                display(fig)

            # Show poly repr for x in internal (0, 1) domain
            rep = mdl.get_repr(r.params, v.title(), v, onedomain=True)
            reprs.append(rep)
            if _display:
                display(Math(rep))

            # Show poly repr for x in pixel coordinates
            cx = []
            for c in mdl.get_block_coeff(r.params, onedomain=True).T:
                c = transform_poly(c, 1 / mdl._gridscale, 0)
                cx.append(c)

            bp = mdl.breakpoints * mdl._gridscale
            rep2 = get_ppoly_repr(cx, bp, v.title(), v, expmin=3)
            if _display:
                display(Math(rep2))
            print('\n\n')

    def plot_fit_results(self, image, params, modRes=500):

        media, madness, scale = self.get_cross_sections(image)
        # todo may be useful to know which data was used for calculating medians

        figs = []
        for i, mdl in enumerate(self.models):

            # plot rows / column data
            pixels = image
            if i == 1:
                pixels = pixels.T

            # plot model / median data used for fit
            p, mdata, e = params[mdl.name], media[mdl.name], madness[mdl.name]
            if len(p) > mdl.npar:  # tmp HACK for variable breakpoints
                p, bp = np.split(p, [mdl.npar])
            else:
                bp = mdl.breakpoints
            ebc, lines_mdl, lines_bp = mdl.plot_fit_results(p, mdata, e,
                                                            scale,
                                                            modRes, bp)

            # plot pixel values
            # don't scale the limits for the data since there might be large
            # values from cosmic rays a=and whatnot
            ax1 = lines_mdl.axes
            # ax1.autoscale(False)
            # print(1, ax1.viewLim)

            # plot  pixel values below data
            data_lines = ax1.plot(pixels, 'g', alpha=0.35, zorder=-1,
                                  label='Data')
            legArt = [ebc, data_lines[0], lines_mdl, lines_bp]
            legLbl = [_.get_label() for _ in legArt]

            # print(2, ax1.viewLim)

            ax1.legend(legArt, legLbl)
            figs.append(ax1.figure)

            # break
        return figs

# class VignetteTest(BackgroundMixin, Vignette2DCross):
#     def __init__(self, orders, breaks, segmentation):
#         # init model
#         xorders, yorders = orders
#         xbreaks, ybreaks = breaks
#         Vignette2DCross.__init__(self,
#                                  xorders, xbreaks,
#                                  yorders, ybreaks)
#         self.set_grid(segmentation.data)
#
#         # init segmentation methods
#         BackgroundMixin.__init__(self, segmentation)
#
#     def __call__(self, p, labels=None, out=None):
#         labels = self.resolve_labels(labels)
#         assert len(p) == len(labels)
#
#         if out is None:
#             out = np.zeros(self.segm.shape)
#
#         for i, (sly, slx) in enumerate(self.segm.iter_slices(labels)):
#             out[sly, slx] += super()(p[i])  #
#
#         return out


# class Vignette(BackgroundModel):
#     def __init__(self, segmentation, orders, breaks, use_labels=(0,)):
#         # init model
#
#         vignette = Vignette2DCross(orders, breaks)
#         vignette.set_grid(segmentation.data)
#
#         # init segmentation methods
#         BackgroundModel.__init__(self, segmentation, vignette, use_labels)
#
#     def adapt_grid(self, grid):
#         return None

# def __call__(self, p, labels=None, out=None):
#     labels = self.resolve_labels(labels)
#     assert len(p) == len(labels)
#
#     if out is None:
#         out = np.zeros(self.segm.shape)
#
#     for i, (sly, slx) in enumerate(self.segm.iter_slices(labels)):
#         out[sly, slx] += self.model(p[i])  #
#
#     return out


# class SmoothPPolyLM(PPolyModel):
#
#     def make_params(self, p0=None, namebase=None):
#         # create parameters
#         if p0 is None:
#             p0 = np.ones(self.nfree)
#
#         pars = lm.Parameters()
#         # lm doesn't like non alpha numeric characters as parameter names
#         pnames = self.get_pnames(namebase, free=True, latex=False)
#         for pn, p in zip(pnames, p0):
#             pars.add(pn, p)
#         return pars
#
#     def fit(self, data, grid, p0=None, report=False):
#
#         scale = np.ma.median(data)
#         data = data / scale  # HACK empirically better fit results for scaled data
#
#         if p0 is None:
#             pars = self.make_params()  # initializes with all ones
#         else:
#             pars = p0
#
#         result = lm.minimize(self.wrs, pars,
#                              args=(data[~data.mask], grid[~data.mask], None))
#
#         # HACK for scale
#         for pn, pv in result.params.items():
#             pv.value *= scale
#             pv.stderr *= scale
#         # aic bic redchi residuals ??
#
#         # print report
#         if report:
#             # print(' %s \n---\n' % 'YX'[i])
#             print(lm.fit_report(result))
#
#         return result
#
#     __call__ = convert_params(PiecewisePolynomial.__call__)
#     get_repr = convert_params(PiecewisePolynomial.get_repr)
#     get_block_coeff = convert_params(PiecewisePolynomial.get_block_coeff)


# class SmoothPPolyLMCross(SmoothPPolyLM):
#     def __init__(self, orders, breaks, nbc=2, axis=0, coeffNameBase='a'):
#         SmoothPPolyLM.__init__(self, orders, breaks, nbc, coeffNameBase)
#         self._axis = int(axis)
#
#     def plot_fit_results(self, image, params, modRes=500):
#         # plot fit result
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
#                                        sharex=True,
#                                        gridspec_kw=dict(hspace=0,
#                                                         height_ratios=(3, 1)))
#
#         # get data
#         i = self._axis
#         axs = 'yx'[i]
#         grid, data, uncertainty = g, d, e = get_cross_section_data(image, i)
#         scale = 1  # np.ma.median(data)
#
#         # plot fitted data
#         ebc = ax1.errorbar(g, d, e, fmt='mo', zorder=10,
#                            label='Median $\pm$ (MAD)')
#
#         # plot rows / column data
#         pixels = image / scale
#         if i == 1:
#             pixels = pixels.T
#
#         lines = ax1.plot(g, pixels, 'g', alpha=0.35, label='Data')
#
#         # breakpoints
#         # print(axs, 'breaks', db.model[axs].breakpoints)
#         breakpoints = self.breakpoints  # * g.max()
#         lines_bp = ax1.vlines(breakpoints, 0, 1,
#                               linestyle=':', color='0.2',
#                               transform=ax1.get_xaxis_transform(),
#                               label='Break points')
#
#         # model
#         gm = np.linspace(0, g.max(), modRes)  # bg.shape[not bool(i)]
#         dfit = self(params, gm) * np.sqrt(np.ma.median(data))
#         lines_mdl, = ax1.plot(gm, dfit, 'r-', label='model')
#
#         # plot model residuals
#         res = data - self(params, grid) * np.sqrt(np.ma.median(data))
#         ax2.errorbar(g, res, e, fmt='mo')
#         # percentile limits on errorbars (more informative display for large errorbars)
#         lims = np.percentile(res - e, 25), np.percentile(res + e, 75)
#         ax2.set_ylim(*lims)
#
#         ax1.set_title(('%s cross section fit' % axs).title())
#         ax1.set_ylabel('Normalized Counts')
#         legArt = [ebc, lines[0], lines_mdl, lines_bp]
#         legLbl = [_.get_label() for _ in legArt]
#         ax1.legend(legArt, legLbl)
#         ax1.grid()
#
#         ax2.set_ylabel('Residuals')
#         ax2.grid()
#
#         fig.tight_layout()
#         return fig


# class Vignette2DCrossLM(Vignette2DCross):
#
#     def __init__(self, xorders, xbreaks, yorders, ybreaks):
#         # initialize models
#         mdlx = SmoothPPolyLMCross(xorders, xbreaks, 2, 1, 'a')
#         mdly = SmoothPPolyLMCross(yorders, ybreaks, 2, 0, 'r')
#         self.models = ModelContainer(mdly, mdlx)
#
#     def make_params(self, p0=None):
#
#         # create parameters
#         if p0 is not None:
#             py, px = np.split(plist(p0), [self.models.y.nfree])
#         else:
#             py, px = None, None
#
#         params = lm.Parameters()
#         for mdl, p in zip(self.models, (py, px)):
#             for pn, pv in mdl.make_params(p).items():
#                 params.add(pn, pv)
#         return params
#
#     def make_params_from_result(self, result):
#
#         ry, rx = result
#         p = lm.Parameters()
#         p.update(ry.params)
#         p.update(rx.params)
#
#         return p
#
#     def fit(self, image, p0=None, masked_ignore_thresh=0.35, report=False):
#
#         scale = np.ma.median(image)
#         data = image / scale
#
#         results = []
#         for i, result in enumerate(
#                 self.fit_cross(data, p0, masked_ignore_thresh, False)):
#             # HACK for scale
#             for pn, pv in result.params.items():
#                 pv.value *= np.sqrt(scale)
#                 pv.stderr *= np.sqrt(scale)
#
#             if report:
#                 print(' %s \n---\n' % 'YX'[i])
#                 print(lm.fit_report(result))
#
#             results.append(result)
#
#         return results


# class Vignette(PiecewisePolynomial, Model):
#     pass


# if __name__ == '__main__':
#     from matplotlib import pyplot as plt
#
#     orders_x, bpx = (1, 5), (0, 161.5, image.shape[1])
#     orders_y, bpy = (3, 1, 5), (0, 3.5, 17, image.shape[0])
#     v2 = Vignette2DCross(orders_x, bpx, orders_y, bpy)
#     self = mdl = v2.models.x
#
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#
#     p = np.random.randn(mdl.nfree)  # np.ones(mdl.nfree)#
#     coeff = self.get_block_coeff(p, True).T
#     xres = 500
#
#     scale = self.breakpoints.max()
#     bp = self.breakpoints / scale
#     for i, (i0, i1) in enumerate(pairwise(bp)):
#         c = coeff[i]
#         x = np.linspace(i0, i1, xres)
#         y = np.polyval(c, x)
#
#         ax1.plot(x, y)
#
#         o = self.breakpoints[i] / scale
#         s = 1 / scale
#         c2 = transform_poly(c, s, o)
#         x2 = (x - o) / s
#
#         # print(c)
#         # print(x[[0, -1]])
#         # print(x2[[0, -1]])
#
#         y2 = np.polyval(c2, x2)
#         ax2.plot(x2, y2)
#
#     X = np.linspace(self.breakpoints.min(), self.breakpoints.max(), xres)
#     # coeff = self.get_block_coeff(p)
#     # for i, c in enumerate(coeff):
#     #    coeff[i] = transform_poly(c, 1, self.breakpoints[i])
#
#     # pp = PPoly(coeff.T, self.breakpoints)
#     # X = np.linspace(0, 1, xres)
#     ax3.plot(X, self(p, X))
#
#     # cx = mdl.get_block_coeff(r.params)
#     # scale = 1  # / np.max(bpx)
#     # x = mdl.breakpoints[1]
#     # a = np.polyval(cx[0], x)
#     # b = np.polyval(cx[1], 0)
#     # print(a, b)
#     #
#     # # trans
#     # ct = np.empty_like(cx)
#     # for i, c in enumerate(cx):
#     #     o = mdl.breakpoints[i]
#     #     ct[i] = transform_poly(c, 1, -o)
#     #
#     # x = mdl.breakpoints[1]
#     # e = np.polyval(ct[0], x)
#     # f = np.polyval(ct[1], x)
#     # print(e, f)
#     #
#     # ct = np.empty_like(cx)
#     # s = 1 / np.max(bpx)
#     # for i, c in enumerate(cx):
#     #     o = -mdl.breakpoints[i]
#     #     ct[i] = transform_poly(c, s, o)
#     #
#     # x = mdl.breakpoints[1] / s
#     # g = np.polyval(ct[0], x)
#     # h = np.polyval(ct[1], x)
#     # print(g, h)
#
#     ppy = PPolyModel((1, 1, 1), (0, 3, 17, 24), smooth=False)
#
#     # y cross
#     bm = make_border_mask(image, (1, -5, 0, None))
#     m = ndimage.binary_dilation((image > 400), iterations=3)
#     immy = np.ma.masked_where(bm | m, image)
#
#     p0y = np.ones(ppy.npar)
#     gy = np.mgrid[:image.shape[0]].astype(float)
#     dy = np.ma.median(immy, 1).data
#     scale = abs(np.median(dy))
#     dy /= scale
#     py = ppy.fit(p0y, dy, gy)
#
#     fig, ax = plt.subplots()
#     ax.plot(immy)
#     ax.plot(gy, ppy(py, gy) * scale, 'k-', lw=2)
#
#     # x cross
#     bm = make_border_mask(image, (1, None, 4, -6))
#     m = ndimage.binary_dilation((image > 400), iterations=3)
#     immx = np.ma.masked_where(bm | m, image)
#
#     bp = np.divide((0, 162, image.shape[1]), 1)  # image.shape[1])
#     ppx = PPolyModel((1, 3), bp, smooth=True)
#     p0x = np.ones(ppx.npar)
#     gx = np.mgrid[:image.shape[1]].astype(float)
#     # gx = np.linspace(0, 1, image.shape[1])
#     dx = np.ma.median(immx, 0)
#     scale = abs(np.median(dx))
#     dx /= scale
#     px = ppx.fit(p0x, dx, gx)
#
#     fig, ax = plt.subplots()
#     ax.plot(immx.T)
#     ax.plot(gx, ppx(px, gx) * scale, 'k-', lw=2)
