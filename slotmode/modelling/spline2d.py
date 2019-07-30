"""
Piecewise polynomials in 2d with boundary conditions
"""

# builtin libs
import logging
import itertools as itt
from collections import namedtuple

# third-party libs
import numpy as np
import more_itertools as mit

# local libs
from recipes.dict import AttrDict
from .ppoly import PPoly2D, PPoly2D_v2
from obstools.modelling.parameters import Parameters
from obstools.modelling.image import SegmentedImageModel
from obstools.modelling.core import UnconvergedOptimization, nd_sampler
from recipes import pprint

from .ppoly import _check_orders_knots

SEMANTIC_IDX2POS = {(1, 0): 'top',
                    (-1, 0): 'bottom',
                    (0, 1): 'right',
                    (0, -1): 'left'}

# simple container for 2-component objects
yxTuple = namedtuple('yxTuple', ['y', 'x'])


def _yx_str(obj):
    return f'(y={obj.y!s}, x={obj.x!s})'


def random_walk_gen(n, n_dims=1, step_size=None, origin=None):
    """
    Basic random walk generator

    Parameters
    ----------
    n
    n_dims
    step_size
    origin

    Returns
    -------

    """
    n = int(n)
    n_dims = int(n_dims)

    if step_size is None:
        step_size = 1. / np.sqrt(n_dims)

    if origin is None:
        origin = np.zeros(n_dims)
    else:
        origin = np.asanyarray(origin)
        assert len(origin) == n_dims, 'Origin has incorrect dimensions'

    step_set = [-1, 0, 1]
    pos = origin
    for i in range(n):
        δ = np.random.choice(a=step_set, size=n_dims) * step_size
        pos = pos + δ
        yield pos


# from IPython import embed


# def _gen_order_tuple(low, high):
#     assert len(low) == len(high)
#     return itt.product(*map(range, low, high))
#
#
# def _gen_order_tuples(o_range_y, o_range_x):
#     return itt.product(*map(_gen_order_tuple, *zip(o_range_y, o_range_x)))


# def spline_order_search(image, o_range_y, o_range_x):
#     """Hyper parameter optimization for polynomial spline orders"""


class Spline2D(SegmentedImageModel):

    def __init__(self, orders, knots, smooth=True, continuous=True,
                 primary_segment=None, poly_name_base='p'):

        # TODO smoothness / continuity at each knot ?

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
        if primary_segment is None:
            # The segment with the largest area will be the primary one
            sizes = list(itt.product(*map(np.diff, self.knots)))
            sizes = np.reshape(sizes, self.n_polys + (2,))
            primary_segment = divmod(sizes.prod(-1).argmax(), self.n_polys[1])
            # todo: maybe use self.segm.area.argmax()

        self.primary = pri = primary_segment
        yi = sorted(range(n_polys[0]), key=lambda i: np.abs(i - pri[0]))
        xi = sorted(range(n_polys[1]), key=lambda j: np.abs(j - pri[1]))
        # re-order the sequence of polynomials so that the primary one
        # (largest area) comes first in the list of models followed by its
        # neighbours
        self.iorder = list((i[::-1] for i in itt.product(xi, yi)))
        reorder = tuple(np.transpose(self.iorder))
        self.rorder = np.empty(self.n_polys, int)
        self.rorder[reorder] = np.arange(np.product(n_polys)) + 1

        # get segmented image
        segm = self.get_segmented_image()
        # init parent
        SegmentedImageModel.__init__(self, segm)

        # calculate domain ranges
        domains = self.get_domain_ranges()
        self.segm.domains = dict(zip(self.segm.labels, domains))
        # self.coord_grids = self.segm.coord_grids

        # create grid of 2D polynomials
        for label, (i, j) in enumerate(self.iorder, 1):
            # create 2d polynomial
            o = self.multi_order(i, j)  # multi-order 2-tuple
            poly = PPoly2D(o, smooth, continuous)
            # ensure unique names
            poly.name = '%s%i%i' % (poly_name_base, i, j)  # p₀₁ ?
            # poly2d_%i%i
            # add to container
            self.add_model(poly, label)
            poly.set_grid(self.segm.coord_grids[label])

        # get neighbours
        for ij, poly in zip(self.iorder, self.models.values()):
            n = self.get_neighbours(*ij)
            poly.set_neighbours(**n)

    def __call__(self, p, grid=None):
        # TODO: with an actual grid ??

        if isinstance(p, np.void):
            p = Parameters(p)

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

    def __str__(self):
        # multi-line string repr
        name = self.__class__.__name__
        ws = ' ' * len(name)
        args = 'orders', 'knots'
        w0 = max(map(len, args))
        oy, ox = map(str, self.orders)
        ky, kx = map(str, self.knots)
        wy = max(map(len, (oy, ky)))
        wx = max(map(len, (ox, kx)))
        s = ''
        for i, (pre, arg, y, x) in enumerate(
                zip((name, ws), args, *zip(self.orders, self.knots))):
            s += f'{pre}({arg:<{w0}}=(y={y!s:<{wy}}, x={x!s:<{wx}})'
            s += ',\n' * int(not i)
        return s
        # return f'{name}({args[0]:<{w0}}=(y={oy!s:<{wy}}, x={ox!s:<{wx}}),\n' \
        #        f'{ws} {args[1]:<{w0}}=(y={ky!s:<{wy}}, x={kx!s:<{wx}}))'
        # return f'{name}(orders={_yx_str(self.orders)},\n' \
        #        f'{ws} knots={_yx_str(self.knots)})'

    # def fit(self, data, stddev=None, **kws):
    #     # median rescale
    #     scale = nd_sampler(data, np.median, 100)
    #     results = SegmentedImageModel.fit(self, data / scale, stddev)
    #     results[...] = tuple(r * scale for r in results.tolist())
    #     return results

    def _check_params(self, p):
        if p.__class__.__name__ == 'Parameters':  # HACK auto-reload
            # isinstance(p, Parameters):
            npar = p.npar
        else:
            npar = len(p)

        if npar != self.dof:
            raise ValueError('Parameter vector size (%i) does not match '
                             'degrees of freedom (%i) for model %r' %
                             (npar, self.dof, self))

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
            splix = np.zeros(len(self.models) + 1, int)
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

            # fill the modelled segment in the output array
            out[slc] = mdl(ppp, grids[lbl])

    def optimize_knots(self, image, range=(-3, 4), info='', report=True, **kws):
        """
        Brute force hyper parameters optimization for knot positions.
        """
        pri_lbl = 1
        labels_neighbours = self.rorder[self.get_dependants(*self.primary)]
        results = self._results_container()
        tbl = []
        knots = self.knots
        # messages = f'\nKnot optimizer: {info}'
        for lbl in labels_neighbours:
            #
            r, k_vals, χ2r, indices = self._optimize_knots(
                    image, [pri_lbl, lbl],
                    range, report=0, **kws)

            if report:
                tbl.append(((pri_lbl, lbl),
                            ' --> '.join(map(str, k_vals[indices])),
                            numeric_repr(χ2r[indices].ptp())))

            # check if the optimization converged
            if r is not None:
                # messages += ('\n' + msg)
                # info.append()
                for l in (lbl, pri_lbl):
                    name = self.models[l].name
                    results[name] = r[name]

            else:
                self.logger.warning(
                        'Knot optimization failed for labels: %s', lbl)

        # report
        changed = (knots != self.knots)
        if report:
            title = f'Knot optimizer: {info}'
            if changed:
                from motley.table import Table
                tbl = Table(tbl,
                            title=title,
                            col_headers=['labels', 'k', 'χ²ᵣ'], )
                self.logger.info('\n' + str(tbl))
            else:
                self.logger.info(title + '; unchanged')

        # # fit segments that are not dependent on primary
        # other_labels = np.setdiff1d(np.arange(1, self.nmodels + 1),
        #                             np.r_[1, labels_neighbours])
        # other_results = self.fit(image, labels=other_labels, **kws)
        # for name, r in other_results.to_dict().items():
        #     results[name] = r
        # return results

    def _optimize_knots(self, data, labels, range_=(-3, 4), report=0,
                        **kws):
        # brute force optimization for knot position (boundary) between two
        # adjacent 2d polynomials in the spline
        # note this method overwrites the segmentation image!

        # TODO: look at scipy.optimize.brute ???

        assert len(labels) == 2, 'Invalid pair of labels'
        # todo: additionally, you may want to check that they are adjacent

        knots = self.knots
        # get polynomial index position in list of models
        pix = np.take(self.iorder, np.subtract(labels, 1), 0)
        i = np.where(np.all(pix[0] == pix, 0))[0]
        io = int(not bool(i))
        j = pix[:, io].max()

        k_start = knots[io][j]
        k_vals = k_start + np.arange(*range_)
        k_vals = k_vals[(0 < k_vals) & (k_vals < data.shape[io])]
        if len(k_vals) == 0:
            raise ValueError('Knot optimization failed due to knots being '
                             'outside of image. Whoops!')

        χ2r = np.ma.empty(np.ptp(range_))
        χ2r.mask = True
        results = {}
        for l, k in enumerate(k_vals):
            knots[io][j] = k
            self.set_knots(knots)  # update segmentation
            try:
                # optimize
                r = self.fit(data, labels=labels, full_output=True, **kws)
            except UnconvergedOptimization:
                pass
                # self.logger.exception('UnconvergedOptimization in '
                #                       '`_optimize_knots` for labels %s',
                #                       labels)
            # except Exception:
            #     from IPython import embed
            #     embed()
            else:
                χ2r[l] = self.redchi(r, data)
                results[k] = r

        i_start = np.where(k_vals == k_start)[0][0]
        i_best = χ2r.argmin()
        k_best = k_vals[i_best]
        r_best = results[k_best]
        knots[io][j] = k_best
        if r_best is not None:
            self.set_knots(knots)  # update segmentation

        unchanged = (k_start == k_best)
        if report is None:
            report = not unchanged
            # if self.logger.getEffectiveLevel() <= logging.INFO:

        # report if changed
        if report:
            msg = f'labels={labels}; '
            if unchanged:
                msg += 'Knot unchanged'
            else:
                from recipes import pprint

                Δχ = χ2r[i_best] - χ2r[k_vals == k_start]
                msg += (f'k: {k_start} --> {k_best}; '
                        f'Δχ²ᵣ = {pprint.numeric(Δχ)}')

                if report > 1:
                    from motley.table import Table
                    title = 'Knot optimizer results: ' + msg

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
                                title=title, title_align='l',
                                width=range(120))
                    tbl.colourise(states, 'g')
                    msg = '\n' + str(tbl)

            #
            self.logger.info(msg)

        return r_best, k_vals, χ2r, [i_start, i_best]

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

    def get_domain_ranges(self, unitary=True):
        # get domain ranges for polys
        # note the domain ranges are a bit weird atm in that the polynomials
        #  that share a LOWER y boundary need to have the same grid value (0) on
        #  the boundary.
        #  fixme. probably fix this so that the entire spline
        #   has a uniform grid. this will also help with non-pixel grids...
        #   this will require the apply_bc methods to be updated. changing
        #   this to arbitrary grid values is probably not advisable since it
        #   will adversely affect convergence properties
        domains = np.zeros((2, 2) + self.n_polys)  #
        # dimensions are lower/upper, y/x, polys

        if unitary:
            domains[1] = 1
        else:
            # here we set the domains to have the same physical scale.  This is
            # needed to have the smoothness condition be physically meaningful
            # in the image space.
            for i in range(2):
                d = np.diff(self.knots[i])
                sizes = np.atleast_2d(d / d[self.primary[i]])
                if i == 0:
                    sizes = sizes.T
                # set upper interval limit
                domains[1][i] = sizes

            #
        prix = np.array(self.primary, ndmin=3).T
        less = self._ix < prix
        # grtr = self._ix > prix
        # interval_10 = np.c_[1, 0].T

        # invert domains below primary (in y direction)
        domains[:, 0, less[0]] = domains[:, 0, less[0]][::-1]
        # domains[:, 1, grtr[1]] = domains[:, 1, grtr[1]][::-1]

        # reorder for labels
        yro, xro = tuple(np.transpose(self.iorder))
        domains = domains[:, :, yro, xro].T

        return domains

    def set_domains(self, domains):
        self.segm.domains = domains
        del self.segm.coord_grids  # delete lazy property
        for label, poly in enumerate(self.models.values(), 1):
            poly.set_grid(self.segm.coord_grids[label])

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
            poly.free_diagonal()

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

    def set_knots(self, knots, preserve_labels=True):
        """

        Parameters
        ----------
        knots: 2-tuple
            y, x positions of knots
        preserve_labels: bool
            preserve any other segments "on top of" background
        Returns
        -------

        """
        assert len(knots) == 2
        # check that the given knot positions are ok
        checker = (self._check_orders_knots(*ok)
                   for ok in zip(self.orders, knots))
        _, self.knots = itt.starmap(yxTuple, zip(*checker))

        # get the updated background segmentation
        seg = self.get_segmented_image()

        if preserve_labels:
            # preserve any other segments "on top of" background
            current = self.segm.labels
            keep = current[current > len(self.models)]
            seg_obj = self.segm.keep(keep)
            mask = (seg_obj != 0)
            seg[mask] = seg_obj[mask]

        # update SegmentationImage
        self.segm.data = seg

        # calculate new domains
        # domains = self.get_domain_ranges()
        # # update grid for each segment
        # self.set_domains(domains)  # note only needed for non-unitary domains

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

    def plot_fit_results(self, image, result, show_knots=True, seg=None):

        from obstools.modelling.image.diagnostics import plot_modelled_image

        fig = plot_modelled_image(self, image, result, seg)

        if result is not None and show_knots:
            ishape = self.segm.shape
            for ax in fig.axes[2:3]:
                ax.vlines(self.knots.x[1:], 0, ishape[0], 'k')
                ax.hlines(self.knots.y[1:], 0, ishape[1], 'k')
                ax.set(xlim=(-0.5, ishape[1] - 0.5),
                       ylim=(-0.5, ishape[0] - 0.5))
        return fig


# def AttrGetter():
#     def attr_getter(self, *attrs):
#         """Fetch attributes from polys return as object array"""
#         return operator.attrgetter(*attrs)(self)


# TODO: ListOf constructor. check consistent type. optional expose
#  certain attributes to vectorized getter


def column_vector(v):
    v = np.asarray(v)
    if v.ndim == 1:
        v = v[None].T
    return v


class ScaledTranslation(object):
    def __init__(self, o, s):
        self.o = column_vector(o)
        self.s = column_vector(s)

    def __str__(self):  # TODO
        return f'{self.__class__.__name__}\ns={self.s!s}\no={self.o!s}'

    def __call__(self, yx):
        return yx * self.s + self.o

    def inverse(self, yx):
        return (yx - self.o) / self.s


from obstools.modelling.core import CompoundModel, StaticGridMixin


class Spline2D_v2(CompoundModel):
    """Non-uniform 2d polynomial spline for image / surface modelling"""

    poly_name_base = 'p'

    def __init__(self, orders, knots, smooth=True, continuous=True,
                 primary=None):

        # TODO smoothness / continuity at each knot ?

        # checks
        assert len(orders) == len(knots) == 2
        checker = itt.starmap(_check_orders_knots, zip(orders, knots))

        # wrap namedtuple for easy attribute access
        self.orders, self.knots = itt.starmap(yxTuple, zip(*checker))
        self.n_polys = n_polys = tuple(map(len, self.orders))
        self.n_knots = np.subtract(n_polys, 1)
        self._ix = np.indices(self.n_polys)  # index grid for polys

        # get polynomial sequence order (iteration starts from primary poly)
        if primary is None:
            # The segment with the largest area will be the primary one
            sizes = list(itt.product(*map(np.diff, self.knots)))
            sizes = np.reshape(sizes, self.n_polys + (2,))
            primary = divmod(sizes.prod(-1).argmax(), self.n_polys[1])
        else:
            if len(primary) == len(knots):
                raise ValueError('Index of `primary` should have same number of'
                                 'dimensions as `knots`: %i' % self.n_knots)
        self.primary = np.asarray(primary)
        self._itr_order = self.get_iter_order()
        domains = self.get_domains()
        # domain transform params
        origins, scale = self.get_domain_transform_params()


        # create polynomials
        polys = np.empty(self.n_polys, 'O')
        for i, j in self._ix.reshape((2, -1)).T:
            # create 2d polynomial
            o = self.multi_order(i, j)  # multi-order 2-tuple
            poly = polys[i, j] = PPoly2D_v2(o, smooth, continuous)
            # ensure unique names
            poly.name = f'{self.poly_name_base}{i}{j}'  # p₀₁ ?
            poly.domain = domains[i, j]
            poly.domain_transform = ScaledTranslation(origins[i, j], scale)
            # poly.origin = origins[i, j, None].T
            # poly.scale = scale[None].T

        # get dependent polys
        used = np.zeros(self.n_polys, bool)
        # self.dependant = {}
        for ij in zip(*self._itr_order):
            b = self._get_neighbours_bool(*ij)
            unused_neigh = b & ~used
            children = list(polys[unused_neigh])
            # positions of neighbours relative to current
            neigh_pos = np.subtract(np.where(unused_neigh), np.atleast_2d(ij).T)
            named_pos = map(SEMANTIC_IDX2POS.get, map(tuple, neigh_pos.T))
            polys[ij].set_neighbours(**dict(zip(named_pos, children)))

            # self.dependant[ij] = list(self.polys[b & ~used])

            used |= b
            used[ij] = True
            if used.all():
                break

        # init parent
        CompoundModel.__init__(self, polys[self._itr_order])

    def __str__(self):
        # multi-line string repr
        name = self.__class__.__name__
        ws = ' ' * len(name)
        args = 'orders', 'knots'
        w0 = max(map(len, args))
        oy, ox = map(str, self.orders)
        ky, kx = map(str, self.knots)
        wy = max(map(len, (oy, ky)))
        wx = max(map(len, (ox, kx)))
        s = ''
        for i, (pre, arg, y, x) in enumerate(
                zip((name, ws), args, *zip(self.orders, self.knots))):
            s += f'{pre}({arg:<{w0}}=(y={y!s:<{wy}}, x={x!s:<{wx}})'
            s += ',\n' * int(not i)
        return s

    def __call__(self, p, grid):

        # dof consistency check etc
        p = self._check_params(p)

        # initialize output array
        out = np.zeros(np.shape(grid)[1:])

        # compute and fill in regions for each poly
        self.eval(p, grid, out)
        return out

    def eval(self, p, grid, out):
        # compute and fill in regions for each poly
        for mdl, pp in zip(self.models, self.split_params(p)):
            if np.isnan(pp).any():
                continue  # nans are ignored. perhaps warn ??

            # fill the modelled segment in the output array
            mdl(pp, grid, out)

    def _check_params(self, p):
        #
        # interpret parameters
        p = np.asanyarray(p)

        # convert rec array to Parameters
        if isinstance(p, np.void):
            p = Parameters(p)

        if p.ndim not in (0, 1):
            raise ValueError('Parameter vector has invalid dimensionality of '
                             '%i' % p.ndim)

        # if p.__class__.__name__ == 'Parameters':  # HACK auto-reload
        if isinstance(p, Parameters):
            n_par = p.npar
        else:
            n_par = len(p)

        if n_par != self.dof:
            raise ValueError('Parameter vector size (%i) does not match '
                             'degrees of freedom (%i) for model %r' %
                             (n_par, self.dof, self))
        return p

    def update_coeff(self, plist):
        # update tied coefficients for `poly` using current variable
        # coefficient values in `poly.coeff`. walk neighbours and do the same
        for poly, p in zip(self.models, plist):
            poly.set_coeff(p)

    def multi_order(self, i, j):
        return self.orders.y[i], self.orders.x[j]  # 2d multi-order

    def _get_neighbours_dist(self, i, j):
        # taxicab/rectilinear/L1/city block/Manhattan distance to neighbours
        return np.abs(self._ix - np.array((i, j), ndmin=3).T).sum(0)

    def _get_neighbours_bool(self, i, j):
        return self._get_neighbours_dist(i, j) == 1

    def get_iter_order(self):
        d = self._get_neighbours_dist(*self.primary)
        return np.divmod(d.ravel().argsort(), len(d))

    def get_grid_order(self):
        gp = np.zeros(self.n_polys, int)
        gp[self._itr_order] = np.arange(np.product(self.n_polys))
        return gp

    def get_domains(self):
        # the yx domains (lower upper) as implied by the knots
        domains = itt.product(*map(mit.pairwise, self.knots))
        return np.reshape(list(domains), self.n_polys + (2, 2))

    def get_slices(self):
        """
        Get an array of `slice` objects for the current knots

        Returns
        -------

        """
        domains = self.get_domains()
        return np.vectorize(slice)(*domains.round().astype(int).T).T

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

        return np.split(p, np.cumsum(self.dofs))

    def get_domain_transform_params(self):
        # get domain ranges for polys
        # For the spline to work with boundary constraints, we need each
        # polynomial to live in it's own independent domain intrinsically.
        # It is otherwise mathematically impossible to have the unique
        # coefficients for neighbouring polynomials in the same domain.
        # The domains are chosen in such a way that the yx value for
        # domain of neighbouring polynomials on the boundary
        # with the primary is always 0.

        # Additionally we need to ensure the domains to have the same physical
        # scale.  This is needed to have the smoothness condition be
        # physically meaningful in the image space.  Here we simply scale by
        # the image dimensions as (implied by the knots).  This means that
        # the internal domain of each polynomial will always be a sub-interval
        # of [0, 1].  This aids numerical stability when fitting since powers
        # of numbers greater than 1 tend to explode.

        dom = self.get_domains()
        less = np.moveaxis(self._ix < np.array(self.primary, ndmin=3).T, 0, -1)
        # invert the domains for polys that are below or left of primary
        dom[less] = dom[less][:, ::-1]
        # zeros points for each poly in global (image) coordinates
        origins = dom[..., 0]
        scale = dom[-1, -1, :, 1]  # *  inv

        # return params as though for inverse transform
        return -origins / scale, 1. / scale

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
            poly.free_diagonal()

        for poly in models:
            poly.set_freedoms()

    # --------------------------------------------------------------------------

    def display(self, p, grid, **kws):
        from graphical.imagine import ImageDisplay
        im = ImageDisplay(self(p, grid), **kws)
        for hv, k in zip('hv', self.knots):
            for kk in k[1:-1]:
                getattr(im.ax, f'ax{hv}line')(kk, color='orange')
        return im


class Spline2DImage(Spline2D_v2, StaticGridMixin):

    def __init__(self, orders, knots, smooth=True, continuous=True,
                 primary=None):
        super().__init__(orders, knots, smooth, continuous, primary)

        # create grid based on extremal knots
        grid = np.indices((self.knots.y[-1], self.knots.x[-1]))
        for poly in self.models:
            # set static grid and static domain mask
            poly.set_grid(grid)

    def __call__(self, p, grid=None):

        # dof consistency check etc
        p = self._check_params(p)

        # initialize output array
        if grid is None:
            out_shape = (self.knots.y[-1], self.knots.x[-1])
        else:
            out_shape = np.shape(grid)[1:]
        out = np.zeros(out_shape)
        # if grid not passed, it will still be `None` here which gets passed
        # through and we let component models set grids if needed

        # compute and fill in regions for each poly
        self.eval(p, grid, out)
        return out

    def get_shape(self):
        return self.knots.y[-1], self.knots.x[-1]

    def get_segmented_image(self):
        """Create segmented image data from knots"""
        seg = np.zeros(self.get_shape(), int)
        slices = self.get_slices()[self._itr_order]
        for k, s in enumerate(slices):
            seg[tuple(s)] = k + 1
        return seg

    def display(self, p, grid=None, **kws):
        return super().display(p, grid, **kws)

    def display_residuals(self, image, result, show_knots=True,
                          knots_line_colour='orange', seg=None):

        from obstools.modelling.image.diagnostics import plot_modelled_image

        fig = plot_modelled_image(self, image, result, seg)

        if result is not None and show_knots:
            shape = self.knots.y[-1], self.knots.x[-1]
            for ax in fig.axes[2:3]:
                ax.vlines(self.knots.x[1:], 0, shape[0], knots_line_colour)
                ax.hlines(self.knots.y[1:], 0, shape[1], knots_line_colour)
                ax.set(xlim=(-0.5, shape[1] - 0.5),
                       ylim=(-0.5, shape[0] - 0.5))
        return fig

# class Poly2DHessian(HessianUpdateStrategy):
#     def initialize(self, n, approx_type):
#         self.h = np.empty((n, n))
#         self.approx_type = approx_type
#
#     def get_matrix(self):
#         return self.h
#
#     def dot(self, p):
#         return self.h.dot(p)
#
#     def update(self, delta_x, delta_grad):
#         pass
