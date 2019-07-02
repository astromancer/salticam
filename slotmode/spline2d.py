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
from salticam.slotmode.ppoly import PPoly2D
from obstools.modelling.parameters import Parameters
from obstools.modelling.image import SegmentedImageModel
from obstools.modelling.core import UnconvergedOptimization, nd_sampler
from recipes import pprint

# simple container for 2-component objects
yxTuple = namedtuple('yxTuple', ['y', 'x'])


def _yx_str(obj):
    return f'(y={obj.y!s}, x={obj.x!s})'


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
        # TODO: with an actual grid ?? OR better yet, with a resolution which
        #  automatically makes a unitary grid

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
        return f'{self.__class__.__name__}' \
            f'(orders={_yx_str(self.orders)}, ' \
            f'knots={_yx_str(self.knots)})'

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
        k_vals = k_vals[(0 < k_vals) | (k_vals < data.shape[io])]

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
            keep = current[current > self.nmodels]
            seg_obj = self.segm.select(keep)
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
