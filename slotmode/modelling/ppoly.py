"""
Fittable piecewise polynomials (a.k.a. non-uniform splines) in 1D and 2D
"""

# std libs
import numbers
import operator
import textwrap
import warnings
import functools
import itertools as itt
from collections import OrderedDict

# third-party libs
import more_itertools as mit
import numpy as np
from numpy.polynomial.polynomial import polyval2d
from scipy.interpolate import PPoly
from scipy.special import factorial
from scipy.optimize import minimize
# from scipy.special import binom
from scipy.linalg import toeplitz  # , circulant

# local libs
from recipes.containers.sets import OrderedSet
from recipes.array.fold import fold
from recipes.language.unicode import (SUP_NRS as UNICODE_SUPER,
                                      SUB_NRS as UNICODE_SUBS
                                      )
from recipes import pprint
from obstools.modelling.core import RescaleInternal
from obstools.modelling.parameters import Parameters
from obstools.modelling import FixedGrid, Model, CompoundModel


def _echo(_):
    return _


def prod(x):
    """Product of a list of numbers; ~40x faster vs np.prod for Python tuples"""
    if len(x) == 0:
        return 1
    return functools.reduce(operator.mul, x)


def falling_factorial(n, k):
    n = np.asarray(n)
    k = np.asarray(k)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        return np.where(n < k, np.zeros_like(n * k),
                        factorial(n) / factorial(n - k))


def poly_derivative_matrix(x, o, nd, increasing=True):
    """
    Get coefficients of the linear system in polynomial parameters for the
    sequence of derivatives of the polynomial of degree `o`, up to derivative
    order `nd` at the point `x`. This system can be solved to reduce the
    dimensionality of the piecewise  polynomial model given the boundary
    conditions at the breakpoint `x`.

    For example:
    For 3rd order (o = 3) polynomial
        p₁(x) = a + bx + cx² + dx³,
    with nd = 3 (continuous, smooth, and curvature matched at boundaries),
    the left-hand side coefficients of the system of linear equations:
        p₁(x)   = p₂(0)
        p₁`(x)  = p₂`(0)
        p₁``(x) = p₂`(0)
    can be written as `P · θ`, where θ = [a, b, c, d], a column vector, and P:
        [[ 1   1 x  1 x²   1 x³  ]                [[ 1 x³  1 x²  1 x  1 ]
         [ 0   1    2 x    3 x²  ]                 [ 3 x²  2 x   1    0 ]
         [ 0   0    2      6 x  ]]                 [ 6 x   2     0    0 ]]
    if increasing is True                     if increasing is False


    Parameters
    ----------
    x : float
        break point
    o : int
        degree (order) of the polynomial
    nd : int
        number of boundary conditions. highest order derivative to set
        equal at boundary. 0 implies continuity, 1 smoothness, 2 matched
        curvature etc..
    increasing : bool
        whether the coefficient matrix should be in increasing in powers of x

    Examples
    --------

    Returns
    -------
    array

    """

    P = np.zeros((nd, o))
    pwrx = np.vander([x, ], N=o, increasing=True)
    # increasing always True, otherwise can't use toeplitz below
    P[:o] = toeplitz(pwrx)[:nd]
    f = falling_factorial(np.arange(o),
                          np.arange(nd)[None].T)
    step = [-1, 1][increasing]
    return (f * P)[:, ::step]


def boundary_constraints_system(x, ncf, nbc, increasing=False,
                                one_domain=False):
    """
    Construct linear system of boundary constraints between polynomials
    pₙ and pₙ₊₁

    Parameters
    ----------
    x:
        coordinate value at break point
    ncf: array-like size 2
        number of coefficients (order + 1) for polynomials
    nbc: int
        highest order derivative that will be equated at boundary
    increasing: bool
        Is polynomial defined with coefficients increasing in order
    one_domain: bool


    Returns
    -------

    """
    assert len(ncf) == 2  # TODO assert len(x) == 2 if not one_domain etc

    # Construct the linear system
    cso = np.cumsum(np.r_[0, ncf])
    # indices = list(mit.pairwise(cso))
    m = sum(ncf)
    A = np.zeros((nbc, m))

    # lhs
    j, o = cso[0], ncf[0]
    A[:, j:j + o] = poly_derivative_matrix(x, o, nbc, increasing)

    # rhs
    if not one_domain:
        x = 0
    j, o = cso[1], ncf[1]
    A[:, j:j + o] = -poly_derivative_matrix(x, o, nbc, increasing)
    # note: x coordinate for each poly relative to break point.

    return A


# alias
bcs = boundary_constraints_system


def get_solvable_indices(A, orders, keep_free=(), solve_high_order=True,
                         return_powers=False):
    # Get indices of parameters to be solved for algebraically by reducing
    # the linear system of parameter constraints

    # number of equations in the linear system (== number of constraints)
    n_eq = len(A)

    if n_eq == 0:
        # all polys are completely free
        if return_powers:
            return np.array([]), np.array([])
        return np.array([])

    # if solve_high_order = False
    # ==> more intuitive approach where we solve lower order
    # coefficients algebraically using the boundary conditions. This method
    # is preferred  when extending to higher dimensions.

    # if solve_high_order = True
    # prefer to fit coefficients corresponding to smaller power in x
    # (empirically found to be more numerically stable for optimization)

    # note:
    # the block below sometimes cannot not eliminate higher powers of x
    # since it leads to a singular linear system (this is largely due to the
    # independent domains of each polynomial (starting from 0) and therefore
    # the large number of 0s in the constraining system. one way past this
    # would be to formulate to transform to a different domain (1,2) maybe,
    # solve the system, and then transform back via binomial equation...
    # This will mean changing coordinates after applying the constraints in
    # the call method

    n_coeffs = np.add(orders, 1)  # OR PASS ncoeff
    cso = np.cumsum(np.r_[0, n_coeffs])
    ix_ranges = fold(cso, 2, 1)

    if solve_high_order:
        # get indices of highest power first
        so = orders.argsort()[::-1]  # sorted orders
        rng = ix_ranges[so]
        o = 1  # this assumes *decreasing* power sequence
    else:
        # order of polys doesn't matter since we are enumerating from lower
        # powers
        rng = ix_ranges
        o = -1  # this assumes *decreasing* power sequence

    seq = mit.roundrobin(*(range(*r)[::o] for r in rng))
    # we now have the sequence of indices in order of (in/de)creasing powers
    if len(keep_free):
        # indices to be solved for can be specified explicitly using this
        seq = OrderedSet(seq) - set(keep_free)

    # iterate through combinations of coefficients and check if we can solve
    # for these parameters based on the boundary constraints
    seq = list(seq)
    for i, ix in enumerate(itt.combinations(seq, n_eq)):
        det = np.linalg.det(A[:, ix])
        if det != 0:  # can solve for these parameters
            ix = np.array(ix)
            break
    else:
        raise ValueError("No combination of %i parameters could be "
                         "eliminated. That's pretty weird...")

    if return_powers:
        pwr = np.r_[tuple(map(np.arange, n_coeffs))][list(ix)]
        return ix, pwr
    return ix


def _reduce_constraints(A, ix_solve, ix_const=(), const_vals=()):
    # reduce the system of constraints in `A` so we can solve for the
    # parameters in `ix_solve`

    nconstraints, ncoeff = A.shape

    s = np.linalg.inv(A[:, ix_solve])
    constraints = -s.dot(A)
    # columns in `constraints` at indices `ix_solve` will now have 1, and all
    #  zeros further.

    p_off = 0
    if len(ix_const):
        # apply constant parameter values
        p_off = constraints[:, ix_const].dot(const_vals)

    ix_have = np.hstack([ix_solve, ix_const]).astype(int)
    ix_free = np.setdiff1d(np.arange(ncoeff), ix_have)

    return constraints[:, ix_free], p_off, ix_free, ix_have
    # To obtain the values of the constrained parameters, simply take the
    # dot product of the matrix returned here with the vector of free
    # parameters


def transform_poly(coeff, scale, offset, increasing=False):
    """
    Compute coefficients for new polynomial with independent variable related
    to the old one by a simple linear transformation.
        `x = α u + β`

              ₙ                   ₙ
    If p(x) = ∑ aᵢ·xⁿ⁻ⁱ = p`(u) = ∑ bᵢ·uⁿ⁻ⁱ
             ᵢ₌₀                 ᵢ₌₀

               ₙ
    Then: bᵢ = ∑ ₙ₋ᵢCₖ·aₖ·αⁿ⁻ⁱ·βⁱ   with    ₙCₖ the binomial coefficient
              ₖ₌₀

    Parameters
    ----------
    coeff
    offset
    increasing

    Examples
    --------
    >>> coeff = np.arange(5)
    >>> x = np.array([0, 1, 2])
    >>> scale, offset = 5, 3
    >>> a = np.polyval(coeff, x)
    >>> b = np.polyval(transform_poly(coeff, scale, offset), (x-offset) / scale)
    >>> assert np.allclose(a, b)


    Returns
    -------

    """
    from scipy.special import binom
    from scipy.linalg import circulant

    step = [-1, 1][increasing]
    rng = np.arange(len(coeff))
    n = rng[::step, None]

    opwrs = np.triu(circulant(rng).T)  # powers of the offset term
    apwrs = np.tril(rng)[::-1, ::-1]  # powers of the scaling term
    b = binom(n.T, n).T  # binomial coefficients for the sequence of powers
    B = b * np.power(scale, apwrs) * np.power(offset, opwrs)
    return np.sum(B * np.array(coeff, ndmin=2).T, 0)  # new coefficients for u


# TODO move pprint functions to new module??
#       also maybe ppoly2d ??

def make_pnames(orders, alpha='a', increasing=False, latex=False, unicode=False,
                nested=False):
    """
    Get a list of parameter names eg: [a0, a1, a2, a3, b0, b1, c0, d0, ...]
    Parameter names will be enumerated as 'a0-an' for the first polynomial,
    'b0-bm' for the second etc.


    Parameters
    ----------

    orders: array-like
        orders of polynomials
    alpha
    increasing
    latex
    unicode
    nested

    Returns
    -------
    list of strings
    """

    # check flags
    assert not (unicode and latex)  # can't do both!

    ch = ord(alpha)
    names = []
    if latex:
        formatter = '${}_{{{}}}$'.format
    elif unicode:
        formatter = unicode_subscript
    else:
        formatter = '{}{}'.format

    if nested:
        aggregate = names.append
    else:
        aggregate = names.extend

    step = [-1, 1][increasing]
    for i, o in enumerate(orders):
        char = chr(ch + i)
        work = map(formatter, itt.repeat(char), range(o + 1)[::step])
        aggregate(list(work))
    return names


def make_pnames1(alpha, *nrs, latex=False, unicode=False,
                 nested=False):
    """
    Get a list of parameter names eg: [a0, a1, a2, a3, b0, b1, c0, d0, ...]
    Parameter names will be enumerated as 'a0-an' for the first polynomial,
    'b0-bm' for the second etc.


    Parameters
    ----------

    orders: array-like
        orders of polynomials
    alpha
    increasing
    latex
    unicode
    nested

    Returns
    -------
    list of strings
    """

    # check flags
    assert not (unicode and latex)  # can't do both!

    ch = ord(alpha)
    names = []
    if latex:
        formatter = '${}_{{{}}}$'.format
    elif unicode:
        formatter = unicode_subscript
    else:
        formatter = '{}{}'.format

    if nested:
        aggregate = names.append
    else:
        aggregate = names.extend

    #
    if isinstance(alpha, str):
        alpha = alpha,
    #
    itr = itt.combinations_with_replacement(alpha, *nrs)

    for char, *ijk in itr:
        formatter(char, )

    return names


def make_pnames_comb(alpha, *nrs, latex=False, unicode=False, flat=False):
    """
    Get a list of parameter names eg: [a0, a1, a2, a3]
    Parameter names will be enumerated as 'a0-an'


    Parameters
    ----------
    alpha
    nrs
    latex
    unicode
    nested


    Returns
    -------
    list of strings
    """

    # check flags
    assert not (unicode and latex)  # can't do both!

    if latex:
        formatter = latex_subscript
    elif unicode:
        formatter = unicode_subscript
    else:
        formatter = ascii_postscript

    #
    if isinstance(alpha, str):
        alpha = alpha,

    #
    itr = itt.product(alpha, *nrs)
    work = itt.starmap(formatter, itr)

    size = len(alpha) + len(nrs) + 5 * latex
    dtype = 'U%s' % size
    names = np.fromiter(work, dtype)
    if flat:
        return names

    shape = np.array([len(alpha)] + list(map(len, nrs)))
    return names.reshape(shape[shape != 1])


def latex_subscript(alpha, *nrs):
    subs = ''.join(map(str, nrs))
    return '$%s_{%s}$' % (alpha, subs)


def unicode_subscript(alpha, *nrs):
    subs = (UNICODE_SUBS[nr] for nr in nrs)
    return alpha + ''.join(subs)


def ascii_postscript(alpha, *nrs):
    return alpha + ''.join(map(str, nrs))


def vector_powers(symbol, n, column=False, squash='01', unicode=None,
                  latex=None):
    """
    Get the sequence:
        ['1', 'y', 'y²', 'y³' ... 'yⁿ']

    Parameters
    ----------
    symbol
    n
    column
    squash

    Returns
    -------

    """

    if (unicode, latex) == (None, None):
        unicode = True  # default is to prefer unicode

    if unicode:
        latex = False
        powers = ''.join(UNICODE_SUPER)
        _exponent = _echo
    else:
        latex = True
        powers = '0123456789'
        _exponent = '^{{{}}}'.format

    # nrs = '0123456789' #''.join(map(chr, range(48, 58)))
    trans = str.maketrans('0123456789', powers, str(squash))

    squash0 = ('0' in squash)
    symbols = itt.chain(iter('1' * squash0),
                        itt.repeat(symbol, n - squash0))

    # pad out to width # todo make optional
    w = len(symbol) + len(str(n)) + 3 * latex  # extra '^{}' characters
    terms = np.fromiter((f'{s}{_exponent(str(i).translate(trans)): <{w}}'
                         for i, s in enumerate(symbols)),
                        dtype=f'U{w}')

    #      make 2d
    return terms[None].T if column else terms


def lincombfmt(coeff, variables, **kws):
    # todo remove

    # number of terms
    n = len(coeff)
    if isinstance(variables, str):
        variables = [variables] * n
    variables = np.array(variables)

    # handle coefficient formatting
    coeff = np.asarray(coeff)
    if np.issubdtype(coeff.dtype, np.number):
        # if we have numeric coefficients
        show = (coeff != 0)
        coeff = coeff[show]
        ones = (coeff == 1)
        # get signs of coefficients
        signs = np.take(['', '+', '-'], np.sign(coeff).astype(int))
        signs = np.char.add(signs, ' ')  # add space for aesthetic
        # convert to nice str format
        coeff = pprint.numeric_array(np.abs(coeff), **kws)
        coeff = np.char.strip(coeff, '$')
        coeff[ones] = ''  # don't need to display ones (implicit) "1x" --> "x"

    else:
        show = (np.char.str_len(coeff) != 0)
        coeff = coeff[show]
        signs = np.array(['+ '] * len(coeff))

    coeff = np.char.add(signs, coeff)
    terms = np.char.add(coeff, variables[show])

    equation = ' '.join(terms)
    return equation.strip('+ ')


# todo: similar for 2D poly
def get_poly_equation(coeff, variable='x', increasing=False,
                      precision=3, significant=3, log_switch=None,
                      compact=True, times='x', unicode=None,
                      latex=None):
    """
    Produce a latex string representing a polynomial

    Parameters
    ----------
    coeff : array-like
        polynomial coefficients
    variable: str
        variable name
    increasing : bool, default False
        If False, (the default) the order of the coefficients are in decreasing
        powers of x
    precision : int
        numerical precision for coefficients
    compact : bool
        whether to represent floats as the shortest possible string (without
        information loss) for given precision.
    log_switch : float
        order of magnitude at which to switch to exponential formatting for
        coefficients
    times: str
        symbol to use for multiplication in scientific representation of numbers

    Returns
    -------
    str

    Examples
    --------
    >>> get_poly_equation([-2, -1, 2, 2.104302], compact=False)
   '- 2.00x^{3} - 1.00x^{2} + 2.00x + 2.10'
    """

    return ' '.join(get_poly_terms(coeff, variable, increasing, precision,
                                   significant, log_switch, compact, times,
                                   unicode, latex))


def get_poly_terms(c, variable='x', increasing=False,
                   precision=3, significant=3, times='x', compact=True,
                   log_switch=None, unicode=None, latex=None):
    if (unicode, latex) == (None, None):
        unicode = True  # default is to prefer unicode
        latex = False

    # number of terms
    n = len(c)
    step = [-1, 1][increasing]

    # get xⁿ terms
    x = vector_powers(variable, n, latex=latex)
    x[0] = ''  # replace "1" or "1^{}" factor with ""
    x = x[::step]

    # get number terms
    zeros = (c == 0)
    ones = (c == 1)
    # get signs of coefficients
    signs = np.take(['', '+', '-'], np.sign(c).astype(int))
    signs = np.char.add(signs, ' ')  # add space for aesthetic
    # convert to nice str format
    c = pprint.numeric_array(np.abs(c), precision=precision,
                             significant=significant, sign='',
                             compact=compact,
                             log_switch=log_switch, times=times,
                             latex=latex, unicode=unicode)
    c = np.char.strip(c, '$')
    c[ones] = ''  # don't need to display ones (implicit) "1x" --> "x"

    terms = np.array(list(map(''.join, zip(signs, c, x))))
    terms[zeros] = ''
    return terms


def get_latex_equation(coeff, breakpoints,
                       name='f', variable='x',
                       increasing=False, onedomain=True, compact=False,
                       # number formatting options follow
                       precision=3, significant=3, times='x', terse=True,
                       log_switch=None):
    """
    Produce a latex equation representing a polynomial

    Parameters
    ----------
    coeff : array-like (N, M)
        block of coefficients for N equations of degree M
    name: str
        name of the function
    variable: str
        variable name
    increasing : bool
        order of the powers
    precision : int
        numerical precision for coefficients
    compact : bool
        whether to represent floats as shortest possible string for given
        precision

    Returns
    -------
    str
    """

    # assert environ in ('cases', 'array')

    nr_format = dict(precision=precision, significant=significant,
                     log_switch=log_switch, compact=terse,
                     times=times)

    n, m = coeff.shape
    intervals = fold(breakpoints, 2, 1, pad=False)
    if onedomain:
        # each variable has its own domain starting from (0, ...)
        # this is the default coordinate system for coeff of
        # `scipy.interpolation.PPoly`
        new_coeff = np.empty_like(coeff)
        for i, c in enumerate(coeff):
            new_coeff[i] = transform_poly(c, 1, breakpoints[i])
        coeff = new_coeff
        variables = [variable] * n
        vname = variable
    else:
        # give each indep. variable a subscript
        variables = [f'{variable}_{{{i}}}' for i in range(n)]
        vname = ', '.join(variables)
        intervals = intervals - intervals[:, [0]]

    # create piecewise latex equation using "array" environment
    # add newlines so we can copy paste to latex and still have nice formatting
    # TODO: bonus points for aligning terms in latex!!
    #  can do with motley.table
    rep = '$$\n'
    rep += rf'{name}\left({vname}\right) = \left\{{ \begin{{array}}'

    if compact:
        sep, ncols = ' ', 6
    else:
        sep, ncols = ' & ', m + 4

    rep += '{*{%d}{r}}' % ncols
    rep += '\n'
    for i, (p, v, inter) in enumerate(zip(coeff, variables, intervals)):
        terms = get_poly_terms(p, v, increasing, **nr_format)

        i0, i1 = pprint.numeric_array(inter, **nr_format, latex=False)
        rep += ' & '.join((sep.join(terms),
                           r'\mbox{{for }} ',
                           rf'{i0} {sep} \leq {v} < {sep} {i1}\\ ' + '\n'))

    rep += rf'\end{{array}}\right.'
    rep += '\n$$'
    return rep


def display_eq(eq, fontsize=20):
    from matplotlib import rc
    from matplotlib.texmanager import TexManager
    from IPython.display import Image

    rc("text.latex", preamble=r'\usepackage{amsmath}')
    #
    tm = TexManager()
    im = Image(filename=tm.make_png(eq, fontsize, 72))
    rc("text.latex", preamble='')  # restore preamble state
    return im


# TODO:
#  class Representation(): ??


def repr_matrix_product(coeff):
    # c = pprint.matrix(coeff)
    yo, xo = coeff.shape  # multi_order
    ypr, xpr = (pprint.nrs.matrix(vector_powers(*_))
                for _ in zip('yx', (yo, xo), (0, 1)))
    # ypr, xpr = map(pprint.nrs.matrix,
    #                map(vector_powers, 'yx', (yo, xo), (0, 1)))
    skip_lines = (yo // 2) - int(not (yo % 2))

    y_space = ypr.index('\n')
    c = pprint.matrix(coeff)
    c_space = max(map(len, c.splitlines()))

    ylist = [''] * skip_lines + ypr.splitlines()
    clist = c.splitlines()
    xlist = xpr.splitlines()
    lines = []
    for ys, cs, xs in itt.zip_longest(ylist, clist, xlist, fillvalue=''):
        lines.append(
                '{:{}s} {:{}s} {:s}'.format(ys, y_space, cs, c_space, xs)
        )
    return '\n'.join(lines)


def repr_sum(coeff):
    s = '''\
        %s %s
         ∑ cᵢⱼ·𝑦ⁱ𝑥ʲ
        ᵢ ⱼ\
        ''' % tuple(UNICODE_SUBS[n] for n in coeff.shape[::-1])
    # ᵢ₌₀ ⱼ₌₀
    return textwrap.dedent(s)


# TODO: move to modelling.core ?????
def get_gof(mdl, p, data, grid, stddev=None):
    rchi = mdl.redchi(p, data, grid, stddev)
    cod = mdl.coefficient_of_determination(p, data, grid)
    aic = mdl.aic(p, data, grid, stddev)
    # aicc = mdl.aicc(p, data, grid, stddev)
    return rchi, cod, aic  # , aicc


def get_gof_dict(mdl, p, data, grid, stddev=None):
    names = (r'\chi^2_{\nu}', 'R^2', 'AIC')  # , 'AIC_c')
    return dict(zip(names, get_gof(mdl, p, data, grid, stddev)))


def hyper_objective(data, grid, stddev, orders, breaks, nbc, **kws):
    mdl = PPolyModel(orders, breaks, nbc=nbc, fit_breakpoints=False, **kws)
    p = mdl.fit(data, grid, stddev)

    gof = None
    if p is not None:
        gof = get_gof(mdl, p, data, grid, stddev)  # = rchi, cod, aic, aicc

    # secondary optimization for breakpoints
    # TODO: this is not an effective optimization for breakpoints
    # bp = mdl.fit_bp_only(p, data, grid)
    # gof2 = get_gof(mdl, p, data, grid, stddev)

    # if gof1[0] > gof2[0]:
    #     mdl.breakpoints = mdl.breakpoints0

    return mdl, p, gof


def bruteforce_order_search(oranges, breaks, nbc, data, grid, stddev,
                            plot=True):
    # TODO: try basinhopping for breakpoint optimization
    # ndim = len(oranges)
    search_shape = tuple(np.ptp(o) for o in oranges)
    ix0 = next(zip(*oranges))
    mdl = best_mdl = best_p = None
    n_gof = 3
    gof_shape = (n_gof,) + search_shape
    GoF = np.full(gof_shape, np.nan)
    rchi_min = np.inf
    figures = OrderedDict()
    for orders in itt.product(*(range(*r) for r in oranges)):
        mdl, p, gof = hyper_objective(data, grid, stddev, orders,
                                      breaks, nbc,
                                      solve_high_order=False)
        if gof is not None:
            indices = (slice(None),) + tuple(np.subtract(orders, ix0))
            GoF[indices] = gof
            rchi, cod, aic = gof  # , aicc
            # print(o, gof1[0] > gof2[0], gof1[0] - gof2[0],
            #       (mdl.breakpoints0 - mdl.breakpoints)[1:-1])

            if rchi < rchi_min:
                rchi_min = rchi
                best_p = p
                best_mdl = mdl
                print('best!', mdl, rchi_min)
                if plot:
                    art = mdl.plot_fit_results(p, data, grid, stddev)
                    figures[orders] = art[1].figure

    print('Best model:', best_mdl, rchi_min)
    return best_mdl, best_p, rchi_min, GoF, figures


class PiecewisePolynomial(object):  # Spline1dBase
    # FIXME:           degrees, knots     ?
    def __init__(self, orders, breakpoints, continuous=True, smooth=True,
                 use_scale=True, coeff_name_base='a', solve_high_order=True,
                 nbc=None):
        """
        Create a piecewise polynomial from orders, break points and boundary
        conditions.

        # TODO: docstring equations: see scipy.interpolate.PPoly
        # constraints


        Parameters
        ----------
        orders: array-like
            Polynomial orders
        breakpoints : array-like
            Location of break points (boundaries between different polynomials)
        smooth: bool
            smoothness boundary conditions at each of the breakpoints.
            True ==> 1st derivative of polynomials are equal at break point
                  for all adjoining polynomials.
        continuous: bool
            continuity conditions at each of the breakpoints.
            True ==> 1st and 2nd derivative of polynomials are equal at break
                      point for all adjoining polynomials.


        Attributes
        ----------
        todo

        Methods
        -------
        todo

        See also
        --------
        scipy.interpolate.PPoly

        Notes
        -----
        Important: Even though the break points passed to this class are
        given wrt the same coordinate system (that of the first polynomial),
        each polynomial segment is internally calculated in it's own domain
        (x-coordinates) ranging from zero, to the value of the next break point.
        This is is the same as `scipy.interpolate.PPoly`

        """

        # use boundary conditions between adjacent polys to constrain parameter
        # space

        orders, breakpoints = self._check_ord_bp(orders, breakpoints)

        self.orders = np.array(orders)
        self.npoly = npoly = len(orders)
        ncoeffs = self.ncoeffs = self.orders + 1
        # number of polynomial coefficients parameters (unconstrained)
        self.ncoeff = ncoeffs.sum()

        # handle constraints
        nbp = npoly - 1  # number of boundaries between polys
        # number of boundary conditions at each breakpoint
        self.nbc = np.zeros(nbp, int)

        if nbc is None:
            for i, spec in enumerate((continuous, smooth)):
                if isinstance(spec, bool):
                    if spec:
                        items = slice(None)
                    else:
                        continue
                else:
                    items = list(spec)
                self.nbc[items] = i + 1
        else:
            # if nbc is given, ignore `continuous` and `smooth`
            if isinstance(nbc, numbers.Integral):
                self.nbc[:] = nbc
            else:
                nbc = np.array(nbc).astype(int)
                assert len(nbc) == nbp
                self.nbc = nbc

        # total number of boundary constraint conditions
        self.nconstraints = self.nbc.sum()
        # check for over constrained
        if self.nconstraints > self.ncoeff:
            raise ValueError('Over constrained')

        # block coeff for `scipy.interpolate.PPoly`
        self._coeff_shape = (ncoeffs.max(), npoly)
        self._coeff = np.zeros(self._coeff_shape)

        # cumulative sum of polynomial orders starting at 0 (used for indexing)
        self.cso = np.cumsum(np.r_[0, ncoeffs])
        self._ix_coeff_rng = fold(self.cso, 2, 1)
        self.csc = np.cumsum(np.r_[0, self.nbc])

        # for coefficient (parameter) names
        self.coeff_name_base = str(coeff_name_base)

        # optimizations more stable if independent variable is scaled to unity
        bp = np.asarray(breakpoints, float)
        self.use_scale = bool(use_scale)
        self._xscale = 1
        # scale breakpoints
        if use_scale:
            self._xscale = bp.max()
        self.breakpoints = bp / self._xscale
        self.intervals = np.array(list(mit.pairwise(self.breakpoints)))

        # apply the boundary conditions between the polys and decide which
        # coefficients to solve for, which to fit if not specified
        self.solve_high_order = bool(solve_high_order)
        self.ix_const = ()
        self.const_vals = ()
        self.p_off = 0

        # self.ix_const, = np.where(is_const)
        self._make_solving()

    @property
    def dof(self):
        # number of free parameters
        return self.ncoeff - self.nconstraints - len(self.ix_const)

    def _fix_coeff(self, indices, values):
        ix = np.array(indices)
        l = ix < 0
        if l.any():
            ix[l] += self.ncoeff

        self._make_solving(None, ix, values)

    def _make_solving(self, ix_solve=None, ix_const=(), const_vals=()):

        A = self._get_linear_system(self.breakpoints)
        self._A = A

        if ix_solve is None:
            ix_solve = get_solvable_indices(A, self.orders, ix_const,
                                            self.solve_high_order)
        else:
            ix_solve = np.array(ix_solve)
            det = np.linalg.det(A[:, ix_solve])
            if det == 0:  # non invertible sub-matrix
                raise ValueError('Singular')

        # constraints
        self._constraints, self.p_off, ix_free, ix_have = \
            _reduce_constraints(A, ix_solve, ix_const, const_vals)

        # indices wrt stacked coefficient list
        self.ix_solve = ix_solve
        self.ix_free = ix_free
        self.ix_const = ix_const
        self.const_vals = np.array(const_vals)

        # map coefficients to block matrix (for numerical convenience)
        mxo = self.ncoeffs.max()  #
        # Create map between parameter vectors for this class and the
        # block array of coefficients accepted by
        # `scipy.interpolate.PPoly` that will be used to do the computation
        coeff = np.ones((self.npoly, mxo))
        for i, o in enumerate(self.orders):
            e = max(mxo - o - 1, 0)
            coeff[i, :e] = 0
        self.coeff_map = np.array(np.where(coeff))[::-1]

        # this is an array with equation number in first row and corresponding
        # parameter number in second row. The columns therefor map to a index
        # in the coefficient block
        self._ix_free = tuple(self.coeff_map[:, ix_free])  # _cix_free ??
        self._ix_solve = tuple(self.coeff_map[:, ix_solve])
        self._ix_const = tuple(self.coeff_map[:, ix_const])

        # set constant coeffs!
        self._coeff[self._ix_const] = self.const_vals

        # the number of free parameters per poly
        self._dof_each = np.unique(self._ix_free[1], return_counts=True)[1]

    def _check_params(self, p):
        n_ok = self.dof
        if len(p) != n_ok:
            raise ValueError('%r takes %i free parameters, %i given.'
                             % (self.__class__.__name__, n_ok, len(p)))
        return p

    def __call__(self, p, grid):
        """
        Evaluate the piecewise polynomial with parameters `p` at grid
        position(s) `grid`
        """
        self._check_params(p)

        # construct scipy.interpolate.PPoly
        coeff = self.get_block_coeff(p)
        pp = PPoly(coeff, self.breakpoints, extrapolate=False)
        return pp(grid)

    def __repr__(self):
        # quick and dirty repr
        return '%s%s' % (self.__class__.__name__, tuple(self.orders))

    def residuals(self, p, data, grid):
        """inject fast evaluation for fitting"""
        return data - self.evaluate(p, grid)

    def evaluate(self, p, grid):
        return self._evaluate_fast(p, grid, self.breakpoints)

    def _evaluate_fast(self, p, grid, breakpoints, nu=0):
        """
        Faster evaluation that skips checks. ~4x faster than __call__.
        used for fitting

        Parameters
        ----------
        p:
            parameters
        grid:
            points at which to evaluate
        breakpoints:
            delineates the domains of each polynomial

        nu:
            derivative order

        Returns
        -------

        """

        coeff = self.get_block_coeff(p)
        pp = PPoly.construct_fast(coeff, breakpoints, extrapolate=False)
        out = np.empty((len(grid), prod(coeff.shape[2:])), dtype=coeff.dtype)
        pp._evaluate(grid, nu, False, out)  # fast evaluation in parent
        return out.reshape(grid.shape + coeff.shape[2:])

    def _get_linear_system(self, breakpoints=None):
        # Construct the N x M linear system imposing the coefficient constraints
        # NOTE: This is for simultaneous fitting

        # coefficients are given in decreasing powers of x
        # increasing = False
        if breakpoints is None:
            breakpoints = self.breakpoints

        # Construct the linear system that imposes constraints in N x M
        A = np.zeros((self.nconstraints, self.ncoeff))
        rows = list(map(slice, *zip(*mit.pairwise(self.csc))))
        # col_slices = list(map(slice, *zip(*mit.pairwise(self.cso))))
        δx = np.diff(breakpoints[:-1])
        for n, (i, j, x, nbc) in enumerate(zip(rows, self.cso, δx, self.nbc)):
            o = self.ncoeffs[n:n + 2]
            A[i, j:j + o.sum()] = bcs(x, o, nbc)

        return A

    def get_indices_of(self, powers):

        z = self.orders[None].T - powers
        i = np.ma.masked_where(z < 0, z) + self.cso[None, :-1].T
        if not np.ma.is_masked(i):
            return i.data

        return i.squeeze()

    def get_powers_of(self, indices=..., free=False, solve=False, const=False):
        """
        Get the power of the coefficient(s) corresponding to indices in
        the parameter vector.
        """
        if indices is not ...:
            if free:
                indices = self.ix_free
            elif solve:
                indices = self.ix_solve
            elif const:
                indices = self.ix_const

        return self.get_powers()[indices]

    def get_powers(self, nested=False, increasing=False):
        # free=True  - only include free parameters

        o = 1 if increasing else -1
        pwr_seqs = tuple(np.arange(n)[::o] for n in self.ncoeffs)

        if nested:
            return pwr_seqs
        else:
            return np.r_[pwr_seqs]

    # def __str__(self):
    #     p = self.get_pnames(free=True, latex=False)
    #     return self.get_repr(p)

    def get_pnames(self, alpha=None, free=True, increasing=False,
                   latex=False, unicode=False, nested=False):
        """
        Get a list of parameter names. Parameter names will start from 'a0' for
        the first polynomial, 'b0' for the second etc.

        Parameters
        ----------
        alpha : str
            letter to start with for names
        free : bool
            whether to return only the free parameter names, or the full set

        Returns
        -------
        names: list of strings or list of lists
            eg: ['a0', 'a1' 'a2', 'b0', 'b1', 'c0', 'd0']

        """
        alpha = alpha or self.coeff_name_base
        names = make_pnames(self.orders, alpha, increasing, latex,
                            unicode, False)

        if free:
            names = list(np.take(names, self.ix_free))

        if nested:
            ix = self._dof_each if free else self.ncoeffs
            splitting = np.cumsum(ix)
            names = np.split(names, splitting)[:-1]
            names = list(map(list, names))  # make list of lists

        return names

    def get_repr(self, coeff, name='f', variable='x', increasing=False,
                 onedomain=False, precision=2, minimalist=True, expmin=None,
                 times='\cdot'):
        # FIXME!!
        """
        Produce a latex string representing a polynomial

        Parameters
        ----------
        coeff : array-like
            coefficients
        variable: str
            variable name
        increasing : bool
            order of the powers
        precision : int
            numerical precision for coefficients
        minimalist : bool
            whether to represent floats as shortest possible string for given
            precision

        Returns
        -------
        str

        Returns
        -------
        str or IPython display object
        """

        coeff = np.asarray(coeff)
        if np.issubdtype(coeff.dtype, np.number):
            # numerical coefficients
            block = self.get_block_coeff(coeff)
        else:
            # str coefficients (parameter names)
            # get expression for derived coefficients
            dcoeff = []
            for i, cr in enumerate(self._constraints):
                q = lincombfmt(cr, coeff)
                if np.sum(cr != 0) > 1:
                    q = '(%s)' % q
                dcoeff.append(q)

            dtype = 'U%i' % np.char.str_len(dcoeff).max()
            block = np.empty((self.npoly, self.ncoeffs.max()), dtype)
            iy, ix = self.coeff_map[:, self.ix_solve]
            block[iy, ix] = dcoeff
            #
            iy, ix = self.coeff_map[:, self.ix_free]
            block[iy, ix] = coeff

        return get_latex_equation(block.T, self.breakpoints, name, variable,
                                  increasing, onedomain, precision, minimalist,
                                  expmin, times)

    def get_constrained(self, p):
        """
        Derive the parameters fixed by the boundary constraints from the free
        parameters. Linear combination.

        Parameters
        ----------
        p

        Returns
        -------
        coeff: array
            values of constrained parameters
        """
        return self._constraints.dot(p) + self.p_off

    def get_block_coeff(self, p):
        """
        Get the full set of coefficients for all the polynomials as an MxN
        array, where M is the degree of the highest order polynomial and N is
        the  number of polynomials (pieces). Those coefficients which can
        be algebraically determined by the constraints are filled. This set of
        coefficients returned by this method can be passed to
        `scipy.interpolation.PPoly` to construct a smooth piecewise polynomial
        (see `__call__` method).

        Parameters
        ----------
        p:  array
            parameter values

        Returns
        -------
        coeff : array
        """

        coeff = self._coeff
        coeff[self._ix_free] = p
        coeff[self._ix_solve] = self.get_constrained(p)
        # coeff[self._ix_const] = self.const_vals

        return coeff

    def check_bc(self, p=None):
        # check that boundary conditions are satisfied

        if self.nconstraints == 0:
            # no boundary conditions
            return True

        if p is None:
            # use random parameter vector
            p = np.random.randn(self.dof)

        coeff = self.get_block_coeff(p)
        lhs = np.polyval(coeff, 0)[1:]
        rhs = list(map(np.polyval, coeff.T, np.diff(self.breakpoints[:-1])))
        assert np.allclose(lhs, rhs)

    def _check_ord_bp(self, orders, breakpoints, scale=True):
        bp = breakpoints[1:]  # FIXME: ??
        # no constraint provided by 1st breakpoint (edge)
        if isinstance(orders, float):
            orders = int(orders)
        if isinstance(orders, int):
            orders = [orders] * len(bp)
        if len(orders) != len(bp):
            raise ValueError('Order / breakpoint size mismatch: %i, %i. '
                             'Breakpoints should have size `len(orders) + 1`'
                             % (len(orders), len(breakpoints)))

        # Might want to check the orders - if all 1, (straight lines) and
        # continuous = True, this is actually just a straight line, and it's
        # pointless to use a PiecewisePolynomial, but OK!
        return np.asarray(orders), np.asarray(breakpoints)


class PiecewiseLinear(PiecewisePolynomial):
    def __init__(self, breakpoints, continuous=True, scale=True,
                 coeff_name_base='a'):
        smooth = False  # smooth piecewise linear same as just plain linear
        orders = np.ones(len(breakpoints) - 1)
        super().__init__(orders, breakpoints, continuous, smooth, scale,
                         coeff_name_base)


class PPolyModel(FixedGrid, PiecewisePolynomial, RescaleInternal, Model):

    # TODO Spline1D / NonUniform1dSpline

    def __init__(self, orders, breakpoints, continuous=True, smooth=True,
                 scale_x=True, scale_y=True, fit_breakpoints=False,
                 coeff_name_base='a', solve_high_order=True, nbc=None):
        """
        Allow fitting breakpoints

        Parameters
        ----------
        fit_breakpoints: bool
            Whether the breakpoints are considered as parameters of the model.
            If true, `breakpoints` will be used as the initial guess, and each
            breakpoint parameter is now allowed to vary on a bounded interval
            surrounding the initial value so long as it does not cross more
            than halfway the distance to its neighbouring breakpoint,
            or fall outside the interval (0, 1).
        """
        PiecewisePolynomial.__init__(self, orders, breakpoints, continuous,
                                     smooth, scale_x, coeff_name_base,
                                     solve_high_order, nbc)
        #
        self.fit_breakpoints = fit_breakpoints
        # self._grid = None
        self.breakpoints0 = self.breakpoints.copy()
        self._var_bp = self.breakpoints.copy()

        if not scale_y:
            self._yscale = 1
        # else: will be set upon first call to fit

    @property
    def dof(self):
        # number of free parameters
        dof = super().dof
        if self.fit_breakpoints:
            return dof + self.npoly - 1
        return dof

    def __call__(self, p, grid):
        # keep the call method as a convenience method and do all the
        # idiot checks here, while keeping the `evaluate` method for the actual
        # work.
        # NOTE: this is different from the parent Model, which uses __call__
        # for the main work.
        #  FIXME: maybe change to having `evaluate` as the main workhorse
        # method in `Model`

        if isinstance(p, Parameters):
            p = p.flattened  # flatten nested parameters here

        self._check_params(p)
        return self.evaluate(p, grid)

    def evaluate(self, p, grid):
        # note: This method expects flattened parameter array
        # split parameters into coeff, breakpoints
        if self.fit_breakpoints:
            p, bp = np.split(p, [super(PPolyModel, self).dof])
            self.breakpoints[1:-1] = bp

        return self._evaluate_var_bp(p, grid, self.breakpoints)

    def _evaluate_var_bp(self, p, grid, breakpoints):
        # since the breakpoints are changing, so too the constraints matrix
        A = self._get_linear_system(breakpoints)

        # constraints
        self._constraints, self.p_off, _, _ = _reduce_constraints(
                A, self.ix_solve, self.ix_const, self.const_vals)

        return self._evaluate_fast(p, grid, breakpoints)

    def get_bounds(self, inf=None, delta=2):
        """These represent the assumption of uniform priors"""

        if not self.fit_breakpoints:
            return

        if inf is None:
            unbound = (None, None)
        elif isinstance(inf, numbers.Real):
            unbound = (-inf, inf)
        else:
            raise ValueError

        bounds = [unbound] * self.dof
        bpb = self.get_bp_bounds(delta)
        bounds[-len(bpb):] = bpb
        return bounds

    def get_bp_bounds(self, delta=2):
        # using (0, 1) here may lead to singular (non-invertible) system
        # each breakpoint parameter is now allowed to vary on a bounded
        # interval surrounding the initial value so long as it does not
        #  cross more than halfway the distance to its neighbouring
        # breakpoint, or fall outside the interval (0, 1).
        # assert bpv < 0.5
        # bp = self.breakpoints
        return tuple(self.breakpoints[None, 1:-1].T
                     + np.multiply(delta / self._xscale, [-1, 1]))

    def p0guess(self, *args, nested=False):

        pdict = {}
        ch = ord('f')
        pdict.update(coeff={chr(ch + i): np.zeros(o)
                            for (i, o) in enumerate(self._dof_each)})

        if self.fit_breakpoints:
            # use given breakpoints as initial guess
            pdict.update(breakpoints=self.breakpoints0[1:-1])

        p0 = Parameters(**pdict)

        if nested:
            return p0

        return p0.flattened

    def set_grid(self, data):
        self._grid = np.arange(len(data), dtype=float) / self._xscale
        # return self.grid

    def inverse_transform(self, p):
        if self.fit_breakpoints:
            # make sure we don't rescale the breakpoints
            coeff = p[:self.dof]
        else:
            coeff = p

        coeff *= self._yscale
        return p

    def fit(self, data, grid=None, stddev=None, p0=None, *args, **kws):
        if self.fit_breakpoints:
            # using (0, 1) here may lead to singular (non-invertible) system
            bounds = self.get_bounds()
            kws.setdefault('bounds', bounds)

        return super().fit(data, grid, stddev, p0, *args, **kws)

    def fit_bp_only(self, coeff, data, grid):
        bounds = self.get_bp_bounds()
        r = minimize(self.objective_bp_only, self.breakpoints0[1:-1],
                     (coeff, data, grid), bounds=bounds)
        return r.x

    def objective_bp_only(self, bp, coeff, data, grid):
        # split parameters into coeff, breakpoints
        self.breakpoints[1:-1] = bp
        v = self._evaluate_var_bp(coeff, grid, self.breakpoints)
        return np.square(data - v).sum()

    def get_pnames(self, alpha=None, free=True, increasing=False, latex=False,
                   unicode=False, nested=False):

        # get coefficient names
        names = super().get_pnames(alpha, free, increasing, latex,
                                   unicode, nested)

        if self.fit_breakpoints:
            n_free_breaks = len(self.breakpoints) - 2
            breakpoint_names = [f'bp{i}' for i in range(n_free_breaks)]

            aggregate = getattr(names, 'append' if nested else 'extend')
            aggregate(breakpoint_names)

        return names

    def plot_fit_results(self, p, data, grid=None, std=None, yscale=1,
                         modRes=500):
        # TODO suppress UserWarning: Warning: converting a masked element to nan
        import matplotlib.pyplot as plt
        from recipes.pprint import nrs

        def gof_text(stat, name, xpos=0):
            v = stat(p, data, grid, std)
            s = nrs.nr(v, significant=3, latex=True)
            txt = '$%s = %s$' % (name, s.strip('$'))
            # print(txt)
            return axTxt.text(xpos, 0.05, txt, fontsize=14, va='bottom',
                              transform=axTxt.transAxes)

        # fig = mdl.plot_fit_results(image, params[mdl.name], modRes)
        # figs.append(fig)

        # plot fit result
        fig, axes = plt.subplots(3, 1,
                                 figsize=(10, 8),
                                 sharex=True,
                                 gridspec_kw=dict(hspace=0,
                                                  height_ratios=(0.25, 3, 1)))
        axTxt, axMdl, axResi = axes

        # scale data,
        data = data * yscale
        if std is not None:
            std = std * yscale
        if grid is None:
            grid = self.grid

        gsc = grid * self._xscale

        # model
        model_colour = 'darkgreen'
        p = p.squeeze()
        x_mdl = np.linspace(grid[0], grid[-1], modRes)
        dfit = self(p, x_mdl) * np.sqrt(yscale)
        line_mdl, = axMdl.plot(x_mdl * self._xscale, dfit, '-',
                               color=model_colour, label='model',
                               zorder=100)

        # plot fitted data
        data_colour = 'royalblue'
        ebMdl = axMdl.errorbar(gsc, data, std,
                               fmt='o', color=data_colour, zorder=10,
                               label=r'Median $\pm$ (MAD)')

        # residuals
        res = data - self(p, grid) * np.sqrt(yscale)
        ebRes = axResi.errorbar(gsc, res, std, fmt='o', color=data_colour)
        ebars = [ebMdl, ebRes]

        # get the breakpoints
        if isinstance(p, Parameters):
            p = p.view((float, p.npar))

        self._check_params(p)

        # breakpoints
        bp = p[1 - self.npoly:] if self.fit_breakpoints else self.breakpoints
        breakpoints = bp * self._xscale
        lineCols = [ax.vlines(breakpoints, 0, 1,
                              linestyle=':', color='0.2',
                              transform=ax.get_xaxis_transform(),
                              label='Break points')
                    for ax in axes[1:]]

        # axes labels
        axTxt.set_title(f'{self.name} Fit', dict(fontweight='bold'))
        axMdl.set_ylabel('Counts (ADU)')
        axResi.set(xlabel='pixels',
                   ylabel='Residuals')
        axMdl.grid()
        axResi.grid()

        # ylims
        w = data.ptp() * 0.2
        axMdl.set_ylim(data.min() - w, data.max() + w)

        w = res.ptp()
        ymin = res.min() - w
        ymax = res.max() + w
        if std is not None:
            ymin = min(ymin, (res - std).min() * 1.2)
            ymax = max(ymax, (res + std).max() * 1.2)
        axResi.set_ylim(ymin, ymax)

        # GoF statistics
        # Remove all spines, ticks, labels, etc for `axTxt`
        axTxt.set_axis_off()

        positions = np.linspace(0, 1, 4, endpoint=False) + 0.05
        texts = [gof_text(f, name, xpos)
                 for xpos, (f, name) in
                 zip(positions, {self.redchi: r'\chi^2_{\nu}',
                                 self.rsq: 'R^2',
                                 self.aic: 'AIC',
                                 self.aicc: 'AIC_c'}.items())]

        # turn the ticks back on for the residual axis
        for tck in axResi.xaxis.get_major_ticks():
            tck.label1.set_visible(True)

        fig.tight_layout()
        #
        return ebars, line_mdl, lineCols, texts  # art

    def animate_fit(self, p0, data, grid=None, std=None, **kws):

        # run the minimization and save parameter array at each step
        plist = []
        if p0 is None:
            p0 = self.p0guess(data)

        result = self.fit(p0, data, grid, callback=plist.append, **kws)
        return result, self.animate(plist, data, grid, std)

    def animate(self, plist, data, grid, std):
        """
        Animate the optimization process for this model :))

        Parameters
        ----------
        p0
        data
        std

        Returns
        -------

        """
        import time
        from matplotlib.animation import FuncAnimation
        from recipes.pprint import nrs

        # plot the result
        ebars, line_mdl, lineCols, chiTxt = \
            self.plot_fit_results(plist[0], data, grid, std)
        #
        if std is not None:
            ebars[1][-1][0].set_visible(False)

        # add text for step number
        frame_fmt = 'i = %i'
        ax = line_mdl.axes
        frame_txt = ax.text(0, 1, '', transform=ax.transAxes)

        def update(i):
            if i == len(plist):
                # pause for a bit before restarting the animation
                time.sleep(5)
                return

            # update model line
            p = plist[i]
            x_mdl = line_mdl.get_xdata()
            y_mdl = self(p, x_mdl / self._xscale)  # * self._yscale
            line_mdl.set_data(x_mdl, y_mdl)
            #
            if i == 0:
                line_mdl.set_color('r')
            # last item ==> final result
            if i == len(plist) - 1:
                line_mdl.set_color('g')  # converged solution in green

            # update breakpoints
            if self.fit_breakpoints:
                bp = p[self.dof:] * self._xscale
                nbp = self.npoly - 1
                segs = np.dstack((np.tile(bp, (nbp, 1)).T, np.eye(nbp)))
                for lcbp in lineCols:
                    lcbp.set_segments(segs)

            # update residuals
            res = data - self(p, grid) * self._yscale
            line_resi = ebars[1][0]
            line_resi.set_ydata(res)

            # update text
            frame_txt.set_text(frame_fmt % i)
            rchisq = self.redchi(p, data, grid)
            txt = nrs.numeric(rchisq, latex=True)
            txt = txt.replace('$', '$\chi^2_r = ', 1)
            chiTxt.set_text(txt)

        fig = line_mdl.figure
        return FuncAnimation(fig, update, range(len(plist) + 1))


class PartialPoly(Model):
    # helper for computing polynomials with algebraic relations between
    # coefficients
    def __init__(self, powers, constraints, c_offset=0):
        self.powers = np.sort(powers)[::-1]  # decreasing power
        self.degree = self.powers[0]
        self.p = np.empty(self.degree + 1)

        self.ix_free = np.sort(abs(self.powers - self.degree))
        self.ix_omit = list(set(range(self.degree + 1)) - set(self.ix_free))

        self.c_offset = c_offset
        self.constraints = np.array(constraints)
        assert self.constraints.shape[1] == len(self.ix_omit)

    def __call__(self, coeff, grid):
        self.p[self.ix_free] = coeff
        self.p[self.ix_omit] = self.constraints.dot(coeff) + self.c_offset
        return np.polyval(self.p, grid)


class PartialPPoly(PPolyModel):
    """Wrapper for allowing constant parameter values"""

    def __init__(self, orders, breakpoints, continuous=True, smooth=True,
                 scale_x=True, scale_y=True, fit_breakpoints=False,
                 coeff_name_base='a', solve_high_order=True, nbc=None,
                 keep_free=()):
        super().__init__(orders, breakpoints, continuous, smooth, scale_x,
                         scale_y, fit_breakpoints, coeff_name_base,
                         solve_high_order, nbc, keep_free)

        self._p = np.empty(self.dof)

    def set_constant_params(self, indices, values):
        self._ix_const = np.array(indices, int)
        self._ix_free = np.setdiff1d(np.arange(self.dof), self._ix_const)
        self._p[self._ix_const] = np.array(values, float)
        return self._p

    def get_block_coeff(self, p):
        self._p[self._ix_free] = p
        return super().get_block_coeff(self._p)

    def _check_p(self, p):
        n_ok = len(self._ix_free)
        if self.fit_breakpoints:
            n_ok += self.npoly - 1

        if len(p) != n_ok:
            raise ValueError('%r takes %i free parameters, %i given.'
                             % (self.__class__.__name__, self.dof, len(p)))


# # constant parameter values
# is_const = np.zeros(self.ncoeff, bool)
# if constant_coeffs is not None:
#     for n, cix in enumerate(constant_coeffs):
#         i = cix + self.cso[n]
#         is_const[i] = True


# class PPolyModelSequential():
# """Sequential fitting instead of simultaneous"""
#
# def __init__(self, orders, breakpoints, continuous=True, smooth=True,
#              scale_x=True, scale_y=True, fit_breakpoints=False,
#              coeff_name_base='a', primary=0):
#     #
#     self.orders, breakpoints = _check_ord_bp(orders, breakpoints)
#     if scale_x:
#         self.breakpoints = breakpoints / breakpoints.max()
#
#     # primary - which poly one will be fit first
#     self.primary = int(primary)
#     # self._ix_const = []
#
#     self.pp = []
#     for i, o in enumerate(mit.pairwise(orders)):
#         bp = breakpoints[i: i + 3]
#
#         cso = np.cumsum(np.r_[0, np.add(o, 1)])
#         i0 = self.primary - i
#         keep_free = np.arange(*cso[i0: i0 + 2])
#         # keep_free = np.arange(*pp._ix_coeff_rng[self.primary - i])
#         # self._ix_const.append(keep_free)
#
#         pp = PartialPPoly(o, bp, continuous, smooth, scale_x,
#                           scale_y, fit_breakpoints, coeff_name_base,
#                           keep_free=keep_free)
#         self.pp.append(pp)
#
# def fit0(self, p0, data, grid, stddev=None, *args, **kws):
#
#     from scipy.optimize import minimize
#
#     # if grid is None:
#     #     grid = self.static_grid
#
#     primary = self.primary
#     others = np.setdiff1d(np.arange())
#     splitting = np.searchsorted(grid, self.breakpoints[1:-1])
#     ranges = mit.pairwise([0] + list(splitting) + [None])
#     slices = list(map(slice, *zip(*ranges)))
#
#     # least squares quick fix
#     selection = slices[primary]
#     # independent domains
#     g = grid[selection]    # - self.breakpoints[primary]
#     p00 = np.polyfit(g, data[selection], self.orders[primary])
#
#     for i, pp in enumerate(self.pp):
#         pri = self.primary - i
#         rng = np.cumsum(pp._dof_each[pri:])
#         ix_const = np.arange(*rng)
#         pp.set_constant_params(ix_const, p00)


# def make_partial_poly(order, nbc, plhs=None, prhs=None):
#
#     # FIXME: this is really a dynamic method to change constraints system
#
#     # assert plhs, prhs != (None, None)    # not both None
#     # assert None not in (plhs, prhs)          # not both given
#
#     if plhs is not None:
#         orders = len(plhs), order
#         ix_const = np.arange(order)
#
#     elif prhs is not None:
#         orders = order, len(prhs)
#         ix_const = np.arange(order, sum(orders))
#     else:
#         raise ValueError
#
#     # pri = self.primary - i
#     ncf = np.add(orders, 1)
#     # rng = np.cumsum(np.r_[0, ncf])[pri: pri + 2]
#
#     A = boundary_constraints_system(x, ncf, nbc)
#
#     # ix_const = np.arange(*rng)
#     ix_solve = sorted(get_solvable_indices(A, orders, ix_const, False))
#     # solve for lower order indices so that we fit for higher curvatures
#     ix_fixed = np.sort(np.r_[ix_const, ix_solve])
#     ix_free = sorted(set(range(sum(ncf))) - set(ix_fixed))
#     # print('const', ix_const, 'solve', ix_solve, 'fixed', ix_fixed,
#     # 'free', ix_free)
#     # S.extend(ix_solve + self.cso[i])
#
#     # constraints
#     s = np.linalg.inv(Asub[:, ix_solve])
#     constraints = s.dot(Asub)
#
#     p_offset = -constraints[:, ix_const].dot(p00)
#     # constraints = -np.delete(constraints, ix_fixed, axis=1)
#
#     # print('constraints')
#     # print(constraints)
#     pwrs = self.get_powers(nested=True)[i]
#     powers = np.r_[pwrs[i: i + 2]][ix_free]
#
#     # return p00, constraints, powers, ix_const, ix_solve, ix_free
#
#     return PartialPoly(powers, -constraints[:, ix_free], p_offset)


class PPolyModelSequential(PPolyModel):
    """
    Sequential fitting instead of simultaneous for speedy hyper parameter
    searches etc.
    """

    def __init__(self, orders, breakpoints, continuous=True, smooth=True,
                 scale_x=True, scale_y=True, fit_breakpoints=False,
                 coeff_name_base='a', solve_high_order=True, nbc=None,
                 keep_free=(), primary=None):

        # primary - which poly one will be fit first
        self.primary = int(primary)

        super().__init__(orders, breakpoints, continuous, smooth, scale_x,
                         scale_y, fit_breakpoints, coeff_name_base,
                         solve_high_order, nbc, keep_free)

    def get_partial_poly(self, i, p00):

        o = self.orders[i: i + 2]

        i0, i1 = self.csc[i:i + 2]
        j0, j1 = self.cso[[i, i + 2]]
        Asub = self._A[i0:i1, j0:j1]
        # print('Asub', Asub)

        pri = self.primary - i
        ncf = np.add(o, 1)
        rng = np.cumsum(np.r_[0, ncf])[pri: pri + 2]
        ix_const = np.arange(*rng)
        ix_solve = sorted(get_solvable_indices(Asub, o, ix_const, False))
        # solve for lower order indices so that we fit for higher curvatures
        ix_fixed = np.sort(np.r_[ix_const, ix_solve])
        ix_free = sorted(set(range(sum(ncf))) - set(ix_fixed))
        # print('const', ix_const, 'solve', ix_solve, 'fixed', ix_fixed,
        # 'free', ix_free)
        # S.extend(ix_solve + self.cso[i])

        # constraints
        s = np.linalg.inv(Asub[:, ix_solve])
        constraints = s.dot(Asub)

        p_offset = -constraints[:, ix_const].dot(p00)
        # constraints = -np.delete(constraints, ix_fixed, axis=1)

        # print('constraints')
        # print(constraints)
        pwrs = self.get_powers(nested=True)[i]
        powers = np.r_[pwrs[i: i + 2]][ix_free]

        # return p00, constraints, powers, ix_const, ix_solve, ix_free

        return PartialPoly(powers, -constraints[:, ix_free], p_offset)

    def make_ppoly(self, data, grid):

        p = np.empty(self.dof)

        primary = self.primary
        # others = np.setdiff1d(np.arange())
        splitting = np.searchsorted(grid, self.breakpoints[1:-1])
        ranges = mit.pairwise([0] + list(splitting) + [None])
        slices = list(map(slice, *zip(*ranges)))

        # least squares quick fix
        selection = slices[primary]
        # independent domains
        g = grid[selection] - self.breakpoints[primary]
        p00 = np.polyfit(g, data[selection], self.orders[primary])
        p[np.arange(*self._ix_coeff_rng[primary])] = p00

        PP = []
        # R = []
        # S = []
        pwrs = self.get_powers(nested=True)

        for i, o in enumerate(mit.pairwise(self.orders)):

            i0, i1 = self.csc[i:i + 2]
            j0, j1 = self.cso[[i, i + 2]]
            Asub = self._A[i0:i1, j0:j1]
            # print('Asub', Asub)

            pri = primary - i
            ncf = np.add(o, 1)
            rng = np.cumsum(np.r_[0, ncf])[pri: pri + 2]
            ix_const = np.arange(*rng)
            ix_solve = sorted(get_solvable_indices(Asub, o, ix_const, False))
            # solve for lower order indices so that we fit for higher curvatures
            ix_fixed = np.sort(np.r_[ix_const, ix_solve])
            ix_free = sorted(set(range(sum(ncf))) - set(ix_fixed))
            # print('const', ix_const, 'solve', ix_solve, 'fixed', ix_fixed,
            # 'free', ix_free)
            # S.extend(ix_solve + self.cso[i])

            # constraints
            s = np.linalg.inv(Asub[:, ix_solve])
            constraints = s.dot(Asub)

            p_offset = -constraints[:, ix_const].dot(p00)
            # constraints = -np.delete(constraints, ix_fixed, axis=1)

            # print('constraints')
            # print(constraints)

            powers = np.r_[pwrs[i: i + 2]][ix_free]

            # return p00, constraints, powers, ix_const, ix_solve, ix_free

            pp = PartialPoly(powers, -constraints[:, ix_free], p_offset)
            PP.append(pp)

        return PP

        #     ix = 2 if i else 0
        #     s = slices[ix]
        #     g = grid[s] - self.breakpoints[ix]
        #     p0 = np.ones(len(ix_free))
        #     r = pp.fit(p0, data[s], g)
        #     R.append(r)
        #
        # R.insert(primary, p00)
        #
        # return PP, R, S


class _PPolyModel2D(PPolyModel):
    # Patch to optimize `PPolyModel` for use in 2D optimization problems.
    # We need to change a number of attributes so that the `get_block_coeff`
    # method calculates the constrained parameters correctly and returns
    # the right coefficients with the constant parameters zeroed.

    def __init__(self, orders, breakpoints, continuous=True, smooth=True,
                 scale_x=True, scale_y=True, fit_breakpoints=False,
                 coeff_name_base='a'):
        #
        super().__init__(orders, breakpoints, continuous, smooth, scale_x,
                         scale_y, fit_breakpoints, coeff_name_base)

        # since we are combining 2 piecewise polynomial models, we can
        # eliminate an additional parameter (the constant value is specified in
        # both models) and thus also eliminate a degeneracy in the parameter
        # space of the 2d model.  We arbitrarily choose the x
        # PiecewisePolynomial for this.

    def _check_p(self, p):
        return super()._check_p(p)

    def _get_linear_system(self, breakpoints):
        A = PiecewisePolynomial._get_linear_system(self, self.breakpoints)
        i = self.cso[1:] - 1  # indices of constant terms
        A[0, i] = 0  # zero the constant terms in the constraint system
        return A


class PPoly2dOuter(FixedGrid, CompoundModel):  # RescaleInternal
    """
    Two-dimensional polynomial object with no mixed coefficient terms. Such
    polynomials can be expressed as a (outer) sum of 1D polynomials in x and y.
    """

    # TODO: make fit_breakpoints a property

    @classmethod
    def from_orders(cls, orders, breaks, continuous=True, smooth=True,
                    scale_x=True, scale_y=True, fit_breakpoints=False,
                    coeff_name_base=('a', 'u'), solve_high_order=True):
        """

        Parameters
        ----------
        orders: xorders, yorders
        breaks: xbreaks, ybreaks

        all other arguments can be bool or 2 tuple of bool
        """

        # HACK to allow all arguments to be 2-tuples or boolean
        constructor = PPoly2dOuter.from_orders  # self-reference
        code = constructor.__code__
        arg_names = code.co_varnames[:code.co_argcount]
        defaults = constructor.__defaults__
        # default arg values taken from method definition above

        kwargs = args_y, args_x = {}, {}
        for argname, dflt in zip(arg_names[-len(defaults):], defaults):
            value = eval(argname)
            if isinstance(value, (list, tuple)):
                assert len(value) == 2, 'Sequence arguments must be of length 2'
                args_x[argname] = value[1]
                args_y[argname] = value[0]

            elif isinstance(value, bool):
                args_y[argname] = args_x[argname] = value
            else:
                raise TypeError('Value for %r is invalid type %r' %
                                (argname, type(value)))

        print('init')
        print(args_y)
        print(args_x)

        # # save init_args for pickling
        # self._init_args = tuple(map(eval, code.co_varnames[1:code.co_argcount]))

        # initialize models
        polyy, polyx = (PPolyModel(o, b, **kws)
                        for (o, b, kws) in zip(orders, breaks, kwargs))

        # init base class
        return cls(y=polyy, x=polyx)

    def __init__(self, models=(), **kws):

        CompoundModel.__init__(self, models, **kws)

        # since we are combining 2 piecewise polynomial models, we can
        # eliminate an additional parameter (the constant value is specified in
        # both models) and thus also eliminate a degeneracy in the parameter
        # space of the 2d model.  We arbitrarily choose the x
        # PiecewisePolynomial for this.
        mx = self.x
        powers = mx.get_powers()
        # indices of constant (power 0) coeff wrt stacked coefficient array
        ix_const_coeff, = np.where(powers == 0)
        # remove possible constant coefficient that is being solved for
        # ix_const_coeff = np.setdiff1d(ix_const_coeff, mx.ix_solve)
        primary = 1
        mx._make_solving(mx.ix_solve, [ix_const_coeff[primary]], [0])

        # ensure unique coefficient names
        if self.x.coeff_name_base == self.y.coeff_name_base:
            self.x.coeff_name_base = 'u'

    def __call__(self, p, grid=None):
        if grid is None:  # todo delegate to FixedGrid
            grid = self.static_grid

        if isinstance(p, Parameters):
            # note: flatten nested parameters here
            p = p.flattened

        # all idiot checks performed inside constituent models __call__
        zy = self.y(p[:self.y.dof], grid[0])
        zx = self.x(p[self.y.dof:], grid[1])
        return zy[:, None] + zx[None]  # outer summation

    @property
    def dof(self):
        return sum(m.dof for m in self.models)

    def __reduce__(self):
        # overwrites OrderedDict.__reduce__.
        # since the base class implements __reduce__, simply overwriting
        # __getnewargs__ will not work
        return (PPoly2dOuter, (tuple(self.models),),
                dict(static_grid=self.static_grid))

    def get_pnames(self, alpha=(), free=True, increasing=False,
                   latex=False, unicode=False, nested=False):

        pnames = []
        aggregate = pnames.append if nested else pnames.extend
        for mdl, a in itt.zip_longest(self.models, alpha, fillvalue=None):
            aggregate(
                    mdl.get_pnames(a, free, increasing, latex, unicode, nested)
            )
        return pnames

    def format_params(self, values, names=None, precision=2, switch=3, sign=' ',
                      times='x', compact=True, unicode=True, latex=False,
                      engineering=False):

        # TODO: move to model / compound model

        #
        dof = self.dof
        if len(values) != dof:
            raise ValueError('Invalid number of parameters (%i) for %s  '
                             'object with %i degrees of freedom' %
                             (len(values), self.__class__.__name__, dof))
        if names is None:
            names = self.get_pnames(unicode=unicode, latex=latex)
        else:
            assert len(names) == len(dof)

        # order of magnitude
        # todo: align on decimal point.
        # oom = np.log10(np.abs(p)).astype(int)

        # numeric repr
        s = np.vectorize(pprint.numeric, ['U10'])(
                values, precision, switch, sign, times, compact, unicode,
                latex, engineering)
        # if uncert is not None:
        #     raise NotImplementedError  # TODO

        return list(map('%s = %s'.__mod__, zip(names, s)))

    # def p0guess(self, data, grid=None, stddev=None, **kws):
    #
    #     # Good guess for poly coeff can come from fitting median cross sections
    #     # of the data initially
    #     media = {'x': np.ma.median(data, 0),
    #              'y': np.ma.median(data, 1)}
    #
    #     p0 = super().p0guess(data, grid, stddev, **kws)
    #     #
    #     for yx in 'yx':
    #         r = self[yx].fit(None, media[yx])
    #         iconst = np.cumsum(self[yx]._dof_each) - 1
    #         # print(iconst)
    #         r[iconst] /= 2
    #         p0[yx].flattened[:] = r
    #
    #     if kws.get('nested'):
    #         return p0
    #     else:
    #         return p0.flattened

    def evaluate(self, p, grid):
        zy = self.y.evaluate(p[:self.y.dof], grid[0])
        zx = self.x.evaluate(p[self.y.dof:], grid[1])
        return zy[:, None] + zx[None]

    # def residuals(self, p, data, grid=None):
    #     """inject fast evaluation for fitting"""
    #     return data - self.evaluate(p, grid)

    def set_grid(self, data):
        self.static_grid = []
        for i, mdl in enumerate(self.models):
            order = -1 if i else 1
            ix = (slice(None), 0)[::order]
            grid = mdl.set_grid(data[ix])
            self.static_grid.append(grid)

    def fit(self, data, grid=None, stddev=None, p0=None, *args, **kws):
        return Model.fit(self, data, grid, stddev, p0, *args, **kws)


# def animate_fit(mdl, p0, data, std):
#     from matplotlib.animation import FuncAnimation
#
#     # run the minimization and save parameter array at each step
#     plist = []
#     result = my.fit(p0, data, callback=plist.append)
#
#     # plot the result
#     ebc, line_mdl, lineCol_bp = mdl.plot_fit_results(p0, data, std)
#
#     def update(p):
#         # update model line
#         y = mdl._evaluate_var_bp(p, mdl.static_grid) * mdl._yscale
#         x = mdl.static_grid * mdl._xscale
#         line_mdl.set_data(x, y)
#
#         # update breakpoints
#         bp = p[mdl.dof:] * mdl._xscale
#         segs = np.dstack((np.tile(bp, (mdl.npoly - 1, 1)).T, np.eye(2)))
#         lineCol_bp.set_segments(segs)
#
#         # TODO: update residuals. converged result in green
#
#     fig = line_mdl.figure
#     ani = FuncAnimation(fig, update, plist)


# def yield_free():

# The terms "order" and "degree" are used as synonymns throughout
# https://en.wikipedia.org/wiki/Degree_of_a_polynomial
# https://en.wikipedia.org/wiki/Order_of_a_polynomial

def _check_order(o, ndim):
    assert len(o) == ndim, 'Order tuple has incorrect number of items. ' \
        f'Expected {ndim}, received {len(o)}'


class Poly2D(FixedGrid, Model):
    """Fittable 2d polynomial"""

    def __init__(self, multi_order):
        """
        A fittable 2-dimensional polynomial.

        Parameters
        ----------
        multi_order: 2-tuple of int
            The order / degree of the polynomial along each dimension.
            i.e. The dimensions of the coefficient matrix less one.

        """
        # TODO: attributes

        self.n_coeff = self.coeff = self.free = None  # place holders for init
        self.set_orders(multi_order)
        self._yixj = None
        # self.fit_variance = False

    # def __call__(self, p, grid=None):
    #     grid = self._check_grid(grid)
    #     return self.eval(p, grid)

    def eval(self, p, grid):
        self.set_coeff(p)  # coefficients increasing in power (y, x)
        return polyval2d(*grid, self.coeff)

    def __str__(self):
        return repr_matrix_product(self.coeff)

    def __repr__(self):
        return '%s%s' % (self.__class__.__name__, tuple(self.n_coeff - 1))

    def set_orders(self, multi_order):

        _check_order(multi_order, 2)
        self.n_coeff = np.add(multi_order, 1)  # sy, sx

        # coefficients increasing in power (y, x)
        self.coeff = np.zeros(self.n_coeff)

        # mask over coefficient matrix for free / const parameters
        self.free = np.ones(self.n_coeff, bool)

    def set_grid(self, grid):
        FixedGrid.set_grid(self, grid)
        # grid-dependent yⁱxʲ terms used to calculate jacobian
        self._yixj = self.power_products(grid)  # TODO: lazy prop?

    def pre_process(self, p0, data, grid=None, stddev=None, *args, **kws):
        # remove masked data
        p0, data, grid, stddev, args, kws = super().pre_process(
                p0, data, grid, stddev, *args, **kws)
        if kws.get('method') == 'leastsq':
            # compute yⁱxʲ terms used to calculate jacobian. Since masked data
            # is removed above, these terms depend on data as well as grid when
            # optimization routine is leastsq
            self._yixj = self.power_products(grid)
            kws.update(Dfun=self.jacobian_fwrs,
                       col_deriv=True)

        return p0, data, grid, stddev, args, kws

    def power_products(self, grid):  # jac_comp / jacobian_components
        # grid-dependent yⁱxʲ terms used to calculate jacobian
        ij = np.indices(self.n_coeff)
        ixr = (slice(None), self.free) + (None,) * (grid.ndim - 1)
        return np.power(grid[:, None], ij[ixr]).prod(0)

    def _jacobian(self, p, data, grid, stddev=None):
        # jacobian (without 2 multiplier term)
        # Optimization for computing jacobian / hessian components that only
        # depend on the grid, and not on the parameter vector. This allows a
        # ~3-4x speedup for model fitting. yay!
        r = self.residuals(p, data, grid)
        if stddev is not None:
            r /= stddev

        return r * self._yixj

    def jacobian_wrss(self, p, data, grid, stddev=None):
        # jacobian derivative vector for wrss
        # computing Jacobian analytically improves convergence statistics
        # (most frequent reason for non-convergence is precision loss due to
        # point-to-point jacobian estimate)
        # Furthermore, this yields a performance optimization of ~20 %
        # compared to when jacobian is estimated from finite
        # differences of the objective function.
        return -2 * self._jacobian(p, data, grid, stddev).sum((-1, -2))

    def jacobian_fwrs(self, p, data, grid, stddev=None):
        return -2 * self._jacobian(p, data, grid, stddev).reshape(self.dof, -1)

    # def hessian_wrss(self, p, data, grid, stddev=None):
    #     # hessian derivative matrix for wrss (independent of model parameters!)
    #     # return 2 * (self._yixj[None] * self._yixj[:, None]).sum((-1, -2))
    #     return self._hess

    def p0guess(self, data, grid=None, stddev=None):
        # TODO inherit docstring
        p0coeff = np.zeros(self.dof)
        # if self.fit_variance:
        #     return np.r_[p0coeff, 1]
        return p0coeff

    def set_coeff(self, p):
        """
        Set the values of the free parameters (polynomial coefficients)
        """
        # check
        psize = np.size(p)
        if psize != self.free.sum():
            raise ValueError('Parameter vector (length %i) has incorrect shape '
                             'for %s with %i free parameters' %
                             (psize, self.__class__.__name__, self.free.sum()))
        # set
        self.coeff[self.free] = p

    def set_const(self, indices, values=0):
        assert isinstance(indices, tuple)
        assert len(indices) == 2

        self.free[indices] = False
        self.coeff[indices] = values

    @property
    def dof(self):
        return self.free.sum()  # + self.fit_variance

    @property
    def const_coeff_vals(self):
        return self.coeff[~self.free]

    def get_pnames(self, alpha='a', free=True, latex=False, unicode=False):
        nrs = map(range, self.n_coeff)
        names = make_pnames_comb(alpha, *nrs, latex=latex,
                                 unicode=unicode)

        if free:
            return names[self.free]
        return names


class PPoly2D(Poly2D):  #
    """
    Two-dimensional polynomial object that shares a boundary with
    a neighbouring 2d polynomial. Automatically update the coefficient
    matrix for dependent polynomials.

    Dependent polys have restricted parameter spaces. These constraints are
    imposed by (optional) continuity and smoothness conditions on the boundary
    between polynomials.


    """

    # TODO: SemanticNeighbours mixin?
    SEMANTIC_POS = ('top', 'left', 'bottom', 'right')

    def __init__(self, orders, smooth=True, continuous=True):
        """
        Initialize with 2-tuple of polynomial multi-order. There are up to four
        possible neighbours that depend on this polynomial.

        Parameters
        ----------
        orders
        """

        # init parent
        super().__init__(orders)

        # neighbours
        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
        self.depends_on = None
        self._tied_rows_cols = (0, 0)

        # set boundary conditions flags
        self.continuous = bool(continuous)
        self.smooth = bool(smooth) or self.continuous
        # must be smooth if continuous

        # set parameter freedom
        self.set_freedoms()

    def set_coeff(self, p):
        #
        super().set_coeff(p)
        self.apply_bc()

    def apply_bc(self):
        """
        Update coefficient matrices of neighbours applying the (optional)
        smoothness and continuity conditions
        """

        # TODO: include domain ranges here so you don't have to

        c = self.coeff

        for k in range(self.smooth + self.continuous):
            if k == 0:
                # continuity
                j, i = 1, 1
            else:
                # smoothness
                j, i = np.ogrid[tuple(map(slice, self.n_coeff))]
                # 1st derivative multiplier :)

            # set coefficients for x-only terms
            if self.top:
                self.top.coeff[k] = np.sum(c * j, 0)

            if self.bottom:
                self.bottom.coeff[k] = c[k]

            # set coefficients for y-only terms
            if self.left:
                self.left.coeff[:, k] = c[:, k]

            if self.right:
                self.right.coeff[:, k] = np.sum(c * i, 1)

    def get_neighbours(self):
        n = {}
        for s in self.SEMANTIC_POS:
            poly = getattr(self, s)
            if poly is not None:
                n[s] = poly
        return n

    def set_neighbours(self, **n):
        for s, o in n.items():
            if s not in self.SEMANTIC_POS:
                raise KeyError('%s is not a valid neighbour position' % s)

            if o.__class__.__name__ != 'PPoly2D':  # note auto-reload hack
                raise TypeError(
                        'Neighbouring polynomials should be `None` or `PPoly2D`'
                        ' instances, not %r.' % o.__class__.__name__)

            # set neighbours as object attributes
            setattr(o, 'depends_on', self)
            setattr(self, s, o)
            l = self.continuous + self.smooth

            if s in ['left', 'right']:
                o._tied_rows_cols = (0, l)  # (slice(None), slice(l, None))
            else:
                o._tied_rows_cols = (l, 0)  # (slice(l, None), slice(None))

        # set the tied coefficients in neighbouring polys
        self.apply_bc()
        self.set_freedoms()

    def set_freedoms(self):  # todo better name. update_neighbours?
        for i in range(self.smooth + self.continuous):
            # set coefficients for x-only terms
            if self.top:
                self.top.free[i] = False

            if self.bottom:
                self.bottom.free[i] = False

            # set coefficients for y-only terms
            if self.left:
                self.left.free[:, i] = False

            if self.right:
                self.right.free[:, i] = False

    def free_diagonal(self):
        i = self.smooth + self.continuous
        r, c = np.diag_indices(self.n_coeff.min())
        rm, cm = self.n_coeff
        self.free[r[i:rm], c[i:cm]] = True

    def yield_coeff_subspaces(self, diagonal=True, max_dof_lsq=None):

        yo, xo = self.n_coeff

        if diagonal:
            # diagonal expansion into higher multi-orders
            mo = np.c_[self.n_coeff]
            t = np.c_[self._tied_rows_cols].T

            # free = np.zeros((yo + xo, yo, xo), bool)
            for ii, i in enumerate(range(yo - 1, -xo + 1, -1)):
                free = np.zeros((yo, xo), bool)
                # free[ii] = False
                ix = j, k = np.array(np.triu_indices(yo, i))
                ix[1] = yo - k - 1
                ok = np.all(ix < mo, 0) & np.all(ix >= t, 0)
                if not ok.any():
                    continue

                if ok.sum() > max_dof_lsq:
                    continue

                # j, k = ix[:, ok]
                free[tuple(ix[:, ok])] = True
                yield free

        else:
            # order expansion strategy: currently loops through all rectangular
            # sub matrices in the matrix of free coefficients
            j0 = k0 = 0
            if self._tied_rows_cols:
                j0, k0 = self._tied_rows_cols

            j1, k1 = np.add(self.coeff.shape, 1)
            # self.smooth + self.continuous

            order_grid = np.mgrid[j0 + 1:j1, k0 + 1:k1]
            order_pairs = order_grid.reshape(2, -1).T

            # check that we don't try fit more parameters than data points. This
            # is only an issue for least squares
            if max_dof_lsq is not None:
                ok = np.product(order_pairs, 1) <= max_dof_lsq
                order_pairs = order_pairs[ok]

            # free = np.zeros((len(order_pairs), yo, xo), bool)
            for i, (j1, k1) in enumerate(order_pairs):
                # print(i, j0,j1, k0,k1)
                free = np.zeros((yo, xo), bool)
                free[j0:j1, k0:k1] = True

                yield free

    def order_search(self, image, diagonal=True, **kws):

        import motley

        # systematically increase orders
        params = []
        goodness = []  # np.full(len(order_pairs), np.nan)

        rectsub = self.yield_coeff_subspaces(False, image.size)
        diagsub = self.yield_coeff_subspaces(True, image.size)
        subs = mit.interleave_longest(diagsub, itt.islice(rectsub, 1, None))
        # subs = itt.islice(subs, 15, 16)
        bgof, freedom, ibest = np.inf, None, None
        for i, free in enumerate(subs):
            self.free[:] = free
            print(i)
            print(free)
            # continue

            # try:
            gof = None
            r = self.fit(image, self._static_grid, **kws)

            # except Exception as err:
            #     r = None
            #     self.logger.exception('FAIL!')
            # else:
            print('RESULT:', r)
            if r is not None:
                gof = self.redchi(r, image, self._static_grid)
                if gof < bgof:
                    bgof, freedom, ibest = gof, free, i
                    print('GOF:', motley.green(gof))
                else:
                    print('GOF:', gof)
                    # finally:
            params.append(r)
            goodness.append(gof)

        # return

        # # best orders
        # ibest = np.nanargmin(goodness)

        # jb, kb = obest = tuple(order_pairs[ibest])
        # self.free[:] = False
        # self.free[j0:jb, k0:kb] = True
        # embed()

        # propagate coefficients
        self.free = freedom
        self.coeff[~self.free] = 0
        if self.depends_on is not None:
            self.depends_on.apply_bc()  # set tied coeff

        self.set_coeff(params[ibest])

        # log message
        self.logger.info('Optimal multi-order value for segment with shape %s '
                         'is y, x = %s. (%i models tested)',
                         image.shape, 'zz', i + 1)  # todo: label here?

        return params, goodness, ibest


from collections import namedtuple

# simple container for 2-component objects
yxTuple = namedtuple('yxTuple', ['y', 'x'])


def _check_orders_knots(orders, knots):
    # check consistency of knots and polynomial orders
    k = knots[1:]  # no constraint provided by 1st knot (edge)

    if isinstance(orders, float):
        orders = int(orders)

    if isinstance(orders, int):
        orders = [orders] * len(k)

    if len(orders) != len(k):
        raise ValueError('Order / knot vector size mismatch: %i, %i. '
                         'Knots should have size `len(orders) + 1`'
                         % (len(orders), len(knots)))

    # check knots are increasing order
    if np.any(np.diff(knots) < 0):
        raise ValueError('Knots should be of increasing order')

    # TODO
    # Might want to check the orders - if all 1, (straight lines) and
    # continuous = True, this is actually just a straight line, and it's
    # pointless to use a Spline2d, but will still work OK!

    return np.array(orders, int), np.asarray(knots)


from recipes.containers.dicts import AttrReadItem


class PolyNeighbours(AttrReadItem):
    """Helper class for managing neighbouring polynomials in a spline"""

    SEMANTIC_POS = ('top', 'left', 'bottom', 'right')

    def __setitem__(self, key, obj):
        if key not in self.SEMANTIC_POS:
            raise KeyError('%r is not a valid neighbour position' % key)

        if isinstance(obj, PPoly2D):
            raise TypeError(
                    'Neighbouring polynomials should be `None` or `PPoly2D`'
                    ' instances, not %r.' % obj.__class__.__name__)
        #
        super().__setitem__(key, obj)


class NullTransform(object):
    def __call__(self, grid):
        return grid


class DomainTransformMixin(object):
    _domain_transform = NullTransform

    @property
    def domain_transform(self):
        return self._domain_transform

    @domain_transform.setter
    def domain_transform(self, trans):
        self.set_domain_transform(trans)

    def set_domain_transform(self, trans):
        if not callable(trans):
            raise ValueError('transform should be callable')
        self._domain_transform = trans


class PPoly2D_v2(Poly2D, DomainTransformMixin):  #
    """
    Two-dimensional polynomial object that shares a boundary with
    another neighbouring 2d polynomial.

    Automatically update the coefficient matrix for dependent polynomials.

    Dependent polys have restricted parameter spaces. These constraints are
    imposed by (optional) continuity and smoothness conditions on the boundary
    between polynomials.
    """

    def __init__(self, orders, domain, smooth=True, continuous=True):
        """
        Initialize with 2-tuple of polynomial multi-order.


        There are up to four
        possible neighbours that depend on this polynomial.

        Parameters
        ----------
        orders: tuple of int
            The
        """

        # init parent
        Poly2D.__init__(self, orders)

        # neighbours
        self.neighbours = PolyNeighbours()
        self.depends_on = []
        self.domain = np.array(domain)
        # self._domain_mask = None
        # self.origin = 0 # np.array([[0], [0]])
        # self.scale = 1

        # set boundary conditions flags
        self.continuous = bool(continuous)
        self.smooth = bool(smooth) or self.continuous  # smooth if continuous
        self._tied_rows_cols = [0, 0]

        # set parameter freedom
        self.set_freedoms()

    def __call__(self, p, grid=None, out=None):

        p = self._check_params(p)

        # print('call starts', grid)

        if grid is None:
            # get static grid if available.  Assume here internal grid is
            # already correct domain and form for `eval`.  Checks done inside
            # `_check_grid`
            grid = self._check_grid(grid)
            domain_mask = self._domain_mask
        elif grid is not self.grid:
            # external grid provided.  Select domain.
            grid, domain_mask = self.get_domain_mask(grid)
            grid = self.domain_transform(grid)

        # print('delegate', grid)
        y = super().__call__(p, grid)

        if out is not None:
            out[domain_mask] = y
        return y

    def _checks(self, p, grid, *args, **kws):
        return super()._checks(p, grid, *args, **kws)

    def set_grid(self, grid):
        # transform domain
        self._domain_mask = m = self.get_domain_slice(grid)
        Poly2D.set_grid(self, self.domain_transform(grid[m]))

    def get_domain_mask(self, grid):
        # isolate points within the domain of this model from an arbitrary grid
        # of points.

        # The operators used below to check which points fall within the
        # domain of the function, depend on the position of the polynomial in
        # the spline layout.

        # The convention used here is that models are defined on the interval
        # that is closed at the bottom/left and open at the top/right.
        # This doesn't really matter for the case of continuous,
        # smooth polys, but does matter otherwise since the value of the
        # neighbouring polys and their derivatives will differ at the boundary

        if self.domain is None:
            return ...

        lo, hi = self.domain[(None,) * (grid.ndim - 1)].T
        # noinspection PyUnresolvedReferences
        return (lo <= grid) & (grid < hi)  # .all(0)

        # return m
        # return (op.le(lo, grid) & op.lt(grid, hi)).all(0)

    def get_domain_slice(self, grid):
        """
        For an arbitrary ordered grid, select the rectangular sub-region for
        which the function is defined. Return the tuple of slice objects that
        can be used to get the associated grid points
        """
        # This function returns the domain mask as a tuple of slices so that
        # the grid remains shaped
        if self.domain is None:
            return ...

        # lo, hi = self.domain[(None,) * (grid.ndim - 1)].T
        # b = ((lo <= grid) & (grid <= hi))
        # b = list()
        # fixme, if you know the grid is uniform and sorted this can be done
        #  faster

        return (...,) + tuple(
                slice(*(np.nonzero(b.any(int(not i)))[0][[0, -1]] + [0, 1]))
                for i, b in enumerate(self.get_domain_mask(grid)))

    def set_coeff(self, p):
        super().set_coeff(p)
        self.apply_bc()

    def apply_bc(self):
        """
        Update coefficient matrices of neighbours applying the (optional)
        smoothness and continuity conditions
        """

        # get internal domain
        # y1, x1 = self.domain.ptp(1).astype(float)  #
        y1, x1 = self.domain_transform(self.domain[:, 1, None])

        # print('apply_bc')
        # print('y1, x1', y1, x1)

        c = self.coeff
        for k in range(self.continuous + self.smooth):
            # powers xⁱ·yʲ
            j, i = np.ogrid[tuple(map(slice, (k, k), self.n_coeff))]

            for pos, poly in self.neighbours.items():
                # set coefficients for x-only terms
                if pos == 'bottom':
                    poly.coeff[k] = c[k]

                # set coefficients for y-only terms
                if pos == 'left':
                    poly.coeff[:, k] = c[:, k]

                if pos == 'top':  # x-only terms
                    poly.coeff[k] = np.sum(
                            c[k:] * (j ** k) * y1 ** (j - k), 0)

                if pos == 'right':  # y-only terms
                    poly.coeff[:, k] = np.sum(
                            c[:, k:] * (i ** k) * x1 ** (i - k), 1)

    def set_neighbours(self, **n):
        for s, o in n.items():
            self.neighbours[s] = o

            # set neighbours as object attributes
            o.depends_on.append(self)
            # setattr(self, s, o)
            l = self.continuous + self.smooth
            if s in ['left', 'right']:
                o._tied_rows_cols[1] = l  # (slice(None), slice(l, None))
            else:
                o._tied_rows_cols[0] = l  # (slice(l, None), slice(None))

        # set the tied coefficients in neighbouring polys
        self.apply_bc()
        self.set_freedoms()

    def set_freedoms(self):
        n_tied = self.smooth + self.continuous
        for pos, poly in self.neighbours.items():
            for i in range(n_tied):
                # set coefficients for x-only terms
                if pos in ('top', 'bottom'):
                    poly.free[i] = False

                # set coefficients for y-only terms
                if pos in ('left', 'right'):
                    poly.free[:, i] = False

    def order_search(self, data, std=None, max_dof=np.inf,
                     gof=('aic', 'bic', 'redchi'), report=False, **opt_kws):
        """

        Parameters
        ----------
        data
        std
        max_dof
        gof
        report
        opt_kws

        Returns
        -------

        """
        for g in gof:
            assert hasattr(self, g), (f'Goodness-of-fit statistic {g} not '
                                      'understood.')

        # n = np.prod(self.n_coeff - self._tied_rows_cols)
        results = []  # np.ma.masked_all((n,) + self.coeff.shape)
        free = []
        goodness = defaultdict(list)  # np.ma.masked_all((n, len(gof)))
        selector = gof[0]
        # reps = []

        # clean slate
        self.free[:] = False

        best = np.inf
        ibest = None
        cbest = np.zeros(self.n_coeff)

        i = 0
        for i, freedom in enumerate(ParamSubspaces(self).iter_blocks(max_dof)):
            # print(i, free.sum() )
            # free some params
            # ensure the boundary conditions are met
            self.coeff[~freedom] = 0
            self.free = freedom
            for parent in self.depends_on:
                parent.apply_bc()

            # start at current best optimum
            p0 = cbest[freedom]

            # fit
            # self.logger.info('%s: \n%s', i, reps[-1])
            r = self.fit(data, self.grid, p0=p0, **opt_kws)
            self.logger.debug('r = %s', r)

            free.append(freedom.copy())
            results.append(r if r is None else self.coeff[self.free])
            for g in gof:
                if r is None:
                    gg = np.inf
                else:
                    gg = getattr(self, g)(r, data, self.grid, std)
                goodness[g].append(gg)

            # use AIC for gof test
            g = goodness[selector][-1]
            if g < best:
                best, ibest = g, i
                cbest[freedom] = r

        # propagate coefficients
        self.free = free[ibest]
        self.coeff[~self.free] = 0
        # ensure the boundary conditions are met
        for parent in self.depends_on:
            parent.apply_bc()

        # set best fit params
        self.set_coeff(results[ibest])

        # log message
        self.logger.info(
                f'Best fit model to data with shape {data.shape}: '
                f'{self!r} (dof={self.dof})\n'
                f'{self!s}\n '
                f'({i + 1} models tested)'
        )

        if report:
            self.logger.info(
                    f'\n{report_order_search(results, free, goodness)}')  #
            # top=5

        return results, np.array(free), goodness, ibest


from collections import defaultdict


def report_order_search(results, free, goodness, top=None):
    from motley.table import Table
    from obstools.image.segmentation import SegmentedImage

    gof = list(goodness.keys())
    goodness = np.array(list(goodness.values()))
    aic = goodness[0]  # 'aic'
    idx = aic.argsort()[:top]
    idx = [i for i in idx if results[i] is not None]

    goodness = goodness[:, idx]
    aic = aic[idx]

    dofs = np.empty(len(idx))
    blocks = []
    for i, j in enumerate(idx):
        f = free[j]
        dofs[i] = f.sum()

        blocks.append(
                SegmentedImage(f).format_term(show_labels=False,
                                              origin=1, frame=False,
                                              cmap='nipy_spectral'))

    # relative likelihood
    pref = np.exp((aic.min() - aic) / 2) * 100

    footnote = None
    if top:
        footnote = f'{len(results) - top} more nested models...'

    goodness /= goodness.max(1, keepdims=True)
    return Table.from_columns(dofs, blocks, *goodness, pref,
                              col_headers=np.r_[
                                  ['dof', 'block', *gof, 'Likelihood\nratio']],
                              precision=3, minimalist=True, order='r',
                              # hlines=...,
                              # highlight={ibest: 'g'}
                              footnote=footnote
                              )


def print_blocks(blocks, h=None):
    import motley
    from obstools.image.segmentation import SegmentedImage

    tables = [SegmentedImage(b).format_term(show_labels=False, origin=1,
                                            cmap='Greens')
              for b in blocks]
    if h is not None:
        tables[h] = motley.hue(tables[h])
    if len(tables):
        return print(motley.hstack(tables))
    else:
        print('EMTPY!')


class ParamSubspaces(object):
    """
    Parameter subspace iteration helper for PPoly2D
    """

    def __init__(self, pp):
        self.y0, self.x0 = pp._tied_rows_cols
        self.y1, self.x1 = pp.n_coeff

    def runner(self, free, itr, copy=False):
        yields = np.copy if copy else _echo
        # use this to prevent nested conditionals :)
        for i, j in itr:
            free[i, j] = True
            yield yields(free)

    def _get_blocks(self, iters, max_dof=np.inf, collection=None, print_=False):
        # aggregate
        yields = _echo
        if print_:
            from obstools.image.segmentation import SegmentedImage

            def yields(free):
                # representation of parameter space freedom
                print(SegmentedImage(free).format_term(show_labels=False,
                                                       origin=1,
                                                       cmap='nipy_spectral'))
                return free

        #
        for n, f in enumerate(mit.roundrobin(*iters), 1):
            dof = f.sum()
            if dof > max_dof:
                continue
            if collection is not None:
                collection[dof].append(f.copy())
            yield yields(f)

    def iter_blocks(self, max_dof=np.inf):
        yield from self._get_blocks(self.get_iters(), max_dof)

    def get_blocks(self, max_dof=np.inf):
        return list(self._get_blocks(self.get_iters(), max_dof))

    def get_iters(self):
        # create iterables

        shape = (self.y1, self.x1)
        free = np.zeros((self.x1 - self.x0,) + shape, bool)
        # first those for rectangular subspaces
        iters = []
        for j, z in zip(range(self.x0, self.x1), free):
            iters.append(self.runner(z, self.rect(j, self.y0, self.x0)))

        # now diagonals / edges
        free = np.zeros((3,) + shape, bool)
        free[:, self.y0, self.x0] = True
        free[-1, self.y0:self.y0 + 2, self.x0:self.x0 + 2] = True

        idx_itrs = (self.edges(1, 1),
                    self.dia(1, 1),
                    itt.islice(self.edges_dia(), 2, None)
                    )
        iters.extend(map(self.runner, free, idx_itrs))
        return iters

    def _gen_subspaces(self, max_dof=np.inf, collection=None):
        yield from self._get_blocks(self.get_iters(), max_dof, collection)

    def get_subspaces(self, max_dof=np.inf):
        subspaces = defaultdict(list)
        for _ in self._gen_subspaces(max_dof, subspaces):
            pass
        return subspaces

    def row(self, i, j=0):
        return zip(itt.repeat(i), range(self.x0 + j, self.x1))

    def col(self, j, i=0):
        return zip(range(self.y0 + i, self.y1), itt.repeat(j))

    def rect(self, j, i0=0, j0=0):
        for i in range(i0, self.y1):
            yield (slice(i0, i + 1), slice(j0, j + 1))

    def edges(self, i0=0, j0=0):
        for idx in zip(self.row(self.y0, i0), self.col(self.x0, j0)):
            yield tuple(zip(*idx))

    def dia(self, i0=0, j0=0):
        return zip(range(self.y0 + i0, self.y1), range(self.x0 + j0, self.x1))

    def edges_dia(self, i0=0, j0=0):
        for idx in zip(self.row(self.y0), self.col(self.x0), self.dia(i0, j0)):
            yield tuple(zip(*idx))
