# standard library
import multiprocessing as mp
import os
import socket
import sys
from multiprocessing.managers import SyncManager  # , NamespaceProxy
from pathlib import Path

import logging.handlers

# ===============================================================================
# Check input file | doing this before all the slow imports
fitsfile = sys.argv[1]
fitspath = Path(fitsfile)
path = fitspath.parent
if not fitspath.exists():
    raise IOError('File does not exist: %s' % fitsfile)

# execution time stamps
from motley.profiler.timers import Chrono

chrono = Chrono()
# TODO: option for active reporting; inherit from LoggingMixin, make Singleton
chrono.mark('start')

# ===============================================================================
# related third party libs

import numpy as np
from scipy import ndimage
from joblib.pool import MemmappingPool  # Parallel, delayed
from addict.addict import Dict
import more_itertools as mit

chrono.mark('Imports: 3rd party')

# local application libs
from obstools.phot.utils import rand_median
from obstools.phot.proc import FrameProcessor
from obstools.modelling.utils import load_memmap_nans

from recipes.interactive import is_interactive
from recipes.parallel.synced import SyncedCounter

from salticam import slotmode
from  salticam.slotmode.image import SlotBackground

# from slotmode import get_bad_pixel_mask
# * #StarTracker, SegmentationHelper, GraphicalStarTracker
#
# from obstools.modelling.core import *
# from obstools.fastfits import FitsCube
# from obstools.modelling.bg import Poly2D
# from obstools.modelling.psf.models_lm import EllipticalGaussianPSF
from obstools.phot import log  # listener_process, worker_init, logging_config
from obstools.phot.proc import TaskExecutor
from obstools.phot.tracking.core import SegmentationHelper, SlotModeTracker, \
    check_image_drift

# from motley.printers import func2str
from graphical.imagine import ImageDisplay

chrono.mark('Import: local libs')


# ===============================================================================

# class SyncedProgressLogger(ProgressLogger):
#     def __init__(self, precision=2, width=None, symbol='=', align='^', sides='|',
#                  logname='progress'):
#         ProgressLogger.__init__(self, precision, width, symbol, align, sides, logname)
#         self.counter = SyncedCounter()


def display(image, title, **kws):
    if isinstance(image, SegmentationHelper):
        im = image.display(**kws)
    else:
        im = ImageDisplay(image, **kws)

    im.ax.set_title(title)

    if args.live:
        idisplay(im.figure)
    return im


# TODO: colourful logs - like daquiry / technicolor
# TODO: python style syntax highlighting for exceptions in logs ??

__version__ = 3.14519

# class MyManager(Manager):
#     pass

SyncManager.register('Counter', SyncedCounter)


# MyManager.register('ProgressBar', SyncedProgressLogger)

def Manager():
    m = SyncManager()
    m.start()
    return m


def task(size, max_fail=None):
    # a little task factory
    counter = manager.Counter()
    fail_counter = manager.Counter()
    return TaskExecutor(size, counter, fail_counter, max_fail)


if __name__ == '__main__':
    # def main():
    chrono.mark('Main start')

    import argparse

    ncpus = os.cpu_count()

    # parse command line args
    parser = argparse.ArgumentParser(
            'phot',  # fromfile_prefix_chars='@',
            description='Parallelized generic time-series photometry routines')
    parser.add_argument('fitsfile', type=str,  # type=FitsCube,
                        help='filename of fits data cube to process.')
    parser.add_argument('-ch', '--channel', type=int,  # type=FitsCube,
                        help='amplifier channel')
    # TODO: process many files at once
    parser.add_argument(
            '-n', '--subset', nargs='*', type=int,
            help='Data subset to load. Useful for testing/debugging. If not given, '
                 'entire cube will be processed. If a single integer, first `n` '
                 'frames will be processed. If 2 integers `n`, `m`, all frames '
                 'starting at `n` and ending at `m-1` will be processed.')
    parser.add_argument(
            '-j', '--nprocesses', type=int, default=ncpus,
            help='Number of worker processes running concurrently in the pool.'
                 'Default is the value returned by `os.cpu_count()`: %i.'
                 % ncpus)
    parser.add_argument(
            '-k', '--clobber', action='store_true',
            help='Whether to resume computation, or start afresh. Note that the '
                 'frames specified by the `-n` argument will be recomputed if '
                 'overlapping with previous computation irrespective of the '
                 'value of `--clobber`.')

    parser.add_argument(
            '-a', '--apertures', default='circular',
            choices=['c', 'cir', 'circle', 'circular',
                     'e', 'ell', 'ellipse', 'elliptical'],
            help='Aperture specification')
    # TODO: option for opt

    # plotting
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--plot', action='store_true', default=True,
                       help='Do plots')
    group.add_argument('--no-plots', dest='plot', action='store_false',
                       help="Don't do plots")

    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--version', action='version',
                        version='%(prog)s %s')
    args = parser.parse_args(sys.argv[1:])

    # add some args manually
    args.live = is_interactive()

    # data
    cube = np.lib.format.open_memmap(args.fitsfile)  # TODO: paths.input
    ch = args.channel
    cube = cube[:, ch]

    # check / resolve options
    clobber = args.clobber
    # subset of frames for compute
    if args.subset is None:
        subset = (0, len(cube))
    elif len(args.subset) == 1:
        subset = (0, min(args.subset[0], len(cube)))
    elif len(args.subset) == 2:
        subset = args.subset
    else:
        raise ValueError('Invalid subset: %s' % args.subset)
    # number of frames to process
    subsize = np.ptp(subset)

    # ===============================================================================
    # Decide how to log based on where we're running

    if socket.gethostname().startswith('mensa'):
        plot_lightcurves = plot_diagnostics = False
        print_progress = False
        log_progress = True
    else:
        plot_lightcurves = plot_diagnostics = args.plot
        print_progress = True
        log_progress = False

    if args.live:  # turn off logging when running interactively (debug)
        from recipes.interactive import exit_register
        from IPython.display import display as idisplay

        log_progress = print_progress = False
        monitor_mem = False
        monitor_cpu = False
        # monitor_qs = False
    else:
        from atexit import register as exit_register
        from recipes.io.utils import WarningTraceback

        # check_mem = True            # prevent execution if not enough memory available
        monitor_mem = False  # True
        monitor_cpu = False  # True  # True
        # monitor_qs = True  # False#

        # setup warnings to print full traceback
        wtb = WarningTraceback()

    logging.captureWarnings(True)

    # print section timing report at the end
    exit_register(chrono.report)

    # ===============================================================================
    # create folder output data to be saved
    resultsPath = fitspath.with_suffix('.ch%i.phot' % ch)
    # create logging directory
    logPath = resultsPath / 'logs'  # TODO: paths.log
    if not logPath.exists():
        logPath.mkdir(parents=True)

    # create directory for plots
    if plot_diagnostics:
        figPath = resultsPath / 'plots'  # TODO: paths.figures
        if not figPath.exists():
            figPath.mkdir()

    # setup logging processes
    logQ = mp.Queue()  # The logging queue for workers
    # TODO: open logs in append mode if resume
    config_main, config_listener, config_worker = log.config(logPath, logQ)
    logging.config.dictConfig(config_main)
    # create log listner process
    logger = logging.getLogger()
    logger.info('Creating log listener')
    stop_logging_event = mp.Event()
    logListner = mp.Process(target=log.listener_process, name='logListener',
                            args=(logQ, stop_logging_event, config_listener))
    logListner.start()
    logger.info('Log listener active')
    #
    chrono.mark('Logging setup')

    # ===============================================================================
    # Image Processing setup

    # FIXME: any Exception that happens below will stall the log listner indefinitely

    n = len(cube)
    ishape = cube.shape[-2:]
    frame0 = cube[0]

    # bad pixels
    # ----------
    bad_pixel_mask = slotmode.get_bad_pixel_mask(frame0, args.channel + 1)
    # TODO: handle bad_pixels as negative label in SegmentationHelper ???

    # flat fielding
    # -------------
    # Some of the bad pixels can be flat fielded out
    flatpath = fitspath.with_suffix('.flat.npz')  # TODO: paths.calib.flat
    if flatpath.exists():
        logger.info('Loading flat field image from %r', flatpath.name)
        flat = np.load(flatpath)['flat']
        flat_flat = flat[flat != 1]
    else:
        'still experimental'
        # construct flat field for known bad pixels by computing the ratio
        # between the median pixel value and the median of it's neighbours
        # across multiple (default 1000) frames
        # from slotmode import neighbourhood_median
        #
        # flat_cols = [100]
        # flat_ranges = [[14, (130, 136)],
        #                [15, (136, 140)]]
        # flat_mask = slotmode._mask_from_indices(ishape, None, flat_cols,
        #                                         None, flat_ranges)
        # flat = neighbourhood_median(cube, flat_mask)  #
        # flat_flat = flat[flat_mask]

    #
    # calib = (None, flat)
    calib = (None, None)

    # extract flat field from sky
    # flat = slotmode.make_flat(cube[2136:])
    # image /= flat
    # TODO: save this somewhere so it's quicker to resume
    # TODO: same for all classes here

    # create the global background model
    # mdlBG = Poly2D(1, 1)
    # p0bg = np.ones(mdlBG.npar)  # [np.ones(m.nfree) for m in mdlBG.models]
    # edge_cutoffs = slotmode.get_edge_cutoffs(image)
    # border = make_border_mask(image, edge_cutoffs)
    # mdlBG.set_mask(border)

    # for SLOTMODE: fit median cross sections with piecewise polynomial
    # orders_x, orders_y = (1, 5), (4, 1, 5)
    # bpx = np.multiply((0, 0.94, 1), image.shape[1])
    # bpy = np.multiply((0, 0.24, 0.84, 1), image.shape[0])
    # #orders_y, bpy = (3, 1, 5), (0, 3.5, 17, image.shape[0]) # 0.15, 0.7
    # mdlBG = Vignette2DCross(orders_x, bpx, orders_y, bpy)

    # TODO: try fit for the breakpoints ??
    # orders_x, orders_y = (5, 1), (4, 1, 5)
    # bpx = np.multiply((0, 0.94, 1), image.shape[1])
    # sy, sx = ishape
    # # orders = orders_x, orders_y = (1, 3), (3, 1, 3)     # channel 1
    # orders = orders_x, orders_y = (1, 3), (1, 1, 1)
    # breaks = (0, 162, sx), (0, 3, 17, sy)  # 6x6
    # smoothness = (True, False)

    # This for Amp3 (channel 2) binning 3x3
    sy, sx = ishape
    orders = orders_x, orders_y = (5, 1), (3, 1, 3)
    breaks = (0, 10, sx), (0, 10, 38, sy)  # 3x3
    smoothness = (False, False)


    # bpy = np.multiply((0, 0.24, 0.84, 1), image.shape[0])
    # bpy = (0, 10, 37.5, sy)
    # bpy = (0, 3.5, 17.5, sy)  # 6x6
    # orders = (1, 3), (1, 1, 1)
    # breaks = (0, 162, sx), (0, 3.5, 17.5, sy)
    # breaks = bpx, bpy

    # create sample image
    image = rand_median(cube, 5, 100)  # / flat

    # load SementationHelper from pickle
    # segmPath = fitspath.with_suffix('.segm.pkl')

    # First initialize the SlotBackground model from the image
    snr = (10, 7, 5, 3)
    npixels = (7, 5, 3)
    dilate = (5, 1)

    models = []

    mdlr, p0bg, seg_data, mask, groups = SlotBackground.from_image(
            orders, breaks, smoothness, image, bad_pixel_mask,
            snr, npixels, dilate=dilate)

    # TODO: hyperparameter search
    def hyperparameter_search(breaks):
        import itertools as itt
        # given the breakpoints, search polynomial orders
        npy, npx = np.add(map(len, breaks), 1)
        npy0 = np.ones_like(npy)
        npx0 = np.ones_like(npx)

        #
        yrngs, xrngs = (5, 3), (7, 3, 7)
        z = itt.repeat(0)
        ybounds = zip(z, yrngs)
        xbounds = zip(z, xrngs)
        smbounds = (0, 1)

        p0 = np.r_[npy0, npx0, 0 , 0]  # last too zeros for smoothness
        # minimize(hyper_objective, p0, (data, ))

    def hyper_objective(p0, data):
        # init the model and get best fit residuals
        'todo'


    # check for non-convergence
    # failed = [np.isnan(x).any() for x in [p0bg.bleed, p0bg.vignette.y,
    #                                       p0bg.vignette.x]]
    failed = np.isnan(p0bg.bleed).any()
    if failed:
        # replace nans with zero
        p0bg.bleed = 0

    # segmentation (this one only contains the stars, not the frame transfer
    # bleed masks
    segm = SegmentationHelper(seg_data)

    # everything in here will be used for sky
    notSky = seg_data.astype(bool) | bad_pixel_mask
    skyRegion = ~notSky
    skyImageMasked = np.ma.MaskedArray(image, notSky)

    # residuals (for sample image)
    resi = mdlr.residuals(p0bg, image)
    # masked residuals
    skyResiMasked = np.ma.MaskedArray(resi, ~skyRegion)

    # save_pickle(segmPath, segm)  # TODO: paths.segm

    # init the tracker
    # select here groups by snr to do photometry
    use_labels = np.hstack(groups[:2])
    com = segm.com_bg(resi, use_labels)
    # slices = Slices(segm)

    # use_labels = groups[0]
    tracker = SlotModeTracker(com, seg_data, groups, use_labels, bad_pixel_mask)

    # display
    displayed = Dict()
    if args.plot:
        # initial diagnostic images (for the modelled sample image)
        displayed.image = display(image, 'Sample image')
        displayed.segmentation = display(segm, 'Segmentation')
        displayed.residual = display(skyResiMasked, 'Sky Residuals')
        # add CoM ticks
        displayed.residual.ax.plot(*com[:, ::-1].T, 'rx', ms=3)
        # plot cross sections for bg fit
        figs_yx = mdlr.plot_fit_results(skyImageMasked, p0bg)
        for w, fig in zip('yx', figs_yx):
            obj = idisplay(fig) if args.live else fig
            displayed.cross[w] = obj

        # TODO: plot psf for different group detections + models


        #

    # raise SystemExit

    # tracker, image, p0bg = SlotModeTracker.from_cube_segment(cube, 25,
    #                                                          100,
    #                                                          mask=bad_pixel_mask,
    #                                                          bgmodel=mdlBG,
    #                                                          snr=(10, 3),
    #                                                          npixels=(7, 3),
    #                                                          dilate=(4, 3),
    #                                                          deblend=True)
    #
    # if plot_diagnostics: # thread??
    # resi = mdlBG.residuals(p0bg, image)
    # resi_masked = tracker.mask_segments(resi, bad_pixel_mask)
    # resi_clip = sigma_clipping.sigma_clip(resi_masked)

    # check fit
    # resi = mdlBG.residuals(p0bg, image)
    # ImageDisplay(tracker.segm)
    # ImageDisplay(tracker.mask_segments(resi, bad_pixel_mask))
    # mdlBG.plot_fit_results(tracker.mask_segments(image, bad_pixel_mask), p0bg)

    # estimate maximal positional shift of stars by running detection loop on
    # maximal image of 1000 frames evenly distributed across the cube
    mxshift, maxImage, segImx = check_image_drift(cube, 1000, bad_pixel_mask,
                                                   snr=5)


    # TODO: for slotmode: if stars on other amplifiers, can use these to get sky
    # variability and decorellate TS

    # TODO: set p0ap from image
    sky_width, sky_buf = 5, 0.5
    if args.apertures.startswith('c'):
        p0ap = (3,)
    else:
        p0ap = (3, 3, 0)

    # create psf models
    # models = EllipticalGaussianPSF(),
    # models = ()
    # create image modeller
    # mdlr = ImageModeller(tracker.segm, models, mdlBG,
    #                      use_labels=tracker.use_labels)

    # create object that generates the apertures from modelling results
    # cmb = AperturesFromModel(3, (8, 12))

    chrono.mark('Pre-compute')

    # ===============================================================================
    # create shared memory
    # aperture positions / radii / angles
    nstars = tracker.nsegs
    ngroups = tracker.ngroups
    naps = 1  # number of apertures per star

    # TODO: single structured file may be better??
    # TODO: paths.modelling etc..

    # create frame processor
    proc = FrameProcessor()
    # TODO: folder structure for multiple aperture types circular / elliptical
    proc.init_mem(n, nstars, ngroups, naps, resultsPath, clobber=clobber)

    # modelling results
    mdlr.init_mem(n, resultsPath / 'modelling/bg.par', clobber=clobber)
    # BG residuals
    bgResiduPath = resultsPath / 'residual'
    bgshape = (n,) + ishape
    residu = load_memmap_nans(bgResiduPath, bgshape, clobber)

    # aperture parameters
    # cmb.init_mem(n, resultsPath / 'aps.par', clobber=clobber)
    optStatPath = resultsPath / 'opt.stat'
    opt_stat = load_memmap_nans(optStatPath, (n, tracker.ngroups), clobber)

    # tracking data
    cooFile = resultsPath / 'coords'
    coords = load_memmap_nans(cooFile, (n, 2), clobber)

    chrono.mark('Memory alloc')

    # ===============================================================================
    # main compute
    # synced counter to keep track of how many jobs have been completed
    manager = Manager()

    # task executor  # there might be a better one in joblib ??
    Task = task(subsize)
    worker = Task(proc.process)

    chunks = mit.divide(args.nprocesses, range(*subset))
    chunks2 = chunks.copy()
    rngs = mit.pairwise(next(zip(*chunks2)) + (subset[1],))

    mdlr.logger.info('hello world')

    def proc_(irng, data, calib, residu, coords, opt_stat, tracker, mdlr,
              p0bg, p0ap, sky_width, sky_buf):

        #
        logger.info('Model names: %s', tuple(mdl.name for mdl in mdlr.models))
        logger.info(tuple(mdlr.values()))


        # use detections from max image to compute CoM and offset from initial
        # reference image
        i0, i1 = irng
        # labels in tracker.segm corresponding to those in segImx
        lbl = np.sort(ndimage.median(tracker.segm.data, segImx.data,
                                     segImx.labels)).astype(int)
        coo = segImx.com_bg(data[i0])

        # check bad CoM
        lbad = (np.isinf(coo) | np.isnan(coo)).any(1)
        lbad2 = ~segImx.inside_segment(coo)
        ignix = lbl[lbad | lbad2] - 1
        #
        weights = np.zeros(len(tracker.use_labels))
        weights[lbl - 1] = 1
        weights[ignix] = 0

        coo_tr = np.zeros_like(tracker.rcoo)
        coo_tr[lbl - 1] = coo
        off = tracker.update_offset(coo_tr, weights)

        logger.info('Init tracker at offset (%.3f, %.3f) for frame %i',
                    *off, i0)



        for i in range(*irng):
            worker(i, data, calib, residu, coords, opt_stat, tracker, mdlr,
                   p0bg, p0ap, sky_width, sky_buf)


    try:
        # Fork the worker processes to perform computation concurrently
        logger.info('About to fork into %i processes', args.nprocesses)

        # rng = next(rngs)
        # proc_(rng, cube, residu, coords, opt_stat,tracker, mdlr, p0bg, p0ap, sky_width, sky_buf)

        # raise SystemExit

        # NOTE: This is for testing!!
        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker, )) as pool:
        #     results = pool.map(Task(test), range(*subset))

        # NOTE: This is for tracking!!
        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker, )) as pool:
        #     results = pool.starmap(bg_sub,
        #         ((chunk, cube.data, residu, coords, tracker, mdlr)
        #             for chunk in chunks))

        # NOTE: This is for photometry!
        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker,)) as pool:
        #     results = pool.starmap(
        #             Task(proc.proc1),
        #             ((i, cube.data, residu, coords, tracker, optstat,
        #               p0ap, sky_width, sky_buf)
        #              for i in range(*subset)))

        # from IPython import embed
        # embed()
        # raise SystemExit

        # NOTE: chunked sequential mapping (doesn't work if there are frame shifts)
        # chunks = mit.divide(args.nprocesses, range(*subset))
        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker,)) as pool:
        #     results = pool.starmap(proc_,
        #             ((chunk, cube, residu, coords, opt_stat,
        #               tracker, mdlr, p0bg, p0ap,
        #               sky_width, sky_buf)
        #                 for chunk in chunks))

        #

        # raise SystemExit

        with MemmappingPool(args.nprocesses, initializer=log.worker_init,
                            initargs=(config_worker,)) as pool:
            results = pool.starmap(proc_,
                                   ((rng, cube, calib, residu, coords, opt_stat,
                                     tracker, mdlr, p0bg, p0ap, sky_width,
                                     sky_buf)
                                    for rng in rngs))

        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker, )) as pool:
        #     results = pool.starmap(
        #         proc, ((i, cube.data, coords,
        #                 tracker, mdlr, cmb,
        #                 successes, failures,
        #                 counter, prgLog)
        #                 for i in range(*subset)))

        # get frame numbers of successes and failures

    except Exception as err:
        # catch errors so we can safely shut down the listeners
        logger.exception('Exception during parallel loop.')
        plot_diagnostics = False
        plot_lightcurves = False
    else:
        # put code here that that must be executed if the try clause does
        # not raise an exception
        # The use of the else clause is better than adding additional code to
        # the try clause because it avoids accidentally catching an exception
        # that wasn’t raised by the code being protected by the try … except
        # statement.

        # Hang around for the workers to finish their work.
        pool.join()
        logger.info('Workers done')  # Logging in the parent still works
        chrono.mark('Main compute')

        # Workers all done, listening can now stop.
        logger.info('Telling listener to stop ...')
        stop_logging_event.set()
        logListner.join()
    finally:
        # A finally clause is always executed before leaving the try statement,
        # whether an exception has occurred or not.
        # any unhandled exceptions will be raised after finally clause,
        # basically just KeyboardInterrupt for now.

        # check task status
        failures = Task.report()    # FIXME:  stuck here
        # TODO: print opt failures

        chrono.mark('Process shutdown')

        # diagnostics
        if plot_diagnostics:
            # TODO: GUI
            # TODO: if interactive dock figs together
            # dock for figures
            # connect ts plots with frame display

            from obstools.phot.diagnostics import new_diagnostics, save_figures

            figs = new_diagnostics(coords, tracker.rcoo[tracker.ir],
                                   proc.Appars, opt_stat)
            save_figures(figs, figPath)

            # GUI
            from obstools.phot.gui_dev import FrameProcessorGUI

            gui = FrameProcessorGUI(cube, coords, tracker, mdlr, proc.Appars,
                                    residu, clim_every=1e6)

        if plot_lightcurves:
            from obstools.phot.diagnostics import plot_aperture_flux

            figs = plot_aperture_flux(fitspath, proc, tracker)
            save_figures(figs, figPath)

        chrono.mark('Diagnostics')
        chrono.report()  # TODO: improve report formatting

        if not args.live:
            # try:
            # from _qtconsole import qtshell  # FIXME
            # qtshell(vars())
            # except Exception as err:
            from IPython import embed

            embed()

    # with mp.Pool(10, worker_logging_init, (config_worker, )) as pool:   # , worker_logging_init, (q, logmix)
    #     results = pool.starmap(
    #         work, ((i, counter, prgLog)
    #                for i in range(n)))

    # #
    # import sys
    # from recipes.io.utils import TracePrints
    # sys.stdout = TracePrints()

    # n = 50
    # with Parallel(n_jobs=8, verbose=0, initializer=worker_logging_init,
    #               initargs=(counter, config_worker)) as parallel: #)) as parallel:#
    #     results = parallel(
    #         delayed(work)(i)#, cube.data, tracker, mdlr, counter, residu)
    #         for i in range(n))

    # sys.stdout = sys.stdout.stdout

# if __name__ == '__main__':
#     main()
