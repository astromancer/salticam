import logging
from collections import defaultdict, OrderedDict

import motley
import numpy as np
from obstools.phot.segmentation import detect_loop
from obstools.phot.utils import shift_combine
from salticam.slotmode.modelling.image import FrameTransferBleed

PHOTON_BLEED_WIDTH = 12

def deep_detect(images, tracker, xy_offsets, indices_use, bad_pixels,
                report=True):
    # combine residuals
    mr = np.ma.array(images)
    mr.mask = bad_pixels  # BAD_PIXEL_MASK
    xy_off = xy_offsets[indices_use]
    mean_residuals = shift_combine(mr, xy_off, 'median', extend=True)
    # better statistic at edges with median

    # run deep detection on mean residuals
    PHOTON_BLEED_THRESH = 8e4   # FIXME: remove
    NPIXELS = (5, 3, 2)
    DILATE = (2, 1)
    seg_deep, groups_, info_, _, _ = \
        detect_loop(mean_residuals,
                    dilate=DILATE,
                    npixels=NPIXELS,
                    report=True)

    # merge detection groups
    groups = defaultdict(list)
    for inf, grp in zip(info_, groups_):
        groups[str(inf)].extend(grp)

    # relabel bright stars
    counts = seg_deep.count_sort(mean_residuals)
    bright = np.where(counts > PHOTON_BLEED_THRESH)[0]

    ng = 2
    g = groups_[:ng]
    labels_bright = np.hstack(g)
    last = labels_bright[-1]
    cxx = seg_deep.com(mean_residuals, labels_bright)

    if report:
        try:
            from motley.table import Table
            from recipes.pprint import numeric_array

            gn = []
            for i, k in enumerate(map(len, g)):
                gn.extend([i] * k)

            cc = numeric_array(counts[:last], precision=1, significant=3,
                               switch=4).astype('O')
            cc[bright] = list(map(motley.yellow, cc[bright]))
            tbl = Table(np.column_stack([labels_bright, gn, cxx, cc]),
                        title=(f'{last} brightest objects'
                               '\nmean residual image'),
                        col_headers=['label', 'group', 'y', 'x', 'counts'],
                        minimalist=True, align=list('<<>>>'))

            logger = logging.getLogger('root')
            logger.info('\n' + str(tbl))
        except Exception as err:
            from IPython import embed
            import traceback
            import textwrap
            embed(header=textwrap.dedent(
                """\
                Caught the following %s:
                ------ Traceback ------
                %s
                -----------------------
                Exception will be re-raised upon exiting this embedded interpreter.
                """) % (err.__class__.__name__, traceback.format_exc()))
            raise


    # return seg_deep, mean_residuals

    # xy_track = tracker.segm.com(labels=tracker.use_labels)
    # # ix_track = tuple(xy_track.round().astype(int).T)
    # ix_track = tuple(np.round(xy_track + indices_start.min(0)).astype(int).T)
    # old_labels = seg_deep.data[ix_track]
    # new_labels = np.arange(1, old_labels.max() + 1)
    # missing = set(new_labels) - set(old_labels)
    # old_labels = np.hstack([old_labels, list(missing)])

    # return seg_deep, old_labels, new_labels
    # seg_deep.relabel_many(old_labels, new_labels)

    # update tracker segments
    # todo: let tracker keep track of minimal / maximal offsets
    ranges = [np.floor(xy_off.min(0)) - np.floor(xy_offsets.min(0)),
              np.ceil(xy_off.max(0)) - np.ceil(xy_offsets.max(0)) +
              tracker.segm.shape]
    section = tuple(map(slice, *np.array(ranges, int)))

    # get new segments (tracker)
    new_seg = np.zeros_like(tracker.segm.data)
    new_seg[section] = seg_deep.data

    # add ftb regions
    new_seg, labels_streaks = FrameTransferBleed.adapt_segments(
            new_seg, bright + 1, width=PHOTON_BLEED_WIDTH)

    # update tracker
    tracker.segm.data = new_seg.data

    # get new groups
    new_groups = OrderedDict(bright=bright + 1)
    new_groups.update(groups)
    new_groups['streaks'] = labels_streaks
    tracker.groups.update(new_groups)

    return new_seg, mean_residuals, counts, tbl