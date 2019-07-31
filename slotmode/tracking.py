import numpy as np
from obstools.phot.segmentation import SegmentationHelper
from obstools.phot.tracking import StarTracker, GlobalSegmentation

from . import get_bad_pixel_mask
from .modelling.image import FrameTransferBleed, PhotonBleed


class SlotModeGlobalSegmentation(GlobalSegmentation):
    def for_offset(self, xy_offsets, shape, type_=SegmentationHelper):
        seg = super().for_offset(xy_offsets, shape, type_)






class SlotModeTracker(StarTracker):
    PHOTON_BLEED_THRESH = 3.e4
    PHOTON_BLEED_WIDTH = 8  # TODO: kill ??

    @classmethod
    def from_images(cls, images, mask=None, required_positional_accuracy=0.5,
                    centre_distance_max=1, f_detect_measure=0.5,
                    f_detect_merge=0.2, post_merge_dilate=1, flux_sort=True,
                    ft_bleed_threshold=PHOTON_BLEED_THRESH,
                    ft_bleed_width=PHOTON_BLEED_WIDTH,
                    worker_pool=None, report=None, plot=False, **detect_kws
                    ):
        tracker, xy, centres, xy_offsets, counts, counts_med = \
            super().from_images(images, mask, required_positional_accuracy,
                                centre_distance_max, f_detect_measure,
                                f_detect_merge, post_merge_dilate, flux_sort,
                                worker_pool, report, plot, **detect_kws)

        # save object mask before adding streaks
        tracker.masks['sources'] = tracker.masks.all
        tracker.add_photon_bleed(centres, counts_med, ft_bleed_threshold,
                                 ft_bleed_width)

        # remember extremal offsets from construction
        tracker.start_max = xy_offsets.max(0) - tracker.zero_point

        return tracker, xy, centres, xy_offsets, counts, counts_med

    def add_photon_bleed(self, centres, counts, threshold, width):
        # add regions affected by photon bleeding
        bright,  = np.where(counts > threshold)

        loc = centres[bright, 1] - self.segm.zero_point[1]
        _, labels_streaks = PhotonBleed.adapt_segments(
                self.segm, loc=loc, width=width)

        self.groups['bright'] = bright + 1
        self.groups['streaks'] = labels_streaks

    def get_segments(self, start, shape):
        seg = super().get_segments(start, shape)

        # dynamically add photon bleed regions.  This has to be done
        # dynamically since large camera drift (beyond the maximal xy offsets
        # that the GlobalSegmentation were constructed with) will cut the
        # photon bleed regions before the edges of the image, and will
        # therefore not be correct.

        # check if this y offset is extremal
        if np.any(start[0] < 0):
            ''

        if start >


        seg, labels_streaks = PhotonBleed.adapt_segments(
                self.segm, loc=loc, width=width, copy=True)

        return seg

    # snr_cut = 1.5
    #
    # @classmethod
    # def from_image(cls, image, bgmodel=None, snr=3., npixels=7,
    #                edge_cutoffs=None, deblend=False, dilate=1, flux_sort=True,
    #                bad_pixel_mask=None, streak_threshold=1e3):
    #     #
    #
    #     # get the bad pixel mask
    #     if bad_pixel_mask is None:
    #         bad_pixel_mask = get_bad_pixel_mask(image)
    #
    #     # create edge mask for sky photometry
    #     if edge_cutoffs is None:
    #         edge_cutoffs = get_edge_cutoffs(image)
    #
    #     # init parent class
    #     tracker, p0bg = super(SlotModeTracker, cls).from_image(
    #             image, bgmodel, snr, npixels, edge_cutoffs, deblend, dilate,
    #             flux_sort, bad_pixel_mask)
    #
    #     # mask streaks
    #     # tracker.bright = tracker.bright_star_labels(image)
    #     # tracker.streaks = False
    #     # tracker.streaks = tracker.get_streakmasks(
    #     #         tracker.rcoo[tracker.bright - 1])
    #     return tracker, p0bg

    # @classmethod
    # def from_measurements(cls, segs, xy, counts, f_accept=0.2,
    #                       post_merge_dilate=1, required_positional_accuracy=0.5,
    #                       masked_ratio_max=0.9, bad_pixel_mask=None,
    #                       ft_bleed_threshold=FT_BLEED_THRESH_COUNTS,
    #                       ft_bleed_width=FT_BLEED_WIDTH):
    #
    #     # tracker, xy, centres, xy_offsets, counts, counts_med = \
    #     return \
    #         super().from_measurements(segs, xy, counts, f_accept,
    #                                   post_merge_dilate,
    #                                   required_positional_accuracy,
    #                                   masked_ratio_max,
    #                                   bad_pixel_mask)
    #
    #     # save object mask before adding streaks
    #     tracker.masks['objects'] = tracker.masks.all  #
    #     tracker.add_ft_regions(centres, counts_med, ft_bleed_threshold,
    #                            ft_bleed_width)
    #
    #     return tracker, xy, centres, xy_offsets, counts, counts_med

    def add_ft_regions(self, centres, counts, ft_bleed_threshold,
                       ft_bleed_width):
        # add regions affected by frame transfer bleeding
        bright = np.where(counts > ft_bleed_threshold)[0]

        # # HACK for if positions not accurate enough # FIXME
        # if not len(self.use_labels):
        #     self.use_labels = bright

        loc = centres[bright, 1] - self.segm.zero_point[1]
        _, labels_streaks = FrameTransferBleed.adapt_segments(
                self.segm, loc=loc, width=ft_bleed_width)

        self.groups['bright'] = bright + 1
        self.groups['streaks'] = labels_streaks

    def bright_star_labels(self, image, flx_thresh=1e3):
        flx = self.segm.flux(image)
        w, = np.where(flx > flx_thresh)
        w = np.setdiff1d(w, self.ignore_labels)
        coo = self.rcoo[w]
        bsl = self.segm.data[tuple(coo.round(0).astype(int).T)]
        return bsl

    # def __init__(self, coords, segm, label_groups=None, use_labels=None,
    #              bad_pixel_mask=None, edge_cutoffs=None, reference_index=0,
    #              background_estimator=np.ma.median, weights=None,
    #              update_rvec_every=100, update_weights_every=10):
    #     #
    #     super().__init__(coords, segm, label_groups, use_labels, bad_pixel_mask,
    #                      edge_cutoffs, reference_index, background_estimator,
    #                      weights, update_rvec_every, update_weights_every)
    #     # self.mask_streaks = None

    # def prep_masks_phot(self, labels=None, ft_bleed_width=FT_BLEED_WIDTH):
    #
    #     super().prep_masks_phot(labels)
    #
    #     # seg_streak, lbls = FrameTransferBleed.adapt_segments(
    #     #         self.segm, width=ft_bleed_width, copy=True)
    #     # self.mask_streaks = seg_streak.to_bool()
    #     not_streaks = ~self.mask_streaks
    #     self.masks_phot &= not_streaks
    #     self.masks_sky &= not_streaks
    #
    #     return self.masks_phot, self.masks_sky, self.mask_all  # ,
    # self.streak_mask

    # def get_position_residuals(self, coms):
    #
    #     xy_offsets = (coms - self.coords).mean(1)
    #     position_residuals = coms - xy_offsets[:, None] - centres
    #     σ_pos = position_residuals.std(0)

    # def __call__(self, image, mask=None):
    #
    #     # mask = self.bad_pixel_mask | self.streaks
    #     com = StarTracker.__call__(self, image, mask)
    #     # return com
    #
    #     # update streak mask
    #     # self.streaks = self.get_streakmasks(com[:len(self.bright)])
    #     return com

    # def mask_segments(self, image, mask=None):
    #     imbg = StarTracker.mask_segments(self, image, mask)
    #     # imbg.mask |= self.streaks
    #     return imbg

    # def get_streakmasks(self, coo, width=6):
    #     # Mask streaks
    #
    #     w = np.multiply(width / 2, [-1, 1])
    #     strkRng = coo[:, None, 1] + w
    #     strkSlc = map(slice, *strkRng.round(0).astype(int).T)
    #
    #     strkMask = np.zeros(self.segm.data.shape, bool)
    #     for sl in strkSlc:
    #         strkMask[:, sl] = True
    #
    #     return strkMask

    # return grid, data arrays
    # return [(np.where(m), image[m]) for m in reg]

    # def get_edgemask(self, xlow=0, xhi=None, ylow=0, yhi=None):
    #     """Edge mask"""
    #     # can create by checking derivative of vignette model
    #     # return

    #     def add_streakmasks(self, image, flx_thresh=2.5e3, width=6):

    #         strkMask = self.get_streakmasks(image, flx_thresh, width)
    #         labelled, nobj = ndimage.label(strkMask)
    #         novr = strkMask & (self.segm.data != 0) # non-overlapping streaks
    #         labelled[novr] = 0

    #         streak_labels = self.segm.add_segments(labelled)
    #         self.ignore_labels = np.union1d(self.ignore_labels, streak_labels)
    #         return streak_labels

    # def refine(self, image, bgmodel=None, use_edge_mask=True, snr=3., npixels=6,
    #            edge_cutoff=0, deblend=False, flux_sort=False, dilate=1):
    #     # mask bad pixels / stars for bg fit
    #     # TODO: combine use_edge_mask with edge_cutoff
    #     # TODO: incorporate in from_image method??
    #
    #     # mask stars, streaks, bad pix
    #     imm = self.segm.mask_segments(image)  # stars masked
    #     # starmask = imm.mask.copy()
    #     use_edge_mask = use_edge_mask and self.edge_mask is not None
    #     if self.bad_pixel_mask is not None:
    #         imm.mask |= self.bad_pixel_mask
    #     if use_edge_mask:
    #         imm.mask |= self.edge_mask
    #     # if self.streaks is not None:
    #     #     imm.mask |= self.streaks
    #
    #     if bgmodel:
    #         # fit background
    #         results = bgmodel.fit(np.ma.array(imm, mask=self.streaks))  # = (ry, rx)
    #         # input image with background model subtracted
    #         im_bgs = bgmodel.residuals(results, image)
    #
    #         # do detection run without streak masks since there may be faint but
    #         # detectable stars partly in the streaks
    #         resi_sm = np.ma.array(im_bgs, mask=imm.mask)
    #         # prepare frame: fill bright stars
    #         resif = np.ma.filled(resi_sm, np.ma.median(resi_sm))
    #     else:
    #         resif = np.ma.filled(imm, np.ma.median(imm))
    #         im_bgs = image
    #
    #     # detect faint stars
    #     new_segm = SegmentationHelper.from_image(resif, snr, npixels,
    #                                              edge_cutoff, deblend,
    #                                              flux_sort, dilate)
    #     # since we dilated the detection masks, we may now be overlapping with
    #     # previous detections. Remove overlapping pixels here
    #     if dilate:
    #         overlap = new_segm.data.astype(bool) & self.segm.data.astype(bool)
    #         new_segm.data[overlap] = 0
    #
    #     # remove detections with small areas:
    #     # l = tracker.segm.areas[1:] < npixels
    #
    #     # add new detections
    #     faint_labels = self.segm.add_segments(new_segm)
    #     # calculate centroids of new detections
    #     if len(faint_labels):
    #         coords = self.segm.com_bg(im_bgs, mask=self.bad_pixel_mask)
    #         # ignore the faint stars for subsequent frame shift calculations
    #         self.ignore_labels = np.union1d(self.ignore_labels, faint_labels)
    #
    #         # update coords
    #         self.yx0 = coords[self.ir]
    #         self.rvec = coords - self.yx0
    #
    #         # set bg model
    #         self.bgmodel = bgmodel
    #
    #         return faint_labels, results, im_bgs
    #     else:
    #         return faint_labels, None, image
