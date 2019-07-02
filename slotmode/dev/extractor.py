


class Extractor(LoggingMixin):
    def __init__(self, filelist, **kw):
        # WARNING:  Turns out this sequential extraction thing is SUPER slow!!
        # TODO: Checkout ipython-progress bar??????????
        """
        Extract frames from list of fits files.
        """
        self.start = start = kw.setdefault('start', 0)
        self.step = step = kw.setdefault('step', 1)
        self.clobber = kw.setdefault('clobber', True)

        self.keygrab = kw.setdefault('keygrab', None)
        self.headkeys = []

        self.filelist = parse.to_list(filelist)
        self.outfiles = []



        first = pyfits.open(filelist[0], memmap=True)
        pheader = first[0].header

        # Assume all files have same number of extensions and calculate total number of frames.
        Next = pheader.get('nextend')
        self.nfiles = nfiles = len(filelist)
        self.ntot = ntot = (nfiles * Next - start) // step
        self.stop = stop = kw.setdefault('stop', ntot)

        self.padwidth = len(str(ntot))  # for numerical file naming
        self.count = 0

        self.bar = kw.pop('progressbar', ProgressBar())  # initialise progress bar unless False

        # create master output file (HduList with header of first file)
        primary = pyfits.PrimaryHDU(header=pheader)
        self.master = pyfits.HDUList(primary)  # Use this as the master header for the output file
        self.mheader = self.master[0].header

        # TODO:update master header

    def get_ext_elements(self, hdulist, i):
        # map from integer extension number to (header, data) element
        return hdulist[i + 1].header, hdulist[i + 1].data
        # return header and data for extension i

    def loop(self, func):
        """
        loop through the files and extract selected frames.
        """
        start, stop, step = self.start, self.stop, self.step
        # create progressbar
        if self.bar:
            self.bar.create(stop)

        # loop through file list
        for i, f in enumerate(self.filelist):
            # first file start at start, thereafter continue sequence of steps
            start = start if i == 0 else start % step
            with pyfits.open(f, memmap=True) as hdulist:
                end = len(hdulist) - 1  # end point for each multi-extension cube

                # map to header, frame
                datamap = map(functools.partial(self.get_ext_elements, hdulist),
                              range(start, end, step))

                for j, (header, data) in enumerate(datamap):
                    # exit clause
                    if self.count >= stop:
                        return

                    func(data, header)

                    # grab header keys
                    if not self.keygrab is None:
                        self.headkeys.append([header.get(k) for k in self.keygrab])

                    # show progress
                    if self.bar:
                        self.bar.progress(self.count)

                    self.count += 1

    def _burst(self, data, header):
        numstr = '{1:0>{0}}'.format(self.padwidth, self.count)
        fn = self.naming.format(numstr)
        self.outfiles.append(fn)
        pyfits.writeto(fn, data, header)

    def _multiext(self, data, header):
        self.master.append(pyfits.ImageHDU(data, header))

    def _cube(self, data, header):
        hdu = self.master[0]
        if hdu.data is None:
            hdu.data = data
        else:
            hdu.data = np.r_['0,3', hdu.data, data]  # stack data along 0th axis
            # pyfits.HeaderDiff

    def burst(self, naming='sci{}.fits'):
        """Save files individaully.  This is probably quite inefficient (?)"""
        self.naming = naming
        self.loop(self._burst)

        return self.outfiles

    def multiext(self):
        """Save file as one big multi-extension FITS file."""
        master = self.master
        header = self.mheader

        # work
        self.loop(self._multiext)

        # update header info
        header['nextend'] = header['NSCIEXT'] = len(master)
        # optional to update NCCDS, NSCIEXT here

        # verify FITS compliance
        master.verify()

        return master

    def cube(self):
        """Save file as one big 3D FITS data cube."""
        master = self.master
        header = self.mheader

        # work
        self.loop(self._cube)

        # update header info
        header.remove('nextend')
        header.remove('NSCIEXT')

        # verify FITS complience
        master.verify()

        return master

        # @path.to_string.auto
        # def write(self, outfilename):
        #     # optional to update NCCDS, NSCIEXT here
        #
        #     master.writeto(outfilename, clobber=self.clobber)
        #     master.close()