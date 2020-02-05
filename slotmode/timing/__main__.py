import numpy as np




if __name__ == '__main__':
    # load previously extracted data from file
    ext = SlotModeExtract.fileExtensionHeaderInfo
    try:
        info_file = next(out_path.glob(f'*{ext}'))
    except StopIteration as err:
        raise IOError(
                'Previously extracted header info file (extension %r) '
                'could not be found at location: %r' %
                (str(ext), out_path)) from err

    # noinspection PyTypeChecker
    info = np.rec.array(np.load(info_file, 'r'))


    # fix timestamps
    time_file = str(fx.basename.with_suffix('.time'))
    fix_timing(info, coords, time_file)

    # graphics(data, save=['png'], outpath=fig_path)
    plt.show()
