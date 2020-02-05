# # test stuff
# from pathlib import Path
# path = Path('/media/Oceanus/UCT/Observing/data/sources/V2400_Oph/SALT/20140921/reduced/')
# fn0 = path / 's20140921.fits'
# fn1 = path / 'test.fits'
#
# hdu = fits.open(fn0, ignore_missing_end=True)
# hdu.pop(-1)
# hdu
#
# import more_itertools as mit
#
# s = []
# for fn in (fn0, fn1):
#     with fn.open('rb') as fp0:
#         s.append(np.frombuffer(fp0.read(), dtype='S1'))
# a, b = s
# print('sizes', a.size, b.size)
# n = min(a.size, b.size)
# l = (a[:n] != b[:n])
# w, = np.where(l)
# w.writeto(fn1, overwrite=True)