import os
from collections import defaultdict
from pprint import pprint

from salticam.slotmode.extract.extract import _parse_head
from astropy.io.fits.header import Header

import motley

import sys

path = sys.argv[1]

obsdur = defaultdict(list)
obsdates = defaultdict(list)
for root, dirs, files in os.walk(path):
    fitsfinder = lambda f: (f.startswith('bxgp') & f.endswith('.fits'))
    fitsfiles = sorted(filter(fitsfinder, files))
    if len(fitsfiles) > 1 and not root.endswith(('raw', 'finders', 'new')):
        try:
            first, last = [os.path.join(root, f) for f in
                           (fitsfiles[0], fitsfiles[-1])]
            nfiles = len(fitsfiles)

            ext_start, mhead, ext1head, header_size = _parse_head(first)
            header = Header.fromstring(mhead)
            n_ex = header['NEXTEND']
            texp = header['EXPTIME']

            nframes = nfiles * n_ex / 4
            tot_time = texp * nframes

            print()
            print('-' * 100)
            print('OBJECT', motley.green(header['OBJECT']))
            print('DATE-OBS', header['DATE-OBS'])
            print('EXPTIME', texp)
            print('NFRAMES', nframes)
            print('TOT TIME:', tot_time)
            print('-' * 100)
            print()
        except:
            print(motley.red('PROBLEMS'), root)

        # obj = quickheadkeys(first, 'object')
        # date0, ut0, = headhunter(first, ('date-obs', 'utc-obs'))
        #
        # date1, ut1 = headhunter(last, ('date-obs', 'utc-obs'))
        # ut1, date1 = filtermore(lambda s: s != '23:59:59.972000', ut1, date1)
        #
        # if None in ut0:
        #     print('PROBLEMS!!', first)
        # else:
        #     t0s = 'T'.join(nthzip(0, date0, ut0))
        #     t0 = Time(t0s)
        #
        #     t1s = 'T'.join(nthzip(0, date1, ut1))
        #     t1 = Time(t1s)
        #     dt = t1 - t0
        #
        #     # print( root )
        #     # print( obj[0], dt*24 )
        #     obsdur[obj[0]].append(dt.value * 24)
        #     obsdates[obj[0]].append(t0)

obsdurtot = {k: round(sum(v), 5) for k, v in obsdur.items()}
pprint(obsdurtot)
