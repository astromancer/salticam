

def extractor(chunk):
    """
    Find all matches in the chunk of file given

    Parameters
    ----------
    chunk

    Returns
    -------
    _sre.SRE_Match object

    """
    return matcher.findall(filemap, *chunk)


def cleaner(b):
    return b.strip(b"' ").decode()


def parse_and_clean(s):
    parts = card_parser.match(s).groups()
    key, val, comment = map(cleaner, parts)
    return key, val, comment


def parse_and_clean_no_comment(s):
    return parse_and_clean(s)[:2]


def merge2dict(raw, keys, defaults=[]):
    # NOTE: THIS MERGER WILL BE REDUNDANT IF YOU CAN MAKE THE REGEX CLEVER ENOUGH TO MATCH KEYWORD,VALUE,COMMENTS
    # TODO: BENCHMARK!
    # TODO: Optimize for speed!
    """
    Merge the results from the multiprocessing pool into a dictionary of lists.
    First identifying the END cards and split the results accordingly (by header).
    Loop through headers. Split each keyword into its parts (key, val, comment).
    Substitute missing values from given defaults (None) before aggregating the
    results.

    Parameters
    ----------
    raw:
        List of matched lines from headers
    keys:
        sequence of keywords by with to key resulting dictionary
    defaults:
        Values to substitute (same order as keys) in case of missing keys.

    Returns
    -------
    D:          Dictionary of lists
    """

    keys = map(str.upper, keys)

    raw = np.asarray(raw)  # sometimes this is faster than np.array(_)
    w, = np.where(raw == end_str)  # location of END matches

    D = DefaultOrderedDict(list)
    # default key values (None if not given)
    defaults = OrderedDict(
            itt.zip_longest(keys, defaults))  # preserve order of keys

    # iterate through headers. extract keyword values. if keyword missing, sub default. aggregate
    for seg in np.split(raw, w[
                             :-1] + 1):  # split the raw data (last item in each segment is the end string )
        # write the extracted key-val pairs to buffer and sub defaults
        buffer = defaults.copy()
        buffer.update(map(parse_and_clean_no_comment, seg[:-1]))
        # append to the aggregator
        for key in defaults.keys():
            D[key].append(buffer[key])

    return D


def aggregate(raw, keys=None, defaults=[], return_type='raw'):
    if return_type == 'raw':
        return raw

    md = merge2dict(raw, keys, defaults)
    if return_type == 'dict':
        return md

    tmd = tuple(md.values())
    if return_type == 'list':
        return tmd

    if return_type == 'array':
        return np.asarray(tmd)


# @profile( follow=(merge2dict,) )
def headhunter(filename, keys, defaults=[], **kw):
    # TODO: BENCHMARK! Nchunks, filesize
    # TODO: OPTIMIZE?
    # TODO:  Read first N keys from multi-ext header???
    # WARNING:  No error handling implemented!  Use with discretion!

    """
    Fast extraction of keyword-value from FITS header(s) using multiprocessing
    and memory mapping.

    Parameters
    ----------
    filename:   str
        file to search
    keys:       sequence
        keywords to search for
    defaults:   sequence
        optional defaults to substitute in case of missing keywords

    Keywords
    --------
    nchunks:            int;    default 25
        Number of chunks to split the file into.  Tweaking this number can yield
        faster computation times depending on how many cores you have.
    with_parent:        bool;   default False
        whether the key values from the parent header should be kept in the
        results
    return_type:        str;   options: 'raw', 'dict', 'list', 'array'
        How the results should be merged:
        raw -->         raw matched strings are returned.
        dict -->        return dict of lists keyed on keywords
        list -->        return tuple of lists
        array -->       return 2D array of data values

    Returns
    -------
    dict of lists / list of str depending on the value of `merge`
    """

    nchunks = kw.get('nchunks', 25)
    with_parent = kw.get('with_parent', False)
    return_type = kw.get('return_type', 'list')

    assert return_type in ('raw', 'dict', 'list', 'array')

    # if `keys` is a regex pattern matcher, use it
    if isinstance(keys, re._pattern_type):
        matcher = keys
    else:
        # build regex pattern matcher
        if isinstance(keys, str):
            keys = keys,
        matcher = match_maker(*keys)

    # divide file into chunks
    chunksize = max(1, os.path.getsize(filename) // nchunks)

    # fork extraction process
    pool = Pool(initializer=init, initargs=[filename, matcher])
    raw = pool.imap(extractor, getchunks(filename, chunksize))
    # chunksize=10 (can this speed things up??)
    pool.close()
    pool.join()

    # concatenate the list of lists into single list (a for loop seems to be
    # *the* fastest way of doing this!!)
    # TODO: you can avoid this by using globally shared memory!!!
    results = []
    for r in raw:
        results.extend(r)

    # without parent header values (this is needed for the superheadhunter)
    if not with_parent:
        ix = results.index(end_str)
        results = results[ix + 1:]

    return aggregate(results, keys, defaults, return_type)


# @profile()# follow=(headhunter,)dd
def superheadhunter(filelist, keys, defaults=[], **kw):
    # TODO: BENCHMARK! nchunks, nfiles
    # TODO: OPTIMIZE?
    """Headhunter looped over a list of files."""

    nchunks = kw.get('nchunks', 25)
    with_parent = kw.get('with_parent', False)
    return_type = kw.get('return_type', 'list')

    hunt = functools.partial(headhunter,
                             keys=keys,
                             nchunks=nchunks,
                             return_type='raw',
                             with_parent=False)

    pool = Pool()
    raw = pool.map(hunt, filelist)
    pool.close()
    # pool.join()

    # Flatten the twice nested list of string matches (this is the fastest way of doing this!!)
    results = []
    for r in raw:
        results.extend(r)

    return aggregate(results, keys, defaults, return_type)



# ****************************************************************************************************
class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# ****************************************************************************************************
# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class Pool(mp.pool.Pool):
    Process = NoDaemonProcess
