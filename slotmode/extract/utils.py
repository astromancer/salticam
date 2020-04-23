import re
from pathlib import Path
import logging

import numpy as np
from recipes.introspection.utils import get_module_name

# module level logger
logger = logging.getLogger(get_module_name(__file__))

# Template for patterns matching FITS header keys-value pair ignoring comments
TEMPLATE_PATTERN_HEADER_KEYS = r"""
    (%s%s)          # keywords (non-capture group if `capture_keywords` False)
    \s*?=\s+        # (optional) whitespace surrounding value indicator '='
    '?([^'/]*)'?  # value associated with any of the keys (un-quoted)
    """


def regex_maker(keys, capture_keywords=True, encode=True, compile=True,
                terse=False):
    """
    Dynamically generate pre-compiled regex matcher for finding keywords in FITS
    header.

    Parameters
    ----------
    keys:       str, bytes, tuple of str,
        sequence of keys to match
    capture_keywords:   bool, default True
        Whether to make capture groups for the keys
    encode:     bool, default True
        Whether to encode the regex so it works on bytes
    compile:    bool, default True
        whether to compile the regex and return `_sre.SRE_Pattern` object, or
         otherwise just return the patters str / bytes
    terse:      bool, default False
        If true, will strip all the verbose content (comments) in the pattern
         and return only the regex code bits

    Returns
    -------
    str or `_sre.SRE_Pattern`
    """

    if keys is None or (len(keys) == 0):
        return  # _NullMatcher()

    if isinstance(keys, (str, bytes)):
        keys = keys,

    types = set(map(type, keys))
    assert len(types) == 1, 'Mixed str / bytes keys not supported'
    type_, = types
    if type_ is bytes:
        keys = map(bytes.decode, keys)

    regex_any_key = '|'.join(keys)
    regex = (TEMPLATE_PATTERN_HEADER_KEYS %
             (['?:', ''][capture_keywords], regex_any_key))
    if terse:
        from recipes.regex import terse
        regex = terse(regex)
    if encode:
        regex = regex.encode()
    if compile:
        return re.compile(regex, re.X | re.I)  # VERBOSE, IGNORECASE
    return regex


def prepare_path(name, folder, overwrite):
    path = Path(name)
    if path.parent == Path():
        # `name` is intended as a filename str not a path - make relative to
        # input data location
        path = folder / path
        # path = folder

    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    if path.exists:
        if overwrite:
            logger.info('The file %r will be overwritten', path)
        else:
            raise IOError('Operation will overwrite the file %r. To avoid this '
                          'error pass the argument `overwrite=False`'
                          % str(path))

    return path


def truncate_memmap(data, n_frames):
    """

    Parameters
    ----------
    data
    n_frames

    Returns
    -------

    """
    offset = data.offset
    n, n_ch, n_rows, n_cols = data.shape
    image_size_bytes = n_rows * n_cols * data.itemsize
    remove_bytes = image_size_bytes * n_ch * n_frames
    new_size = offset + data.nbytes - remove_bytes
    new_shape = ((len(data) - n_frames),) + data.shape[1:]
    dtype = data.dtype
    filename = data.filename

    # de-allocate memory for `data`
    del data

    with open(filename, 'rb+') as fp:
        fp.truncate(new_size)

    # return new memmap
    return np.memmap(filename, dtype, 'r', offset, new_shape)

    #


class _NullMatcher:
    """Implement null object pattern for re SRE_Pattern matcher"""

    # return a null matcher that mimics a compiled regex (good enough for our
    # purpose here) but always returns empty list.  useful in that it avoids
    # unnecessary if statements inside the main extraction loop and thus
    # performance gain.

    def findall(self, _):
        return []
