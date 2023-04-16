import os
import lzma
import re
import time
import json
import math
import pickle
import random
import logging
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import faiss

import src.constants as co
import src.utility as util


class SplitFile(object):
    def __init__(self, name_pattern, chunk_size=40 * 1024 ** 3):
        self.name_pattern = name_pattern
        self.chunk_size = chunk_size
        self.file = None
        self.part = -1
        self.offset = None

    def write(self, bytes):
        if not self.file:  self._split()
        while True:
            l = len(bytes)
            wl = min(l, self.chunk_size - self.offset)
            self.file.write(bytes[:wl])
            self.offset += wl
            if wl == l: break
            self._split()
            bytes = bytes[wl:]

    def _split(self):
        if self.file:  self.file.close()
        self.part += 1
        self.file = open(self.name_pattern % self.part, "wb")
        self.offset = 0

    def close(self):
        if self.file:  self.file.close()

    def __del__(self):
        self.close()


def pickle_into_several_parts(folder: Path, name: str, obj: object, parts: int):
    content = pickle.dumps(obj)
    leng = len(content)
    plen = int(math.floor(leng/parts))
    for ii in range(parts):
        if ii == parts -1:
            part = content[ii * plen:]
        else:
            part = content[ii * plen: (ii + 1) * plen]

        file_name = name + '.part%02d.pkl.xz' % ii
        path = folder / file_name
        with lzma.open(path, 'wb') as f:
            f.write(part)


def pickle_from_several_parts(folder: Path, name: str):
    pattern = name + r'\.part\d{2}.pkl.xz'
    files = sorted([x for x in os.listdir(folder) if re.match(pattern, str(x))])

    content = b''
    for file in files:
        with lzma.open(folder / file, 'rb') as f:
            content += f.read()

    return pickle.loads(content)


def save_vectors():
    df = util.load_data()
    vecs = util.get_vectors(df)
    for ii in range(len(vecs)):
        sp = co.DATA_FOLDER / 'vectors' / '{}.npy'.format(ii)
        vec = vecs[ii]
        np.save(sp, vec)





def main():
    vec = load_vector(0)
    print(vec)


if __name__ == "__main__":
    ss = dt.now().strftime('%H:%M:%S'); start_time = time.time()
    main()
    es = dt.now().strftime('%H:%M:%S'); l = ''.join(['-'] * 37); elapsed = time.time() - start_time
    print("\n%s\n-------- %s - %s --------\n--------- % 9.3f seconds ---------\n%s" % (l, ss, es, elapsed, l))
