import random
import numpy as np
import pickle

hash_file_name = r'hash_map.pkl'
hash_map = {}
id_file_name = r'file_name.pkl'
id_map = {}

BIT_PAIRS = 10  # up to 12
BATCH_COUNT = 3  # bits duplicater

BITS = BIT_PAIRS * 2
BATCH_HASH_COUNT = int(512 // BITS)
BATCH_HASH_SIZE = BIT_PAIRS * BATCH_HASH_COUNT


# this generate a shuffled index to take out different combination of bits

def save():
    print(end="saving hash table... ")
    afile = open(hash_file_name, 'wb')
    pickle.dump(hash_map, afile)
    afile.close()

    afile = open(id_file_name, 'wb')
    pickle.dump(id_map, afile)
    afile.close()
    print(f"{len(hash_map)} hashs, {len(id_map)} videos")


def load():
    print(end="loading hash table...")
    global hash_map, id_map
    try:
        hash_map = pickle.load(open(hash_file_name, "rb"))
    except FileNotFoundError:
        hash_map = {}
    try:
        id_map = pickle.load(open(id_file_name, "rb"))
    except FileNotFoundError:
        id_map = {}
    print(f"{len(hash_map)} hashs, {len(id_map)} videos")


def reset():
    print("reset hash table")
    global hash_map, id_map
    hash_map = {}
    id_map = {}
    save()


random.seed(0)
r_index = []
for i in range(BATCH_COUNT):
    r1 = list(range(512))
    random.shuffle(r1)
    r_index.append(np.array(r1[:BATCH_HASH_SIZE], dtype=np.int32).reshape(BATCH_HASH_COUNT, BIT_PAIRS))
r_index = np.concatenate(r_index)
r_multi = [4 ** e for e in np.arange(BIT_PAIRS)]


def generate_hashs(v):
    s = np.empty((512,), dtype=np.int64)
    for ii, b in enumerate(v):
        if b < -0.035:
            s[ii] = 0
        elif b > 0.035:
            s[ii] = 2
        else:
            s[ii] = 1
    return (np.sum((s[r_index] * r_multi), axis=-1) << 8) + np.arange(len(r_index))


def store_hashs(hashs, id1, id2):
    global hash_map
    id_ = (id1, id2)
    for hash_ in hashs:
        if hash_ in hash_map:
            hash_map[hash_].append(id_)
        else:
            hash_map[hash_] = [id_]


def find_hashs(hashs, frame_sensitivity):
    d = {}
    used_hashs = 0
    for n in hashs:
        w = len(hash_map.get(n, []))
        if w < 128:
            used_hashs += 1
        if 0 < w < 128:
            ww = 1.  # /(w **0.25)
            for e in hash_map[n]:
                if e != i:
                    if e in d:
                        d[e] += ww
                    else:
                        d[e] = ww

    if not d:
        return []
    # compensate ignored hashs
    frame_sensitivity *= len(hashs) / used_hashs

    biggest_k = max(d, key=d.get)
    if d[biggest_k] > frame_sensitivity:
        # l=[(k,(v/frame_sensitivity)**.25) for k,v in d.items() if v>frame_sensitivity]
        return [(biggest_k, (d[biggest_k] / frame_sensitivity) ** .25)]
    else:
        return []
