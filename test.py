import cv2
import time
import tqdm
import argparse

from rmac_hash import generate_hashs, find_hashs
import rmac_hash
from rmac_vgg import check, load_RMAC
from utils import ptime

tmatches = 0
ttime = 0.


def scan_references(references, second):
    global tmatches, ttime
    for k, v in list(references.items()):
        for kk, vv in list(v.items()):
            vv[2] *= .8
            if vv[2] < 0.2:
                if vv[1] > 7.0:
                    print(
                        f"at {ptime(vv[0])}-{ptime(second)}({ptime(second - vv[0])}) found in video nr.{k} at {ptime(kk)} ")
                    ttime += second - vv[0]
                    tmatches += 1
                del references[k][kk]
                if not references[k]:
                    del references[k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="""INPUT:
    ======
    A video (--source ) 
    
    """)
    parser.add_argument('--source', required=True, type=str, metavar="source",
                        help='video')
    parser.add_argument('--frequency', required=False, type=float, metavar="frequency", default=1.,
                        help='create hash every x seconds')
    parser.add_argument('--debug', required=False, type=bool, metavar="debug", default=False,
                        help='show each frame detection')
    args = parser.parse_args()

    rmac_hash.load()

    print(f"open: {args.source}")
    cap = cv2.VideoCapture(args.source)

    regions, model = load_RMAC()

    start = time.time()
    flag, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    references = {}
    nomatch_c = 0
    check_evey_n_frame = int(fps * 2 / 3)

    print(f"check every {check_evey_n_frame} frame")

    second = 0
    for i in tqdm.tqdm(range(frame_count)):
        flag, frame1 = cap.read()
        if not flag:
            break
        second = i / fps
        if i % check_evey_n_frame == 0:

            dat = check(frame1, regions, model).flatten()
            hashs = generate_hashs(dat).flatten()
            best, fids = find_hashs(hashs)
            nomatch_c += 1
            scan_references(references, second)

            for fidk, fidv in fids:
                fidk1, fidk2 = fidk
                if fidk1 in references:  # other video already in list, find best time-match
                    found_match = False
                    for kk, vv in references[fidk1].items():
                        # how far in past is kk.vv[0] vs fid2.second
                        time_distance = ((fidk2 - kk) - (second - vv[0])) ** 4
                        w = .5 / (.5 + time_distance)
                        if w > (.5 / (3 ** 4)):  # up to 4 seconds distance to count as match
                            found_match = True
                            references[fidk1][kk][1] += fidv * (w)
                            references[fidk1][kk][2] += fidv * (w)
                    if not found_match:
                        references[fidk1][fidk2] = [second, fidv, fidv]
                else:
                    references[fidk1] = {fidk2: [second, fidv, fidv]}  # (second, weight_akk, weight_now)
                if args.debug:
                    print("frame-match:", fidk1, ptime(fidk2), "at ",
                          ptime(i / fps), " after:", nomatch_c, "frames")
                nomatch_c = 0

    real_duration=time.time() - start
    virtual_duration=frame_count / fps
    print(f"{ptime(real_duration)} for a {ptime(virtual_duration)} video. speed: {real_duration/virtual_duration:.2f}X")

    while references:
        scan_references(references, second)

    print(f"\nfound: {tmatches} video segments with total time:{ptime(ttime)}  ")
