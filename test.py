import cv2
import time
import tqdm
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from rmac_hash import generate_hashs, find_hashs
import rmac_hash
from rmac_vgg import check, load_RMAC
from utils import ptime


def scan_references(references, second, keep_alive):
    sequences=[]
    for k, v in list(references.items()):
        for kk, vv in list(v.items()):
            vv[2] *= keep_alive
            if vv[2] < 0.2:
                if vv[1] > 7.0:
                    print(
                        f"at {ptime(vv[0])}-{ptime(second)}({ptime(second - vv[0])}) sequence found at {ptime(kk)} in ..{rmac_hash.id_map[k][-20:]} tscore{vv[1]:.2f}")
                    sequences += (second - vv[0],second,k,kk,   vv[1])
                del references[k][kk]
                if not references[k]:
                    del references[k]
    return sequences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="""INPUT:
    ======
    A video (--source ) 
    
    """)
    parser.add_argument('--source', required=True, type=str, metavar="source",
                        help='video file to analyse')
    parser.add_argument('--frequency', required=False, type=float, metavar="frequency", default=1.,
                        help='create hash every x seconds')
    parser.add_argument('--keep', required=False, type=float, metavar="keep_alive", default=0.8,
                        help='range from 0.1 to 0.9, keep video sequence active after positive detections')
    parser.add_argument('--detect', required=False, type=float, metavar="detect_threshold", default=20,
                        help='minimum matching score to count a frame as detected ')

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

    print(f"check every {check_evey_n_frame} frames")

    second = 0
    secuences=[]
    for i in tqdm.tqdm(range(int(frame_count))):
        flag, frame1 = cap.read()
        if not flag:
            break
        second = i / fps
        if i % check_evey_n_frame == 0:

            dat = check(frame1, regions, model).flatten()
            hashs = generate_hashs(dat).flatten()
            fids = find_hashs(hashs, args.detect_threshold)
            nomatch_c += 1
            secuences += scan_references(references, second, args.keep_alive)

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

    real_duration = time.time() - start
    virtual_duration = frame_count / fps
    print(
        f"\nTested in {ptime(real_duration)} for a {ptime(virtual_duration)} video. speed: {virtual_duration / real_duration:.2f}X")

    while references:
        secuences += scan_references(references, second, args.keep_alive)

    total_time = sum([s[0] for s in secuences])

    print(f"found: {len(secuences)} video segments with total time: {ptime(total_time)}")
    if args.debug:
        print(secuences)