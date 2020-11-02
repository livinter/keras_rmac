import cv2
import time
import tqdm
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from rmac_hash import generate_hashs, store_hashs
import rmac_hash
from rmac_vgg import check, load_RMAC
from utils import ptime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="""INPUT:
    ======
    A video (--source ) to add to the training-database.
    --reset to start a new training database
    
    """)
    parser.add_argument('--source', required=True, type=str, metavar="source",
                        help='video')
    parser.add_argument('--reset', required=False, type=bool, metavar="reset",default=False,
                        help='reset the hash db')
    parser.add_argument('--frequency', required=False, type=float, metavar="frequency",default=1.,
                        help='create hash every x seconds')
    args = parser.parse_args()
    if args.reset:
        rmac_hash.reset()
    rmac_hash.load()
    if rmac_hash.id_map:
        next_id=max(rmac_hash.id_map.keys())+1
    else:
        next_id=1

    print(f"open: {args.source}")
    cap = cv2.VideoCapture(args.source)

    regions, model = load_RMAC()

    start = time.time()
    flag, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"processing with {1./args.frequency:.2f} FPS hashs")
    for i in tqdm.tqdm(range(int(frame_count))):
        flag, frame1 = cap.read()
        if not flag:
            break
        if i % int(fps*args.frequency) == 0:
            dat = check(frame1,regions, model).flatten()
            hashs = generate_hashs(dat)
            store_hashs(hashs, id1=next_id, id2=i // int(fps))

    real_duration=time.time() - start
    virtual_duration=frame_count / fps
    print(f"{ptime(real_duration)} for a {ptime(virtual_duration)} video. speed: {virtual_duration/real_duration:.2f}X")
    rmac_hash.id_map.update({next_id:args.source})
    rmac_hash.save()