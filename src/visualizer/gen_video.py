#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import cv2 as cv
from scipy import signal
from pathlib import Path
import h5py
from tqdm import tqdm
import scipy.io as sio

# Maps file extension to four character code (fourCC)
ext_to_fcc = {"mp4": cv.VideoWriter_fourcc(*"mp4v"),
              "avi": cv.VideoWriter_fourcc(*"mjpg"),
              "mkv": cv.VideoWriter_fourcc(*"x264"),
              }

def gen_video(video, filename, predictions=None, fps=15):
    _, ext = filename.split('.')


    img = []
    frames = []
    fig = plt.figure()

    for k in tqdm(range(len(video))):
        # frame = video[k, :512, :] # crop to only have face
        frame = video[k, :] # crop to only have face
        # breakpoint()
        # frame = np.flip(frame, axis=0) # flip to be right way up
        # frame = 255 * (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        rows, cols = frame.shape
        if predictions is not None:
            coords = (rows//10, cols//10)
            label = predictions[idx] if (idx := k // 15) < len(predictions) else predictions[-1] # update every 15 frames
            cv.putText(frame, f"STRESS: {label}", coords, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # plt.text(rows//10, cols//10, f"STRESS: {label}", bbox=dict(fill=False, edgecolor='red', linewidth=2))
        frames.append([plt.imshow(frame, cmap=cm.Greys_r,animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    print("done")
    ani.save(filename)

    # vw = cv.VideoWriter(filename, ext_to_fcc[ext], fps, video[0,...].shape[::-1])

    # for k in range(len(video)):
    #     frame = video[k, :512, :] # crop to only have face
    #     frame = np.flip(frame, axis=0) # flip to be right way up
    #     frame = 255 * (frame - np.min(frame)) / np.max(frame)
    #     rows, cols = frame.shape
    #     coords = (rows//10, cols//10)
    #     label = predictions[idx] if (idx := k // 15) < len(predictions) else predictions[-1] # update every 15 frames
    #     # print(label)
    #     cv.putText(frame, f"STRESS: {label}", coords, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    #     if k % 100 == 0:
    #         plt.imshow(frame)
    #         plt.show()
    #     vw.write(cv.cvtColor(np.uint8(frame),cv.COLOR_GRAY2BGR))

    # vw.release()

if __name__=="__main__":
    filepath = Path(sys.argv[1])
    extension = sys.argv[2]

    if not extension:
        extension = "mp4"

    # predictions = [int(x) for x in sys.stdin.read().split()]
    predictions = None
    # video = h5py.File(filepath)['data']
    video = sio.loadmat(filepath)['data'].squeeze()
    video = np.transpose(video, (2, 0, 1))
    final = "labeled_" + filepath.stem + "." + extension
    print(final)
    gen_video(video, predictions=predictions, filename=final)
