import pyVHR as vhr
import numpy as np
from pyVHR.analysis.pipeline import Pipeline
from pyVHR.plot.visualize import *
import os
import plotly.express as px
from pyVHR.utils.errors import getErrors, printErrors, displayErrors

# import PySimpleGUI as sg
# sg.home()


# dataset_name = 'PURE'                      # the name of the python class handling it 
# video_DIR = '/var/datasets/VHR1/PURE/'  # dir containing videos
# BVP_DIR = '/var/datasets/VHR1/PURE/'    # dir containing BVPs GT

dataset_name = 'UBFC2'                      # the name of the python class handling it 
video_DIR = '/home/mind/dev/datasets/ubfc-2/'  # dir containing videos
BVP_DIR = '/home/mind/dev/datasets/ubfc-2/'    # dir containing BVPs GT

dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
allvideo = dataset.videoFilenames

# print the list of video names with the progressive index (idx)
for v in range(len(allvideo)):
  print(v, allvideo[v])


# -- PARAMETER SETTING and GT

wsize = 30        # seconds of video processed (with overlapping) for each estimate 
video_idx = 2    # index of the video to be processed
fname = dataset.getSigFilename(video_idx)
sigGT = dataset.readSigfile(fname)
test_bvp = sigGT.data
bpmGT, timesGT = sigGT.getBPM(wsize)  # GT
print('GT BPM: ', bpmGT)
videoFileName = dataset.getVideoFilename(video_idx)
print('Video to process: ', videoFileName)
fps = vhr.extraction.get_fps(videoFileName)
print('Video frame rate: ',fps)




#### TEST holistic approach

# params
roi_approach = 'holistic'   # 'holistic' or 'patches'
bpm_est = 'median'         # BPM final estimate, if patches choose 'medians' or 'clustering'
method = 'cpu_CHROM'       # one of the methods implemented in pyVHR
pipe = Pipeline()          # object to execute the pipeline

# run
bvps, timesES, bpmES = pipe.run_on_video(videoFileName,
                                        winsize=wsize, 
                                        roi_method='convexhull',
                                        roi_approach=roi_approach,
                                        method=method,
                                        estimate=bpm_est,
                                        patch_size=40, 
                                        RGB_LOW_HIGH_TH=(5,230),
                                        Skin_LOW_HIGH_TH=(5,230),
                                        pre_filt=True,
                                        post_filt=True,
                                        cuda=True, 
                                        verb=True
)

# ERRORS
RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps, fps, bpmES, bpmGT, timesES, timesGT)
#printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)
#displayErrors(bpmES, bpmGT, timesES, timesGT)
print("bpmES: ", bpmES)



#### TEST median approach

# params
roi_approach = 'patches'   # 'holistic' or 'patches'
bpm_est = 'median'         # BPM final estimate, if patches choose 'medians' or 'clustering'
method = 'torch_CHROM'       # one of the methods implemented in pyVHR
pipe = Pipeline()          # object to execute the pipeline

# run
bvps, timesES, bpmES = pipe.run_on_video(videoFileName,
                                        winsize=wsize, 
                                        roi_method='convexhull',
                                        roi_approach=roi_approach,
                                        method=method,
                                        estimate=bpm_est,
                                        patch_size=40, 
                                        RGB_LOW_HIGH_TH=(5,230),
                                        Skin_LOW_HIGH_TH=(5,230),
                                        pre_filt=True,
                                        post_filt=True,
                                        cuda=True, 
                                        verb=True
)

# ERRORS
RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps, fps, bpmES, bpmGT, timesES, timesGT)
#printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)
#displayErrors(bpmES, bpmGT, timesES, timesGT)
print("bpmES: ", bpmES)
