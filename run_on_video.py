import pyVHR as vhr
import numpy as np
from pyVHR.analysis.pipeline import Pipeline
from pyVHR.utils.errors import getErrors

dataset_name = 'UBFC2'                      # the name of the python class handling it 
video_DIR = '/home/saturngtxd/dev/datasets/ubfc-2/' # dir containing videos
BVP_DIR = '/home/saturngtxd/dev/datasets/ubfc-2/'    # dir containing BVPs GT
# path_dir = '/home/saturngtxd/dev/parser/videos/test5.mp4'  # Ensure the correct file extension

dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
allvideo = dataset.videoFilenames

# print the list of video names with the progressive index (idx)
for v in range(len(allvideo)):
  print(v, allvideo[v])

print("Number of videos:", len(dataset.videoFilenames))
# print("Number of BVP signals:", len(dataset.sigFilenames))

# assert len(dataset.videoFilenames) == len(dataset.sigFilenames), "Mismatch between video and signal files"

# -- PARAMETER SETTING and GT

wsize = 1        # seconds of video processed (with overlapping) for each estimate 
video_idx = 0
fname = dataset.getSigFilename(video_idx)
sigGT = dataset.readSigfile(fname)
test_bvp = sigGT.data
bpmGT, timesGT = sigGT.getBPM(wsize)  # GT
videoFileName = dataset.getVideoFilename(video_idx)
fps = vhr.extraction.get_fps(videoFileName)

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

# Show the estimated BPM
print("bpmES: ", bpmES)
# Find the frame in the video that gives a value from bpmES that is greater than 0
for i, bpm in enumerate(bpmES):
    if bpm > 0:
        frame_number = int(timesES[i] * fps)
        print(f"Frame number with BPM > 0: {frame_number}")
        break