import numpy as np
from pyVHR.datasets.dataset import Dataset
from pyVHR.BPM.BPM import BVPsignal


class UBFC2(Dataset):
    """
    UBFC2 Dataset

    .. UBFC dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |   |-- SubjDIR1/
    ..     |       |-- vid.avi
    ..     |...
    ..     |   |-- SubjDIRM/
    ..     |       |-- vid.avi
    """
    name = 'UBFC2'
    signalGT = 'BVP'     # GT signal type
    numLevels = 2        # depth of the filesystem collecting video and BVP files
    numSubjects = 26     # number of subjects
    video_EXT = 'avi'    # extension of the video files
    frameRate = 30       # vieo frame rate
    VIDEO_SUBSTRING = 'vid'  # substring contained in the filename
    SIG_EXT = 'txt'     # extension of the BVP files
    SIG_SUBSTRING = ''  # substring contained in the filename
    SIG_SampleRate = 30  # sample rate of the BVP files
    skinThresh = [40, 60]  # thresholds for skin detection

    def readSigfile(self, filename):
        """ 
        Load signal from file.
        
        Returns:
            a :py:class:`pyVHR.BPM.BPM.BVPsignal` object that can be used to extract BPM signal from ground truth BVP signal.
        """
        gtTrace = []
        gtTime = []
        gtHR = []
        with open(filename, 'r') as file:
            x = file.readlines()

        s = x[0].split(' ')
        s = list(filter(lambda a: a != '', s))
        s = [value for value in s if value.replace('.', '', 1).isdigit()]
        gtTrace = np.array(s).astype(np.float64)

        t = x[2].split(' ')
        t = list(filter(lambda a: a.replace('.', '', 1).isdigit(), t))
        gtTime = np.array(t).astype(np.float64)

        hr = x[1].split(' ')
        hr = list(filter(lambda a: a != '' and a != '\n', hr))
        gtHR = np.array(hr).astype(np.float64)

        time = np.array(gtTime)
        self.SIG_SampleRate = np.round(1/np.mean(np.diff(time)))

        return BVPsignal(gtTrace, self.SIG_SampleRate)
