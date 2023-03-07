#import torch
#from torchvision import transforms
#from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from glob import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
#import torch.nn as nn
#import torch.nn.functional as F
from tqdm import tqdm
import os 
from sklearn.metrics import confusion_matrix
from scipy.stats import logistic
import imageio
from videoio import videosave, videoread
from moviepy.editor import *
import json
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pylab
import imageio
from tqdm import tqdm
from joblib import Parallel, delayed
from videoio import videosave, videoread
import os
import fadesUtilities as fades_util
import opticalFlowUtilities as util
import fitCNNUtilities as cnn_utils
import sys

def fades_comp(filename = "/ml_data_fuse/cookup/The_Cook_Up_S01E36_ryffcopy.mp4", output_path0 = 'results_MR_hist_eq_test_vid_orig_rgb', cleanup = False, start_frame = 0, end_frame = 500, seek_forward = 1):

    hist_eq0 = False
    if not os.path.exists(output_path0):
        os.mkdir(output_path0)

    fades_obj = fades_util.detectFades(filename, output_path0, start_frame, end_frame, rgb=True, seek_forward=seek_forward)

    print('Step 1: Creating JSON files detailing histogram overlaps')
    fades_obj.runHistogramOverlapCalc(INTERVAL = 30)

    frames_identified = fades_obj.groupJSONsIdentifyTransitions(thold=0.90, display_transitions=False)

    print('Step 2: Writing the output transition frames for histogram check')
    fades_obj.writeTransitionFrames(hist_eq0)
    

def cnn_component(filename = "/ml_data_fuse/cookup/The_Cook_Up_S01E36_ryffcopy.mp4", output_path0_model = "results_MR_hist_eq", output_path0 = 'results_MR_hist_eq_test_vid_orig_rgb', hist_eq0 = False, cleanup = False, start_frame = 0, end_frame = 500, seek_forward_cnn = 1, approach = 3):

    hist_eq0 = True
    opt_flow_test = util.generateOpticalFlowResults(filename, output_path = output_path0, hist_eq = hist_eq0, end_frame=end_frame, start_frame=start_frame, seek_forward=seek_forward_cnn)
    cnn_obj = cnn_utils.analysisWithCNN(output_path0_model, output_path0, hist_eq0, rgb=False, seek_forward=seek_forward_cnn)
    
    print('Step 3: Run optical flow for each frame of the test video')
    opt_flow_test.runOpticalFlow()
    
    print('Step 4: Run CNN on each optical flow result')
    output_path0_ref = 'results_MR_hist_eq'
    cnn_obj.runModelOnTestData(filename, output_path0, output_path0_ref, end_frame=end_frame,start_frame=start_frame)
    
    print('Step 5: Writing the output transition frames detected by the CNN')
    cnn_obj.createVideosWithTransitions(filename, INTERVAL = 400, approach = 3, trans_thold=0.2)
    
    print('Step 6: Combining everything into one video file')
    cnn_obj.combineVideos()
    
    # perform cleanup of intermediate files
    if cleanup:
        print('Step 7: Deleting all intermediate files')
        opt_flow_test.cleanupFiles(prefix='out0_')
        opt_flow_test.cleanupFiles(prefix='vid0_')
        opt_flow_test.cleanupFiles(prefix='vid_t'+str(approach)+'_')
        opt_flow_test.cleanupFiles(prefix='frame_',suffix='png')
        opt_flow_test.cleanupFiles(prefix='optical_flow_',suffix='png')
        opt_flow_test.cleanupFiles(prefix='np_flow',suffix='npy')

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        fades_comp()
        cnn_component()
    elif len(sys.argv) == 9:
        filename = sys.argv[1] 
        output_path0_model = sys.argv[2] 
        output_path0 = sys.argv[3] 
        hist_eq0 = int(sys.argv[4])==1
        cleanup = int(sys.argv[5])==1
        start_frame = int(sys.argv[6])
        
        if sys.argv[7] == 'None':
            end_frame = None
        else:
            end_frame = int(sys.argv[7])
        print(end_frame)
        
        seek_forward = int(sys.argv[8])
        seek_forward_cnn = 1
        
        fades_comp(filename, output_path0, cleanup, start_frame, end_frame, seek_forward)

        cnn_component(filename, output_path0_model, output_path0, hist_eq0, cleanup, start_frame, end_frame, seek_forward_cnn, approach = 3)
        
    else:
        print('Not enough arguments')
    
