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
#from sklearn.metrics import confusion_matrix
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
import fadesUtilities_frame_by_frame as fades_util
import opticalFlowUtilities_frame_by_frame as util
import fitCNNUtilities_frame_by_frame as cnn_utils
import sys
import time
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import shutil

def findFrameNumber(video_path):

    vid_capture = cv2.VideoCapture(video_path)
    max_frames = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    return max_frames, fps

def videoRead_batches(video_path, start_frame, end_frame, INTERVAL):

    """
    Function returns shot frames from defined frame range, as well as the video fps
    """

    # Create a video capture object, in this case we are reading the video from a file
    vid_capture = cv2.VideoCapture(video_path)

    if (vid_capture.isOpened() == False):
        print(f"Error opening the video file in {video_path}")
    else:
        # Get frame rate information
        fps = vid_capture.get(cv2.CAP_PROP_FPS)

    frames=[]
    count=-1
    kk10 = 0
    while(vid_capture.isOpened()):
        #print(count)
        count+=1
        # vid_capture.read() methods returns a tuple, first element is a bool and the second is frame
        ret, frame = vid_capture.read()
        if count >= start_frame and count < end_frame:
            # Append all frames to a list
            if ret == True:
                frames.append( cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) )
            else:
                break
        elif count > end_frame:
            break

        if len(frames)==INTERVAL:
            
            print(start_frame, end_frame)
            writeVideos(str(kk10*INTERVAL)+'_'+str((kk10+1)*INTERVAL), frames, root='batch_videos')
            kk10+=1
            frames = []
            
    # Release the video capture object
    vid_capture.release()
    
    return frames, fps

def videoRead(video_path, start_frame, end_frame):

    """
    Function returns shot frames from defined frame range, as well as the video fps
    """

    # Create a video capture object, in this case we are reading the video from a file
    vid_capture = cv2.VideoCapture(video_path)

    if (vid_capture.isOpened() == False):
        print(f"Error opening the video file in {video_path}")
    else:
        # Get frame rate information
        fps = vid_capture.get(cv2.CAP_PROP_FPS)

    frames=[]
    count=-1

    while(vid_capture.isOpened()):
        #print(count)
        count+=1
        # vid_capture.read() methods returns a tuple, first element is a bool and the second is frame
        ret, frame = vid_capture.read()
        if count >= start_frame and count < end_frame:
            # Append all frames to a list
            if ret == True:
                frames.append( cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) )
            else:
                break
        elif count > end_frame:
            break

    # Release the video capture object
    vid_capture.release()
    
    return frames, fps

def computeTransitionPredictions(filename, thold_hist, thold_cnn, fades_obj, opt_flow_test, cnn_obj, frame_no, frame_img, frame_of, frame_cnn, debug=False):
    
    if debug:
        fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(15,45))
        ax[0].imshow(frame_img)
        ax[1].imshow(frame_of)
        ax[2].imshow(frame_cnn)
        plt.show()
    
    intersect = fades_obj.calculateHistogramOverlapSingle(frame_img, frame_of)
    
    if intersect < thold_hist:
        if debug: print('inter: '+str(intersect))
        return frame_no

    flow = opt_flow_test.performOpticalFlowAndSave(frame_img, frame_cnn)
    
    if debug:
        fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,40))
        ax[0].imshow(flow[:,:,0])
        ax[1].imshow(flow[:,:,1])

    pred = float(cnn_obj.runModelOnTestData(filename, flow))
    
    if pred > thold_cnn:
        if debug: print('cnn_pred: '+str(pred))
        return frame_no
    
def extractTransitionFrames(frames, seek_forward_of, seek_forward_cnn, fades_obj, opt_flow_test, cnn_obj, thold_hist, thold_cnn, filename):
    max_iter = len(frames)-max(seek_forward_of, seek_forward_cnn)
    a = Parallel(n_jobs=-1)(delayed(computeTransitionPredictions)(filename, thold_hist, thold_cnn, fades_obj, opt_flow_test, cnn_obj, frame_no, frames[frame_no], \
                                                             frames[frame_no+seek_forward_of], frames[frame_no+seek_forward_cnn]) \
                                                             for frame_no in tqdm(range(max_iter)))
    transition_frame_list = [i for i in a if i is not None]
    return transition_frame_list

def writeTransitions(frames, transition_frame_list):
    frames2 = frames
    for start_frame in transition_frame_list:

        frames2[start_frame][:100,:,:] = 0
        frames2[start_frame][-100:,:,:] = 0
        frames2[start_frame][:,:100,:] = 0
        frames2[start_frame][:,-100:,:] = 0
        frames2[start_frame][:100,:,:2] = 255
        frames2[start_frame][-100:,:,:2] = 255
        frames2[start_frame][:,:100,:2] = 255
        frames2[start_frame][:,-100:,:2] = 255
    frames_stacked = np.stack(frames2,0)
    return frames_stacked

def writeVideos(kk10, frames_stacked2, root='output_videos'):

    out = cv2.VideoWriter(root+'/'+str(kk10)+'.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (1920,1080))

    for i in tqdm(range(0,len(frames_stacked2))):
        out.write(frames_stacked2[i][:,:,::-1])

    out.release()
    
def outerLoop(filename, start_frame, end_frame, seek_forward_cnn, seek_forward_of, kk10, fades_obj, opt_flow_test, cnn_obj, thold_hist, thold_cnn):
    frames, fps = videoRead(filename, start_frame, end_frame+max(seek_forward_cnn, seek_forward_of))
    transitions = extractTransitionFrames(frames, seek_forward_of, seek_forward_cnn, fades_obj, opt_flow_test, cnn_obj, thold_hist, thold_cnn, filename)
    #frames_stacked = writeTransitions(frames, transitions)
    #print(frames_stacked.shape)
    #writeVideos(kk10, frames_stacked)
    return transitions

def extractVideoClips(filename, starttime, max_frame, INTERVAL, i, fps, start_frame):

    # Replace the filename below.
    required_video_file = filename
    
    diff = 5000.0/fps
    endtime = starttime + diff - 1/fps
    
    #start_frame = initial_frame + i*INTERVAL
    
    #end_frame   = initial_frame + (i+1)*INTERVAL
    #end_frame   = min(end_frame, max_frame)
    
    #starttime = start_frame/fps
    #endtime = end_frame/fps
    
    print(starttime, endtime, (endtime-starttime)*fps)
    
    ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname='batch_videos/temp.mov')
    
    max_frame, fps = findFrameNumber('batch_videos/temp.mov')
    
    end_frame = start_frame + max_frame
    shutil.move('batch_videos/temp.mov','batch_videos/'+str(start_frame)+"_"+str(end_frame-1)+".mov")
    
    return endtime+1/fps, end_frame
    
    #ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname='batch_videos/'+str(i*INTERVAL)+"_"+str((i+1)*INTERVAL)+".mov")
    

def splitVideo(filename, start_frame, max_frame, INTERVAL, fps):    

    starttime = 0
    start_frame = 0
    for i in range(1+(max_frame//INTERVAL)):
        
        starttime, start_frame = extractVideoClips(filename, starttime, max_frame, INTERVAL, i, fps, start_frame)
        