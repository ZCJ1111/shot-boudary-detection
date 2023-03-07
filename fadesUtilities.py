import numpy as np
from PIL import Image
from glob import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os 
from sklearn.metrics import confusion_matrix
from scipy.stats import logistic
import imageio
from videoio import videosave, videoread
from moviepy.editor import *
import json
import pylab
from joblib import Parallel, delayed
from natsort import natsorted
import pandas as pd
from scipy import ndimage

class detectFades():
    
    def __init__(self, filename, output_path0='results_of_MR', start_frame=0, end_frame=None, rgb=False, seek_forward = 1):
        
        vid_0 = imageio.get_reader(filename,  'ffmpeg')
        self.ref_img = vid_0.get_data(start_frame)

        if end_frame == None:
            end_frame = vid_0.count_frames()-seek_forward
        else:
            assert end_frame <= vid_0.count_frames()-seek_forward
            
        self.max_frame = vid_0.count_frames()-seek_forward
        self.filename = filename
        self.output_path0 = output_path0
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.rgb = rgb
        self.seek_forward = seek_forward
        
        if not os.path.exists(self.output_path0+'/jsons'):
            os.mkdir(self.output_path0+'/jsons')
            
    def showFrameAndTransition(self,frame):
        
        vid0 = imageio.get_reader(self.filename,  'ffmpeg') # hack around a bug in the implementation
        vid1 = imageio.get_reader(self.filename,  'ffmpeg')

        max_frame = vid0.count_frames()
        if (chunk+1)*INTERVAL > max_frame:
            return

        if self.rgb:
            prev1 = vid0.get_data(frame)
            next1 = vid1.get_data(frame+self.seek_forward)
        else:
            prev1 = cv.cvtColor(vid0.get_data(frame), cv.COLOR_BGR2GRAY)
            next1 = cv.cvtColor(vid1.get_data(frame+self.seek_forward), cv.COLOR_BGR2GRAY)

        fig, ax = plt.subplots(figsize=(20,25))
        if rgb:
            ax.imshow(np.concatenate((prev1,next1),1), vmin=0, vmax=255)
        else:
            ax.imshow(np.concatenate((prev1,next1),1), cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        ax.set_title('Frame '+str(frame))
                     
    # measure the overlap in the histograms
    def histogram_intersection(self, prev_img, next_img):
        
        n, m = prev_img.shape
        N = n*m
    
        bins = 10
        h1 = np.histogram(prev_img.flatten(), bins=np.arange(0,255.1,255/bins))[0]/N
        h2 = np.histogram(next_img.flatten(), bins=np.arange(0,255.1,255/bins))[0]/N

        bins = np.ones((bins,1))

        sm = 0
        for i in range(len(bins)):
            sm += np.minimum(bins[i]*h1[i], bins[i]*h2[i])
        return sm

    # measure the overlap in the histograms
    def histogram_intersection_rgb(self, prev_img_rgb, next_img_rgb):
        
        sm_rgb = np.zeros((3,1))
        for chan in range(int(prev_img_rgb.shape[-1])):
            
            prev_img = prev_img_rgb[:,:,chan]
            next_img = next_img_rgb[:,:,chan]
            
            n, m = prev_img.shape
            N = n*m

            bins = 10
            h1 = np.histogram(prev_img.flatten(), bins=np.arange(0,255.1,255/bins))[0]/N
            h2 = np.histogram(next_img.flatten(), bins=np.arange(0,255.1,255/bins))[0]/N

            bins = np.ones((bins,1))

            sm = 0
            for i in range(len(bins)):
                sm += np.minimum(bins[i]*h1[i], bins[i]*h2[i])
                
            sm_rgb[chan] = sm
            
        return sm_rgb.min()

    def runHistogramOverlapCalc(self, INTERVAL):
        
        Parallel(n_jobs=-1)(delayed(self.calculateHistogramOverlapBatch)(i, INTERVAL, self.max_frame) for i in tqdm(range(1+int(np.ceil((self.end_frame-self.start_frame-self.seek_forward)/INTERVAL)))))
        
    def calculateHistogramOverlapBatch(self, chunk, INTERVAL, max_frame):
    
        for frame in range(chunk*INTERVAL, (chunk+1)*INTERVAL):
            
            vid0 = imageio.get_reader(self.filename,  'ffmpeg') # hack around a bug in the implementation
            vid1 = imageio.get_reader(self.filename,  'ffmpeg')
            
            if self.start_frame+frame+self.seek_forward >= max_frame:
                return

            if not os.path.exists(self.output_path0+'/jsons/'+str(self.start_frame+frame)+'_s'+str(self.seek_forward)+'.json'):

                if self.rgb:
                    prev_img = vid0.get_data(self.start_frame+frame)
                    next_img = vid1.get_data(self.start_frame+frame+self.seek_forward)
                    intersect = float(self.histogram_intersection_rgb(prev_img, next_img))
                else:
                    prev_img = cv.cvtColor(vid0.get_data(self.start_frame+frame), cv.COLOR_BGR2GRAY)
                    next_img = cv.cvtColor(vid1.get_data(self.start_frame+frame+self.seek_forward), cv.COLOR_BGR2GRAY)
                    intersect = float(self.histogram_intersection(prev_img, next_img))
                    
                json_dict = {}
                json_dict[self.start_frame+frame] = intersect
                with open(self.output_path0+'/jsons/'+str(self.start_frame+frame)+'_s'+str(self.seek_forward)+'.json', 'w') as outfile:
                    json.dump(json_dict, outfile)
                    
    def calculateHistogramOverlapSingle(self, frame):
    
        vid0 = imageio.get_reader(self.filename,  'ffmpeg') # hack around a bug in the implementation
        vid1 = imageio.get_reader(self.filename,  'ffmpeg')

        if self.rgb:
            prev_img = vid0.get_data(frame)
            next_img = vid1.get_data(frame+self.seek_forward)
            intersect = float(self.histogram_intersection_rgb(prev_img, next_img))
        else:
            prev_img = cv.cvtColor(vid0.get_data(frame), cv.COLOR_BGR2GRAY)
            next_img = cv.cvtColor(vid1.get_data(frame+self.seek_forward), cv.COLOR_BGR2GRAY)
            intersect = float(self.histogram_intersection(prev_img, next_img))
        
        json_dict = {}
        json_dict[frame] = intersect
        with open(self.output_path0+'/jsons/'+str(frame)+'_s'+str(self.seek_forward)+'.json', 'w') as outfile:
            json.dump(json_dict, outfile)

        return prev_img, next_img
    
    def groupJSONsIdentifyTransitions(self, thold=0.5, display_transitions=False):
        
        dfs = [] # an empty list to store the data frames
        for file in tqdm(glob(self.output_path0+'/jsons/*')):
            data = pd.read_json(file, lines=True) # read data frame from json file
            dfs.append(data.transpose()) # append the data frame to the list

        intersect_df = pd.concat(dfs) # concatenate all the data frames in the list.
        
        frames_identified_df = intersect_df.loc[intersect_df[0]<thold]
        frames_identified = list(frames_identified_df[0].index)
        thresholds_identified = list(frames_identified_df[0])
        
        if display_transitions:
        
            for frm,th0 in zip(frames_identified,thresholds_identified):

                vid0 = imageio.get_reader(self.filename,  'ffmpeg') # hack around a bug in the implementation
                vid1 = imageio.get_reader(self.filename,  'ffmpeg')

                if self.rgb:
                    prev1 = vid0.get_data(frm)
                    next1 = vid1.get_data(frm+self.seek_forward)
                else:
                    prev1 = cv.cvtColor(vid0.get_data(frm), cv.COLOR_BGR2GRAY)
                    next1 = cv.cvtColor(vid1.get_data(frm+self.seek_forward), cv.COLOR_BGR2GRAY)
                    
                fig, ax = plt.subplots(figsize=(20,25))
                ax.imshow(np.concatenate((prev1,next1),1),vmin=0,vmax=255,cmap='gray')
                ax.axis('off')
                ax.set_title('Frame '+str(frm)+' thold: '+str(th0))
                
        frames_identified = [i for i in frames_identified if i>=self.start_frame]
        frames_identified = [i for i in frames_identified if i<=self.end_frame]
        self.frames_identified = frames_identified

        return frames_identified
    
    def identifyLargestMovements(self, frame, mag_thresh=10):

        prev1 = cv.imread(self.output_path0+'/frame_nhe_'+str(frame)+'.png')
        next1 = cv.imread(self.output_path0+'/frame_nhe_'+str(frame+self.seek_forward)+'.png')
        flow = np.load(self.output_path0+'/np_flow_s'+str(self.seek_forward)+'_'+str(frame)+'.npy')
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        magnitude[magnitude == np.inf] = 0
        magnitude[magnitude == -np.inf] = 0
        thold = np.percentile(magnitude.flatten(),99)

        #print(thold)
        if thold < mag_thresh:
            return

        img = magnitude>thold
        # smooth the image (to remove small objects)
        imgf = ndimage.gaussian_filter(img, 1)
        threshold = 0.5

        # find connected components
        labeled, nr_objects = ndimage.label(imgf > threshold) 
        #print("Number of objects is {}".format(nr_objects))

        obj_sizes = [(label, (labeled==label).sum()) for label in range(1,labeled.max()+1)]

        largest_2_comps = list(pd.DataFrame(np.array(obj_sizes), columns={'label','size'}).set_index('label').sort_values('size',ascending=False).iloc[:2].index)

        self.largest1_comp = (labeled==largest_2_comps[0])
        self.largest2_comp = (labeled==largest_2_comps[1])
        
        self.findCentroid()
        
        min_row_1, max_row_1, min_col_1, max_col_1 = self.findBoundingBox(self.row_center_1, self.col_center_1, self.largest1_comp, bb_rel_size = 0.2)
        min_row_2, max_row_2, min_col_2, max_col_2 = self.findBoundingBox(self.row_center_2, self.col_center_2, self.largest2_comp, bb_rel_size = 0.2)
        
        prev_cropped1 = prev1[min_row_1:max_row_1, min_col_1:max_col_1, 0]
        prev_cropped2 = prev1[min_row_2:max_row_2, min_col_2:max_col_2, 0]

        next_cropped1 = next1[min_row_1:max_row_1, min_col_1:max_col_1, 0]
        next_cropped2 = next1[min_row_2:max_row_2, min_col_2:max_col_2, 0]
        
        inter1 = self.histogram_intersection(prev_cropped1, next_cropped1)
        inter2 = self.histogram_intersection(prev_cropped2, next_cropped2)
        #inter = np.minimum(inter1, inter2)
        inter = inter1
        
        return inter, inter1, inter2, magnitude, prev_cropped1, prev_cropped2, next_cropped1, next_cropped2, self.largest1_comp, self.largest2_comp, prev1, next1
        
    def findCentroid(self):
        
        self.row_center_1, self.col_center_1 = np.argwhere(self.largest1_comp==1).sum(0)/self.largest1_comp.sum()
        self.row_center_2, self.col_center_2 = np.argwhere(self.largest2_comp==1).sum(0)/self.largest2_comp.sum()
        
    def findBoundingBox(self, row_center_1, col_center_1, largest1_comp, bb_rel_size = 0.2):

        bb_rel_size = 1/bb_rel_size
        min_row_1 = np.maximum(0,int(row_center_1)-int(largest1_comp.shape[0]/(bb_rel_size*2)))
        max_row_1 = min_row_1 + largest1_comp.shape[0]/bb_rel_size
        if max_row_1 > largest1_comp.shape[0]:
            max_row_1 = largest1_comp.shape[0]
            min_row_1 = max_row_1 - largest1_comp.shape[0]/bb_rel_size

        min_col_1 = np.maximum(0,int(col_center_1)-int(largest1_comp.shape[1]/(bb_rel_size*2)))
        max_col_1 = min_col_1 + largest1_comp.shape[1]/bb_rel_size
        if max_col_1 > largest1_comp.shape[1]:
            max_col_1 = largest1_comp.shape[1]
            min_col_1 = max_col_1 - largest1_comp.shape[1]/bb_rel_size

        return int(min_row_1), int(max_row_1), int(min_col_1), int(max_col_1)

    def writeTransitionFrames(self, hist_eq0):
        
        self.hist_eq0 = hist_eq0
        
        KK1 = glob(self.output_path0+'/frame*_tt_s'+str(self.seek_forward)+'_*')
        for kk in KK1:
            os.remove(kk)
            
        KK2_1 = glob(self.output_path0+'/frame_nhe_'+str(self.start_frame)+'*png')
        KK2_2 = glob(self.output_path0+'/frame_nhe_'+str(self.end_frame-1)+'*png')
        
        if len(KK2_1)+len(KK2_2) < 2:
            self.generateFrames()
        
        #print(self.frames_identified)
        for idx in tqdm(self.frames_identified):
            
            kk = self.output_path0+'/np_flow_s'+str(self.seek_forward)+'_'+str(idx)+'.npy'
            #print(kk)
            transition_frame = cv.imread(kk.replace('np_flow_s'+str(self.seek_forward),'frame_nhe').replace('.npy','.png'))
            
            if self.rgb:
                transition_frame[:int(transition_frame.shape[0]//20),:,:2] = 255
                transition_frame[-int(transition_frame.shape[0]//20):,:,:2] = 255
                transition_frame[:,:int(transition_frame.shape[1]//20),:2] = 255
                transition_frame[:,-int(transition_frame.shape[1]//20):,:2] = 255
            else:
                transition_frame[:,:,2] = 0
            
            cv.imwrite(kk.replace('np_flow_s'+str(self.seek_forward),'frame_nhe').replace('.npy','.png').replace('frame_nhe','frame_tt_s'+str(self.seek_forward)),transition_frame)
            
    def generateFrames(self, INTERVAL=100):
           
        Parallel(n_jobs=-1)(delayed(self.generateSubsetOfFrames)(i, INTERVAL, self.max_frame) for i in tqdm(range(1+int(np.ceil((self.end_frame-self.start_frame)/INTERVAL)))))
        
    def generateSubsetOfFrames(self, chunk, INTERVAL, max_frame):
    
        for frame in range(chunk*INTERVAL, (chunk+1)*INTERVAL):
            
            if not os.path.exists(self.output_path0+'/frame_nhe_'+str(self.start_frame+frame)+'.png'):
        
                vid0 = imageio.get_reader(self.filename,  'ffmpeg') # hack around a bug in the implementation

                if self.start_frame+frame >= max_frame:
                    return

                if self.rgb:
                    new_frame_gray = vid0.get_data(self.start_frame+frame)
                else:
                    new_frame_gray = cv.cvtColor(vid0.get_data(self.start_frame+frame), cv.COLOR_BGR2GRAY)

                #print(new_frame_gray.shape)
                cv.imwrite(self.output_path0+'/frame_nhe_'+str(self.start_frame+frame)+'.png',new_frame_gray)

        