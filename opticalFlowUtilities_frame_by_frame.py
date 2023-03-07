import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pylab
import imageio
from tqdm import tqdm
from joblib import Parallel, delayed
from videoio import videosave, videoread
import os
from moviepy.editor import *
import os
from natsort import natsorted
from glob import glob

class generateOpticalFlowResults():

    def __init__(self,filename, hist_eq = False, seek_forward = 1):
        
        #if not os.path.exists(output_path):
        #    os.mkdir(output_path)
            
        self.rgb = False
        self.filename = filename
        #self.output_path = output_path
        self.hist_eq = hist_eq

    def performOpticalFlowAndSave(self, frame_img, frame_cnn, hist_eq=False):
    
        prev_frame_gray = cv.cvtColor(frame_img, cv.COLOR_BGR2GRAY)
        new_frame_gray = cv.cvtColor(frame_cnn, cv.COLOR_BGR2GRAY)

        if hist_eq == True:
            prev_frame_gray = cv.equalizeHist(prev_frame_gray)
            new_frame_gray = cv.equalizeHist(new_frame_gray)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_frame_gray, new_frame_gray, 
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)

        return flow
        
    """

    def showFrameAndTransition(self,frame):
        
        vid0 = imageio.get_reader(self.filename,  'ffmpeg') # hack around a bug in the implementation
        vid1 = imageio.get_reader(self.filename,  'ffmpeg')

        prev1 = cv.cvtColor(vid0.get_data(frame), cv.COLOR_BGR2GRAY)
        next1 = cv.cvtColor(vid1.get_data(frame+self.seek_forward), cv.COLOR_BGR2GRAY)

        fig, ax = plt.subplots(figsize=(20,25))
        ax.imshow(np.concatenate((prev1,next1),1),vmin=0, vmax=255, cmap='gray')
        ax.axis('off')
        ax.set_title('Frame '+str(frame))
        

            
    def dispOpticalFlow(self, Image,Flow,Divisor,name):
        "Display image with a visualisation of a flow over the top. A divisor controls the density of the quiver plot."
        PictureShape = np.shape(Image)
        #determine number of quiver points there will be
        Imax = int(PictureShape[0]/Divisor)
        Jmax = int(PictureShape[1]/Divisor)
        #create a blank mask, on which lines will be drawn.
        mask = np.zeros_like(Image)
        for i in range(1, Imax):
            for j in range(1, Jmax):
                X1 = (i)*Divisor
                Y1 = (j)*Divisor
                X2 = int(X1 + Flow[X1,Y1,1])
                Y2 = int(Y1 + Flow[X1,Y1,0])
                X2 = np.clip(X2, 0, PictureShape[0])
                Y2 = np.clip(Y2, 0, PictureShape[1])
                #add all the lines to the mask
                mask = cv.line(mask, (Y1,X1),(Y2,X2), [255, 255, 255], 1)
        #superpose lines onto image

        Image1 = np.zeros((np.shape(Image)[0],np.shape(Image)[1],3),np.uint8)
        Image1[:,:,0] = Image
        Image1[:,:,1] = Image
        Image1[:,:,2] = Image

        Image1[:,:,1] = cv.add(Image1[:,:,1],mask)

        return Image1/255.0

    def generateOpticalFlowVideo(self):
    
        INTERVAL = 500 # split the video into chunks to conserve memory
        num_chunks = int(np.ceil((self.end_frame-self.start_frame-self.seek_forward)/INTERVAL))

        for chunk in range(num_chunks):
            if not os.path.exists(self.output_path+"/vid0_"+str(chunk)+".mp4"):
            
                first_frame_chunk = self.start_frame+chunk*INTERVAL
                end_frame_chunk = self.start_frame+(chunk+1)*INTERVAL
                if end_frame_chunk >= self.end_frame:
                    end_frame_chunk = self.end_frame - self.seek_forward -1

                print(end_frame_chunk, first_frame_chunk)
                opt_flow = np.zeros((1+end_frame_chunk-first_frame_chunk,self.ref_img.shape[0],self.ref_img.shape[1],self.ref_img.shape[2]),np.uint8)
                video_np = np.zeros((1+end_frame_chunk-first_frame_chunk,self.ref_img.shape[0],self.ref_img.shape[1],self.ref_img.shape[2]),np.uint8)

                kk10 = 0 
                for frame in tqdm(range(first_frame_chunk,end_frame_chunk+1)):

                    img_cv = cv.imread(self.output_path+'/frame_'+self.mid+str(frame)+'.png')
                    opt_cv = cv.imread(self.output_path+'/optical_flow_s'+str(self.seek_forward)+'_'+str(frame)+'.png')

                    opt_flow[kk10,:,:,:] = opt_cv
                    video_np[kk10,:,:,:] = img_cv
                    kk10+=1

                videosave(self.output_path+"/out0_"+str(chunk)+".mp4", opt_flow/255)
                videosave(self.output_path+"/vid0_"+str(chunk)+".mp4", video_np/255)
            
    def fuseVideoFiles(self,prefix='out0_',suffix='mp4'):
        
        if not os.path.exists(self.output_path+'/'+prefix+"fused."+suffix):
        
            L =[]
            KK = glob(self.output_path+'/'+prefix+'*.'+suffix)
            KK = [k for k in KK if 'fused' not in k]

            for i in range(len(KK)):

                files = self.output_path+'/'+prefix+str(i)+'.'+suffix
                video = VideoFileClip(files)
                L.append(video)

            final_clip = concatenate_videoclips(L)
            final_clip.to_videofile(self.output_path+'/'+prefix+"fused."+suffix, fps=24)
        
    def cleanupFiles(self, prefix='out0_',suffix='mp4'):
    
        KK = glob(self.output_path+'/'+prefix+'*.'+suffix)
        KK = [k for k in KK if 'fused' not in k]
        #print(KK)
        for kk in KK:
            os.remove(kk)
            
    def computeMagnitudes(self, approach):
        
        self.approach = approach
        #KK = glob(self.output_path+'/*npy')
        KK = [self.output_path+'/np_flow_s'+str(self.seek_forward)+'_'+str(k)+'.npy' for k in range(1+self.start_frame, self.end_frame- self.seek_forward)]

        magnitude_vect = -1*np.ones((self.end_frame + 1,1))
        magnitude_vect = -1*np.ones((self.end_frame + 1,1))
        
        for kk in tqdm(KK):

            flow = np.load(kk)

            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

            if approach == 1:
                #approach 1
                magnitude_vect[int(kk.split('np_flow_s'+str(self.seek_forward)+'_')[-1].split('.')[0])] = magnitude.sum()
            elif approach == 2:
                #approach 2
                magnitude_vect[int(kk.split('np_flow_s'+str(self.seek_forward)+'_')[-1].split('.')[0])] = (magnitude>10).sum()
            elif approach == 3:
                magnitude_vect[int(kk.split('np_flow_s'+str(self.seek_forward)+'_')[-1].split('.')[0])] = (magnitude>10).sum()
                angles = angle * 360 / (np.pi * 2)
                temp1 = angles*(magnitude>5)
                temp1_flat = temp1.flatten()
                temp1_flat2 = temp1_flat[temp1_flat>0]
                [bincount, edges] = np.histogram(temp1_flat2,36)
                if len(bincount[bincount>1000]) > 25:
                    magnitude_vect[int(kk.split('np_flow_s'+str(self.seek_forward)+'_')[-1].split('.')[0])] = 1e8

        self.magnitude_vect_processed = magnitude_vect[magnitude_vect>-1]
        frame_idx_processed = np.expand_dims(np.arange(self.end_frame + 1),1)
        self.frame_idx_processed = frame_idx_processed[magnitude_vect>-1]
    
    def thresholdMagnitudes(self):
        
        frame_idx_processed2 = self.frame_idx_processed[np.isfinite(self.magnitude_vect_processed)]
        magnitude_vect_processed2 = self.magnitude_vect_processed[np.isfinite(self.magnitude_vect_processed)]

        if self.approach == 1:
            #approach 1
            thold = 5.0e6
        elif self.approach == 2:
            #approach 2
            thold = 1.0e4
        elif self.approach == 3:
            #approach 3
            thold = 5.0e6

        self.frame_idx_processed3 = frame_idx_processed2[magnitude_vect_processed2>thold]
        self.magnitude_vect_processed3 = magnitude_vect_processed2[magnitude_vect_processed2>thold]

        return self.frame_idx_processed3
    
    def writeTransitionFrames(self):
        
        KK1 = glob(self.output_path+'/frame*_t_s'+str(self.seek_forward)+'*')
        for kk in KK1:
            os.remove(kk)
        
        for idx in tqdm(self.frame_idx_processed3):
            kk = self.output_path+'/np_flow_s'+str(self.seek_forward)+'_'+str(idx)+'.npy'
            transition_frame = cv.imread(kk.replace('np_flow_s'+str(self.seek_forward),'frame_'+self.mid[:-1]).replace('.npy','.png'))
            transition_frame[:,:,2] = 0
            cv.imwrite(kk.replace('np_flow_s'+str(self.seek_forward),'frame_'+self.mid[:-1]).replace('.npy','.png').replace('frame','frame_t_s'+str(self.seek_forward)),transition_frame)
            
    def createTransitionVideos(self):
        
        KK2 = glob(self.output_path+'/frame_nhe_*')
        print(len(KK2), self.end_frame, self.start_frame, self.end_frame - self.start_frame)
        if len(KK2) < self.end_frame - self.start_frame:
            self.generateFrames()

        INTERVAL = 400
        num_chunks = int(np.ceil((self.end_frame-self.start_frame)/INTERVAL))
        Parallel(n_jobs=-1)(delayed(self.createVid)(chunk, INTERVAL) for chunk in tqdm(range(num_chunks)))
        
    def createVid(self,chunk, INTERVAL):

        if not os.path.exists(self.output_path+"/vid_t"+str(self.approach)+"_"+str(chunk)+".mp4"):

            first_frame_chunk = self.start_frame+chunk*INTERVAL
            end_frame_chunk = self.start_frame+(chunk+1)*INTERVAL
            if end_frame_chunk >= self.end_frame:
                end_frame_chunk = self.end_frame - 1 - self.seek_forward

            video_np = np.zeros((end_frame_chunk-first_frame_chunk,self.ref_img.shape[0],self.ref_img.shape[1],self.ref_img.shape[2]),np.uint8)

            kk10 = 0 
            for frame in tqdm(range(first_frame_chunk+1,end_frame_chunk+1)):

                Kt = glob(self.output_path+'/frame_t_s*_'+str(frame)+'.png')
                Ktt = glob(self.output_path+'/frame_tt_s*_'+str(frame)+'.png')

                if len(Kt) > 0:
                    img_cv = cv.imread(Kt[0])
                elif len(Ktt):
                    img_cv = cv.imread(Ktt[0])
                else:
                    img_cv = cv.imread(self.output_path+'/frame_nhe_'+str(frame)+'.png')

                video_np[kk10,:,:,:] = img_cv
                kk10+=1

            videosave(self.output_path+"/vid_t"+str(self.approach)+"_"+str(chunk)+".mp4", video_np/255)
                
                
    def generateFrames(self, INTERVAL=100):
           
        #print(self.max_frame)
        Parallel(n_jobs=-1)(delayed(self.generateSubsetOfFrames)(i, INTERVAL, self.end_frame) for i in tqdm(range(1+int(np.ceil((self.end_frame-self.start_frame-self.seek_forward)/INTERVAL)))))
        
    def generateSubsetOfFrames(self, chunk, INTERVAL, max_frame):
    
        for frame in range(chunk*INTERVAL, (chunk+1)*INTERVAL):
            
            if not os.path.exists(self.output_path+'/frame_nhe_'+str(self.start_frame+frame)+'.png'):
        
                vid0 = imageio.get_reader(self.filename,  'ffmpeg') # hack around a bug in the implementation

                if self.start_frame+frame >= max_frame:
                    return

                if self.rgb:
                    new_frame_gray = vid0.get_data(self.start_frame+frame)
                else:
                    new_frame_gray = cv.cvtColor(vid0.get_data(self.start_frame+frame), cv.COLOR_BGR2GRAY)

                #print(new_frame_gray.shape)
                cv.imwrite(self.output_path+'/frame_nhe_'+str(self.start_frame+frame)+'.png',new_frame_gray)
    """