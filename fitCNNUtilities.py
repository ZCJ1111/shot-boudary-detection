import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from glob import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os 
from sklearn.metrics import confusion_matrix
from scipy.stats import logistic
import imageio
from videoio import videosave, videoread
from moviepy.editor import *
from joblib import Parallel, delayed

class analysisWithCNN():
    
    def __init__(self, output_path0, output_path0_model, hist_eq0, rgb=False, seek_forward = 1):
        
        if hist_eq0 == True:
            self.mid = 'he_'
        else:
            self.mid = 'nhe_'
            
        self.rgb = rgb
        self.output_path0_model = output_path0_model
        self.output_path0 = output_path0
        if not torch.cuda.is_available():
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823), (0.2470, 0.2435))])
        self.seek_forward = seek_forward
        
        if hist_eq0 == False:
            self.transition_frames_2 = [ 199,  297,  641,  689,  737,  785,  833,  881,  929,  977, 1197,
                                       1241, 1289, 1357, 1453, 1517, 1589, 1673, 1778, 1882, 2000, 2123,
                                       2165, 2215, 2268, 2312, 2356, 2380, 2427, 2475, 2555, 2626, 2692,
                                       2736, 2790, 2840, 2883, 2931, 2981, 3051, 3147, 3242, 3292, 3422,
                                       3518, 3550, 3758, 3808, 4093, 4185, 4240, 4320, 4343, 4369, 4413,
                                       4470, 4520, 4535, 4588, 4634, 4855, 4960, 5152, 5265, 5311]
        elif hist_eq0 == True:
            self.transition_frames_2 = [ 32, 199,  297,  626,  641,  689,  737,  785,  833,  881,  929, 
                                    977, 1121, 1141, 1164, 1197, 1241, 1289, 1357, 1411, 1425, 1453,
                                   1517, 1589, 1673, 1778, 1882, 2000, 2050, 2123, 2165, 2215, 2268,
                                   2312, 2356, 2380, 2427, 2475, 2511, 2525, 2555, 2626, 2692, 2736,
                                   2790, 2840, 2883, 2931, 2981, 3051, 3147, 3202, 3242, 3292, 3422,
                                   3518, 3550, 3611, 3758, 3808, 3982, 4093, 4185, 4240, 4320, 4343,
                                   4369, 4413, 4470, 4520, 4535, 4588, 4634, 4855, 4960, 5152, 5265,
                                   5311, 5535 ]
            
        self.transition_frames_2 = [i-1 for i in self.transition_frames_2] # correcting for change to seek forward
    
    def showTrainingData(self, max_show = 10):
        
        if max_show == None:
            max_show = len(self.transition_frames_2)
            
        for frm in self.transition_frames_2[:max_show]:

            prev1 = cv.imread(self.output_path0+'/optical_flow_s'+str(self.seek_forward)+'_'+str(frm)+'.png')
            next1 = cv.imread(self.output_path0+'/optical_flow_s'+str(self.seek_forward)+'_'+str(frm+self.seek_forward)+'.png')
            fig, ax = plt.subplots(figsize=(20,25))
            ax.imshow(np.concatenate((prev1,next1),1))
            ax.axis('off')
            ax.set_title('Frame '+str(frm))

    def defineTrainValDatasets(self):
    
        KK = glob(self.output_path0+'/np_flow_s'+str(self.seek_forward)+'_*npy')
        KK_t0 = [self.output_path0+'/np_flow_s'+str(self.seek_forward)+'_'+str(k)+'.npy' for k in self.transition_frames_2]
        KK_nt0 = list(set(KK) - set(KK_t0))

        val_percent = self.val_percent

        KK_t_train = KK_t0[:int((1-val_percent)*len(KK_t0))]
        KK_t_val = KK_t0[int((1-val_percent)*len(KK_t0)):]

        KK_nt_train = KK_nt0[:int((1-val_percent)*len(KK_nt0))]
        KK_nt_val = KK_nt0[int((1-val_percent)*len(KK_nt0)):]

        KK_nt_val = KK_nt_val[::15]
        KK_nt_train = KK_nt_train[::15]

        data_train = KK_t_train + KK_nt_train
        targets_train = list(np.ones((len(KK_t_train))))+list(np.zeros((len(KK_nt_train))))

        self.data_val = KK_t_val + KK_nt_val
        self.targets_val = list(np.ones((len(KK_t_val))))+list(np.zeros((len(KK_nt_val))))

        idx = np.arange(len(data_train),dtype=int)
        np.random.shuffle(idx)

        self.data_train = [data_train[k] for k in idx]
        self.targets_train = [targets_train[k] for k in idx]
        
    def createDataloader(self):
    
        dataset_train = self.MyDataset(self.data_train, self.targets_train, transform=self.transform)
        self.dataloader_train = DataLoader(dataset_train, batch_size=10)

        dataset_val = self.MyDataset(self.data_val, self.targets_val, transform=self.transform)
        self.dataloader_val = DataLoader(dataset_val, batch_size=5)
        
    def trainCNN(self, val_percent = 0.2, lrate = 1e-2, n_epochs = 100000):
    
        self.val_percent = val_percent
        self.lrate = lrate
        self.n_epochs = n_epochs
        
        self.defineTrainValDatasets()
        self.createDataloader()
        
        # defining the model
        self.model = self.Net()
        # defining the loss function
        self.criterion = nn.BCEWithLogitsLoss()
        # checking if GPU is available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        # create a new model with these weights
        self.model.apply(self.weights_init_uniform_rule)

        self.opt_val_loss = 100000
        self.counter = 0
        self.flag_early_stop = False

        for epoch in range(n_epochs):
            self.train_loop(epoch)
            if self.flag_early_stop: break
        
    class MyDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = data
            self.targets = torch.FloatTensor(targets)
            self.transform = transform

        def __getitem__(self, index):
            path = self.data[index]
            y = self.targets[index]

            if self.transform:
                x = np.load(path)
                x = self.transform(x)

            return x, y

        def __len__(self):
            return len(self.data)

    def weights_init_uniform_rule(self,m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    class Net(nn.Module):   
        def __init__(self):
            super(analysisWithCNN.Net, self).__init__()

            self.cnn_layers = nn.Sequential(
                # Defining a 2D convolution layer
                nn.Conv2d(2, 4, kernel_size=5, stride=2, padding=1),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Defining another 2D convolution layer
                nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Defining another 2D convolution layer
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Defining another 2D convolution layer
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.linear_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.Linear(50, 1)
            )

        # Defining the forward pass    
        def forward(self, x):
            #print(x.shape)
            x = self.cnn_layers(x)
            #print(x.shape)
            x = x.view(x.size(0), -1)
            #print(x.shape)
            x = self.linear_layers(x)
            #print(x.shape)
            return x
        
    def train_loop(self, epoch): #, model, epoch, opt_val_loss, counter, flag_early_stop, lrate

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lrate)

        self.model.train()

        tr_loss = 0
        val_loss = 0
        train_losses = []
        val_losses = []

        for sample, target in tqdm(self.dataloader_train):

            if torch.cuda.is_available():
                x_train = sample.cuda()
                y_train = target.cuda()
            else:
                x_train = sample
                y_train = target

            y_train = torch.unsqueeze(y_train,1)

            # clearing the Gradients of the model parameters
            optimizer.zero_grad()

            # prediction for training and validation set
            output_train = self.model(x_train)

            # computing the training and validation loss
            loss_train = self.criterion(output_train, y_train)
            train_losses.append(float(loss_train.cpu().detach().numpy()))

            # computing the updated weights of all the model parameters
            loss_train.backward()
            optimizer.step()
            tr_loss = loss_train.item()

        self.model.eval()

        for sample, target in tqdm(self.dataloader_val):

            if torch.cuda.is_available():
                x_val = sample.cuda()
                y_val = target.cuda()
            else:
                x_val = sample
                y_val = target

            self.x_val = x_val
            y_val = torch.unsqueeze(y_val,1)

            # prediction for training and validation set
            output_val = self.model(x_val)

            # computing the training and validation loss
            loss_val = self.criterion(output_val, y_val)
            val_losses.append(float(loss_val.cpu().detach().numpy()))

        if np.mean(np.array(val_losses)) < self.opt_val_loss:
            self.opt_val_loss = np.mean(np.array(val_losses))
            torch.save(self.model, self.output_path0_model+'/optimal_model.ckpt')
            self.counter = 0
        else:
            self.counter +=1

        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'train_loss :', np.mean(np.array(train_losses)), '\t', 'val_loss :', np.mean(np.array(val_losses)), '\t', 'opt_val_loss :', self.opt_val_loss, '\t', 'lrate :', self.lrate)

        if self.counter == 10:
            self.lrate /= 10
            self.counter = 0
            self.model = torch.load(self.output_path0_model+'/optimal_model.ckpt')
            self.opt_val_loss = 100000

            if self.lrate < 1e-5:
                self.flag_early_stop = True
  
    def computeValPredictions(self):
        
        if torch.cuda.is_available():
            model = torch.load(self.output_path0_model+'/optimal_model.ckpt')
        else:
            model = torch.load(self.output_path0_model+'/optimal_model.ckpt', map_location=torch.device('cpu'))

        kk10 = 0
        for sample, target in tqdm(self.dataloader_val):

            if torch.cuda.is_available():
                x_val = sample.cuda()
                y_val = target.cuda()
            else:
                x_val = sample
                y_val = target

            # prediction for training and validation set
            output_val = model(x_val)

            if kk10 == 0:
                pred_val_concat = output_val
                y_val_concat = y_val
            else:
                pred_val_concat = torch.concat([pred_val_concat,output_val])
                y_val_concat = torch.concat([y_val_concat,y_val])

            kk10 +=1
            
        self.preds_vect = logistic.cdf(pred_val_concat.cpu().detach().numpy())
        self.true_vect = y_val_concat.cpu().detach().numpy()
        
        plt.scatter(self.true_vect, self.preds_vect)
        
    def displayValTransitions(self, thold=0.1):
        
        thresholded_frames = [self.data_val[k] for k in range(len(self.data_val)) if self.preds_vect[k]>thold]
        thresholded_vals = [self.preds_vect[k] for k in range(len(self.data_val)) if self.preds_vect[k]>thold]
        
        assert self.seek_forward==1
        for frm0,th0 in zip(thresholded_frames,thresholded_vals):

            frm = int(frm0.split('.')[0].split('_')[-1])

            prev1 = cv.imread(self.output_path0+'/optical_flow_s'+str(self.seek_forward)+'_'+str(frm)+'.png')
            next1 = cv.imread(self.output_path0+'/optical_flow_s'+str(self.seek_forward)+'_'+str(frm+self.seek_forward)+'.png')
            fig, ax = plt.subplots(figsize=(20,25))
            ax.imshow(np.concatenate((prev1,next1),1))
            ax.axis('off')
            ax.set_title('Frame '+str(frm)+' thold: '+str(th0))
            
    def showConfusionMatrix(self,thold):
        
        C = confusion_matrix(self.true_vect, self.preds_vect>thold)
        print(C)
        
    def createTestDataLoader(self):
    
        self.data_test = [self.output_path0_test+'/np_flow_s'+str(self.seek_forward)+'_'+str(k)+'.npy' for k in list(np.arange(self.start_frame, self.end_frame-self.seek_forward))]

        targets_test = np.zeros(len(self.data_test))
        dataset_test = self.MyDataset(self.data_test, targets_test, transform=self.transform)
        self.dataloader_test = DataLoader(dataset_test, batch_size=1)

    def runModelOnTestData(self, filename_test, output_path0_test, output_path0_ref, end_frame, start_frame = 0):

        vid_0 = imageio.get_reader(filename_test,  'ffmpeg')
        if end_frame == None:
            end_frame = vid_0.count_frames()
        else:
            assert end_frame <= vid_0.count_frames()

        self.max_frame = int(vid_0.count_frames())
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.output_path0_test = output_path0_test
        self.output_path0_ref = output_path0_ref
        self.createTestDataLoader()
        
        if os.path.exists(output_path0_test+'/test_preds.npy'):
            aa = np.load(output_path0_test+'/test_preds.npy')
            self.preds_test_vect = np.array([k[0] for k in aa])
            
        else:

            if torch.cuda.is_available():
                model = torch.load(self.output_path0+'/optimal_model.ckpt')
            else:
                model = torch.load(self.output_path0+'/optimal_model.ckpt', map_location=torch.device('cpu'))

            example_train_data = glob(self.output_path0_ref+'/frame_*png')[0]
            exam_img = cv.imread(example_train_data)

            n, m = exam_img.shape[-3:-1]
            kk10 = 0
            for sample, target in tqdm(self.dataloader_test):

                if torch.cuda.is_available():
                    x_test = sample.cuda()
                    y_test = target.cuda()
                else:
                    x_test = sample
                    y_test = target

                x_test = torch.squeeze(x_test)
                x_test_res_0 = cv.resize(x_test[0,:,:].cpu().detach().numpy(), (m,n), interpolation = cv.INTER_AREA)
                x_test_res_1 = cv.resize(x_test[1,:,:].cpu().detach().numpy(), (m,n), interpolation = cv.INTER_AREA)

                x_test_res2_0 = torch.unsqueeze(torch.Tensor(x_test_res_0),0)
                x_test_res2_1 = torch.unsqueeze(torch.Tensor(x_test_res_1),0)

                if torch.cuda.is_available():
                    x_test = torch.unsqueeze(torch.cat([x_test_res2_0,x_test_res2_1],0),0).cuda()
                else:
                    x_test = torch.unsqueeze(torch.cat([x_test_res2_0,x_test_res2_1],0),0)

                # prediction for training and validation set
                output_test = model(x_test)

                if kk10 == 0:
                    pred_test_concat = output_test.cpu().detach().numpy()
                else:
                    pred_test_concat = np.concatenate([pred_test_concat,output_test.cpu().detach().numpy()])

                kk10 +=1

            self.preds_test_vect = logistic.cdf(pred_test_concat)
            np.save(output_path0_test+'/test_preds.npy',self.preds_test_vect)

    def createTestTransitionFrames(self):
        
        KK1 = glob(self.output_path0_test+'/frame*_t_s'+str(self.seek_forward)+'_*')
        for kk in KK1:
            os.remove(kk)

        transition_potential = [self.data_test[k] for k in range(len(self.preds_test_vect)) if self.preds_test_vect[k]>self.trans_thold]
        transition_prob = [self.preds_test_vect[k] for k in range(len(self.preds_test_vect)) if self.preds_test_vect[k]>self.trans_thold]
        idx_trans = [int(k.split('.')[0].split('_')[-1]) for k in transition_potential]

        #for i,k in zip(transition_potential,transition_prob):
        #    print(i,k )
            
        for idx in tqdm(idx_trans):
            kk = self.output_path0_test+'/np_flow_s'+str(self.seek_forward)+'_'+str(idx)+'.npy'
            transition_frame = cv.imread(kk.replace('np_flow_s'+str(self.seek_forward),'frame_'+self.mid[:-1]).replace('.npy','.png'))
            transition_frame[:,:,2] = 0
            cv.imwrite(kk.replace('np_flow_s'+str(self.seek_forward),'frame_'+self.mid[:-1]).replace('.npy','.png').replace('frame_'+self.mid[:-1],'frame_t_s'+str(self.seek_forward)),transition_frame)

    def createVideosWithTransitions(self, filename_test, INTERVAL = 400, approach = 3, trans_thold=0.1, preview=False, workers=4):
        
        #KK2 = glob(self.output_path+'/frame_nhe_*')
        #print(len(KK2), self.end_frame, self.start_frame, self.end_frame - self.start_frame)
        #if len(KK2) < self.end_frame - self.start_frame:
        #    self.generateFrames()
        #INTERVAL = 400
        #num_chunks = int(np.ceil((self.end_frame-self.start_frame)/INTERVAL))
        
        self.trans_thold = trans_thold
        self.createTestTransitionFrames()
        self.approach = approach
        self.preview = preview
        
        if INTERVAL > self.max_frame:
            INTERVAL = self.max_frame

        vid_0 = imageio.get_reader(filename_test,  'ffmpeg')
        self.ref_img = vid_0.get_data(0)
        num_chunks = int(1+np.ceil((self.end_frame-self.start_frame)/INTERVAL))
        
        Parallel(n_jobs=workers)(delayed(self.createVid)(chunk, INTERVAL) for chunk in tqdm(range(num_chunks)))
        
    def createVid(self,chunk, INTERVAL):
            
        if not os.path.exists(self.output_path0_test+"/vid_t"+str(self.approach)+"_"+str(chunk)+".mp4") or self.preview == True:

            first_frame_chunk = self.start_frame+chunk*INTERVAL
            end_frame_chunk = self.start_frame+(chunk+1)*INTERVAL

            if first_frame_chunk > self.end_frame:
                return
            if end_frame_chunk > self.end_frame:
                end_frame_chunk = self.end_frame

            if self.preview == False:
                video_np = np.zeros((end_frame_chunk-first_frame_chunk,self.ref_img.shape[0],self.ref_img.shape[1],self.ref_img.shape[2]),np.uint8)

            kk10 = 0 

            for frame in tqdm(range(first_frame_chunk,end_frame_chunk)):

                if frame >= self.end_frame:
                    continue

                Kt = glob(self.output_path0_test+'/frame_t_s'+str(self.seek_forward)+'_'+str(frame)+'.png')
                Ktt = glob(self.output_path0_test+'/frame_tt_s*'+'_'+str(frame)+'.png')

                flag_trans = True
                if len(Kt)>0:
                    img_cv = cv.imread(Kt[0])
                elif len(Ktt)>0:
                    img_cv = cv.imread(Ktt[0])
                else:
                    img_cv = cv.imread(self.output_path0_test+'/frame_nhe_'+str(frame)+'.png')
                    flag_trans = False

                if flag_trans == True and self.preview==True:                
   
                    KK_t = Kt + Ktt
                    prev1 = cv.imread(KK_t[0])
                    next1 = cv.imread(self.output_path0_test+'/frame_nhe_'+str(frame+1)+'.png')
                    fig, ax = plt.subplots(figsize=(20,25))
                    ax.imshow(np.concatenate((prev1,next1),1))
                    ax.axis('off')
                    ax.set_title('Frame: '+str(frame)+', Pred: '+str(self.preds_test_vect[frame]))
                    plt.show()

                if self.preview == False:
                    video_np[kk10,:,:,:] = img_cv
                kk10+=1

            if self.preview == False:
                videosave(self.output_path0_test+"/vid_t"+str(self.approach)+"_"+str(chunk)+".mp4", video_np/255)

        else:
            pass

    def combineVideos(self):

        L =[]
        KK = glob(self.output_path0_test+'/vid_t3_*.mp4')
        KK = [k for k in KK if 'fused' not in k]

        for i in tqdm(range(len(KK))):

            files = self.output_path0_test+'/vid_t3_'+str(i)+'.mp4'
            video = VideoFileClip(files)
            L.append(video)

        final_clip = concatenate_videoclips(L)
        final_clip.to_videofile(self.output_path0_test+'/vid_t3_fused.mp4', fps=24)