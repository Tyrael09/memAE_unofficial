import os
import numpy as np
import cv2
from tqdm import tqdm, trange
import pandas as pd
from PIL import Image
import imageio.v3 as iio

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from extract_frames import extract_frames

'''
My own Dataloader. Takes a set of mp4 videos and a csv file called "header.csv" containing an ID (same as path minus ".mp4" at the end..), 
a label (0 = regular, 1 = irregular), the total frame count and the video path, both from the same directory, applies some transformations 
and returns the Dataset (consisting of the ID, label and tensor representations of the input frames)
'''
class VideoDataset(Dataset):
    def __init__(self,
                 root_dir='/local/scratch/Cataract-1K-Hendrik/',
                 dataset_name='regular_videos_long/train/',
                 transform_fn=None):

        self.load_clip = self.load_clip_ucfc
            
        self.path = f'{root_dir}/{dataset_name}'
        self.header_df = pd.read_csv(self.path+'header.csv')
        self.lenght = len(self.header_df)
        
        im_resize = transforms.Resize((256, 256),
                                      interpolation=transforms.InterpolationMode.BICUBIC,
                                      antialias=True)
        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                         std=[0.22803, 0.22145, 0.216989])
        if transform_fn:
            self.transform_fn = transforms.Compose([
                im_resize,
                transforms.ToTensor(),
                transform_fn,
                normalize,
                ])
        else:
            self.transform_fn = transforms.Compose([
                im_resize,
                transforms.ToTensor(),
                normalize,
                ])
    
    def __len__(self):
        return self.lenght
    
    def __getitem__(self, index):
        return self.load_clip(index)
    
    def load_clip_ucfc(self, idx):
        video_id, label, frame_count, video_path = self.header_df.iloc[idx][['video_id', 'label', 'frame_count', 'video_path']]
        file_path = os.path.join(self.path, video_path)
        video = []
        # Read video using imageio.v3
        frames = iio.imread(file_path, plugin='pyav')

        for frame in frames:
            image = Image.fromarray(frame)
            image = self.transform_fn(image)
            video.append(image)
        return {'clip_id': video_id, 'label':label, 'data': torch.stack(video, dim=0)}

class VideoClipDataset(Dataset):
    def __init__(self,
                 root_dir='/local/scratch/Cataract-1K-Full-Videos/',
                 dataset_name='train',
                 clip_len=16,
                 split='train',
                 load_reminder=False,
                 transform_fn=None):
        self.path = f'{root_dir}/{dataset_name}'
        self.clip_len = clip_len
        # self.vid_header_df = pd.read_csv(self.path+f'/header_{split}.csv')
        self.vid_header_df = pd.read_csv(f'/local/scratch/hendrik/video_annotations_full.csv') # TODO: use correct annotations for train/test set!!
        if load_reminder:
            self.header_df = pd.read_csv(self.path+f'/reminders_{split}.csv')
        else:
            self.header_df = self.vid_header_df #  pd.read_csv(self.path+f'/header_{split}.csv') # was '/splits_header_{split}.csv'
        self.length = len(self.header_df)
        
        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                         std=[0.22803, 0.22145, 0.216989])
        if transform_fn:
            self.transform_fn = transforms.Compose([
                transforms.ToTensor(),
                transform_fn,
                normalize,
                ])
        else:
            self.transform_fn = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.load_clip(index)

    def frame_generator(self, video_path, start_frame, end_frame, clip_length=16):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # Extract the numeric ID using split()
        video_id = int(os.path.basename(video_path).split('_')[1].split('.')[0])
        clip_count = 0
        max_clip_count = 50 # increase value to cover entire video
        while start_frame + clip_length <= end_frame and clip_count < max_clip_count:
            clip = []
            for _ in range(0, clip_length * 2, 4): # start, end, step -> skips every 2nd frame
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    return
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = self.transform_fn(image)
                clip.append(image)
            yield {'clip_id': video_id, 'start': start_frame, 'data': torch.stack(clip, dim=0)}
            clip_count += 1
            start_frame += clip_length * 4 # reduce number of clips per video
        cap.release()

    def load_clip(self, idx):
        frame_rate = 60  # Assuming a frame rate of 60 fps
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        start_frame = int(start * frame_rate)
        end_frame = int(end * frame_rate)
        video_path = os.path.join(self.path, f'{video_id}.mp4')
        clips = []
        for clip in self.frame_generator(video_path, start_frame, end_frame):
            clips.append(clip)
            #if len(clip) > 4: 
            #    break
        return clips
    
    def load_clip_2(self, idx):
        frame_rate = 60  # Assuming a frame rate of 60 fps
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        start_frame = int(start * frame_rate)
        end_frame = int(end * frame_rate)
        
        video_path = os.path.join(self.path, f'{video_id}.mp4')
        cap = cv2.VideoCapture(video_path)
        
        clip_length = 16  # Number of frames in each clip
        clip_start = start_frame
        clips = []
        
        while clip_start + clip_length <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
            clip = []
            for frame_idx in range(clip_start, clip_start + clip_length):
                ret, frame = cap.read()
                if not ret:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = self.transform_fn(image)
                clip.append(image)
            
            if len(clip) == clip_length:
                clips.append({'clip_id': video_id, 'start': clip_start, 'data': torch.stack(clip, dim=0)})
            
            clip_start += clip_length  # Move to the next segment
        
        cap.release()
        return clips


    def load_clip_old(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        file_path = os.path.join(self.path, 'frames', video_id)

        clip = []
        for i in range(int(start*60), int(end*60)):
            img_path = f'{file_path}/{i}.jpg'
            with Image.open(img_path) as image:
                image = self.transform_fn(image)
            clip.append(image)
        return {'clip_id': video_id, 'start': start, 'data': torch.stack(clip, dim=0)}

if __name__ == '__main__':
    print('Hello there!')
    # root_dir='../data'
    ds = VideoDataset() 
    clip = ds[0]['data']
    print(clip.shape)
    print('Obiwan Kenobi!')
