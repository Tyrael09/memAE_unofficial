import numpy as np
from collections import OrderedDict
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2

rng = np.random.RandomState(2020)

class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._time_step = time_step
        self._num_pred = num_pred
        self.videos, video_string = setup(self.dir, self.videos)
        #elif type(video_folder) is list:
        #    self.videos, video_string = setup_multiple(self.dir, self.videos)
        self.samples = self.get_all_samples(video_string)
        self.video_name = video_string

    def get_all_samples(self, video_string):
        samples = []
        for video in video_string:
            for i in range(len(self.videos[video]['length']) - self._time_step):
                samples.append((video, i))
        return samples
    
    def load_frames_from_video(self, video_path, start_frame, num_frames):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        
        cap.release()
        return frames

    def __getitem__(self, index):
        video_name, start_frame = self.samples[index]
        video_path = self.videos[video_name]['path'] + '.mp4'  # Assuming videos are stored in .mp4 format
        num_frames = self._time_step + self._num_pred
        frames = self.load_frames_from_video(video_path, start_frame, num_frames)
        
        batch = []
        for frame in frames:
            if self.transform is not None:
                batch.append(self.transform(frame))
        
        return torch.cat(batch, 0)

    def __len__(self):
        return len(self.samples)
    
def setup(path, videos):
    video_files = sorted([v for v in os.listdir(path) if v.endswith('.mp4')])
    video_string = [os.path.splitext(v)[0] for v in video_files]
    
    for video in video_string:
        video_path = os.path.join(path, video + '.mp4')
        videos[video] = {'path': video_path, 'length': get_video_length(video_path)}
    
    return videos, video_string

def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length
