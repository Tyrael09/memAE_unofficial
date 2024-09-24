import pandas as pd
import numpy as np
from collections import OrderedDict
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
from PIL import Image

rng = np.random.RandomState(2020) # not sure why this is here but will use it anyway

class MyDataset(data.Dataset):
    def __init__(self, video_folder, transform, csv_file, time_step=15, num_pred=1, frame_skip=2):
        self.dir = video_folder # currently pointing to all 1000 videos
        self.transform = transform
        self._time_step = time_step
        self._num_pred = num_pred
        self._frame_skip = frame_skip  # Add frame skipping parameter
        self.csv_file = csv_file  # Path to the CSV file
        self.videos = OrderedDict()
        self.frame_ranges = self.load_csv_ranges(self.csv_file)
        self.videos, video_string = self.setup(self.dir, self.videos, self.frame_ranges) # Does this happen for all videos or only those specified in my csv?
        self.samples = self.get_all_samples(video_string)
        self.video_name = video_string

    def load_csv_ranges(self, csv_file):
        df = pd.read_csv(csv_file)
        frame_ranges = {}
        for index, row in df.iterrows():
            video_name = row['video_id']
            start_frame = row['start']
            end_frame = row['end']
            split = row['split']
            if video_name not in frame_ranges:
                frame_ranges[video_name] = []
            frame_ranges[video_name].append((start_frame, end_frame))
        return frame_ranges

    def setup(self, path, videos, frame_ranges): # Get the list of video names from the CSV file (i.e., the keys of frame_ranges)
        csv_video_names = set(frame_ranges.keys())
        video_string = sorted(csv_video_names)  # Only include the videos specified in the CSV
        for video in video_string:
            video_path = os.path.join(path, video + '.mp4')  # Adjust extension as needed
            if os.path.exists(video_path):  # Check if the file exists before adding
                videos[video] = {'path': video_path, 'frame_ranges': frame_ranges.get(video, [])}
            else:
                print(f"Warning: {video} not found in directory.")
        return videos, video_string

    def get_all_samples(self, video_string): # could this be problematic? Depends on what's in video_string/self.videos[video][frame_ranges]
        frames = []
        for video in video_string:
            for start_frame, end_frame in self.videos[video]['frame_ranges']:
                start_frame = int(start_frame * 60)
                end_frame = int(end_frame * 60)
                for i in range(start_frame, end_frame - self._time_step * self._frame_skip, self._frame_skip): # adjusted frame skip logic
                    frames.append((video, i))
        return frames

    def load_image(self, video_path, frame_idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print(f"Failed to load frame {frame_idx} from video {video_path}")
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def __getitem__(self, index): # add parameter for dummy tensor size
        video_name, start_frame = self.samples[index]
        video_path = self.videos[video_name]['path'] # does this actually contain anything?
        batch = []
        for i in range(self._time_step + self._num_pred):
            frame_idx = start_frame + i * self._frame_skip  # Skip frames based on the frame_skip factor
            image = self.load_image(video_path, frame_idx)
            print(f"Loading frame {frame_idx} from video {video_name}", flush=True)
            if image is not None and self.transform is not None:
                try:
                    image = self.transform(image)
                except Exception as e:
                    print(f"Transform error: {e}")
                    continue
                print(f"Image shape before append: {image.shape}")
                batch.append(image)
        batch_len = len(batch)
        if len(batch) != 16: # Handle the case where no valid frames are found
            print(f"No valid images found for index {index}")
            return torch.ones(16, 1, 128, 128) # TODO: handle this better by just skipping the batch entirely, 
            # but first check the video_name and start_frame using the breakpoint right above to see whether 
            # the video is corrupted or something like that
        print(len(batch))
        return torch.stack(batch, 0) # or use torch.cat() ? 

    def __len__(self):
        return len(self.samples)

def give_frame_trans(imshape): # The returned transformation operation reshapes input frames to 256x256 and returns grayscale/normalised frames. 
    height, width = imshape
    print("--There is no other augmentation except resizing, grayscale and normalization--")
    frame_trans = transforms.Compose([transforms.Resize([height, width]),
                                      transforms.Grayscale(num_output_channels=1),
                                      transforms.ToTensor(), # could change to only run this on cpu and rest on gpu.. might be faster
                                      transforms.Normalize([0.5], [0.5])])
    return frame_trans
