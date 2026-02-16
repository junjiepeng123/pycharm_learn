import os
hr_dir= './Datasets/VSR/REDS/train/GT'
video_folders = sorted([f for f in os.listdir(hr_dir)
                                if os.path.isdir(os.path.join(hr_dir, f))])
#print(video_folders)
vid_folder='002'
hr_vid_path = os.path.join(hr_dir, vid_folder)
hr_frames = sorted([f for f in os.listdir(hr_vid_path)
                                if f.endswith(('.png', '.jpg', '.jpeg'))])

print(hr_frames)