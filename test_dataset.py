from video_dataset import VideoDataset

dataset = VideoDataset("extracted_frames")

sequence, label = dataset[0]

print(sequence.shape)
print(label)