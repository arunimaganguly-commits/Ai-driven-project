import cv2
import os

# -----------------------------
# Input and output folders
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

datasets = [
    (
        os.path.join(base_dir, "dataset", "real_videos"),
        os.path.join(base_dir, "extracted_frames", "real")
    ),
    (
        os.path.join(base_dir, "dataset", "fake_videos"),
        os.path.join(base_dir, "extracted_frames", "fake")
    )
]
# datasets = [
#     ("dataset/videos/train", "dataset/frames/train"),
#     ("dataset/videos/test", "dataset/frames/test")
# ]

# -----------------------------
# Extract frames
# -----------------------------

for input_folder, output_folder in datasets:

    os.makedirs(output_folder, exist_ok=True)

    for video_name in os.listdir(input_folder):

        video_path = os.path.join(input_folder, video_name)

        # create subfolder for each video
        video_output_folder = os.path.join(
            output_folder,
            os.path.splitext(video_name)[0]
        )

        os.makedirs(video_output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            save_path = os.path.join(
                video_output_folder,
                f"frame_{frame_count}.jpg"
            )

            cv2.imwrite(save_path, frame)

            frame_count += 1

        cap.release()

        print(f"Extracted frames from: {video_name}")

print("All frame extraction completed")