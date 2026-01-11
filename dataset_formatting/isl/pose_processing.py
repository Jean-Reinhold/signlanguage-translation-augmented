import os, pickle
import numpy as np


POSES_DIR = "/mnt/disk3Tb/slt-datasets/ISL/poses"
# get all poses files recursibly in the directory
poses_files = []
for root, dirs, files in os.walk(POSES_DIR):
    for file in files:
        if file.endswith(".pkl"):
            poses_files.append(os.path.join(root, file))


def format_pose(pose):
    return np.expand_dims(pose["keypoints"], axis=1)


# store all files as npy files in a directory called poses_procesed

POSES_PROCESSED_DIR = "/mnt/disk3Tb/slt-datasets/ISL/poses_processed"
os.makedirs(POSES_PROCESSED_DIR, exist_ok=True)

# remove from the list the files that are already processed
poses_files = [
    file
    for file in poses_files
    if not os.path.exists(
        os.path.join(
            POSES_PROCESSED_DIR, os.path.basename(file).replace(".pkl", ".npy")
        )
    )
]

sorted_pose_files = sorted(poses_files)
for idx, file in enumerate(sorted_pose_files):
    if idx % 100 == 0:
        print(f"Processing file {idx}/{len(sorted_pose_files)}")
    try:
        pose = pickle.load(open(file, "rb"))
        pose["keypoints"] = np.expand_dims(pose["keypoints"], axis=1)
        np.save(
            os.path.join(
                POSES_PROCESSED_DIR, os.path.basename(file).replace(".pkl", ".npy")
            ),
            pose["keypoints"],
        )
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        continue
