import pandas as pd
import h5py
import numpy as np
from ast import literal_eval
import os


DIR = "/mnt/disk3Tb/slt-datasets/LSAT"
annotations = pd.read_csv(f"{DIR}/annotations.csv")

with h5py.File(f"{DIR}/keypoints.h5", "r") as f:
    for id, row in annotations.iterrows():
        out_path = f"{DIR}/poses/{row['id']}.npy"
        print(f"Processing {row['id']}")
        if os.path.exists(out_path):
            continue
        id = row["id"]
        people = f[f"{id}.mp4"]
        infered_signer = row["infered_signer"]
        movement_per_signer = literal_eval(row["movement_per_signer"])
        if pd.isna(infered_signer):
            infered_signer = "signer_0"
        elif len(movement_per_signer) <= len(people):
            infered_signer = row["infered_signer"]
        else:
            infered_signer_pos = int(row["infered_signer"].split("_")[1])
            # count how many 0.0 are in the movement_per_signer before the position of the infered_signer
            for i, m in enumerate(movement_per_signer):
                if i == int(row["infered_signer"].split("_")[1]):
                    break
                if m == 0.0:
                    infered_signer_pos -= 1
            infered_signer = f"signer_{infered_signer_pos}"
        try:
            boxes, keypoints = (
                people[infered_signer]["boxes"],
                people[infered_signer]["keypoints"],
            )
        except:
            print(row)
            print(len(movement_per_signer), movement_per_signer)
            print(infered_signer)
            print([k for k in people])
            continue
        keypoints = np.array(keypoints)
        # keypoints is a list of shape (frames, keyp_len * 4), as we have x,y,z,c for each keypoint. i need to reshape it to (frames, 1, keyp_len, 4). the 1 is for people dimension
        keypoints = keypoints.reshape(keypoints.shape[0], -1, 4)
        # add people dimension
        keypoints = keypoints[:, np.newaxis, :, :]
        # take only the x,y,z columns
        keypoints = keypoints[:, :, :, :-1]

        np.save(f"{DIR}/poses/{id}.npy", keypoints)
