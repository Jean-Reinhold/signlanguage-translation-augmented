import pickle
import torch
import gzip
from tqdm import tqdm

from slt_datasets.SLTDataset import SLTDataset


DATASET = "RWTH_PHOENIX_2014T"
DATA_DIR = "/mnt/disk3Tb/augmented-slt-datasets/"
INPUT_MODE = "pose"
OUTPUT_MODE = "text"
OUTPUT_DIR = "/mnt/disk3Tb/exported-slt-datasets"



def format_pose(pose):
	pose = pose[:, 0, :, :2]
	return pose.reshape(pose.shape[0], -1)

def export_dataset(dataset: SLTDataset, output_path: str):
	samples = []
	for i in tqdm(range(len(dataset))):
		pose = format_pose(dataset[i][0])
		# pose is pytorch tensor shape (N, 1086), remove items from position 34 to 95 (face keypoints) to make it (N, 1024)
		pose = torch.cat((pose[:, :34], pose[:, 96:]), dim=1)
		pose = torch.nan_to_num(pose, nan=0.0)
		text = dataset.get_item_raw(i)[1]
		name = dataset.annotations.iloc[i]['id']
		samples.append({
			"sign": pose,
			"text": text,
			"gloss": "",
			"signer": "",
			"name": name + str(i)
		})
	print(f"Saving dataset to {output_path}")
	with gzip.open(output_path, "wb") as f:
		pickle.dump(samples, f)		
	print(f"Dataset saved to {output_path}")


for SPLIT in ["train", "val", "test"]:
	dataset = SLTDataset(
		data_dir=DATA_DIR + DATASET,
		input_mode=INPUT_MODE,
		output_mode=OUTPUT_MODE,
		split=SPLIT
	)
	export_dataset(dataset, f"{OUTPUT_DIR}/{DATASET}.pami0.{SPLIT}")	