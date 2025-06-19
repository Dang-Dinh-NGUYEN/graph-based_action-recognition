"""
Customized dataset class for the NTU RGB+D 3D Action Recognition Dataset.
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset, Data
from data_generator.utils.read_skeleton_file import read_skeleton_file
from tools.rotation import *

# ==================== Configuration ====================
ROOT_DIR = "data/"
RAW_DIR = "data/nturgb+d_skeletons/"

IGNORED_SAMPLE_PATH = os.path.join(ROOT_DIR, "NTU_RGBD_samples_with_missing_skeletons.txt")
with open(IGNORED_SAMPLE_PATH, "r") as f:
    IGNORED_SAMPLES = set(line.strip() for line in f.readlines()[3:])

CONNECTING_JOINT = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11,
                    1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12] - np.array(np.ones(25))
DIRECTED_EDGES = [(i, CONNECTING_JOINT[i]) for i in range(len(CONNECTING_JOINT))]
BIDIRECTED_EDGES = DIRECTED_EDGES + [(v, u) for u, v in DIRECTED_EDGES]
EDGE_INDEX = torch.tensor(BIDIRECTED_EDGES, dtype=torch.long).t()

TRAINING_SUBJECTS = [
    1
]

"""
TRAINING_SUBJECTS = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
    31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
    58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
    93, 94, 95, 97, 98, 100, 103
]
"""

TRAINING_CAMERAS = [2, 3]

MAX_BODY_PER_FRAME = 2
MAX_BODY_DETECTED = 4
MAX_FRAME = 300
NUM_JOINTS = 25

# ==================== Data Object ====================

class NTU_Data():
    def __init__(self, file_name: str):
        self.file_name = file_name

        action_class = int(file_name[file_name.find('A') + 1:file_name.find('A') + 4])
        self.y = torch.tensor(action_class, dtype=torch.long)

        self.x = torch.zeros((3, MAX_FRAME, NUM_JOINTS, MAX_BODY_PER_FRAME), dtype=torch.float32)

        data = self._extract_node_features()  # shape: (3, T, V, M)

        C, T, V, M = data.shape
        self.x[:, :T, :V, :M] = data


    def _get_nonzero_std(self, frame):
        idx = frame.sum(-1).sum(-1) != 0 # Keep frame with at least one skeleton
        frame = frame[idx]
        if len(frame) != 0:
            frame = frame[:, :, 0].std() + frame[:, :, 1].std() + frame[:, :, 2].std()
        else:
            frame = 0
        return frame

    def _extract_node_features(self):
        seq_info = read_skeleton_file(self.file_name)
        x = np.zeros((MAX_BODY_DETECTED, len(seq_info), NUM_JOINTS, 3))
        for n, f in enumerate(seq_info):
            for m, b in enumerate(f):
                for j, v in enumerate(b.joints):
                    if m  < MAX_BODY_DETECTED and j < NUM_JOINTS:
                        x[m, n, j, :] = [v.x, v.y, v.z]
                    else:
                        pass

        # Select two main skeletons
        skeletons = np.array([self._get_nonzero_std(f) for f in x])
        idx = skeletons.argsort()[::-1][0:MAX_BODY_PER_FRAME]
        x = x[idx]

        return torch.tensor(x.transpose(3, 1, 2, 0))

# ==================== Dataset Class ====================

class NTU_Dataset(InMemoryDataset):
    def __init__(self, root=ROOT_DIR, transform=None, pre_transform=None, pre_filter=None,
                 modality="joint", benchmark="xview", part="eval"):
        self.modality = modality
        self.benchmark = benchmark
        self.part = part
        super(NTU_Dataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return RAW_DIR
    
    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith(".skeleton")]
    
    @property
    def processed_dir(self):
        return os.path.join(os.path.dirname(ROOT_DIR), f"ntu_rgbd_{self.benchmark}")

    @property
    def processed_file_names(self):
        return [f"{self.part}_{self.modality}.pt"]
        
    def process(self):
        data_list = []
        for file_name in tqdm(self.raw_file_names, desc="Processing skeletons"):
            full_path = os.path.join(self.raw_dir, file_name)
            sample_id = os.path.splitext(file_name)[0]

            # Skip sample if it does not pass the pre_filter
            if self.pre_filter is not None:
                if not self.pre_filter(sample_id, self.benchmark, self.part):
                    continue
            
            data = NTU_Data(full_path)

            # Apply optional transform
            if self.pre_transform is not None:
                data.x = self.pre_transform(data.x)

            data_list.append(Data(x = data.x.unsqueeze(0), y= data.y, edge_index=EDGE_INDEX.unsqueeze(-1)))

        self.save(data_list, self.processed_paths[0])

    @staticmethod
    def __nturgbd_pre_filter__(sample_id, benchmark, part):
        if sample_id in IGNORED_SAMPLES:
            return False
        
        subject_id = int(sample_id[sample_id.find('P') + 1:sample_id.find('P') + 4])
        camera_id =  int(sample_id[sample_id.find('C') + 1:sample_id.find('C') + 4])

        is_sample = False
        is_training = False

        if benchmark == "xview":
            is_training = (camera_id in TRAINING_CAMERAS)
        elif benchmark == "xsub":
            is_training = (subject_id in TRAINING_SUBJECTS)
        else:
            raise ValueError(f"Invalid `benchmark`: expected 'xsub' or 'xview', got '{benchmark}'.")

        if part == "train":
            is_sample = is_training
        elif part in {"val", "test"}:
            is_sample = not is_training
        else:
            raise ValueError(f"Invalid `part`: expected 'train', 'val' or 'test', got '{part}'.")

        return is_sample
    

    @staticmethod
    def __nturgbd_pre_transformer__(data, zaxis=[0, 1], xaxis=[8, 4], modality="joint"):
        # data: [C, T, V, M] (single sample)
        data = data.clone().detach().cpu().numpy()
        C, T, V, M = data.shape

        # Transpose to [M, T, V, C] for easier manipulation
        s = data.transpose(3, 1, 2, 0)  # [M, T, V, C]

        # 1. Pad null frames with previous valid frames
        for i_p, person in enumerate(s):
            frame_valid = person.sum(axis=(1, 2)) != 0
            if not frame_valid.any():
                continue
            valid_frames = person[frame_valid]
            num_valid = len(valid_frames)
            if num_valid < T:
                pad = np.tile(valid_frames, (int(np.ceil(T / num_valid)), 1, 1))[:T]
                s[i_p] = pad

        # 2. Subtract center joint (joint 1)
        center_joint = s[:, :, 1:2, :]  # [M, T, 1, C]
        mask = (s.sum(-1, keepdims=True) != 0)  # [M, T, V, 1]
        s = (s - center_joint) * mask

        # 3. Align hip-to-spine to z-axis
        joint_bottom = s[:, 0, zaxis[0], :]  # [M, C]
        joint_top = s[:, 0, zaxis[1], :]     # [M, C]
        bone_vec = joint_top - joint_bottom
        z_target = np.array([0, 0, 1], dtype=np.float32)

        for i_p in range(M):
            vec = bone_vec[i_p]
            if np.linalg.norm(vec) == 0:
                continue
            axis = np.cross(vec, z_target)
            angle = angle_between(vec, z_target)
            matrix_z = rotation_matrix(axis, angle)
            for t in range(T):
                if s[i_p, t].sum() == 0:
                    continue
                s[i_p, t] = s[i_p, t] @ matrix_z.T

        # 4. Align shoulders to x-axis
        joint_r = s[:, 0, xaxis[0], :]  # [M, C]
        joint_l = s[:, 0, xaxis[1], :]  # [M, C]
        shoulder_vec = joint_r - joint_l
        x_target = np.array([1, 0, 0], dtype=np.float32)

        for i_p in range(M):
            vec = shoulder_vec[i_p]
            if np.linalg.norm(vec) == 0:
                continue
            axis = np.cross(vec, x_target)
            angle = angle_between(vec, x_target)
            matrix_x = rotation_matrix(axis, angle)
            for t in range(T):
                if s[i_p, t].sum() == 0:
                    continue
                s[i_p, t] = s[i_p, t] @ matrix_x.T

        # Back to [C, T, V, M]
        s = s.transpose(3, 1, 2, 0)
        s = torch.tensor(s, dtype=torch.float32)

        if modality == "bone":
            for v1, v2 in DIRECTED_EDGES:
                s[:, :, v1, :] = s[:, :, v1, :] - s[:, :, v2, :]

        return s


# ==================== CLI Interface ====================

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="NTU RGB+D Dataset Processing Script")
     parser.add_argument("--data_path", help="Path towards the original dataset")
     parser.add_argument("--modality", default="joint", choices=["joint", "bone"], help="Skeleton's modality")
     parser.add_argument("--benchmark", default="xsub", choices=["xsub", "xview"], help="Benchmark strategy")
     parser.add_argument("--part", default="train", choices=["train", "val", "test"], help="Dataset split")
     args = parser.parse_args()

     dataset = NTU_Dataset(modality=args.modality, benchmark=args.benchmark, part=args.part, 
                           pre_transform=NTU_Dataset.__nturgbd_pre_transformer__,
                           pre_filter=NTU_Dataset.__nturgbd_pre_filter__)
     print(f"Total samples in {args.part}: {dataset.len()}")
    
     print(dataset._data)
     print(dataset.get(0))
     print(dataset.get_summary())
     
