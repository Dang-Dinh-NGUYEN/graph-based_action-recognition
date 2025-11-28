from collections import defaultdict
import os
import random
from typing import List
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
from data_generator.utils.rotation import angle_between, rotation_matrix

# ==================== Constants and Config ====================

ROOT_DIR = "data/"
RAW_DIR = "data/nturgb+d_skeletons/"

NTU_RGBD_IGNORED_SAMPLE_PATH = os.path.join(ROOT_DIR, "NTU_RGBD_samples_with_missing_skeletons.txt")
NTU_RGBD120_IGNORED_SAMPLE_PATH = os.path.join(ROOT_DIR, "NTU_RGBD120_samples_with_missing_skeletons.txt")

with open(NTU_RGBD_IGNORED_SAMPLE_PATH, "r") as f:
    NTU_RGBD_IGNORED_SAMPLES = set(line.strip() for line in f.readlines())
with open(NTU_RGBD120_IGNORED_SAMPLE_PATH, "r") as f:
    NTU_RGBD120_IGNORED_SAMPLES = set(line.strip() for line in f.readlines())

CONNECTING_JOINT = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10,
                    0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]
DIRECTED_EDGES = [(i, CONNECTING_JOINT[i]) for i in range(len(CONNECTING_JOINT))]
BIDIRECTED_EDGES = DIRECTED_EDGES + [(v, u) for u, v in DIRECTED_EDGES]
EDGE_INDEX = torch.tensor(BIDIRECTED_EDGES, dtype=torch.long).t()
EDGES = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        (22, 23), (23, 8), (24, 25), (25, 12)]

TRAINING_SUBJECTS = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
    31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
    58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
    93, 94, 95, 97, 98, 100, 103
]
TRAINING_CAMERAS = [2, 3]

MAX_BODY_PER_FRAME = 2
MAX_BODY_DETECTED = 4
MAX_FRAME = 50
NUM_JOINTS = 25

# ==================== Utils ====================
class NTU_Body:
    def __init__(self):
        self.joints = []

class NTU_Joint:
    def __init__(self, jointinfo, tracking_state):
        self.x, self.y, self.z = jointinfo[0:3]
        self.depthX, self.depthY = jointinfo[3:5]
        self.colorX, self.colorY = jointinfo[5:7]
        self.orientationW = jointinfo[7]
        self.orientationX = jointinfo[8]
        self.orientationY = jointinfo[9]
        self.orientationZ = jointinfo[10]
        self.trackingState = tracking_state


@staticmethod
def read_ntu_skeleton(filename: str):
    with open(filename, "r") as file:
        def read_int():
            return int(file.readline().strip())

        def read_line_as_floats():
            return list(map(float, file.readline().strip().split()))

        frame_count = read_int()
        skeleton_sequence = []

        for _ in range(frame_count):
            frame_info = []
            body_count = read_int()

            for _ in range(body_count):
                body = NTU_Body()
                values = read_line_as_floats()  # read 10 values

                body.bodyID = int(values[0])
                body.clipedEdges = int(values[1])
                body.handLeftConfidence = int(values[2])
                body.handLeftState = int(values[3])
                body.handRightConfidence = int(values[4])
                body.handRightState = int(values[5])
                body.isRestricted = int(values[6])
                body.leanX = values[7]
                body.leanY = values[8]
                body.trackingState = int(values[9])

                joint_count = read_int() # number of joints (25)
                body.jointCount = joint_count

                for _ in range(joint_count):
                    values = read_line_as_floats()  # read 12 values
                    jointinfo = values[0:11]
                    tracking_state = int(values[11])
                    joint = NTU_Joint(jointinfo, tracking_state)
                    body.joints.append(joint)

                frame_info.append(body)
            skeleton_sequence.append(frame_info)

    return skeleton_sequence

def normal_distribution_sampling(num_frames, MAX_FRAME, std_dev_factor=0.2):
    """
    Resize a sequence to MAX_FRAME using a normal distribution sampling strategy
    with a higher density of samples around the middle.
    
    num_frames: Original sequence length.
    MAX_FRAME: Target sequence length.
    std_dev_factor: Controls the "spread" of the sampling (higher = wider spread).
    """
    # Define mean (middle of the sequence) and standard deviation
    mean = num_frames // 2
    std_dev = num_frames * std_dev_factor  # Standard deviation scales with sequence length

    # Generate the normal distribution PDF centered at the mean
    x = np.linspace(0, num_frames - 1, num_frames)
    pdf = np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    
    # Normalize the PDF to sum to 1
    pdf /= pdf.sum()
    
    # Sample `MAX_FRAME` frames using the PDF (higher probability for middle frames)
    selected_idx = np.random.choice(x, size=MAX_FRAME, p=pdf, replace=False)
    
    # Sort the selected indices to maintain temporal order
    selected_idx.sort()
    
    return selected_idx.astype(int)

def linear_interpolation_sampling(num_frames, MAX_FRAME):
    """
    Resize a sequence to MAX_FRAME using linear interpolation sampling.
    Selects frames evenly spaced between 0 and num_frames - 1.
    """
    # Generate evenly spaced positions
    selected_idx = np.linspace(0, num_frames - 1, MAX_FRAME, dtype=int)
    return selected_idx


# ==================== NTU Data Object ====================

class NTU_Data:
    def __init__(self, file_name: str, modality: str):
        self.id = file_name
        self.modality = modality

        action_class = int(file_name[file_name.find('A') + 1:file_name.find('A') + 4])
        self.y = torch.tensor(action_class, dtype=torch.long)

        x_joint = torch.zeros((3, MAX_FRAME, NUM_JOINTS, MAX_BODY_PER_FRAME), dtype=torch.float32)
        joints = self._get_joints()
        C, T, V, M = joints.shape
        x_joint[:, :T, :V, :M] = joints

        self.x_joint = x_joint

        # Assign self.x depending on modality
        if modality == "joint":
            self.x = x_joint
        elif modality == "bone":
            self.x = self._get_bone(x_joint)
        elif modality == "joint_bone":
            bone = self._get_bone(x_joint)
            self.x = torch.cat([x_joint, bone], dim=0)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def _get_nonzero_std(self, frame):
        idx = frame.sum(-1).sum(-1) != 0
        frame = frame[idx]
        return frame[:, :, 0].std() + frame[:, :, 1].std() + frame[:, :, 2].std() if len(frame) else 0

    def _get_joints(self):
        seq_info = read_ntu_skeleton(self.id)
        num_frames = len(seq_info)
        x = np.zeros((MAX_BODY_DETECTED, num_frames, NUM_JOINTS, 3))

        # Fill the array with joint data
        for n, f in enumerate(seq_info):
            for m, b in enumerate(f):
                for j, v in enumerate(b.joints):
                    if m < MAX_BODY_DETECTED and j < NUM_JOINTS:
                        x[m, n, j, :] = [v.x, v.y, v.z]

        # Choose the most active bodies
        skeletons = np.array([self._get_nonzero_std(f) for f in x])
        idx = skeletons.argsort()[::-1][:MAX_BODY_PER_FRAME]
        x = x[idx]  # Shape: [M, T, V, C]

        # Frame selection logic
        if num_frames < MAX_FRAME:
            # Repeat frames to reach MAX_FRAME
            reps = (MAX_FRAME + num_frames - 1) // num_frames  # ceil division
            x = np.tile(x, (1, reps, 1, 1))[:, :MAX_FRAME, :, :]  # repeat along time axis
        else:
            selected_idx = linear_interpolation_sampling(num_frames, MAX_FRAME)
            x = x[:, selected_idx, :, :]  # select along time axis

        return torch.tensor(x.transpose(3, 1, 2, 0))  # [C, T, V, M]

    def _get_bone(self, joint_data):
        bone = torch.zeros_like(joint_data)
        for v1, v2 in EDGES:
            bone[:, :, v1 - 1, :] = joint_data[:, :, v1 - 1, :] - joint_data[:, :, v2 - 1, :]
        return bone

# ==================== NTU Dataset ====================

class NTU_Dataset(InMemoryDataset):
    def __init__(self, root=ROOT_DIR, transform=None, pre_transform=None, pre_filter=None,
                 modality="joint", benchmark="xsub", part="train", extended=False, force_reload=False):
        self.modality = modality
        self.benchmark = benchmark
        self.part = part
        self.extended = extended
        super(NTU_Dataset, self).__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return RAW_DIR

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith(".skeleton")]

    @property
    def processed_dir(self):
        return os.path.join(os.path.dirname(ROOT_DIR), f"ntu_rgbd{120 if self.extended else ''}_{self.benchmark}")

    @property
    def processed_file_names(self):
        return [f"{self.part}_{self.modality}.pt"]
    
    @property
    def num_features(self):
        return self.get(0).x.shape[1]
    
    @property
    def num_classes(self):
        return super().num_classes - 1
    
    @property
    def length(self):
        return len(self)
    
    def print_summary(self):
        name = f"NTU_RGB+D{"_120" if self.extended else ""}"
        print(f"{name} Dataset")
        print(f"Benchmark: {self.benchmark} - Modality: {self.modality} - Part: {self.part}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Number of samples: {self.length}")
        print(f"Features' dimension: {self.num_features}")
    
    def process(self):
        data_list = []
        for file_name in tqdm(self.raw_file_names, desc="Processing skeletons"):
            sample_id = os.path.splitext(file_name)[0]
            full_path = os.path.join(self.raw_dir, file_name)

            if self.pre_filter and not self.pre_filter(sample_id, self.benchmark, self.part, self.extended):
                continue

            sample = NTU_Data(full_path, modality="joint")  # always load joint first

            if self.pre_transform:
                sample.x_joint = self.pre_transform(sample.x_joint)

            if self.modality == "joint":
                x = sample.x_joint
            elif self.modality == "bone":
                x = sample._get_bone(sample.x_joint)
            elif self.modality == "joint_bone":
                x = torch.cat([sample.x_joint, sample._get_bone(sample.x_joint)], dim=0)
            else:
                raise ValueError(f"Invalid modality: {self.modality}")

            data = Data(id=sample.id, x=x.unsqueeze(0), y=sample.y, edge_index=EDGE_INDEX.unsqueeze(-1))
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    # ==================== Filters ====================
    @staticmethod
    def __nturgbd_pre_filter__(sample_id, benchmark, part, extended):
        if sample_id in NTU_RGBD120_IGNORED_SAMPLES:
            return False

        subject_id = int(sample_id[sample_id.find('P') + 1:sample_id.find('P') + 4])
        camera_id = int(sample_id[sample_id.find('C') + 1:sample_id.find('C') + 4])
        setup_id = int(sample_id[sample_id.find('S') + 1:sample_id.find('S') + 4])
        action_class = int(sample_id[sample_id.find('A') + 1:sample_id.find('A') + 4])

        if not extended and action_class > 60:
            return False

        if benchmark == "xview":
            is_training = (camera_id in TRAINING_CAMERAS)
        elif benchmark == "xsub":
            is_training = (subject_id in TRAINING_SUBJECTS)
        elif benchmark == "xsetup":
            is_training = (setup_id % 2 == 0)
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}")

        return is_training if part == "train" else not is_training

    # ==================== Transforms ====================
    @staticmethod
    def __nturgbd_pre_transformer__(data, zaxis=[0, 1], xaxis=[8, 4]):
        data = data.clone().detach().cpu().numpy()
        C, T, V, M = data.shape
        s = data.transpose(3, 1, 2, 0) # C, T, V, M  to  M, T, V, C

        for i_p, person in enumerate(s):
            valid = person.sum((1, 2)) != 0
            if not valid.any():
                continue
            valid_frames = person[valid]
            pad = np.tile(valid_frames, (T // len(valid_frames) + 1, 1, 1))[:T]
            s[i_p] = pad

        center = s[:, :, 1:2, :]
        mask = (s.sum(-1, keepdims=True) != 0)
        s = (s - center) * mask

        joint_b, joint_t = s[:, 0, zaxis[0]], s[:, 0, zaxis[1]]
        vec = joint_t - joint_b
        z_target = np.array([0, 0, 1], dtype=np.float32)
        for i in range(M):
            if np.linalg.norm(vec[i]) == 0:
                continue
            axis = np.cross(vec[i], z_target)
            angle = angle_between(vec[i], z_target)
            Rz = rotation_matrix(axis, angle)
            for t in range(T):
                if s[i, t].sum() == 0:
                    continue
                s[i, t] = s[i, t] @ Rz.T

        joint_r, joint_l = s[:, 0, xaxis[0]], s[:, 0, xaxis[1]]
        vec = joint_r - joint_l
        x_target = np.array([1, 0, 0], dtype=np.float32)
        for i in range(M):
            if np.linalg.norm(vec[i]) == 0:
                continue
            axis = np.cross(vec[i], x_target)
            angle = angle_between(vec[i], x_target)
            Rx = rotation_matrix(axis, angle)
            for t in range(T):
                if s[i, t].sum() == 0:
                    continue
                s[i, t] = s[i, t] @ Rx.T

        s = torch.tensor(s.transpose(3, 1, 2, 0), dtype=torch.float32)
        return s
    