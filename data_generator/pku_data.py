from collections import defaultdict
import os
import random
import re
from typing import List
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
from data_generator.ntu_data import linear_interpolation_sampling, normal_distribution_sampling
from data_generator.utils.rotation import angle_between, rotation_matrix

# ==================== Constants and Config ====================
PKU_ROOT_DIR = "data/PKUMMD"
PKU_MAX_FRAME = 50
PKU_NUM_JOINTS = 25
PKU_MAX_BODY_PER_FRAME = 2

CONNECTING_JOINT = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10,
                    0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]
DIRECTED_EDGES = [(i, CONNECTING_JOINT[i]) for i in range(len(CONNECTING_JOINT))]
BIDIRECTED_EDGES = DIRECTED_EDGES + [(v, u) for u, v in DIRECTED_EDGES]
EDGE_INDEX = torch.tensor(BIDIRECTED_EDGES, dtype=torch.long).t()


with open(os.path.join(PKU_ROOT_DIR, f"cross_subject.txt"), 'r') as xsub:
    xsub.readline()
    PKU_TRAINING_SUBJECTS = [x.strip() for x in re.split(r'[,\n]+', xsub.readline()) if x.strip()]

with open(os.path.join(PKU_ROOT_DIR, f"cross_subject_v2.txt"), 'r') as xsub_v2:
    xsub_v2.readline()
    PKU_V2_TRAINING_SUBJECTS = [x.strip() for x in re.split(r'[,\n]+', xsub_v2.readline()) if x.strip()]

with open(os.path.join(PKU_ROOT_DIR, f"cross_view.txt"), 'r') as xview:
    xview.readline()
    PKU_TRAINING_VIEWS = [x.strip() for x in re.split(r'[,\n]+', xview.readline()) if x.strip()]

with open(os.path.join(PKU_ROOT_DIR, f"cross_subject_v2.txt"), 'r') as xview_v2:
    xview_v2.readline()
    PKU_V2_TRAINING_VIEWS = [x.strip() for x in re.split(r'[,\n]+', xview_v2.readline()) if x.strip()]

# ==================== Utils ====================
class PKU_Body:
    def __init__(self):
        self.joints = []

class PKU_Joint:
     def __init__(self, jointinfo):
        self.x, self.y, self.z = jointinfo

@staticmethod
def read_pku_skeleton(skeleton_file_path, label_file_path):
    skeleton_file_name = os.path.basename(skeleton_file_path)
    # print(f"skeleton file name: {skeleton_file_name}")

    label_file_name = os.path.basename(label_file_path)
    # print(f"label file name: {label_file_name}")

    assert(skeleton_file_name == label_file_name)

    actions = []
    with open(label_file_path, "r") as lf:
        for line in lf.readlines():
            action = line.split(",")
            start_frame, end_frame, label = int(action[1]), int(action[2]), int(action[0])
            # print(f"Action {label} starts from {start_frame}-th frame to {end_frame}-th frame.")
            if start_frame >= end_frame:
                continue

            actions.append((start_frame, end_frame, label))

    # print(f"\n{skeleton_file_name} contains {len(actions)} actions:\n")
    # print(f"{actions}\n")

    skeleton_sequences = []

    with open(skeleton_file_path, "r") as sf:
        frames = sf.readlines()
        for action in actions:
            start_frame = action[0]
            end_frame = action[1]
            label = action[2]
            corresponding_frames = frames[start_frame:end_frame]
            # print(f"Action {action[2]} contains {len(corresponding_frames)} frames")

            skeleton_sequence = []
            for frame in corresponding_frames:
                frame_info = []

                bodies = list(map(float, frame.strip().split()))

                fb = PKU_Body() # First body
                fb_joints = bodies[:75]
                for j in range(0, len(fb_joints), 3):
                    joint = PKU_Joint(fb_joints[j:j+3])
                    fb.joints.append(joint)
                # print(f"First body detected : {fb_joints}\n")
                frame_info.append(fb)

                sb =  PKU_Body() # Second body
                sb_joints = bodies[75:]
                for j in range(0, len(sb_joints), 3):
                    joint = PKU_Joint(sb_joints[j:j+3])
                    sb.joints.append(joint)
                # print(f"Second body detected : {sb_joints}\n")
                frame_info.append(sb)
                skeleton_sequence.append(frame_info)

            skeleton_sequences.append((skeleton_sequence, label))
        
        return skeleton_sequences

# ==================== PKU Data Object ====================

class PKU_Data:
    def __init__(self, skeleton_info, modality: str):
        self.skeleton_info = skeleton_info[0]
        self.skeleton_label = skeleton_info[1]
        self.modality = modality

        x_joint = torch.zeros((3, PKU_MAX_FRAME, PKU_NUM_JOINTS, PKU_MAX_BODY_PER_FRAME), dtype=torch.float32)
        joints = self._get_joints()
        C, T, V, M = joints.shape
        T = min(joints.shape[1], PKU_MAX_FRAME)
        x_joint[:, :T, :, :] = joints[:, :T, :, :]
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

        self.y = torch.tensor(self.skeleton_label, dtype=torch.long)

    def _get_nonzero_std(self, frame):
        idx = frame.sum(-1).sum(-1) != 0
        frame = frame[idx]
        return frame[:, :, 0].std() + frame[:, :, 1].std() + frame[:, :, 2].std() if len(frame) else 0

    def _get_joints(self):
        x = np.zeros((PKU_MAX_BODY_PER_FRAME, len(self.skeleton_info), PKU_NUM_JOINTS, 3))
        # print(x.shape)
        for n, f in enumerate(self.skeleton_info):
            # print(f"{n} {f}")
            for m, b in enumerate(f):
                for j, v in enumerate(b.joints):
                    if m < PKU_MAX_BODY_PER_FRAME and j < PKU_NUM_JOINTS:
                        x[m, n, j, :] = [v.x, v.y, v.z]
        skeletons = np.array([self._get_nonzero_std(f) for f in x])
        # print(skeletons)
        idx = skeletons.argsort()[::-1][:PKU_MAX_BODY_PER_FRAME]
        # print(idx)
        x = x[idx]

        # Frame selection logic
        if len(self.skeleton_info) < PKU_MAX_FRAME:
            # Repeat frames to reach MAX_FRAME
            reps = (PKU_MAX_FRAME + len(self.skeleton_info) - 1) // len(self.skeleton_info)  # ceil division
            x = np.tile(x, (1, reps, 1, 1))[:, :PKU_MAX_FRAME, :, :]  # repeat along time axis
        else:
            selected_idx = linear_interpolation_sampling(len(self.skeleton_info), PKU_MAX_FRAME)
            x = x[:, selected_idx, :, :]  # select along time axis

        return torch.tensor(x.transpose(3, 1, 2, 0))  # [C, T, V, M]
    
    def _get_bone(self, joint_data):
        bone = torch.zeros_like(joint_data)
        for v1, v2 in DIRECTED_EDGES:
            bone[:, :, v1, :] = joint_data[:, :, v1, :] - joint_data[:, :, v2, :]
        return bone
    
# ==================== PKU Dataset ====================

class PKU_Dataset(InMemoryDataset):
    def __init__(self, root=PKU_ROOT_DIR, transform=None, pre_transform=None, pre_filter=None,
                 modality="joint", benchmark="xsub", part="train", extended=False, force_reload=False):
        self.modality = modality
        self.benchmark = benchmark
        self.part = part
        self.extended = extended
        super(PKU_Dataset, self).__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])
        

    @property
    def raw_dir(self):
        raw_dir = os.path.join(PKU_ROOT_DIR, f"PKU_Skeleton{"_v2" if self.extended else ""}")
        return raw_dir
    
    @property
    def label_dir(self):
        label_dir =  os.path.join(PKU_ROOT_DIR, f"PKU_Label{"_v2" if self.extended else ""}")
        return label_dir

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith(".txt")]
    
    @property
    def label_file_names(self):
        return [f for f in os.listdir(self.label_dir) if f.endswith(".txt")]

    @property
    def processed_dir(self):
        return os.path.join(os.path.dirname(PKU_ROOT_DIR), f"PKU_Skeleton{"_v2" if self.extended else ""}_{self.benchmark}")

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
        name = f"PKUMMD{"_v2" if self.extended else ""}"
        print(f"{name} Dataset")
        print(f"Benchmark: {self.benchmark} - Modality: {self.modality} - Part: {self.part}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Number of samples: {self.length}")
        print(f"Features' dimension: {self.num_features}")
    
    def process(self):
        data_list = []

        for skeleton_file_name, label_file_name in tqdm(zip(self.raw_file_names, self.label_file_names), desc="Processing skeletons"):
            sample_id = os.path.splitext(skeleton_file_name)[0]
            skeleton_full_path = os.path.join(self.raw_dir, skeleton_file_name)
            label_full_path = os.path.join(self.label_dir, label_file_name)

            if self.pre_filter and not self.pre_filter(sample_id, self.benchmark, self.part, self.extended):
                continue

            skeleton_sequences = read_pku_skeleton(skeleton_file_path=skeleton_full_path, label_file_path=label_full_path)  

            for skeleton_sequence in skeleton_sequences:
                
                sample = PKU_Data(skeleton_info=skeleton_sequence, modality="joint")  # always load joint first

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

                data = Data(x=x.unsqueeze(0), y=sample.y, edge_index=EDGE_INDEX.unsqueeze(-1))
                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

 # ==================== Filters ====================
    @staticmethod
    def __pku_pre_filter__(sample_id, benchmark, part, extended):
        if not extended:
            TRAINING_SUBJECTS = PKU_TRAINING_SUBJECTS
            TRAINING_VIEWS = PKU_TRAINING_VIEWS
        else:
            TRAINING_SUBJECTS = PKU_V2_TRAINING_SUBJECTS
            TRAINING_VIEWS = PKU_V2_TRAINING_VIEWS

        if benchmark == "xview":
            is_training = (sample_id in TRAINING_VIEWS)
        elif benchmark == "xsub":
            is_training = (sample_id in TRAINING_SUBJECTS)
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}")

        # print(f"{sample_id} : {is_training if part == "train" else not is_training} ")
        return is_training if part == "train" else not is_training
    
    # ==================== Transforms ====================
    @staticmethod
    def __pku_pre_transformer__(data, zaxis=[0, 1], xaxis=[8, 4]):
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
