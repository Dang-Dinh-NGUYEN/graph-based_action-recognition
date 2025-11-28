import torch
import os
import csv
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from data_generator.ntu_data import linear_interpolation_sampling

# ==================== Constants and Config ====================
MOCAP_ROOT_DIR = "data/MOCAP"

MOCAP_MAX_FRAME = 50
MOCAP_NUM_JOINTS = 15
MOCAP_MAX_BODY_PER_FRAME = 1

CONNECTING_JOINT = [1, 2, 1, 0, 0, 11, 11, 14, 14, 3, 9, 10, 4, 12, 13]
DIRECTED_EDGES = [(i, CONNECTING_JOINT[i]) for i in range(len(CONNECTING_JOINT))]
BIDIRECTED_EDGES = DIRECTED_EDGES + [(v, u) for u, v in DIRECTED_EDGES]
EDGE_INDEX = torch.tensor(BIDIRECTED_EDGES, dtype=torch.long).t()

with open(os.path.join(MOCAP_ROOT_DIR, f"Annotations.tsv"), newline='') as labels:
    reader = csv.DictReader(labels, delimiter='\t')
    MOCAP_ANNOTATIONS = list(reader)
    MOCAP_ANNOTATED_SAMPLES = [annotation['CSV'] for annotation in MOCAP_ANNOTATIONS]
    # NUM_HANDS_LABELS = []

TRAINING_SUBJECTS = ['LF2', 'MLD', 'LF3']

# ==================== Utils ====================
class MOCAP_Body:
    def __init__(self):
        self.joints = []

class MOCAP_Joint:
     def __init__(self, jointinfo):
        self.x, self.y, self.z = jointinfo

@staticmethod
def read_mocap_skeleton(skeleton_file_path, benchmark):

    skeleton_filename = os.path.basename(skeleton_file_path).replace(".csv", "")
    
    if skeleton_filename not in MOCAP_ANNOTATED_SAMPLES:
        return None

    skeleton_sequence = []
    with open(skeleton_file_path, newline='') as sf:
        csv_reader = csv.reader(sf, delimiter='\t')
        
        # First two rows: joint names + header
        joint_names_row = next(csv_reader)
        header_row = next(csv_reader)

        for frame in csv_reader:
            frame_info = []
            body = MOCAP_Body()
            
            # Convert frame string → list of floats
            body_joints = [float(x) for x in frame[0].split(';')[2:]]

            # Group into (x,y,z) joints
            for j in range(0, len(body_joints), 3):
                joint = MOCAP_Joint(body_joints[j:j+3])
                body.joints.append(joint)

            frame_info.append(body)
            skeleton_sequence.append(frame_info)

    annotation = next(ann for ann in MOCAP_ANNOTATIONS if ann["CSV"] == skeleton_filename)

    if benchmark == 'multi_class':
        label = 1 if annotation["a X mains"] == "à une main" else 2
    elif benchmark == 'multi_label':
        label = [
            1 if annotation["a X mains"] == "à une main" else 2,
            1 if annotation["Torse"] == "Non" else 2,
            1 if annotation["Tete"] == "Non" else 2,
            1 if annotation["Mains"] == "Non" else 2,
            1 if annotation["Fixe"] == "Non" else 2,
        ]

    return (skeleton_sequence, label)

# ==================== MOCAP Data Object ====================

class MOCAP_Data:
    def __init__(self, id, skeleton_info, skeleton_label, modality: str):
        self.id = id
        self.skeleton_info = skeleton_info
        self.skeleton_label = skeleton_label
        self.modality = modality

        x_joint = torch.zeros((3, MOCAP_MAX_FRAME, MOCAP_NUM_JOINTS, MOCAP_MAX_BODY_PER_FRAME), dtype=torch.float32)
        joints = self._get_joints()
        C, T, V, M = joints.shape
        T = min(joints.shape[1], MOCAP_MAX_FRAME)
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
        x = np.zeros((MOCAP_MAX_BODY_PER_FRAME, len(self.skeleton_info), MOCAP_NUM_JOINTS, 3))
        for n, f in enumerate(self.skeleton_info):
            for m, b in enumerate(f):
                for j, v in enumerate(b.joints):
                    if m < MOCAP_MAX_BODY_PER_FRAME and j < MOCAP_NUM_JOINTS:
                        x[m, n, j, :] = [v.x, v.y, v.z]
        skeletons = np.array([self._get_nonzero_std(f) for f in x])
        idx = skeletons.argsort()[::-1][:MOCAP_MAX_BODY_PER_FRAME]
        x = x[idx]

        # Frame selection logic
        if len(self.skeleton_info) < MOCAP_MAX_FRAME:
            # Repeat frames to reach MAX_FRAME
            reps = (MOCAP_MAX_FRAME + len(self.skeleton_info) - 1) // len(self.skeleton_info)  # ceil division
            x = np.tile(x, (1, reps, 1, 1))[:, :MOCAP_MAX_FRAME, :, :]  # repeat along time axis
        else:
            selected_idx = linear_interpolation_sampling(len(self.skeleton_info), MOCAP_MAX_FRAME)
            x = x[:, selected_idx, :, :]  # select along time axis

        return torch.tensor(x.transpose(3, 1, 2, 0))  # [C, T, V, M]
    
    def _get_bone(self, joint_data):
        bone = torch.zeros_like(joint_data)
        for v1, v2 in DIRECTED_EDGES:
            bone[:, :, v1, :] = joint_data[:, :, v1, :] - joint_data[:, :, v2, :]
        return bone

# ==================== MOCAP Dataset ====================

class MOCAP_Dataset(InMemoryDataset):
    def __init__(self, root=MOCAP_ROOT_DIR, transform=None, pre_transform=None, pre_filter=None,
                 modality="joint", benchmark="multi_class", part='train', extended=False, force_reload=False):
        self.modality = modality
        assert benchmark in ['multi_class', 'multi_label']
        self.benchmark = benchmark
        self.part = part
        self.extended = extended
        super(MOCAP_Dataset, self).__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])
        
    @property
    def raw_dir(self):
        return MOCAP_ROOT_DIR

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith(".csv")]
    
    @property
    def label_file_names(self):
        return MOCAP_ANNOTATED_SAMPLES
    
    @property
    def processed_dir(self):
        return os.path.join(os.path.dirname(MOCAP_ROOT_DIR), f"MOCAP_{self.benchmark}")

    @property
    def processed_file_names(self):
        return [f"{self.modality}_{self.part}.pt"]
    
    @property
    def num_features(self):
        return self.get(0).x.shape[1]
    
    @property
    def num_classes(self):
        if self.benchmark == 'multi_label':
            return len(self.get(0).y)
        return super().num_classes - 1

    @property
    def length(self):
        return len(self)
    
    def print_summary(self):
        print(f"MOCAP Dataset")
        print(f"Modality: {self.modality}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Number of samples: {self.length}")
        print(f"Features' dimension: {self.num_features}")
    
    def process(self):
        data_list = []

        for skeleton_file_name in tqdm(self.raw_file_names, desc="Processing skeletons"):
            sample_id = os.path.splitext(skeleton_file_name)[0]
            skeleton_full_path = os.path.join(self.raw_dir, skeleton_file_name)

            if self.pre_filter and not self.pre_filter(sample_id, self.part):
                continue

            skeleton_sequence = read_mocap_skeleton(skeleton_file_path=skeleton_full_path, benchmark=self.benchmark)  

            if skeleton_sequence is None:
                continue

            sample = MOCAP_Data(id=sample_id, skeleton_info=skeleton_sequence[0], skeleton_label=skeleton_sequence[1], modality="joint")  # always load joint first

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
    def __mocap_pre_filter__(sample_id, part):
        if sample_id not in MOCAP_ANNOTATED_SAMPLES:
            return False
        
        is_training = (sample_id.split('_')[0] in TRAINING_SUBJECTS)
        
        # print(f"{sample_id} : {is_training if part == "train" else not is_training} ")
        return is_training if part == "train" else not is_training
