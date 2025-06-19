"""
Customized dataset class for the NTU RGB+D 3D Action Recognition Dataset.

This class handles loading, preprocessing, and formatting of skeleton data
for use in action recognition tasks.
"""

import os
import torch
import pickle
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from typing import override

from torch_geometric.data import Dataset, Data 
from torch_geometric.utils import to_networkx


from Src.Skeleton.read_skeleton_file import read_skeleton_file


ROOT_PATH = "Dataset/nturgb+d_skeletons/"
SAVE_PATH = "Dataset/"


IGNORED_SAMPLE_PATH = "Dataset/NTU_RGBD_samples_with_missing_skeletons.txt"
with open(IGNORED_SAMPLE_PATH, "r") as f:
    IGNORED_SAMPLES = set([line.strip() for line in f.readlines()[3:]])


CONNECTING_JOINT = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11,
                    1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]
_directed_edges = [(i + 1, CONNECTING_JOINT[i]) for i in range(0, len(CONNECTING_JOINT))]
_bidirectional_edges = _directed_edges + [(v,u) for u, v in _directed_edges]
EDGE_INDEX  = torch.tensor(_bidirectional_edges, dtype=torch.long).t()


"""
For instance, we choose to use only samples containing a single skeleton to simplify our experiments.
"""
TRAINING_CLASS = list(range(50, 61)) + list(range(106, 121))
TRAINING_SUBJECT = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
#TRAINING_CAMERAS = [2,3]


MAX_BODY = 2
NUM_JOINT = 25
MAX_FRAME = 300


class NTU_Data(Data):
     """
     Initialize a graph-based data object for a skeleton frame from a .skeleton file.

     Arguments:
     - file_name: Full path and filename of the .skeleton file.
     - frame_idx: Index of the frame from which the skeleton is collected.
     - body_info: Skeleton features (joints' position/direcction etc.) to be converted into a torch_geometric.Data object.
     """
     def __init__(self, file_name : str, frame_idx : int, body_info):
          super(NTU_Data, self).__init__()

          self.file_name = file_name

          # Extract possible labels
          self.setup_id = int(file_name[file_name.find('S') + 1:file_name.find('S') + 4])
          self.subject_id = int(file_name[file_name.find('P') + 1:file_name.find('P') + 4])
          self.camera_id = int(file_name[file_name.find('C') + 1:file_name.find('C') + 4])
          self.replication_id = int(file_name[file_name.find('R') + 1:file_name.find('R') + 4])
          self.action_class = int(file_name[file_name.find('A') + 1:file_name.find('A') + 4])

          self.frame_idx = frame_idx # Later servers as timestamp

          self.body_info = body_info # Later serves as graph attributes

          self.edge_index = EDGE_INDEX
          self.x = self.__extract_node_features()
          self.y = torch.tensor(self.action_class, dtype=torch.long)


     def __extract_node_features(self):
          joints = self.body_info.joints
          joint_features = [[joint.x, joint.y, joint.z] for joint in joints]
          return torch.tensor(joint_features, dtype=torch.float)
          

class NTU_Dataset(Dataset):
     """
     Initialize a customized dataset with Data Objects in torch_geometric format. 
     
     Argument :
          - root: Root directory where the dataset should be saved
          - ignored_sample_path: Path towards list of samples to be ignored as defined in the original papers.

     """
     def __init__(self, root=ROOT_PATH, benchmark="xview", part="eval"):
          super(NTU_Dataset, self).__init__(root=root)

     @property
     def raw_dir(self):
          return self.root
     
     @property
     def processed_dir(self):
         base_dir = os.path.dirname(self.root)
         dataset_name = os.path.basename(self.root)
         return os.path.join(base_dir, f"processed_{dataset_name}")
     

     @property
     def raw_file_names(self):
          return [os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith('.skeleton')]
     
     @property
     def processed_file_names(self):
          if not os.path.exists(self.processed_dir):
               return []
          return [f for f in os.listdir(self.processed_dir) if f.endswith('.pt')]
     
     
     def process(self):
          for file_name in self.raw_file_names:
               # Create a subfolder for each processed skeleton file
               file_dir = os.path.join(self.processed_dir, os.path.splitext(os.path.basename(file_name))[0])
               print(file_dir)
               os.makedirs(file_dir, exist_ok=True)

               body_info = read_skeleton_file(file_name)
               for frame_idx in range(1, len(body_info) + 1):
                    data_sample = NTU_Data(file_name, frame_idx, body_info[frame_idx - 1][0])
                    if self.pre_filter is None or self.pre_filter(data_sample):
                         torch.save(data_sample, os.path.join(file_dir, f'{os.path.splitext(os.path.basename(file_name))[0]}_frame_{frame_idx}.pt'))

               return


     def len(self):
          num_samples = 0
          for file_name in self.processed_file_names:
               body_info = read_skeleton_file(file_name)
               num_samples += len(body_info)
          return num_samples


     def get(idx : int):
          pass

     @staticmethod
     def __nturgbd_pre_filter__(sample : NTU_Data):
          if sample.file_name in IGNORED_SAMPLES:
               return False
          
          
               

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customized dataset class for the NTU RGB+D 3D Action Recognition Dataset.")

    arg = parser.parse_args()

    """
    data_path = "Dataset/nturgb+d_skeletons/S001C001P001R001A060.skeleton"
    body = read_skeleton_file(data_path)
    frame_idx = 1
    data_point = NTU_Data(data_path, frame_idx, body[frame_idx][0])
    print(data_point.num_nodes)
    print(data_point.is_directed())
    """

    ntu_rgbd_dataset = NTU_Dataset()
    


