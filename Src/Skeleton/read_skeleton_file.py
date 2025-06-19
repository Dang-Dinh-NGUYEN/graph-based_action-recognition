"""
Read a .skeleton file from "NTU RGB+D 3D Action Recognition Dataset".

Argument:
    filename: full adress and filename of the .skeleton file.

Running :
    ~/graph-based_action-recognition$ python -m Src.Skeleton.read_skeleton_file Dataset/nturgb+d_skeletons/S001C001P001R001A001.skeleton
"""
import sys
from Src.Skeleton.body import Body
from Src.Skeleton.joint import Joint

def read_skeleton_file(filename: str):
    with open(filename, "r") as file:
        def read_int():
            return int(file.readline().strip())

        def read_line_as_floats():
            return list(map(float, file.readline().strip().split()))

        frame_count = read_int()
        body_info = []

        for _ in range(frame_count):
            frame_bodies = []
            body_count = read_int()

            for _ in range(body_count):
                body = Body()
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
                    joint = Joint(jointinfo, tracking_state)
                    body.joints.append(joint)

                frame_bodies.append(body)
            body_info.append(frame_bodies)

    return body_info
    

# TO DO: argparse should be used
"""
if len(sys.argv) != 2:
    print('Need a filename as argument')
    exit(1)

data = read_skeleton_file(sys.argv[1])
print("read_skeleton_file.py Frame count:", len(data))
"""