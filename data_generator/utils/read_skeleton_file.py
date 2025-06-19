"""
Read a .skeleton file from "NTU RGB+D 3D Action Recognition Dataset".

Argument:
    filename: full adress and filename of the .skeleton file.

Running :
    ~/graph-based_action-recognition$ python -m data_generator.utils.read_skeleton_file data/nturgb+d_skeletons/S001C001P001R001A001.skeleton
"""
from data_generator.skeleton.body import Body
from data_generator.skeleton.joint import Joint

def read_skeleton_file(filename: str):
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

                frame_info.append(body)
            skeleton_sequence.append(frame_info)

    return skeleton_sequence