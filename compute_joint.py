import numpy as np


class JointMaker:

    def __init__(self, verts, rot_mat, transl_mat, weights, in_bones ):
        self.verts = verts
        self.R = rot_mat
        self.T = transl_mat
        self.W = weights
        self.F_, self.N_, _ = self.verts.shape
        _, self.B_ = self.W.shape
        self.parent_bones = self.read_bones_file( in_bones )

    def read_bones_file(self, input_bones_file):
        parent_bones = dict()
        with open(input_bones_file, 'r') as f:
            for line in f.readlines():
                label = int( line.split()[0] )
                for child_label in line.split()[1:]:
                    child_label = int(child_label)
                    parent_bones[child_label] = label
        return parent_bones

    def compute(self):
        pass

    def compute_approx_joint(self):
        pass

    def compute_accurate_joint(self):
        pass