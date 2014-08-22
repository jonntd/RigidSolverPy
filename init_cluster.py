import numpy as np
import h5py
from kabsch_algorithm import KabschAlgorithm
import random
from time import time
from geodesic import GeodesicDistanceComputation
from math import sin,cos

class SkinningInitialization:

    def __init__(self, in_mesh, bones_num, iter_times = 5 ):
        self.B_ = bones_num
        self.read_mesh_file(in_mesh)

        #each cluster is represented as a sequence of |t| bone transformations
        self.clust_rot = np.empty( (self.F_, self.B_, 3, 3) ) #Rotation Feature
        self.clust_transl = np.empty( (self.F_, self.B_, 3) ) #Translation Feature

        self.clust_patch = {}

        self.iter_times = iter_times

        with h5py.File( in_mesh, 'r') as f:
            self.compute_distance = GeodesicDistanceComputation( f['verts'].value.astype(np.float)[0], f['tris'].value ) #used for computing geodesic distance
        self.neighbour_num = 20

    def read_mesh_file(self, input_mesh_file):
        with h5py.File(input_mesh_file,'r') as f:
            self. verts = f['verts'].value.astype(np.float)
            self.F_, self.N_, _ = self.verts.shape


    def rand_init_cluster(self):
        '''
            Firstly randomly initialize the clusters
        '''
        seeds = random.sample( range(self.N_),self.B_ )

        for t in range( 1,self.F_ ):
            kabsch_algorithm = KabschAlgorithm( self.verts[0], self.verts[t] )
            for j in range(self.B_ ):
                neighbours = np.argsort( self.compute_distance( seeds[j] ) )[1: (self.neighbour_num+1) ]
                self.clust_rot[ t, j ], self.clust_transl[ t, j ] = kabsch_algorithm(  neighbours  )





    def update_transformation(self):

        for j in range(self.B_ ):
            if self.clust_patch[j] != []:
                for t in range( 1,self.F_ ):
                    kabsch_algorithm = KabschAlgorithm( self.verts[0], self.verts[t] )
                    self.clust_rot[ t, j ], self.clust_transl[ t, j ] = kabsch_algorithm(  np.array(self.clust_patch[j])  )
            else:
                #if some cluster has no members, choose randomly
                print 'no members in %dth bone'%j
                for t in range( 1,self.F_ ):
                    kabsch_algorithm = KabschAlgorithm( self.verts[0], self.verts[t] )
                    random.seed( time())
                    seed = random.randint(0, self.N_ - 1)
                    neighbours = np.argsort( self.compute_distance( seed ) )[1: (self.neighbour_num+1) ]
                    self.clust_rot[ t, j ], self.clust_transl[ t, j ] = kabsch_algorithm(  neighbours  )
                pass


    def update_cluster(self):
        u = self.verts[0]
        patch = {}
        for j in range(self.B_):
            patch[j] = []
        for v_ in range( self.N_ ):
            err_list = []
            for j in range( self.B_ ):
                err = 0
                for t in range(1, self.F_ ):
                    vt = self.verts[t, v_]
                    qt = vt - ( self.clust_rot[ t, j ].dot( u[v_] ) + self.clust_transl[  t, j ] )
                    err += qt.dot(qt)
                err_list.append( err )
            min_reconstruct_err_bone = np.argmin( err_list )
            patch[ min_reconstruct_err_bone ].append( v_ )
        self.clust_patch = patch.copy()

    def compute(self):
        self.rand_init_cluster()

        for i in range( self.iter_times ):
            print 'iterate %dth'%i
            self.update_cluster()
            self.update_transformation()
        return self.clust_rot, self.clust_transl

if __name__ == '__main__':
    solver = SkinningInitialization( r'./input/models_aligned.h5',16 ,20)
    solver.compute()
    with open(r'./result/init_patch','w') as f:
        for p in solver.clust_patch.values():
            for item in p:
                f.write( '%d '%item )
            f.write('\n')


            













