import numpy as np
import argparse
from scipy.linalg import svd,eig
from tvtk.api import tvtk
import h5py
#from tvtk.api import tvtk
from cvxopt import matrix, solvers, spmatrix
import threading
import Queue
import time
from geodesic import GeodesicDistanceComputation
from kabsch_algorithm import KabschAlgorithm
from init_cluster import SkinningInitialization
from inout import save_off
import os
import math

class RigidSolver:

    # R:rotation matrix
    # T:translation matrix
    # V:mesh sequence matrix
    # P:patch
    # F_: number of frames
    # N_: number of vertices
    # P_: number of patches
    # W: weight
    # B: Bones Relationship
    # B_:Bone number
    def __init__(self, in_mesh, out_working_dir, num_bones):

        self.V, self.F_, self.N_ = self.read_mesh_file(in_mesh)

        self.B_ =num_bones
        self.W = np.zeros( (self.N_,self.B_) ) #weight

        self.iter_times = 100
        self.out_working_dir = out_working_dir
        self.K_ = 4

        self.continue_iterate = True
        self.pre_e_rms = -1

        solvers.options[ 'show_progress' ] = False
        solvers.options[ 'maxiters' ] = 200
        solvers.options[ 'feastol' ] = 1e-9

        #args used in update transformation
        self.theta = 3.0

        with h5py.File( in_mesh, 'r') as f:
            self.compute_distance = GeodesicDistanceComputation( f['verts'].value.astype(np.float)[0], f['tris'].value ) #used for computing geodesic distance

        #rotation matrix and translation matrix
        init_solver = SkinningInitialization(in_mesh, self.B_, 5)
        self.R, self.T = init_solver.compute()
        assert self.R.shape == (self.F_, self.B_, 3, 3)
        assert self.T.shape == (self.F_, self.B_, 3)



    def read_mesh_file(self, input_mesh_file):
        with h5py.File(input_mesh_file,'r') as f:
            verts = f['verts'].value.astype(np.float)
            self.tris = f['tris'].value
            frame_num, verts_num, _ = verts.shape
            return verts, frame_num, verts_num



    def update_weight(self ):
        """
            Smoothing Skining Decomposition with Rigid Bones
            Section 3.2 Update Bone-Vertex Weight Map
        """
        #first global compute
        for v_ in range(self.N_):
            self.update_weight_per_vertex(v_, np.arange(self.B_ ) )

            #second local compute, assiociated with only K bones
            pi = self.V[ 0, v_ ]
            effects = []
            for j in range(self.B_):
                M = self.W[ v_, j ] * ( self.R[1:,j].dot(pi) + self.T[1:,j] )
                M = np.sum( M, axis = 0 )
                effects.append(np.sum( M**2 ))
            bones = np.argsort( effects )[::-1][:self.K_]
            bones = np.array(bones)
            self.update_weight_per_vertex(v_, bones)



    def update_weight_per_vertex(self, v_, bones):

        U = self.V[0] #rest pose

        B_ = bones.shape[0]
        ATAi = np.zeros( (B_, B_) )
        bTAi = np.zeros( B_ )
        for f_ in xrange(1, self.F_):
            A = np.zeros( (3, B_) )
            for idx, b_ in enumerate(bones):
                Rt =  self.R[f_, b_,: , :]
                Tt =  self.T[f_, b_, :]
                u =  U[v_,:]
                A[ :,idx ] = Rt.dot(u) + Tt # A = Rt * ui - Tt
            b = self.V[f_, v_]
            ATAi += A.T.dot(A)
            bTAi += b.T.dot(A)
        ATAi =matrix(ATAi)
        bTAi = matrix(bTAi)

        #quadratic programming
        #   min ( xT*Q*x + pT*x )
        #       Ax = b
        #       Gx <= h

        A = matrix(1.0,(1,B_))
        b = matrix(1.0)
        G = matrix( -1.0 * np.identity(B_) )
        h = matrix(0.0, (B_,1))
        sol = solvers.qp(2*ATAi, -2*bTAi, G, h, A, b)
        self.W[ v_ ] = 0
        self.W[ v_, bones.tolist() ] = np.array( sol['x'] ).reshape( B_ )




    def update_transformation_matrix(self):
        """
            Smoothing Skining Decomposition with Rigid Bones
            Section 3.3 Update Bone Transformations
        """


        p = self.V[ 0 ] #rest pose            
        R = np.zeros( (self.F_,self.B_,3,3) )
        T = np.zeros( (self.F_,self.B_,3) )

        insignificant_first_encounter = True #initialize
        cur_max_err = 0

        for j in range(self.B_):

            wj = self.W[:,j]
            sigma_wwj = np.sum( wj**2, axis=0 )
            

            if sigma_wwj > self.theta:
                wwj = wj**2
                p_star = wwj.dot(p) / sigma_wwj
                _p = p - p_star

                P = np.multiply(  wj, _p.T )

                for t in range( 1, self.F_ ):
                    Rt = self.R[ t ]
                    Tt = self.T[ t ]
                    qt = self.V[ t ].copy()
                    for i in range(self.B_):
                        if i == j:continue
                        pt =  p.dot( Rt[ i ].T ) + Tt[ i ]
                        qt -= np.multiply( self.W[:,i] , pt.T ).T
                    qt_star =wj.dot(qt) / sigma_wwj
                    _qt = qt - np.multiply( wj, np.repeat( qt_star[ np.newaxis, : ], self.N_, axis=0 ).T ).T

                    PQT = P.dot(_qt)
                    U, _, Vt = svd(PQT)

                    R[ t, j ] = Vt.T.dot(U.T)
                    T[ t, j ] = qt_star - R[ t, j ].dot( p_star )

            else:
                """
                    insignificant bone, re-initialization, use Kabsch algorithm
                """

                print "insignificant bone %d, using kabsch algorithm"%j

                if insignificant_first_encounter==True: #compute vertex idx with largest reconstruction error
                    err = []
                    for v in range(self.N_):
                        e = 0
                        for t in range(1,self.F_):
                            qt = self.V[t,v].copy()
                            for b in range(self.B_):
                                qt -= self.W[v, b]*( self.R[t,b].dot( self.V[0,v] ) + self.T[t,b] )
                            e += qt.dot(qt)
                        err.append( e )
                    err_sort_list = np.argsort( err )[::-1]
                    insignificant_first_encounter = False



                #search with one bone
                #for t in range( 1,self.F_):
                #    qt = self.V[t] - np.multiply( self.W[:,j], ( p.dot( self.R[ t, j ].T ) + np.repeat( self.T[ t, j ][np.newaxis, :], self.N_, axis= 0  ) ).T ).T
                #    max_err_vtx_idx = np.argmax( np.sum( qt**2, axis=-1 ), axis=0 ) # find the vertex with largest reconstruction error
                max_err_vtx_idx = err_sort_list[cur_max_err]
                cur_max_err += 1
                neighbours = np.argsort( self.compute_distance( max_err_vtx_idx ) )[ 1:21 ]  #find the 20 nearest neighbourhood
                for t in range( 1,self.F_):
                    ka_algo = KabschAlgorithm(p, self.V[t])
                    R[ t, j ], T[t, j] = ka_algo(neighbours)


        self.R = R.copy()
        self.T = T.copy()

    def save_skinning_mesh_to_file(self, path):
        U = self.V[0]
        for f in range(1,self.F_):
            tmp_mesh = np.zeros( [self.N_,3] )
            for i in range(self.N_):
                for j in range(self.B_):
                    tmp_mesh[i,:] += self.W[i,j]  *( self.R[f, j].dot(U[i,:]) + self.T[f,j] )
            save_off( path+r'/out_f%d.off'%f, tmp_mesh, self.tris )


    def save_transformation_mat_to_file_per_frame(self, path, frame_idx):
            with open(path,'w') as f:
                for j in range(self.B_):
                    for r in range(3):
                        for c in range(3):
                            f.write('%f '%self.R[frame_idx,j,r,c])
                    for r in range(3):
                        f.write('%f'%self.T[frame_idx,j,r]);
                    f.write('\n')

    def cal_error(self):
        err = 0
        for v in range(self.N_):
            for t in range(1,self.F_):
                qt = self.V[t,v].copy()
                for b in range(self.B_):
                    qt -= self.W[v, b]*( self.R[t,b].dot( self.V[0,v] ) + self.T[t,b] )
                    err += qt.dot(qt)
        e_rms = math.sqrt( err/(self.N_*self.F_*3 ) )*1000
        if self.pre_e_rms != -1:
            if (1.0 - e_rms/self.pre_e_rms) < 0.01:
                self.continue_iterate = False
        self.pre_e_rms = e_rms




    def compute(self):

        for i in range( self.iter_times ):
            if self.continue_iterate==False:break
            start_time = time.time()
            print  'the %dth iteration'%i
            self.update_weight()
            self.update_transformation_matrix()
            print time.time() - start_time
            with open( self.out_working_dir+r'/weights_iter%d'%(i+1), 'w') as f:
                for v in range( self.N_ ):
                    for j in range( self.B_ ):
                        f.write('%f '%self.W[v,j] )
                    f.write('\n')
            mesh_dir = self.out_working_dir + r'/iter%d_mesh'%i
            if os.path.exists(mesh_dir) == False:
                os.mkdir(mesh_dir)
            self.save_skinning_mesh_to_file(mesh_dir)
            #self.cal_error()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute Rigid Bone Transfromation')
    parser.add_argument('input_file')
    parser.add_argument('output_working_dir')
    parser.add_argument('bones_num')
    args = parser.parse_args()
    solver = RigidSolver(args.input_file, args.output_working_dir, int(args.bones_num))
    solver.compute()