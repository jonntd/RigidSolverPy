
import argparse
import numpy as np
from scipy.linalg import svd, norm, cho_factor, cho_solve
import h5py
from tvtk.api import tvtk

from geodesic import GeodesicDistanceComputation

def normalize(v_arr):
    lens = np.sqrt( v_arr[:,0]**2 + v_arr[:,1]**2 + v_arr[:,2]**2 )
    v_arr[:,0] /= lens
    v_arr[:,1] /= lens
    v_arr[:,2] /= lens
    return v_arr

def cal_mesh_normals(verts, faces):
    norms = np.zeros(verts.shape, dtype=verts.dtype)
    tris = verts[faces]
    n = np.cross( tris[::,1]-tris[::,0], tris[::,2]-tris[::,0])
    n = normalize(n)
    norms[ faces[:,0] ] += n
    norms[ faces[:,1] ] += n
    norms[ faces[:,2] ] += n
    norms = normalize(norms)
    return norms

def cal_mesh_normals_api(verts,tris):
    pd = tvtk.PolyData(points=verts, polys=tris)
    n = tvtk.PolyDataNormals(input=pd,splitting=False)
    n.update()
    norms = n.output.point_data.normals
    #already normalized
    return norms

def corresponding_edge(N, faces):
    #Get edges per vertex for further calculation
    edges = np.empty(N, dtype = int)
    for i in xrange(N):
        for j in xrange(3):
            idx = np.where( faces[:,j]== i )[0]
            if idx.size != 0:
                break
        edges[i] = faces[ idx[0], (j+1)%3 ]
    return edges

#Get local coordinate transform matrix
def w2m_transf_mat(norm, one_edge, translate):
    x_axis = normalize(norm[np.newaxis,:]).reshape(3)
    y_axis = normalize( np.cross(x_axis, one_edge)[np.newaxis,:] ).reshape(3)
    z_axis = normalize( np.cross(x_axis, y_axis)[np.newaxis,:] ).reshape(3)
    transf_mat = np.zeros([4,4])
    #transf_mat[:3,3] = translate
    transf_mat[3,3] = 1
    transf_mat[0,:3] = x_axis
    transf_mat[1,:3] = y_axis
    transf_mat[2,:3] = z_axis
    return transf_mat

def project_weight(x):
    x = np.maximum(0., x)
    max_x = x.max()
    if max_x == 0:
        return x
    else:
        return x / max_x

def prox_l1l2(Lambda, x, beta):
    xlen = np.sqrt((x**2).sum(axis=-1))
    with np.errstate(divide='ignore'):
        shrinkage = np.maximum(0.0, 1 - beta * Lambda / xlen)
    return (x * shrinkage[...,np.newaxis])

def compute_support_map(idx, geodesics, min_dist, max_dist):
    phi = geodesics(idx)
    return (np.clip(phi, min_dist, max_dist) - min_dist) / (max_dist - min_dist)


def main(input_skin_file, input_animation_file, output_sploc_file):

    K = 10 # number of components
    smooth_min_dist = 0.1 # minimum geodesic distance for support map, d_min_in paper
    smooth_max_dist = 0.7 # maximum geodesic distance for support map, d_max in paper
    num_iters_max = 10 # number of iterations to run
    sparsity_lambda = 2. # sparsity parameter, lambda in the paper

    rho = 10.0 # penalty parameter for ADMM
    num_admm_iterations = 10 # number of ADMM iterations

    with h5py.File(input_skin_file,'r') as f:
        skin_verts = f['verts'].value.astype(np.float)
        skin_tris = f['tris'].value


    with h5py.File(input_animation_file,'r') as f:
        verts = f['verts'].value.astype(np.float)
        tris = f['tris'].value

    F,N,_ = verts.shape

    #Preprocess
    skin_norms = np.array( [ cal_mesh_normals_api(skin_verts[f], skin_tris )  for f in xrange(F) ] )
    #skin_norms2 = np.array( [ cal_mesh_normals(skin_verts[f], skin_tris )  for f in xrange(F) ] )
    e = corresponding_edge( N, skin_tris )
    w2m_mat = np.array( [ [ w2m_transf_mat(skin_norms[f][i], skin_verts[f][ e[i] ] - skin_verts[f][i], skin_verts[f][i] ) for i in xrange(N) ] for f in xrange(F) ] )

    compute_geodesic_distance = GeodesicDistanceComputation(verts[0],tris)

    X = verts - skin_verts
    X_ext = np.append(X, np.ones([F,N,1],dtype=X.dtype ), axis=2)
    X_loc = np.empty( X_ext.shape ,dtype=X.dtype )
    for i in xrange(F):
        for j in xrange(N):
            X_loc[i,j,:] = w2m_mat[i,j,:,:].dot( X_ext[i,j,:] )
    X = np.delete(X_loc,3,2)
    pre_scale_factor = 1 / np.std(X)
    X *= pre_scale_factor
    R = X.copy() # residual

    # find initial components explaining the residual
    C = []
    W = []
    for k in xrange(K):
        # find the vertex explaining the most variance across the residual animation
        magnitude = (R**2).sum(axis=2)
        idx = np.argmax(magnitude.sum(axis=0))
        # find linear component explaining the motion of this vertex
        U, s, Vt = svd(R[:,idx,:].reshape(R.shape[0], -1).T, full_matrices=False)
        wk = s[0] * Vt[0,:] # weights
        # invert weight according to their projection onto the constraint set
        # this fixes problems with negative weights and non-negativity constraints
        wk_proj = project_weight(wk)
        wk_proj_negative = project_weight(-wk)
        wk = wk_proj \
                if norm(wk_proj) > norm(wk_proj_negative) \
                else wk_proj_negative
        s = 1 - compute_support_map(idx, compute_geodesic_distance, smooth_min_dist, smooth_max_dist)
        # solve for optimal component inside support map
        ck = (np.tensordot(wk, R, (0, 0)) * s[:,np.newaxis])\
                / np.inner(wk, wk)
        C.append(ck)
        W.append(wk)
        # update residual
        R -= np.outer(wk, ck).reshape(R.shape)
    C = np.array(C)
    W = np.array(W).T

    # prepare auxiluary variables
    Lambda = np.empty((K, N))
    U = np.zeros((K, N, 3))

    # main global optimization
    for it in xrange(num_iters_max):
        # update weights
        Rflat = R.reshape(F, N*3) # flattened residual
        for k in xrange(C.shape[0]): # for each component
            Ck = C[k].ravel()
            Ck_norm = np.inner(Ck, Ck)
            if Ck_norm <= 1.e-8:
                # the component seems to be zero everywhere, so set it's activation to 0 also
                W[:,k] = 0
                continue # prevent divide by zero
            # block coordinate descent update
            Rflat += np.outer(W[:,k], Ck)
            opt = np.dot(Rflat, Ck) / Ck_norm
            W[:,k] = project_weight(opt)
            Rflat -= np.outer(W[:,k], Ck)
        # update spatially varying regularization strength
        for k in xrange(K):
            ck = C[k]
            # find vertex with biggest displacement in component and compute support map around it
            idx = (ck**2).sum(axis=1).argmax()
            support_map = compute_support_map(idx, compute_geodesic_distance,
                                              smooth_min_dist, smooth_max_dist)
            # update L1 regularization strength according to this support map
            Lambda[k] = sparsity_lambda * support_map
        # update components
        Z = C.copy() # dual variable
        # prefactor linear solve in ADMM
        G = np.dot(W.T, W)
        c = np.dot(W.T, X.reshape(X.shape[0], -1))
        solve_prefactored = cho_factor(G + rho * np.eye(G.shape[0]))
        # ADMM iterations
        for admm_it in xrange(num_admm_iterations):
            C = cho_solve(solve_prefactored, c + rho * (Z - U).reshape(c.shape)).reshape(C.shape)
            Z = prox_l1l2(Lambda, C + U, 1. / rho)
            U = U + C - Z
        # set updated components to dual Z,
        # this was also suggested in [Boyd et al.] for optimization of sparsity-inducing norms
        C = Z
        # evaluate objective function
        R = X - np.tensordot(W, C, (1, 0)) # residual
        sparsity = np.sum(Lambda * np.sqrt((C**2).sum(axis=2)))
        e = (R**2).sum() + sparsity
        # TODO convergence check
        print "iteration %03d, E=%f" % (it, e)

    # undo scaling
    C /= pre_scale_factor
    for _, c in enumerate(C):
            c_ext = np.append( c,np.ones( [N,1], dtype = c.dtype ) ,axis=1 )
            c_world = np.empty( c_ext.shape, dtype = c.dtype )
            for i in xrange(N):
                m2w_mat_per_v = np.transpose(w2m_mat[0,i,::])
                c_world[i,:] = m2w_mat_per_v.dot( c_ext[i,:] )
            c = np.delete( c_world, 3, 1)
    Xmean = skin_verts[0]
    with h5py.File(output_sploc_file, 'w') as f:
        f['default'] = Xmean
        f['tris'] = tris
        for i, c in enumerate(C):
            f['comp%03d' % i] = c + Xmean

def save_recover_mesh_to_file(X, W, Xmean):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find Sparse Localized Deformation Components')
    parser.add_argument('input_skin_file')
    parser.add_argument('input_animation_file')
    parser.add_argument('output_sploc_file')
    parser.add_argument('-a', '--output-anim',
                        help='Output animation file (will also save the component weights)')
    args = parser.parse_args()
    main(args.input_skin_file,
        args.input_animation_file,
        args.output_sploc_file)