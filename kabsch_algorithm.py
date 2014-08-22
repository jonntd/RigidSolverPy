import numpy as np
from scipy.linalg import svd, det
class KabschAlgorithm:
    """
        Implementation of KabschAlgorithm
    """
    def __init__(self, _P, _Q):
        """
                _P,_Q : two mesh
        """
        self.P = _P
        self.Q = _Q

    def __call__(self, region):

        V_ = region.shape   # number of vertex
        P = self.P[ region ]
        Q = self.Q[ region ]

        #Step One: Translation, done by subtracting from the point coordinates the coordinates of the respective centroid.
        _p = P.mean(0)
        _q = Q.mean(0)
        P = P - _p
        Q = Q - _q

        #Step Two: Computation of the covariance matrix
        A = Q.T.dot(P)

        #Step Three: Computation of the optimal rotation matrix
        V, _, Wt = svd(A)
        R = Wt.T.dot(V.T)
        S = np.eye(3)
        if det(R) < 0:
            S[2,2] = -1
        elif det(R) == 0:
            S[2,2] = 0
        R = (Wt.T.dot(S)).dot(V.T)

        T = _q - R.dot(_p)

        return R,T




