import numpy as np

def get_nullspace_closed_form(vec):
    N = len(vec)
    N1 = N-1

    xn = vec[-1]
    vec_reduced = vec[:-1]
    Jac = np.zeros((N, N1))

    outer_prod = np.outer(vec_reduced, vec_reduced)

    if xn > 0:
        
        Jac[:N1,:] = np.eye(N1) - outer_prod / (1+xn)
        Jac[-1, :] = -vec_reduced.T
    else:
        Jac[:N1,:] = np.eye(N1) - outer_prod / (1-xn)
        Jac[-1, :] = vec_reduced.T
    return Jac

def norm_spherical(x):
    return x/np.linalg.norm(x)

mean_x = 1.0
mean_y = 2.0
mean_z = 3.0

x = np.array([mean_x, mean_y, mean_z])
E_xx = np.diag([1, 2, 3])


def get_Jac_mu(point):
    mean_x = point[0]
    mean_y = point[1]
    mean_z = point[2]

    norm_mu = (mean_x**2 + mean_y**2 + mean_z**2)**(3/2)

    J_mu = np.zeros((3, 3)) 
    J_mu[0, 0] = (mean_y**2 + mean_z**2) / norm_mu
    J_mu[0, 1] = (-mean_x * mean_y) / norm_mu
    J_mu[0, 2] = (-mean_x * mean_z) / norm_mu

    J_mu[1, 0] = (-mean_x * mean_y) / norm_mu
    J_mu[1, 1] = (mean_x**2 + mean_z**2) / norm_mu
    J_mu[1, 2] = (-mean_y * mean_z) / norm_mu

    J_mu[2, 0] = (-mean_x * mean_z) / norm_mu
    J_mu[2, 1] = (-mean_y * mean_z) / norm_mu
    J_mu[2, 2] = (mean_x**2 + mean_y**2) / norm_mu

    return J_mu



# förstner

def get_Jac_foerstner(point):

    norm_x = np.linalg.norm(point)

    jac = (np.eye(3) - np.outer(point, point) / (point.T @ point)) / norm_x

    return jac

print(get_Jac_mu(x))
print(get_Jac_foerstner(x))

print(np.linalg.matrix_rank(get_Jac_mu(x)))
print(np.linalg.matrix_rank(get_Jac_foerstner(x)))

J_spherical = get_Jac_foerstner(x)

# get reduces covariance in tangent space
xs = norm_spherical(x)
E_xsxs= J_spherical @ E_xx @ J_spherical.T

print(x.T)
J_reduced = get_nullspace_closed_form(x.T)
E_xrxr = J_reduced.T @ E_xsxs @ J_reduced

print(E_xrxr)

# # project point and covariance to image
# covariance3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# mean3d = np.array([1, 2, 3])

# R_c_w = np.eye(3)
# t_c_w = np.zeros(3)
# t_c_w[2] = 1




