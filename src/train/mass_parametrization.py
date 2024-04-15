import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import erf, erfinv


"""
Note:
What you need for the mass parametrization is the functions:
    1) apply_top_decay_mass_param and undo_top_decay_mass_param for the leptonic decay
    2) apply_top_decay_mass_param_hadronic and undo_top_decay_mass_param_hadronic for the hadronic decay
The rest is helper functions. 

What could also be interesting is the functions
    breit_wigner_forward
    breit_wigner_reverse
They map a breit-wigner peak to a gaussian, we use them as preprocessing for the top and W mass channels. 
For top masses we use peak_position=172.7 and width=1, for W masses peak_position=80.4 and width=1
"""


def apply_top_decay_mass_param(p_top, p_w, p_x1):
    # computes top decay mass parameterization
    # expects 4-momenta of top, W, first_decay_product in Eppp parametrization in lab frame
    p_x2 = p_w - p_x1
    p_b = p_top - p_w

    p_w_frame_top = lorentz_boost(p_w, p_top)
    p_x1_frame_w = lorentz_boost(p_x1, p_w)

    m_t = mass(p_top)
    pt_t = pt(p_top)
    eta_t = eta(p_top)
    phi_t = azimuthal_angle(p_top)

    m_w = mass(p_w_frame_top)
    eta_w = eta(p_w_frame_top)
    phi_w = azimuthal_angle(p_w_frame_top)

    eta_x1 = eta(p_x1_frame_w)
    phi_x1 = azimuthal_angle(p_x1_frame_w)

    return np.stack([m_t,
                           pt_t,
                           eta_t,
                           phi_t,
                           m_w,
                           eta_w,
                           phi_w,
                           eta_x1,
                           phi_x1], axis=1)


def undo_top_decay_mass_param(mass_param, return_all = True):
    # undoes top decay mass parameterization
    # expects inputs in the form as output by apply_top_decay_mass_param_single
    # m_t = mass_param[:, 0]
    # pt_t = mass_param[:, 1]
    # eta_t = mass_param[:, 2]
    # phi_t = mass_param[:, 3]
    # m_w = mass_param[:, 4]
    # eta_w = mass_param[:, 5]
    # phi_w = mass_param[:, 6]
    # eta_x1 = mass_param[:, 7]
    # phi_x = mass_param[:, 8]

    # compute top momentum
    p_top = jet2cartesian(mass_param[:, :4])

    # compute w momentum in top rest frame
    pnorm2_w = \
        (mass_param[:, 0]**2 + 4.7**2 - mass_param[:, 4]**2)**2 / (4 * mass_param[:, 0]**2) - 4.7**2
    pt_w = pt_from_norm(np.sqrt(pnorm2_w), mass_param[:, 5])
    p_w = jet2cartesian_single(mass_param[:, 4], pt_w, mass_param[:, 5], mass_param[:, 6])

    # compute up momentum in w rest frame
    E_x1 = mass_param[:, 4] / 2
    m_x1 = np.zeros_like(E_x1)
    pt_x1 = pt_from_norm(E_x1, mass_param[:, 7])
    p_x1 = jet2cartesian_single(m_x1, pt_x1, mass_param[:, 7], mass_param[:, 8])

    p_w = lorentz_boost(p_w, flip(p_top))
    p_x1 = lorentz_boost(p_x1, flip(p_w))

    p_b = p_top - p_w
    p_x2 = p_w - p_x1
    if return_all:
        return p_top, p_b, p_w, p_x1, p_x2
    else:
        return p_top, p_w, p_x1


def apply_top_decay_mass_param_hadronic(p_top, p_w, p_x1):
    # computes top decay mass parameterization
    # expects 4-momenta of top, W, first_decay_product in Eppp parametrization in lab frame
    p_x2 = p_w - p_x1
    p_b = p_top - p_w

    p_w_frame_top = lorentz_boost(p_w, p_top)
    p_x1_frame_w = lorentz_boost(p_x1, p_w)

    m_t = mass(p_top)
    pt_t = pt(p_top)
    eta_t = eta(p_top)
    phi_t = azimuthal_angle(p_top)

    m_w = mass(p_w_frame_top)
    eta_w = eta(p_w_frame_top)
    phi_w = azimuthal_angle(p_w_frame_top)

    m_x1 = mass(p_x1_frame_w, clip=True)
    eta_x1 = eta(p_x1_frame_w)
    phi_x1 = azimuthal_angle(p_x1_frame_w)

    return np.stack([m_t,
                           pt_t,
                           eta_t,
                           phi_t,
                           m_w,
                           eta_w,
                           phi_w,
                           m_x1,
                           eta_x1,
                           phi_x1], axis=1)


def undo_top_decay_mass_param_hadronic(mass_param, return_all = True):
    # undoes top decay mass parameterization
    # expects inputs in the form as output by apply_top_decay_mass_param_single_hadronic
    # m_t = mass_param[:, 0]
    # pt_t = mass_param[:, 1]
    # eta_t = mass_param[:, 2]
    # phi_t = mass_param[:, 3]
    # m_w = mass_param[:, 4]
    # eta_w = mass_param[:, 5]
    # phi_w = mass_param[:, 6]
    # m_x1 = mass_param[:, 7]
    # eta_x1 = mass_param[:, 8]
    # phi_x = mass_param[:, 9]

    # compute top momentum
    p_top = jet2cartesian(mass_param[:, :4])

    # compute w momentum in top rest frame
    pnorm2_w = \
        (mass_param[:, 0]**2 + 4.7**2 - mass_param[:, 4]**2)**2 / (4 * mass_param[:, 0]**2) - 4.7**2
    pt_w = pt_from_norm(np.sqrt(pnorm2_w), mass_param[:, 5])
    p_w = jet2cartesian_single(mass_param[:, 4], pt_w, mass_param[:, 5], mass_param[:, 6])

    # compute up momentum in w rest frame
    m_x1 = mass_param[:, 7]
    pnorm2_x1 = \
        (mass_param[:, 4] ** 2 - mass_param[:, 7] ** 2) ** 2 / (4 * mass_param[:, 4] ** 2)
    pt_x1 = pt_from_norm(np.sqrt(pnorm2_x1), mass_param[:, 8])
    p_x1 = jet2cartesian_single(m_x1, pt_x1, mass_param[:, 8], mass_param[:, 9])

    p_w = lorentz_boost(p_w, flip(p_top))
    p_x1 = lorentz_boost(p_x1, flip(p_w))

    p_b = p_top - p_w
    p_x2 = p_w - p_x1
    if return_all:
        return p_top, p_b, p_w, p_x1, p_x2
    else:
        return p_top, p_w, p_x1



# applies a lorentz boost to p_target, whose velocity is inferred
# from p_frame_particle, such that the new frame becomes the rest
# frame of the particle with momentum p_frame_particle;
# note that this algorithm is numerically unstable to a significant
# extent for highly relativistic events
def lorentz_boost(p_target, p_frame_particle):
    K = np.array([
        [[0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0]],
    ])

    beta = p_frame_particle[...,1:] / p_frame_particle[...,0:1]

    # don't boost momenta when norm(beta) is zero
    boost_mask = np.linalg.norm(beta, axis=-1) != 0.0
    n = beta[boost_mask] / np.linalg.norm(beta[boost_mask], axis=-1, keepdims=True)
    rapidity = np.arctanh(np.sqrt(np.sum(beta[boost_mask]**2, axis=-1)))

    n = np.expand_dims(n, axis=(-1,-2))
    rapidity = np.expand_dims(rapidity, axis=(-1,-2))

    n_dot_K = 0
    for i in range(len(K)):
        n_dot_K += n[...,i,:,:] * K[i]

    B = np.eye(4, 4) - np.sinh(rapidity) * n_dot_K \
        + (np.cosh(rapidity) - 1) * (n_dot_K @ n_dot_K)

    p_boosted = np.full_like(p_target, np.nan)
    p_boosted[boost_mask] = \
        (B @ np.expand_dims(p_target[boost_mask], axis=-1)).squeeze(-1)
    p_boosted[~boost_mask] = p_target[~boost_mask]
    return p_boosted

def flip(p):
    p_flipped = np.full(p.shape, np.nan)
    p_flipped[...,0] = p[...,0]
    p_flipped[...,1:] = -p[...,1:]
    return p_flipped

def mass_squared(p):
    return p[...,0]**2 - np.sum(p[...,1:]**2, axis=-1)

def mass(p, clip=True):
    mass_2 = mass_squared(p)
    if clip:
        return np.sqrt(np.clip(mass_2, a_min=0, a_max=None))
    else:
        return np.sqrt(mass_2)

def norm_squared(p, keepdims=False):
    return np.sum(p[..., 1:]**2, axis=-1, keepdims=keepdims)

def norm(p, keepdims=False):
    return np.sqrt(norm_squared(p, keepdims=keepdims))

def pt2(p):
    return p[...,1]**2 + p[...,2]**2
def pt(p):
    return np.sqrt(pt2(p))
def pt2_from_norm(p_norm_squared, eta):
    return p_norm_squared * (1.0 - np.tanh(eta)**2)
def pt_from_norm(p_norm, eta):
    return p_norm * np.sqrt(1.0 - np.tanh(eta)**2)

def eta(p):
    # p_norm = np.sqrt(np.sum(p[...,1:]**2, axis=-1))
    #return 0.5 * np.log((p[..., 0] + p[..., 3]) / (p[..., 0] - p[..., 3]))
    return np.arctanh(p[..., 3] / np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2 + p[..., 3] ** 2))
def polar_angle(p):
    return np.arccos(p[..., 3] / norm(p))
def azimuthal_angle(p):
    return np.arctan2(p[...,2], p[...,1])

def jet2cartesian(p_m_pt_eta_phi):

    m ,pt, eta, phi = p_m_pt_eta_phi[..., 0], p_m_pt_eta_phi[..., 1], p_m_pt_eta_phi[..., 2], p_m_pt_eta_phi[..., 3]

    p = np.full((*m.shape, 4), np.nan)
    p[..., 1] = pt * np.cos(phi)
    p[..., 2] = pt * np.sin(phi)
    p[..., 3] = pt * np.sinh(eta)

    p[..., 0] = np.sqrt(norm_squared(p) + m**2)
    return p


def jet2cartesian_single(m ,pt, eta, phi):

    p = np.full((*m.shape, 4), np.nan)
    p[..., 1] = pt * np.cos(phi)
    p[..., 2] = pt * np.sin(phi)
    p[..., 3] = pt * np.sinh(eta)

    p[..., 0] = np.sqrt(norm_squared(p) + m**2)
    return p


def cartesian2jet(p_eppp):

    p = np.full(p_eppp.shape, np.nan)
    p[..., 0] = mass(p_eppp)
    p[..., 1] = pt(p_eppp)
    p[..., 2] = eta(p_eppp)
    p[..., 3] = azimuthal_angle(p_eppp)

    return p


def breit_wigner_forward(events, peak_position, width):
    z1 = 1 / np.pi * np.arctan((events - peak_position) / width) + 0.5
    return np.sqrt(2) * erfinv(2 * z1 - 1)


def breit_wigner_reverse(events, peak_position, width):
    a = events/np.sqrt(2)
    a = erf(a)
    a = 0.5*(a+1)
    a = a -0.5
    a = a*np.pi
    a = np.tan(a)
    a = a*width + peak_position
    return a
