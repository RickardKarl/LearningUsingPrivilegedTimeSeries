import numpy as np

from synthethic.system import EasyA, EasyBeta, ProtSystem, Distribution

def create_default_system(dim = 25,
                        seq_length = 10, # Not necessaru if Stationary True
                        componentA = EasyA,
                        componentBeta = EasyBeta,
                        component_args = {},
                        spec_radius = 1.5,
                        input_variance = 5.0,
                        noise_variance = 1.0,
                        target_variance = 1.0,
                        A_scale = 0.2,
                        beta_scale = 0.2,
                        beta2_scale = None, # If not None, then we will break Markov assumption.
                        seed = None,
                        stationary = False) -> base.Data:
    """
    Generates a Markov-Gaussian-linear dynamical system with default configurements
    """

    input_param = {'dist' : np.random.normal, 'dist_params' : [0, np.sqrt(input_variance)]}
    noise_param = {'dist' : np.random.normal, 'dist_params' : [0, np.sqrt(noise_variance)]}
    target_noise_param = {'dist' : np.random.normal, 'dist_params' : [0, np.sqrt(target_variance)]}
    normal_dist = Distribution(input_param, 1) #scalar
    system_noise = Distribution(noise_param, 1)
    reg_noise = Distribution(target_noise_param, 1)
    
    aGen = componentA(dim, scale=A_scale, seed=seed, args=component_args)
    betaGen = componentBeta(dim, scale=beta_scale, seed=seed)

    beta = betaGen.gen()
    if beta2_scale:
        beta2 = beta2_scale*betaGen.gen()
    else:
        beta2 = None
    
    if stationary:
        if spec_radius:
            while True:
                A = aGen.gen()
                A_r = spectral_radius(A)
                A_n = np.linalg.norm(A)
                A = spectral_change(A, spec_radius)
                if np.abs((spectral_radius(A)/A_r) - (np.linalg.norm(A)/A_n)) < 0.01:
                    break
        else:
            A = aGen.gen()
    else:
        A = []
        if seq_length is None:
            raise ValueError('Need to fix seq length for non-stationary system')
        for unused_enumerator_raising_warning in range(seq_length-1):
            if spec_radius is not None:
                while True:
                    A_tmp = aGen.gen()
                    A_tmp_r = spectral_radius(A_tmp)
                    A_tmp_n = np.linalg.norm(A_tmp)
                    A_tmp = spectral_change(A_tmp, spec_radius)

                    if np.abs((spectral_radius(A_tmp)/A_tmp_r) - (np.linalg.norm(A_tmp)/A_tmp_n)) < 0.01:
                        A += [A_tmp]
                        break
            else:
                A += [aGen.gen()]

    dynSystem = ProtSystem(normal_dist, system_noise, reg_noise, {'A' : A, 'beta': beta, 'beta_2': beta2}, break_markov=(beta2 is not None) )

    return dynSystem

def spectral_radius(A : np.array) -> float:
    """ 
    Returns spectral radius of matrix A
    """
    return max(np.real(np.linalg.eigvals(A)))

def spectral_change(A : np.array, r : float) -> np.array:
    """
    Scale spectral radius to r
    """
    w, v = np.linalg.eig(A)
    scale = r/max(np.real(w))
    e = np.zeros((A.shape[0], A.shape[0]), dtype='complex128')
    w = w*scale
    np.fill_diagonal(e, w)
    return np.real(np.matmul(np.matmul(v, e), np.linalg.inv(v)))
