import numpy as np
import awkward as ak

def norm_exponential(scale, N, alpha=0.5):
    x = np.random.exponential(scale, N) ** alpha
    return x / np.max(x)
    
def ak_rand_like(f_rand, array : ak.Array):
    if array.ndim == 1:
        return f_rand(len(array))
    
    elif array.ndim == 2:
        num = ak.num(array)
        rand_array = f_rand(ak.sum(num))
        return ak.unflatten(rand_array, num)
    
def random_vector_like(array : ak.Array):
    # randomize the decay angles in the rest frame
    uniform = lambda n : np.random.uniform(0, 1, n)
    
    u = ak_rand_like(uniform, array)
    v = ak_rand_like(uniform, array)
    
    phi = 2 * np.pi * u
    theta = np.arccos(2 * v - 1)
    
    rvec = ak.zip(dict(
        rho = ak.ones_like(u),
        theta = theta,
        phi = phi,
    ), with_name='Vector3D')
    rvec = rvec / (rvec.x**2 + rvec.y**2 + rvec.z**2)**0.5
    
    return rvec
