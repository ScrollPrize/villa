import torch


'''
Radial basis functions:
'''
eps = 1e-7

def linear(r):
    return -r

def thin_plate_spline(r):
    r = torch.clamp(r, min=eps)
    return r**2 * torch.log(r)

def cubic(r):
    return r**3

def quintic(r):
    return -r**5

def multiquadric(r):
    return -torch.sqrt(r**2 + 1)

def inverse_multiquadric(r):
    return 1/torch.sqrt(r**2 + 1)

def inverse_quadratic(r):
    return 1/(r**2 + 1)

def gaussian(r):
    return torch.exp(-r**2)

RADIAL_FUNCS = {
   "linear": linear,
   "thin_plate_spline": thin_plate_spline,
   "cubic": cubic,
   "quintic": quintic,
   "multiquadric": multiquadric,
   "inverse_multiquadric": inverse_multiquadric,
   "inverse_quadratic": inverse_quadratic,
   "gaussian": gaussian
   }

SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}

MIN_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2
    }