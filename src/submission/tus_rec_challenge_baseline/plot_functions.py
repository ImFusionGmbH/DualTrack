
import numpy as np
import torch

def reference_image_points(image_size, density=2):
    """
    :param image_size: (x, y), used for defining default grid image_points
    :param density: (x, y), point sample density in each of x and y, default n=2
    """
    if isinstance(density,int):
        density=(density,density)

    image_points = torch.flip(torch.cartesian_prod(
        torch.linspace(1, image_size[0], density[0]),
        torch.linspace(1, image_size[1] , density[1])
    ).t(),[0])
    
    image_points = torch.cat([
        image_points, 
        torch.zeros(1,image_points.shape[1])*image_size[0]/2,
        torch.ones(1,image_points.shape[1])
        ], axis=0)
    
    return image_points

def data_pairs_global(num_frames):
    # obtain the data pairs to compute the transfomration between frames and the reference (first) frame
    
    return torch.tensor([[0,n0] for n0 in range(num_frames)])

def data_pairs_local(num_frames):
    # obtain the data pairs to compute the transfomration between frames and the reference (the immediate previous) frame
    
    return torch.tensor([[n0,n0+1] for n0 in range(num_frames)])

def read_calib_matrices(filename_calib):
    # read the calibration matrices from the csv file
    tform_calib = np.empty((8,4), np.float32)
    with open(filename_calib,'r') as csv_file:
        txt = [i.strip('\n').split(',') for i in csv_file.readlines()]
        tform_calib[0:4,:]=np.array(txt[1:5]).astype(np.float32)
        tform_calib[4:8,:]=np.array(txt[6:10]).astype(np.float32)
    return torch.tensor(tform_calib[0:4,:]),torch.tensor(tform_calib[4:8,:]), torch.tensor(tform_calib[4:8,:] @ tform_calib[0:4,:])
