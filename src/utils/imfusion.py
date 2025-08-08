import numpy as np
from src.utils.utils import imfusion


import h5py
import imfusion


def h5_to_imfusion_sweep(input_file, output_file):
    with h5py.File(input_file, "r") as F:
        images = F["images"][:]
        tracking = F["tracking"][:]
        spacing = F["spacing"][:]
        dimensions = F["dimensions"][:]

    descriptor = imfusion.ImageDescriptor()
    descriptor.set_dimensions(dimensions)
    descriptor.set_spacing(spacing, True)

    N, H, W = images.shape
    sweep = imfusion.UltrasoundSweep()

    sweep.set_timestamp(False)
    ts = imfusion.TrackingSequence()

    for i in range(N):
        sweep_i = imfusion.SharedImage(descriptor)
        sweep_i.assign_array(images[i][:, :, None])
        sweep_i.descriptor.set_spacing(spacing, True)
        sweep.add(sweep_i)
        ts.add(tracking[i])

    sweep.add_tracking(ts)

    print(sweep.descriptor())
    print(spacing)
    sweep.descriptor().set_spacing(spacing, True)
    print(sweep.descriptor())
    sweep.descriptor().set_dimensions(dimensions)

    imfusion.save([sweep], output_file)


def imfusion_sweep_to_h5(input_file, output_file) -> bool:
    try:
        (sweep,) = imfusion.load(input_file)
    except Exception as e:
        return False

    with h5py.File(output_file, "w") as F:
        N, _, H, W, _ = sweep.shape
        images = np.zeros((N, H, W), dtype=np.uint8)
        for i in range(N):
            images[i] = np.array(sweep[i])[:, :, 0]
        F.create_dataset("images", data=images)
        tracking = np.zeros((N, 4, 4), dtype=np.float32)
        for i in range(N):
            tracking[i] = sweep.matrix(i)
        F.create_dataset("tracking", data=tracking)
        F.create_dataset("spacing", data=sweep.descriptor().spacing)
        F.create_dataset("dimensions", data=sweep.descriptor().dimensions)
        F.create_dataset("pixel_to_image", data=sweep.get().pixel_to_world_matrix)

    return True