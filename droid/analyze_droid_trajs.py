import h5py
import numpy as np

np.set_printoptions(suppress=True)


path = "/data/droid_raw/1.0.1/AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023/trajectory.h5"

with h5py.File(path, "r") as f:
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    f.visititems(visit)

with h5py.File(path, "r") as f:
    c = f["observation/robot_state/cartesian_position"][-100:]
    print("cartesian_position", c.shape, c.dtype, np.round(c, 3))

    g = f["observation/robot_state/gripper_position"][-100:]
    print("gripper_position", g.shape, g.dtype, np.round(g, 3))
