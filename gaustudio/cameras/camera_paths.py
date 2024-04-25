import json
import copy 
import torch
from gaustudio.utils.cameras_utils import JSON_to_camera
from gaustudio.utils.pose_utils import get_interpolated_poses, quaternion_from_matrix, quaternion_matrix

def get_path_from_json(json_path):
    print("Loading camera data from {}".format(json_path))
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    cameras = []
    for camera_json in camera_data:
        camera = JSON_to_camera(camera_json, "cuda")
        cameras.append(camera)
    return cameras

def upsample_cameras(cameras, steps_per_transition=30):
    new_cameras = []
    total_idx = 0
    for idx in range(len(cameras)-1):
        current_camera = cameras[idx]
        next_camera = cameras[idx+1]
        
        intermediate_cameras = get_interpolated_poses(current_camera.extrinsics.cpu().numpy(), 
                                          next_camera.extrinsics.cpu().numpy(), steps=steps_per_transition)
        for intermediate_camera in intermediate_cameras:
            view_new = copy.deepcopy(current_camera)
            view_new.extrinsics = intermediate_camera
            view_new.image_name = str(total_idx).zfill(8)
            new_cameras.append(view_new)
            total_idx+=1
    return new_cameras

from scipy.signal import savgol_filter
import numpy as np
import numpy.linalg as la
def unwrap_quaternions(qvecs):
    qvecs_unwrapped = np.zeros_like(qvecs)
    qvecs_unwrapped[0] = qvecs[0]
    for i in range(1, qvecs.shape[0]):
        dot = np.clip(np.sum(qvecs_unwrapped[i-1] * qvecs[i]), -1.0, 1.0)
        qvecs_unwrapped[i] = (qvecs[i] if dot > 0 else -qvecs[i])
    return qvecs_unwrapped

def smoothen_cameras(cameras, window_length=9, polyorder=2):
    new_cameras = []
    total_idx = 0
    translates = torch.stack([camera.extrinsics[:3, 3] for camera in cameras]).cpu().numpy()
    qvecs = np.stack([quaternion_from_matrix(camera.extrinsics[:3, :3].cpu().numpy()) for camera in cameras])
    qvecs = unwrap_quaternions(qvecs)
    for dim in range(translates.shape[1]):
        translates[:, dim] = savgol_filter(translates[:, dim], window_length, polyorder)
    
    for dim in range(qvecs.shape[1]):
        qvecs[:, dim] = savgol_filter(qvecs[:, dim], window_length, polyorder)

    for camera, smooth_translate, smooth_qvec in zip(cameras, translates, qvecs):
        smooth_qvec /= la.norm(smooth_qvec)  # Normalize quaternion
        camera_new = copy.deepcopy(camera)
        updated_extrinsics = quaternion_matrix(smooth_qvec)
        updated_extrinsics[:3, 3] = smooth_translate
        camera_new.extrinsics = updated_extrinsics
        new_cameras.append(camera_new)
        total_idx += 1

    return new_cameras