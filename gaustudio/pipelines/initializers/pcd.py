import numpy as np
import torch
import trimesh
import os
import open3d as o3d
from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.base import BaseInitializer

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

def normal2rotation(n):
    n = torch.nn.functional.normalize(n)
    w0 = torch.tensor([[1, 0, 0]]).expand(n.shape)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 *= torch.sign(R0[:, :1])
    R0 = torch.nn.functional.normalize(R0)
    R1 = torch.cross(n, R0)

    R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    R = torch.stack([R0, R1, n], -1)
    q = rotmat2quaternion(R)

    return q

def rotmat2quaternion(R, normalize=False):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    q = torch.stack([
        r,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * r)
    ], -1)
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q

@initializers.register('pcd')
class PcdInitializer(BaseInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.model_path = initializer_config.get('model_path', None)

    def __call__(self, model, dataset=None, overwrite=False):
        if not os.path.exists(self.model_path) or overwrite:
            self.cache_dataset(dataset)
            self.process_dataset()
        model = self.build_model(model)
        return model
    
    def cache_dataset(self, dataset=None):
        pass
    
    def process_dataset(self):
        pass
    
    def build_model(self, model):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"PCD file not found: {self.model_path}")

        pcd = o3d.io.read_point_cloud(self.model_path)
        
        points = torch.tensor(np.asarray(pcd.points)).float()
        colors = torch.tensor(np.asarray(pcd.colors)).float() if len(pcd.colors) > 0 else None
        normals = torch.tensor(np.asarray(pcd.normals)).float() if len(pcd.normals) > 0 else None

        if normals is not None:
            rotations = normal2rotation(normals)
        else:
            rotations = None

        scales = None
        opacity = inverse_sigmoid(0.1 * np.ones((points.shape[0], 1)))
        
        model.create_from_attribute(xyz=points, rgb=colors, scale=scales, opacity=opacity, rot=rotations)
        return model