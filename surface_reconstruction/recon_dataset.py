import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import open3d as o3d
import trimesh


class ReconDataset(data.Dataset):
    # A class to generate synthetic examples of basic shapes.
    # Generates clean and noisy point clouds sampled  + samples on a grid with their distance to the surface (not used in DiGS paper)
    def __init__(self, file_path, n_points, n_samples=128, res=128, sample_type='grid', grid_range=1.1):
        self.file_path = file_path
        self.n_points = n_points
        self.n_samples = n_samples
        # load data
        self.o3d_point_cloud = o3d.io.read_point_cloud(self.file_path)
        self.grid_range = grid_range

        # extract center and scale points and normals
        self.points, self.mnfld_n = self.get_mnfld_points()
        self.bbox = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)]).transpose()
        self.bbox_trimesh = trimesh.PointCloud(self.points).bounding_box.copy()

        self.point_idxs = np.arange(self.points.shape[0], dtype=np.int32)
        # record sigma
        self.sample_gaussian_noise_around_shape()

    def get_mnfld_points(self):
        # Returns points on the manifold
        points = np.asarray(self.o3d_point_cloud.points, dtype=np.float32)
        normals = np.asarray(self.o3d_point_cloud.normals, dtype=np.float32)
        if normals.shape[0] == 0:
            normals = np.zeros_like(points)
        # center and scale data/point cloud
        self.cp = points.mean(axis=0)
        points = points - self.cp[None, :]
        self.scale = np.abs(points).max()
        points = points / self.scale

        return points, normals

    def sample_gaussian_noise_around_shape(self):
        kd_tree = spatial.KDTree(self.points)
        # query each point for sigma
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]
        self.sigmas = sigmas
        return

    def __getitem__(self, index):
        manifold_idxes_permutation = np.random.permutation(self.points.shape[0])
        mnfld_idx = manifold_idxes_permutation[:self.n_points]
        manifold_points = self.points[mnfld_idx]  # (n_points, 3)
        manifold_normals = self.mnfld_n[mnfld_idx]  # (n_points, 3)

        nonmnfld_points = np.random.uniform(-self.grid_range, self.grid_range,
                                            size=(self.n_points, 3)).astype(np.float32)  # (n_points // 2, 3)

        near_points = (manifold_points + self.sigmas[mnfld_idx] * np.random.randn(manifold_points.shape[0],
                                                                                  manifold_points.shape[1])).astype(
            np.float32)
        return {'points': manifold_points, 'mnfld_n': manifold_normals, 'nonmnfld_points': nonmnfld_points,
                'near_points': near_points}

    def get_train_data(self, batch_size):
        manifold_idxes_permutation = np.random.permutation(self.points.shape[0])
        mnfld_idx = manifold_idxes_permutation[:batch_size]
        manifold_points = self.points[mnfld_idx]  # (n_points, 3)
        near_points = (manifold_points + self.sigmas[mnfld_idx] * np.random.randn(manifold_points.shape[0],
                                                                                  manifold_points.shape[1])).astype(
            np.float32)

        return manifold_points, near_points, self.points

    def gen_new_data(self, dense_pts):
        self.points = dense_pts
        kd_tree = spatial.KDTree(self.points)
        # query each point for sigma^2
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]
        self.sigmas = sigmas

    def __len__(self):
        return self.n_samples
