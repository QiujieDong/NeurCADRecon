import os
import sys

import tqdm
import open3d as o3d
import trimesh

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.utils_mp import start_process_pool
import utils.utils as utils

def data_no_filter(mesh_path, input_path, sample_pt_num=10000):
    print(mesh_path)

    try:
        mesh = trimesh.load_mesh(mesh_path, process=False, force='mesh')
        mesh = utils.normalize_mesh_export(mesh)
        pts, idx = trimesh.sample.sample_surface(mesh, count=sample_pt_num)
        normals = mesh.face_normals[idx]
        pts_o3d = o3d.geometry.PointCloud()
        pts_o3d.points = o3d.utility.Vector3dVector(pts)
        pts_o3d.normals = o3d.utility.Vector3dVector(normals)

        f_name = os.path.splitext(mesh_path.split('/')[-1])[0]
        o3d.io.write_point_cloud(os.path.join(input_path, f_name + '.ply'), pts_o3d)

        return
    except Exception as e:
        print(e)
        print('error', mesh_path)


if __name__ == "__main__":
    gt_path = '/home/qiujie/NeurCADRecon_public/data/fandisk/gt'
    os.makedirs(gt_path, exist_ok=True)

    input_path = '/home/qiujie/NeurCADRecon_public/data/fandisk/input'
    os.makedirs(input_path, exist_ok=True)

    num_processes = 16
    sample_pt_num = 30000

    call_params = list()
    for f in tqdm.tqdm(sorted(os.listdir(gt_path))):
        if os.path.splitext(f)[1] not in ['.ply', '.obj', '.off']:
            continue

        mesh_path = os.path.join(gt_path, f)
        call_params.append((mesh_path, input_path, sample_pt_num))

    start_process_pool(data_no_filter, call_params, num_processes)
