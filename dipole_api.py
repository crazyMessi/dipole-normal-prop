import torch
from options import get_parser
import options
import util
from util import orient_center
from inference_utils import load_model_from_file, fix_n_filter, voting_policy
import field_utils
torch.manual_seed(1)
from field_utils import *

estimate_normals = True


def orient_large(opts):
    export_path: Path = opts.export_dir
    export_path.mkdir(exist_ok=True)

    max_patch_size = 500
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    MyTimer = util.timer_factory()
    with MyTimer('load pc', count=False):
        input_pc = util.xyz2tensor(open(opts.pc, 'r').read(), append_normals=False).to(device)
    print(input_pc.shape)

    input_pc, transform = util.Transform.trans(input_pc)

    if opts.estimate_normals:
        with MyTimer('estimating normals'):
            input_pc = util.estimate_normals(input_pc, max_nn=opts.n)

    softmax = torch.nn.Softmax(dim=-1)


    n_models = len(opts.models)
    models = [load_model_from_file(opts.models[i], device) for i in range(n_models)]

    with MyTimer('divide patches'):
        patch_indices = util.divide_pc(input_pc[:, :3], opts.number_parts,
                                       min_patch=opts.minimum_points_per_patch)
        all_patches_indices = [x.clone() for x in patch_indices]

    with MyTimer('filter patches'):
        patch_indices = fix_n_filter(input_pc, patch_indices, opts.curvature_threshold)

    num_patches = len(patch_indices)
    num_all_patches = len(all_patches_indices)
    print(f'number of patches {num_patches}/{num_all_patches}')

    with MyTimer('orient center'):
        for i, p in patch_indices:
            input_pc[p] = orient_center(input_pc[p])

    with MyTimer('find reps'):
        represent = []
        for p in all_patches_indices:
            permutation = torch.randperm(p.shape[0])
            represent.append((p[permutation[:max_patch_size]], p[permutation[max_patch_size:]]))

    pc_probs = torch.ones_like(input_pc[:, 0])

    with MyTimer('network orientation'):
        for i, _ in patch_indices:
            with torch.no_grad():
                current_reps, non_reps_points = represent[i]
                data = input_pc[current_reps]
                data = data.to(device)
                for _ in range(opts.iters):
                    votes = [model(data.clone()) for model in models]
                    vote_probabilities = [softmax(scores)[:, 1] for scores in votes]
                    flip, probs = voting_policy(vote_probabilities)
                    pc_probs[current_reps] = probs
                    input_pc[current_reps[flip], 3:] *= -1


    [model.to('cpu') for model in models]
    with MyTimer('propagating field'):
        strongest_field_propagation_reps(input_pc, represent, diffuse=True)

    with MyTimer('fix global orientation'):
        if field_utils.measure_mean_potential(input_pc) < 0:
            # if average global potential is negative, flip all normals
            input_pc[:, 3:] *= -1

    with MyTimer('exporting result', count=False):
        util.export_pc(transform.inverse(input_pc).transpose(0, 1), export_path / f'final_result.xyz')

    MyTimer.print_total_time()


import argparse
from pathlib import Path


base_path = "D:/Documents/zhudoongli/CG/project/NormalEstimation/dipole-normal-prop"
input_pc = Path(base_path + "/data/scene37_0.001_gt68_2.xyz")
models = [Path(base_path + "/pre_trained/hands2.pt"), Path(base_path + "/pre_trained/hands.pt"), Path(base_path + "/pre_trained/manmade.pt")]
export_dir = Path(base_path + "/data/output/")



def get_parser():

    #python -u D:\Documents\zhudoongli\CG\project\NormalEstimation\dipole-normal-prop/orient_large.py 
    # --pc D:\Documents\zhudoongli\CG\project\NormalEstimation\dipole-normal-prop/data/ok.xyz 
    # --export_dir D:\Documents\zhudoongli\CG\project\NormalEstimation\dipole-normal-prop/demos/ok 
    # --models D:\Documents\zhudoongli\CG\project\NormalEstimation\dipole-normal-prop/pre_trained/hands2.pt 
    #          D:\Documents\zhudoongli\CG\project\NormalEstimation\dipole-normal-prop/pre_trained/hands.pt 
    #          D:\Documents\zhudoongli\CG\project\NormalEstimation\dipole-normal-prop/pre_trained\manmade.pt 
    # --iters 5 
    # --propagation_iters 4 
    # --number_parts 41 
    # --minimum_points_per_patch 100 
    # --diffuse 
    # --weighted_prop 
    # --n 50

    parser = argparse.ArgumentParser()
    parser.add_argument('--pc', type=Path, default=input_pc, help='path to the point cloud')
    parser.add_argument('--models', type=Path, nargs='+', default=models,
                        help='path to the models')    
    parser.add_argument('--export_dir', type=Path, default=export_dir, help='path to the output directory')
    parser.add_argument('--number_parts', type=int, default=10, help='number of parts to divide the point cloud')
    parser.add_argument('--minimum_points_per_patch', type=int, default=100, help='minimum number of points per patch')
    parser.add_argument('--curvature_threshold', type=float, default=0.0, help='curvature threshold')
    parser.add_argument('--iters', type=int, default=5, help='number of iterations')
    parser.add_argument('--diffuse', default=True, action='store_true', help='diffuse')
    parser.add_argument('--weighted_prop', default=True, action='store_true', help='weighted propagation')
    parser.add_argument('--estimate_normals', default=True, action='store_true', help='estimate normals')
    parser.add_argument('--n', type=int, default=30, help='number of nearest neighbors for normal estimation')

    
    return parser

if __name__ == '__main__':
    opts = get_parser().parse_args()
    orient_large(opts)