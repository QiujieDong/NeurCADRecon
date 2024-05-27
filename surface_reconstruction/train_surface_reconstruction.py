import os
import sys
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import trimesh
import torch
import torch.optim as optim
from torchinfo import summary

from models import Network, MorseLoss
import utils.utils as utils
import utils.visualizations as vis
import surface_recon_args
import recon_dataset as dataset

args = surface_recon_args.get_args()

file_name = os.path.splitext(args.data_path.split('/')[-1])[0]
gt_path = glob.glob(os.path.join(args.mesh_dir, os.path.splitext(args.data_path.split('/')[-1])[0] + '*'))
logdir = os.path.join(args.logdir, file_name)
os.makedirs(logdir, exist_ok=True)

# set up logging
log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(logdir, args)
os.system('cp %s %s' % (__file__, logdir))  # backup the current training file
os.system('cp %s %s' % ('recon_dataset.py', logdir))  # backup the current training file
os.system('cp %s %s' % ('../models/DiGS.py', logdir))  # backup the models files
os.system('cp %s %s' % ('../models/losses.py', logdir))  # backup the losses files

device = 'cpu' if not torch.cuda.is_available() else 'cuda'

# get data loaders
utils.same_seed(args.seed)
train_set = dataset.ReconDataset(args.data_path, args.n_points, args.n_samples, args.grid_res,
                                 args.nonmnfld_sample_type)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
# get model
net = Network(in_dim=3, decoder_hidden_dim=args.decoder_hidden_dim, nl=args.nl,
              decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type,
              sphere_init_params=args.sphere_init_params, udf=args.udf)
net.to(device)
if args.load_path is not None:
    net.load_state_dict(torch.load(args.load_path))
    print('Loaded model from %s' % args.load_path)
summary(net.decoder, (1, 1024, 3))

n_parameters = utils.count_parameters(net)
utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)

# Setup Adam optimizers
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
n_iterations = args.n_samples * (args.num_epochs)
print('n_iterations: ', n_iterations)

net.to(device)

criterion = MorseLoss(weights=args.loss_weights, loss_type=args.loss_type, div_decay=args.morse_decay,
                      div_type=args.morse_type, bidirectional_morse=args.bidirectional_morse, udf=args.udf)

num_batches = len(train_dataloader)
refine_flag = True
min_cd = np.inf
max_f1 = -np.inf
# For each epoch
for epoch in range(args.num_epochs):
    # For each batch in the dataloader
    for batch_idx, data in enumerate(train_dataloader):
        if batch_idx != 0 and (batch_idx % 500 == 0 or batch_idx == len(train_dataloader) - 1):
            output_dir = os.path.join(logdir, 'vis')
            os.makedirs(output_dir, exist_ok=True)
            vis.plot_cuts_iso(net.decoder, save_path=os.path.join(output_dir, str(batch_idx) + '.html'))
            try:
                shapename = file_name
                output_dir = os.path.join(logdir, 'result_meshes')
                os.makedirs(output_dir, exist_ok=True)
                cp, scale, bbox = train_set.cp, train_set.scale, train_set.bbox
                mesh_dict = None
                mesh = utils.implicit2mesh(net.decoder, None,
                                           args.grid_res,
                                           translate=-cp,
                                           scale=1 / scale,
                                           get_mesh=True, device=device, bbox=bbox)

                pred_mesh = mesh.copy()
                cd_l1 = np.inf
                f1_mu = np.inf
                if len(gt_path) != 0:
                    print('gt_path:', gt_path[0])
                    gt = trimesh.load(gt_path[0], process=False)
                    if isinstance(gt, trimesh.Scene):
                        gt = trimesh.load(gt_path[0], process=False, force='mesh')
                    nc_ind, cd_l1, cd_l2, f1_mu, euler_num = utils.eval_reconstruct_gt(pred_mesh, gt)
                    print(
                        'iter: {}, cd_l1: {:.7f}, cd_l2: {:.7f}, f1_mu: {:.7f}, euler_num: {:.7f}'.format(batch_idx,
                                                                                                          cd_l1, cd_l2,
                                                                                                          f1_mu,
                                                                                                          euler_num),
                        log_file)
                    utils.log_string(
                        'iter: {}, cd_l1: {:.7f}, cd_l2: {:.7f}, f1_mu: {:.7f} , euler_num: {:.7f}'.format(batch_idx,
                                                                                                           cd_l1, cd_l2,
                                                                                                           f1_mu,
                                                                                                           euler_num),
                        log_file)
                    if cd_l1 < min_cd:
                        min_cd = cd_l1
                        # save model
                        utils.log_string(
                            "saving model to file :{}".format(
                                '%s_generator_model_%d.pth' % (args.model_name, batch_idx)),
                            log_file)
                        torch.save(net.state_dict(),
                                   os.path.join(model_outdir,
                                                '%s_model_%d_%.5f.pth' % (args.model_name, batch_idx, min_cd)))
                        best_path = os.path.join(output_dir,
                                                 shapename + '_iter_{}_euler_{}_cdl1_{}_min_f1_{}.ply'.format(batch_idx,
                                                                                                              euler_num,
                                                                                                              str(min_cd),
                                                                                                              f1_mu))
                        mesh.export(best_path)

                if args.output_any:
                    output_ply_filepath = os.path.join(output_dir,
                                                       shapename + '_iter_{}_euler_{}_cdl1_{}_f1_{}.ply'.format(
                                                           batch_idx, euler_num, str(cd_l1), f1_mu))
                    print('Saving to ', output_ply_filepath)
                    mesh.export(output_ply_filepath)
            except Exception as e:
                print(e)
                print('Could not generate mesh\n')

        net.zero_grad()
        net.train()

        mnfld_points, mnfld_n_gt, nonmnfld_points, near_points = data['points'].to(device), data['mnfld_n'].to(device), \
            data['nonmnfld_points'].to(device), data['near_points'].to(device)

        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()
        near_points.requires_grad_()

        output_pred = net(nonmnfld_points, mnfld_points, near_points=near_points if args.morse_near else None)
        loss_dict, _ = criterion(output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt,
                                 near_points=near_points if args.morse_near else None)
        lr = torch.tensor(optimizer.param_groups[0]['lr'])
        loss_dict["lr"] = lr
        utils.log_losses(log_writer_train, epoch, batch_idx, num_batches, loss_dict, args.batch_size)

        loss_dict["loss"].backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)

        optimizer.step()

        # Output training stats
        if batch_idx % 10 == 0:
            weights = criterion.weights
            utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                             'L_NonMnfld: {:.5f} + L_Nrml: {:.5f} + L_Eknl: {:.5f} + L_Div: {:.5f} + L_Morse: {:.5f}'.format(
                epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / len(train_dataloader),
                loss_dict["loss"].item(), weights[0] * loss_dict["sdf_term"].item(),
                       weights[1] * loss_dict["inter_term"].item(),
                       weights[2] * loss_dict["normals_loss"].item(), weights[3] * loss_dict["eikonal_term"].item(),
                       weights[4] * loss_dict["div_loss"].item(), weights[5] * loss_dict['morse_term'].item(),
            ),
                log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Unweighted L_s : L_Mnfld: {:.5f},  '
                             'L_NonMnfld: {:.5f},  L_Nrml: {:.5f},  L_Eknl: {:.5f}, L_Morse: {:.5f}'.format(
                epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / len(train_dataloader),
                loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                loss_dict["normals_loss"].item(), loss_dict["eikonal_term"].item(),
                loss_dict['morse_term'].item()),
                log_file)
            utils.log_string('', log_file)

        criterion.update_morse_weight(epoch * args.n_samples + batch_idx, args.num_epochs * args.n_samples,
                                      args.decay_params)  # assumes batch size of 1
