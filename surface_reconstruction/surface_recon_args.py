import argparse


def add_args(parser):
    parser.add_argument('--logdir', type=str, default='./log_fandisk', help='log directory')
    parser.add_argument('--model_name', type=str, default='model', help='trained model name')
    parser.add_argument('--seed', type=int, default=3627473, help='random seed')
    parser.add_argument('--data_path', type=str,
                            default='../data/fandisk/input/fandisk.ply',
                        help='path to input dir')
    parser.add_argument('--mesh_dir', type=str, default='../data/fandisk/gt', help='path to the gt folder')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='numbers of epochs')
    parser.add_argument('--n_points', type=int, default=20000, help='number of points in each point cloud')
    parser.add_argument('--grid_res', type=int, default=256, help='uniform grid resolution')
    parser.add_argument('--nonmnfld_sample_type', type=str, default='gaussian',
                        help='how to sample points off the manifold - grid | gaussian | combined')

    # training parameters
    parser.add_argument('--num_epochs', type=int, default=1, help='always be 1')
    parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
    parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Value to clip gradients to')
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples in a minibatch')
    parser.add_argument('--load_path', type=str, default=None)

    # Network architecture and loss
    parser.add_argument('--init_type', type=str, default='siren',
                        help='initialization type siren | geometric_sine | geometric_relu | mfgi')
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='length of decoder hidden dim')
    parser.add_argument('--decoder_n_hidden_layers', type=int, default=4, help='number of decoder hidden layers')
    parser.add_argument('--latent_size', type=int, default=0)
    parser.add_argument('--nl', type=str, default='sine', help='type of non linearity sine | relu')
    parser.add_argument('--sphere_init_params', nargs='+', type=float, default=[1.6, 0.1],
                        help='radius and scaling')
    parser.add_argument('--udf', action='store_true')
    parser.add_argument('--output_any', action='store_true')

    parser.add_argument('--loss_type', type=str, default='siren_wo_n_w_morse')
    parser.add_argument('--decay_params', nargs='+', type=float, default=[10, 0.2, 10, 0.5, 0.001, 0],
                        help='epoch number to evaluate')
    parser.add_argument('--morse_type', type=str, default='l1', help='divergence term norm l1 | l2')
    parser.add_argument('--morse_decay', type=str, default='linear',
                        help='divergence term importance decay none | step | linear')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[7e3, 6e2, 1e2, 5e1, 0, 10],
                        help='loss terms weights sdf | inter | normal | eikonal | div | morse')
    parser.add_argument('--bidirectional_morse', action='store_true',
                        help='if true, add morse constraints to both input point and sampling point')
    parser.add_argument('--morse_near', default=True)
    parser.add_argument('--weight_for_morse', action='store_true',
                        help='if true, Weighting A according to the distance of the sampling point')
    parser.add_argument('--use_morse_nonmnfld_grad', type=bool, default=False,
                        help='if True, use morse loss on nonmnfld')
    parser.add_argument('--use_relax_eikonal', type=bool, default=False,
                        help='if True, use relax eikonal loss on nonmnfld')
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    return args
