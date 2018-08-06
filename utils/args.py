from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Image Autoencoder Experiment')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sparse', dest='sparse', action='store_true')
    parser.add_argument('--vae', dest='vae', action='store_true')
    parser.add_argument('--d_hidden', type=int, default=8 * 3 * 3)
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--beta', type=int, default=3)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume_path', type=str, default='./checkpoints/model_checkpoint.pth.tar')
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--resume_snapshot', dest='resume_snapshot', action='store_true')
    parser.set_defaults(resume_snapshot=False)
    args = parser.parse_args()
    return args
