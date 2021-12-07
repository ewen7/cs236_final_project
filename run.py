import argparse
import random
from main import main
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset_type', type=str, default='MNIST', help='dataset to utilize')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--start', type=int, default=200, help='start size')
    parser.add_argument('--iter-max', type=int, default=20000, metavar='IN',
                        help='number of VAE iterations to train (default: 14)')
    parser.add_argument('--random', action='store_true', default=False,
                        help='generate random queries from generative model')
    parser.add_argument('--sa', type=int, default=100, metavar='IN',
                        help='number sa iters')
    parser.add_argument('--all_epochs', type=int, default=2, metavar='AN',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--query-size', type=int, default=50,
                        help='query size')
    parser.add_argument('--mc-samp', type=int, default=10,
                        help='MC samples to use')
    parser.add_argument('--horizon', type=int, default=10,
                        help='horizon')
    parser.add_argument('-z', type=float, default=3, metavar='Z',
                        help='latent dimension')
    parser.add_argument('--device', type=str, default='cpu', metavar='DEV',
                        help='device')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--trials', type=int, default=1, metavar='T',
                        help='number of trials')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='seed to use')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='run baseline')
    args = parser.parse_args()

    random.seed(args.seed)
    seeds = [random.randrange(1_000_000_000) for _ in range(args.trials)]

    results = np.array([main(args, seed) for seed in seeds])
    print(results)

    print(results.mean(0))
    print(results.std(0) / np.sqrt(args.trials))
    
