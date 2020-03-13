
import argparse

parser = argparse.ArgumentParser(description='LR Range test by S. Gugger')
parser.add_argument('-wd','--weight-decay', help='weight decay (default: 0)')
args = parser.parse_args()
print(args)
print(args.weight_decay)
print(type(args.weight_decay))