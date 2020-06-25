import sys
import os
import argparse

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_type", type=str, default='atomic',
                    choices=["atomic", "conceptnet"])
parser.add_argument("--experiment_num", type=str, default="0")
parser.add_argument("--pickled_data", type=str, default=None)
parser.add_argument("--save_path", type=str, default=None, help='save path override')
parser.add_argument("--train_comet_loss", action='store_true', default=False, help='train with comet loss')
parser.add_argument("--eval_comet_loss", action='store_true', default=False, help='eval with comet loss')

args = parser.parse_args()

if args.experiment_type == "atomic":
    from main_atomic import main
    main(args.experiment_num, args)
if args.experiment_type == "conceptnet":
    from main_conceptnet import main
    main(args.experiment_num)
