import argparse
import ipdb
import sys
import os

sys.path.append(os.getcwd())

import src.data.data as data
from utils.utils import DD
import utils.utils as utils
import random
from src.data.utils import TextEncoder
from tqdm import tqdm
import torch

# Manually change the set of categories you don't want to include
# if you want to be able to train on a separate set of categories
categories = []
categories += ["oEffect"]
categories += ["oReact"]
categories += ["oWant"]
categories += ["xAttr"]
categories += ["xEffect"]
categories += ["xIntent"]
categories += ["xNeed"]
categories += ["xReact"]
categories += ["xWant"]

parser = argparse.ArgumentParser()
parser.add_argument("--pickled_data", type=str)
parser.add_argument("--n_train", type=int, default=10000)
parser.add_argument("--n_dev", type=int, default=1000)
parser.add_argument("--n_test", type=int, default=1000)
parser.add_argument("--max_path_len", type=int, default=25)
parser.add_argument("--n_per_node", type=int, default=3)

args = parser.parse_args()

opt = DD()
opt.dataset = "atomic"
opt.exp = "generation"
opt.data = DD()
opt.data.categories = sorted(categories)

# Additionally added
opt.pickled_data = args.pickled_data
opt.n_train = args.n_train
opt.n_dev = args.n_dev
opt.n_test = args.n_test
opt.max_path_len = args.max_path_len
opt.n_per_node = args.n_per_node

encoder_path = "model/encoder_bpe_40000.json"
bpe_path = "model/vocab_40000.bpe"

text_encoder = TextEncoder(encoder_path, bpe_path)

encoder = text_encoder.encoder
n_vocab = len(text_encoder.encoder)

special = [data.start_token, data.end_token]#, data.sep_token]
special += ["<{}>".format(cat) for cat in categories]
special += [data.blank_token]

for special_token in special:
    text_encoder.decoder[len(encoder)] = special_token
    encoder[special_token] = len(encoder)

save_path = "data/atomic/processed/{}".format(opt.exp)
utils.mkpath(save_path)

#ipdb.set_trace()
if opt.pickled_data:
    save_name = opt.pickled_data
else:
    save_name = os.path.join(save_path, "{}.pickle".format(utils.make_name_string(opt.data)))

data_loader = data.make_data_loader(opt, categories)
data_loader.load_data("data/atomic/")
random.shuffle(data_loader.data["dev"]["total"])

data_loader.make_tensors(text_encoder, special, test=False)
data_loader.reset_offsets()


opt.data.maxe1 = data_loader.max_event
opt.data.maxe2 = data_loader.max_effect
opt.data.maxr = 1

if opt.pickled_data:
    save_name = opt.pickled_data
else:
    save_name = os.path.join(save_path, "{}.pickle".format(utils.make_name_string(opt.data)))

print("Data Loader will be saved at: {}".format(save_name))

torch.save(data_loader, save_name)
