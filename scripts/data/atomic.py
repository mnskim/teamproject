import argparse
import pprint
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

import networkx as nx
import numpy as np

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


opt = DD()
opt.dataset = "atomic"
opt.exp = "generation"
opt.data = DD()
opt.data.categories = sorted(categories)

encoder_path = "model/encoder_bpe_40000.json"
bpe_path = "model/vocab_40000.bpe"

text_encoder = TextEncoder(encoder_path, bpe_path)

encoder = text_encoder.encoder
n_vocab = len(text_encoder.encoder)

special = [data.start_token, data.end_token]
special += ["<{}>".format(cat) for cat in categories]
special += [data.blank_token]

for special_token in special:
    text_encoder.decoder[len(encoder)] = special_token
    encoder[special_token] = len(encoder)

save_path = "data/atomic/processed/{}".format(opt.exp)
utils.mkpath(save_path)

save_name = os.path.join(
    save_path, "{}.pickle".format(utils.make_name_string(opt.data)))

data_loader = data.make_data_loader(opt, categories)
data_loader.load_data("data/atomic/")
random.shuffle(data_loader.data["dev"]["total"])

train_triples = data_loader.data['train']['total']

G=nx.Graph()

entities = set()
for triple in train_triples:
    #ipdb.set_trace()
    m1, rel, m2 = triple
    entities.add(m1)
    entities.add(m2)
    G.add_node(m1)
    G.add_node(m2)
    G.add_edge(m1, m2, rel=rel)

"""
Functions
"""

class Path(object):
    def __init__(self, start_node):
        self.nodes = set()
        self.nodes.add(start_node)
        #self.rels = 
        self.walk = [start_node]

    def update(self, node, rel):
        if not node in self.nodes:
            self.walk.append(rel)
            self.walk.append(node)
            self.nodes.add(node)

            return 1

        else:
            return 0


def single_step(start_node, G):
    # Start from a node and randomly sample a path
    edges = list(G[start_node].items())
    rand_id = np.random.randint(len(edges))
    edge = edges[rand_id]
    obj = edge[0]
    relation = edge[1]['rel']

    return obj, relation


"""
Generate
"""
def main(args):
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    output_f = os.path.join(args.output, 'train.txt')
    output = open(output_f, 'w')
    print("\n###### Generating ATOMIC data ######\n")
    pprint.pprint(vars(args))
    print("\nSaving outputs to: {}".format(output_f) + '\n')

    #n_per_node = 3
    examples = []
    all_nodes = list(G.nodes())
    random.shuffle(all_nodes)
    for node in all_nodes:
        unique_paths = set() # Use for filtering out duplicate paths starting from the same start_node

        for _ in range(args.n_per_node):
            curr_node = node
            path = Path(curr_node) 

            n_attempts = 0
            while len(path.walk) * 2 < args.max_path_len:
                obj, relation = single_step(curr_node, G)
                updated = path.update(obj, relation)
                if updated:
                    curr_node = obj
                else:
                    #ipdb.set_trace()
                    n_attempts += 1
                #print(path.walk)

                if n_attempts > 10 :
                    break

            if not ' '.join(path.walk) in unique_paths:
                examples.append(path.walk)
                unique_paths.add(' '.join(path.walk))

            if len(examples) % 500 == 0:
                print("\nGenerated {} examples".format(len(examples)))
                print(path.walk)

        if len(examples) >= args.n_train:
            break

    examples = examples[:args.n_train]
    for ex in examples:
        output.write(' {} '.format(args.separator).join(ex) + '\n')
    output.close()
    ipdb.set_trace()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data/atomic/processed/lm')
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--separator", type=str, default='[MASK]')
    parser.add_argument("--max_path_len", type=int, default=25)
    parser.add_argument("--n_per_node", type=int, default=3)

    args = parser.parse_args()

    #ipdb.set_trace()
    main(args)

