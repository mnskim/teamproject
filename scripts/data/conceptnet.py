import argparse
import pprint
import numpy as np
import os
import sys
import ipdb

sys.path.append(os.getcwd())

import networkx as nx

import torch
import src.data.conceptnet as cdata
import src.data.data as data

from utils.utils import DD
import utils.utils as utils
import random
from src.data.utils import TextEncoder
from tqdm import tqdm


train_triples = open('./data/conceptnet/train100k.txt').readlines()

G=nx.Graph()

entities = set()
for triple in train_triples:
    rel, m1, m2, weight = triple.rstrip('\n').split('\t')
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
    print("\n###### Generating ConceptNet data ######\n")
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
    parser.add_argument("--output", type=str, default='data/conceptnet/processed/lm')
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--separator", type=str, default='[MASK]')
    parser.add_argument("--max_path_len", type=int, default=25)
    parser.add_argument("--n_per_node", type=int, default=3)

    args = parser.parse_args()

    #ipdb.set_trace()
    main(args)

