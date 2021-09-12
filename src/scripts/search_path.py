import argparse
from itertools import islice

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        default='zr-sample',
        help='Select the target dataset.',
    )
    parser.add_argument(
        '--attfile',
        type=str,
        default='attention_kgat+_epoch020210830-111759.txt',
        help='Select the triplets & attention file.',
    )
    parser.add_argument(
        '--remapfile',
        type=str,
        # required=False, # TODO: False & not default
        default='remap_relationid.txt',
        help='Select the remap relationid text file.',
    )
    parser.add_argument(
        '--userid',
        type=int,
        required=True,
        help='source (recommended) userid.'
    )
    parser.add_argument(
        '--itemid',
        type=int,
        required=True,
        help='target (recommended) itemid.'
    )
    # parser.add_argument(
    #     '--max_depth',
    #     type=int,
    #     default=4,
    #     help='max size of depth from user node to item node.'
    # )
    # parser.add_argument(
    #     '--max_paths',
    #     type=int,
    #     default=30,
    #     help='max number of paths.'
    # )
    args = parser.parse_args()
    args.path_att = './src/outputs/{0}/result/{1}'.format(
        args.dataset, args.attfile)

    return args


def load_graph(path_att: str) -> nx.DiGraph:
    df = pd.read_csv(path_att, delimiter=' ')
    heads = list(df['head'])
    tails = list(df['tail'])
    weights = list(df['att'])
    relations = list(df['relation'])
    edges = [(heads[i], tails[i], {
              'weight': weights[i], 'type': relations[i]}) for i in range(len(df))]

    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def search_path(graph: nx.DiGraph, userid: int, itemid: int, max_depth: int = 4, max_paths: int = 30):
    count = 0
    paths, lens, weights = [], [], []
    for path in islice(nx.all_simple_paths(graph, source=userid, target=itemid, cutoff=max_depth), max_paths):
        count += 1
        str_path = str(path[0])
        len_path = len(path)
        weight = 1.0
        for i in range(len(path) - 1):
            r = graph[path[i]][path[i + 1]]['type']
            str_path += f" -({r})-> {path[i+1]}"
            weight *= float(graph[path[i]][path[i + 1]]['weight'])
        paths.append(str_path)
        lens.append(len_path)
        weights.append(weight)

    sotred_idx = list(np.argsort(-np.array(weights)))  # desc
    paths = np.array(paths)[sotred_idx]
    lens = np.array(lens)[sotred_idx]
    weights = np.array(weights)[sotred_idx]

    for i in range(len(sotred_idx)):
        print(f"len: {lens[i]} | paths: {paths[i]} | weight: {weights[i]}")

    if count == 0:
        print('no paths.')

    if count >= max_depth:
        print('over max_depth: too many paths were found.')


if __name__ == '__main__':
    args = parse_args()
    g = load_graph(args.path_att)
    # TODO: remap relationid -> name
    search_path(g, args.userid, args.itemid)
