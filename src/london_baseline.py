# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import torch
from tqdm import tqdm
import sys
sys.path.append('./.')
import dataset
import run

from multiprocessing import Process, freeze_support


if __name__=="__main__":
    freeze_support()
    argp = argparse.ArgumentParser()
    argp.add_argument('pretrain_corpus_path',
        help="Path of the corpus to pretrain on", default=None)
    argp.add_argument('--eval_corpus_path',
        help="Path of the corpus to evaluate on", default=None)
    args = argp.parse_args()

    block_size = 128
    text = open(args.pretrain_corpus_path, encoding='utf-8').read()
    pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

    assert args.eval_corpus_path is not None
    correct = 0
    total = 0
    predictions = []
    for line in tqdm(open(args.eval_corpus_path)):
        x = line.split('\t')[0]
        x = x + '⁇'
        x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None, ...]
        pred = "London"
        predictions.append(pred)
    total, correct = run.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('Predictions written to {}; no targets provided'
                        .format(args.outputs_path))

