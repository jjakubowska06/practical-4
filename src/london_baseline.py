# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

random.seed(0)
import sys
sys.path.append('./.')
import dataset
from mingptdemo.mingpt import model, utils
import trainer
import dataset

import logging
logging.basicConfig(level = logging.INFO)

def evaluate_places(filepath, predicted_places):
  """ Computes percent of correctly predicted birth places.

  Arguments:
    filepath: path to a file with our name, birth place data.
    predicted_places: a list of strings representing the 
        predicted birth place of each person.

  Returns: 
    (total, correct), floats
  """
  with open(filepath) as fin:
    lines = [x.strip().split('\t') for x in fin]
    if len(lines[0]) == 1:
      print('No gold birth places provided; returning (0,0)')
      return (0,0)
    true_places = [x[1] for x in lines]
    total = len(true_places)
    assert total == len(predicted_places)
    correct = len(list(filter(lambda x: x[0] == x[1],
      zip(true_places, predicted_places))))
    return (float(total),float(correct))


from multiprocessing import Process, freeze_support

if __name__=="__main__":
    freeze_support()

    argp = argparse.ArgumentParser()
    argp.add_argument('function',
        help="Whether to pretrain, finetune or evaluate a model",
        choices=["pretrain", "finetune", "evaluate"])
    argp.add_argument('variant',
        help="Which variant of the model to run ('vanilla')",
        choices=["vanilla"])
    argp.add_argument('pretrain_corpus_path',
        help="Path of the corpus to pretrain on", default=None)
    argp.add_argument('--reading_params_path',
        help="If specified, path of the model to load before finetuning/evaluation",
        default=None)
    argp.add_argument('--writing_params_path',
        help="Path to save the model after pretraining/finetuning", default=None)
    argp.add_argument('--finetune_corpus_path',
        help="Path of the corpus to finetune on", default=None)
    argp.add_argument('--eval_corpus_path',
        help="Path of the corpus to evaluate on", default=None)
    argp.add_argument('--outputs_path', default=None)
    args = argp.parse_args()

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    block_size = 128
    text = open(args.pretrain_corpus_path, encoding='utf-8').read()
    pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

    mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
            n_layer=4, n_head=8, n_embd=256)

    gpt_model = model.GPT(mconf)

    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    gpt_model.load_state_dict(torch.load(args.reading_params_path))
    gpt_model = gpt_model.to(device)
    correct = 0
    total = 0
    print(len(pretrain_dataset.data))
    #baseline = 
    # with open(args.outputs_path, 'w') as fout:
    #             predictions = []
    #             for line in tqdm(open(args.eval_corpus_path)):
    #                 x = line.split('\t')[0]
    #                 x = x + '⁇'
    #                 x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)
    #                 pred = utils.sample(gpt_model, x, 32, sample=False)[0]
    #                 completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
    #                 pred = completion.split('⁇')[1]
    #                 predictions.append(pred)
    #                 fout.write(pred + '\n')
    #             total, correct = evaluate_places(args.eval_corpus_path, predictions)
    #         if total > 0:
    #             print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    #         else:
    #             print('Predictions written to {}; no targets provided'
    #                     .format(args.outputs_path))

