# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
STS-{2012,2013,2014,2015,2016} (unsupervised) and
STS-benchmark (supervised) tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging
import pandas as pd

from scipy.stats import spearmanr, pearsonr
import torch

from senteval.utils import cosine
from senteval.sick import SICKEval
from datasets import Dataset

# def preprocess_function(dataset):
#     t = dataset['text']
#     t = '@user' if t.startswith('@') and len(t) > 1 else t
#     t = 'http' if t.startswith('http') else t
#     dataset['text'] = t
#     return dataset

def preprocess(x):
    t = str(x)
    t = '@user' if t.startswith('@') and len(t) > 1 else t
    t = 'http' if t.startswith('http') else t
    return t

#class CLSTSEval(ClstsEval):
class CLSTSEval(object):
    def __init__(self, taskpath, task_name, seed=2022):
        logging.debug('*********Sentiment dataset evaluate******')
        self.seed=seed
        self.datasets=[task_name]
        self.loadFile(taskpath)

    def loadFile(self, fpath):
        logging.info(f'***** Transfer task : cl {self.datasets[0]} sts *****\n\n')
        self.data={}
        self.samples = []

        for dataset in self.datasets:
            df = pd.read_csv(fpath+'SgTS-%s.csv'% dataset,encoding='utf8')
            sent1 = df['text1']
            sent2 = df['text2']
            
            sent1 = sent1.map(preprocess)
            sent2 = sent2.map(preprocess)
            gs_scores = df['label']

            not_empty_idx = gs_scores !=''
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]

            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
    
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))
            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)
    
    def run(self, params, batcher):
        def align_loss(x, y, alpha=2):
            # x=x/torch.norm(x, p=2, dim=-1, keepdim=True)
            # y=y/torch.norm(y, p=2, dim=-1, keepdim=True)
            
            return (x - y).norm(p=2, dim=1).pow(alpha).mean()

        def uniform_loss(x, t=2):
            #x = x / torch.norm(x, p=2, dim=-1, keepdim=True)
            return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
        
        results = {}
        all_sys_scores = []
        all_gs_scores = []
        ################# newly added
        all_loss_align = []
        all_loss_uniform = []
        #################
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]
                # batch_gs_scores = gs_scores[ii:ii + params.batch_size]  # newly added

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    ################# newly added
                    #pos_indices = [i for i in range(len(batch_gs_scores)) if batch_gs_scores[i] >= 4.0]
                    enc1_norm = enc1/torch.norm(enc1, p=2, dim=-1, keepdim=True)
                    enc2_norm = enc2/torch.norm(enc2, p=2, dim=-1, keepdim=True)
                    # enc1_pos = enc1_norm[pos_indices]
                    # enc2_pos = enc2_norm[pos_indices]
                    # print('pos_indices:', pos_indices)
                    # print('enc1:', enc1_pos)
                    # print('enc2:', enc2_pos)
                    loss_align = align_loss(enc1_norm, enc2_norm)
                    loss_uniform = uniform_loss(torch.cat((enc1_norm, enc2_norm), dim=0))
                    all_loss_align.append(loss_align)
                    all_loss_uniform.append(loss_uniform)
                    #################
                    
                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)
            all_sys_scores.extend(sys_scores)
            all_gs_scores.extend(gs_scores)
            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores),
                                'align_loss': float(np.mean(all_loss_align)),  # newly added
                                'uniform_loss': float(np.mean(all_loss_uniform))}  # newly added
            logging.debug('%s : pearson = %.4f, spearman = %.4f, align_loss = %.4f, uniform_loss = %.4f' %
                          (dataset, results[dataset]['pearson'][0],
                           results[dataset]['spearman'][0], results[dataset]['align_loss'],
                           results[dataset]['uniform_loss']))

        weights = [results[dset]['nsamples'] for dset in results.keys()]
        # for dset in results.keys():
        #     print("dset이 이거" ,dset)
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)
        all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        all_spearman = spearmanr(all_sys_scores, all_gs_scores)
        results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                        'wmean': wavg_spearman},
                          'align_loss': float(np.mean(all_loss_align)),
                          'uniform_loss': float(np.mean(all_loss_uniform))
                        }
        logging.debug('ALL : Pearson = %.4f, \
            Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))
        return results


