# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
emoji - binary classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np
from datasets import load_dataset, DatasetDict

from senteval.tools.validation import SplitClassifier


def preprocess(dataset):
    t = dataset['text']
    t = '@user' if t.startswith('@') and len(t) > 1 else t
    t = 'http' if t.startswith('http') else t
    dataset['text'] = t
    return dataset

class SentimentEval(object):
    def __init__(self, task_path, nclasses=20, seed=2022):
        logging.info(f'***** Transfer task : {task_path} classification *****\n\n')
        self.seed = seed
        self.task_path=task_path
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'
        logging.debug(f'***** Transfer task : {task_path}  classification *****\n\n', self.task_name)
    
        
        dataset = load_dataset(task_path)
        if task_path=="imdb":
            # # for train
            # imdb_train = load_dataset("imdb",  split='train[10%:90%]')
            # imdb_vallid = load_dataset("imdb",  split='train[:10%]+train[-10%:]')
            # for eval
            imdb_train = load_dataset("imdb",  split='train[5%:95%]')
            imdb_vallid = load_dataset("imdb",  split='train[:5%]+train[-5%:]')
            
            dataset = DatasetDict({'train':imdb_train, 'validation':imdb_vallid, 'test':load_dataset(task_path)['test']})
        elif "yelp" in task_path:
            # for train
            #yelp_train = load_dataset(task_path,  split='train[:10000]')
            #yelp_valid = load_dataset(task_path,  split='train[-3000:]')
            
            # for eval
            yelp_train = load_dataset(task_path,  split='train[:90%]')
            yelp_valid = load_dataset(task_path,  split='train[-10%:]')
            dataset = DatasetDict({'train':yelp_train, 'validation':yelp_valid, 'test':load_dataset(task_path)['test']})
        
        
        # dataset = dataset.map(preprocess)
        if "sst2" in task_path:
            train =  {'X': [e.split() for e in dataset['train']['sentence']], 'y': dataset['train']['label']} 
            dev = {'X': [e.split() for e in dataset['validation']['sentence']], 'y': dataset['validation']['label']} 
            test = {'X': [e.split() for e in dataset['test']['sentence']], 'y': dataset['test']['label']} 
        else:
            train =  {'X': [e.split() for e in dataset['train']['text']], 'y': dataset['train']['label']} 
            dev = {'X': [e.split() for e in dataset['validation']['text']], 'y': dataset['validation']['label']} 
            test = {'X': [e.split() for e in dataset['test']['text']], 'y': dataset['test']['label']} 

        self.emoji_data = {'train': train, 'dev': dev, 'test': test}
        

    def do_prepare(self, params, prepare):
        samples = self.emoji_data['train']['X'] + self.emoji_data['dev']['X'] + \
                  self.emoji_data['test']['X']
        return prepare(params, samples)
    
    def run(self, params, batcher,data_name=None):
        
        if "imdb" in self.task_path:
            data_name="IMDB"
        elif "yelp" in self.task_path:
            data_name="Yelp2"
        elif "rotten" in self.task_path:
            data_name="MR"
        elif "sst2" in self.task_path:
            data_name="SST2"
        else:
            data_name="Wrong!"
        
        print('run_started')
        emoji_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size
        
        for key in self.emoji_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_data = sorted(zip(self.emoji_data[key]['X'],
                                     self.emoji_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.emoji_data[key]['X'], self.emoji_data[key]['y'] = map(list, zip(*sorted_data))

            emoji_embed[key]['X'] = []
            for ii in range(0, len(self.emoji_data[key]['y']), bsize):
                batch = self.emoji_data[key]['X'][ii:ii + bsize]
                embeddings = batcher(params, batch)
                emoji_embed[key]['X'].append(embeddings)
            emoji_embed[key]['X'] = np.vstack(emoji_embed[key]['X'])
            emoji_embed[key]['y'] = np.array(self.emoji_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}
        

        clf = SplitClassifier(X={'train': emoji_embed['train']['X'],
                                 'valid': emoji_embed['dev']['X'],
                                 'test': emoji_embed['test']['X']},
                              y={'train': emoji_embed['train']['y'],
                                 'valid': emoji_embed['dev']['y'],
                                 'test': emoji_embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.info('\nDev acc : {0} Test acc : {1} for \
             {2} classification\n'.format(devacc, testacc, self.task_name))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(emoji_embed['dev']['X']),
                'ntest': len(emoji_embed['test']['X'])}
