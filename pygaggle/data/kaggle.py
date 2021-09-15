from collections import OrderedDict, defaultdict
from typing import List
import json
import logging
import random

from pydantic import BaseModel
import scipy.special as sp
import numpy as np

from .relevance import RelevanceExample, Cord19DocumentLoader
from pygaggle.model.tokenize import SpacySenticizer
from pygaggle.rerank.base import Query, Text


__all__ = ['MISSING_ID', 'LitReviewCategory', 'LitReviewAnswer',
           'LitReviewDataset', 'LitReviewSubcategory']


MISSING_ID = '<missing>'


class LitReviewAnswer(BaseModel):
    id: str
    title: str
    exact_answer: str


class LitReviewSubcategory(BaseModel):
    nq_name: str
    kq_name: str
    answers: List[LitReviewAnswer]


class LitReviewCategory(BaseModel):
    name: str
    sub_categories: List[LitReviewSubcategory]


class LitReviewDataset(BaseModel):
    version: str
    categories: List[LitReviewCategory]

    @classmethod
    def from_file(cls, filename: str) -> 'LitReviewDataset':
        with open(filename, encoding='utf-8') as f:
            return cls(**json.load(f))

    def query_answer_pairs(self, split: str = 'nq'):
        return ((subcat.nq_name if split == 'nq' else subcat.kq_name, ans)
                for cat in self.categories
                for subcat in cat.sub_categories
                for ans in subcat.answers)

    def to_senticized_dataset(self,
                              index_path: str,
                              split: str = 'nq') -> List[RelevanceExample]:
        loader = Cord19DocumentLoader(index_path)
        tokenizer = SpacySenticizer()
        example_map = OrderedDict()
        rel_map = OrderedDict()

        #len_doc = []
        #len_sents = []
        for query, document in self.query_answer_pairs(split=split):
            context_list = []
            if document.id == MISSING_ID:
                logging.warning(f'Skipping {document.title} (missing ID)')
                continue
            key = (query, document.id)
            try:
                doc = loader.load_document(document.id)
                example_map.setdefault(key, tokenizer(doc.all_text))
                context_list.append(doc.all_text)
            except ValueError as e:
                logging.warning(f'Skipping {document.id} ({e})')
                continue
            sents = example_map[key]

            '''len_doc.append(len(sents))
            sents_len = list(len(k) for k in sents)
            len_sents.append(np.mean(sents_len))  '''

            '''sents_3 = []
            if (len(sents) > 2):
                for s in range(len(sents)-2):
                    sents_3.append(' '.join([sents[s], sents[s+1], sents[s+2]]))
                sents_3.append(' '.join([sents[-2], sents[-1]]))
                sents_3.append(' '.join([sents[-1]]))
            elif (len(sents) == 2):
                sents_3.append(' '.join([sents[0], sents[1]]))
                sents_3.append(' '.join([sents[-2], sents[-1]]))
            else:
                sents_3 = sents'''

            '''sents_3 = []
            if (len(sents) > 2):
                sents_3.append(' '.join([sents[0], sents[1]]))
                for s in range(len(sents)-2):
                    sents_3.append(' '.join([sents[s], sents[s+1], sents[s+2]]))
                sents_3.append(' '.join([sents[-2], sents[-1]]))
            elif (len(sents) > 1):
                sents_3.append(' '.join([sents[0], sents[1]]))
                sents_3.append(' '.join([sents[-2], sents[-1]]))
            else:
                sents_3 = sents'''

            rel_map.setdefault(key, [False] * len(sents))
            for idx, s in enumerate(sents):
                if document.exact_answer in s:
                    rel_map[key][idx] = True
            
            #example_map[key] = sents_3
        '''print(np.mean(len_doc))
        print(np.mean(len_sents))'''
        mean_stats = defaultdict(list)
        for (_, doc_id), rels in rel_map.items():
            int_rels = np.array(list(map(int, rels)))
            p = int_rels.sum()
            mean_stats['Average spans'].append(p)
            mean_stats['Random P@1'].append(np.mean(int_rels))
            n = len(int_rels) - p
            N = len(int_rels)
            mean_stats['Random R@3'].append(1 - (n * (n - 1) * (n - 2)) / (N * (N - 1) * (N - 2)))
            numer = np.array([sp.comb(n, i) / (N - i) for i in range(0, n + 1)]) * p
            denom = np.array([sp.comb(N, i) for i in range(0, n + 1)])
            rr = 1 / np.arange(1, n + 2)
            rmrr = np.sum(numer * rr / denom)
            mean_stats['Random MRR'].append(rmrr)
            if not any(rels):
                logging.warning(f'{doc_id} has no relevant answers')
        for k, v in mean_stats.items():
            logging.info(f'{k}: {np.mean(v)}')


        return [RelevanceExample(Query(query), list(map(lambda s: Text(s,
                dict(docid=docid)), sents)), rels)
                for ((query, docid), sents), (_, rels) in
                zip(example_map.items(), rel_map.items())]
