from collections import OrderedDict, defaultdict
from typing import List
import json
import logging

import time
from pydantic import BaseModel
import scipy.special as sp
import numpy as np

from .relevance import RelevanceExample
from pygaggle.model.tokenize import SpacySenticizer
from pygaggle.rerank.base import Query, Text


__all__ = ['MISSING_ID', 'SquadAnswer', 'SquadDataset',
            'SquadQas', 'SquadSubcategory', 'SquadData']

           
MISSING_ID = '<missing>'


class SquadAnswer(BaseModel):
    answer_start: int
    text: str


class SquadQas(BaseModel):
    answers: List[SquadAnswer]
    question: str
    id: str


class SquadSubcategory(BaseModel):
    context: str
    qas: List[SquadQas]


class SquadData(BaseModel):
    title: str
    paragraphs: List[SquadSubcategory]


class SquadDataset(BaseModel):
    data: List[SquadData]
    version: str

    @classmethod
    def from_file(cls, filename: str) -> 'SquadDataset':
        with open(filename, encoding='utf-8') as f:
            print(type(json.load(f)))
            return cls(**json.load(f))

    def query_answer(self):
        return ((qas.question, qas.answers, paragraphs.context, qas.id)
                for sqdata in self.data
                for paragraphs in sqdata.paragraphs
                for qas in paragraphs.qas           
                )

    def to_senticized_dataset(self
                              ) -> List[RelevanceExample]:
        tokenizer = SpacySenticizer()
        example_map = OrderedDict()
        rel_map = OrderedDict()
        
        count = 0
        for query, answers, context, id in self.query_answer():
            key = (query, id)
            answer_check = False

            try:
                example_map.setdefault(key, tokenizer(context))
            except ValueError as e:
                logging.warning(f'Skipping {id} ({e})')
                continue
            sents = example_map[key]
            rel_map.setdefault(key, [False] * len(sents))
            
            total_len = 0
            for idx, s in enumerate(sents):
                for a in answers:
                    if (a.answer_start >= total_len and a.answer_start <= total_len+len(s)):
                        rel_map[key][idx] = True
                        answer_check = True
                total_len += len(s)
            
            if answer_check == False:
                rel_map.pop(key)
                example_map.pop(key)
                logging.warning(f'Skipping {id} (answer error)')

        mean_stats = defaultdict(list)
        for (_, doc_id), rels in rel_map.items():
            int_rels = np.array(list(map(int, rels)))
            p = int_rels.sum()
            mean_stats['Average spans'].append(p)
            mean_stats['Random P@1'].append(np.mean(int_rels))
            n = len(int_rels) - p
            N = len(int_rels)
            if (N * (N - 1) * (N - 2)) != 0:
                r3 = 1 - (n * (n - 1) * (n - 2)) / (N * (N - 1) * (N - 2))
            else:
                r3 = 1 - (n * (n - 1) * (n - 2))
            mean_stats['Random R@3'].append(r3)
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
                dict(docid=id)), sents)), rels)
                for ((query, id), sents), (_, rels) in
                zip(example_map.items(), rel_map.items())]

