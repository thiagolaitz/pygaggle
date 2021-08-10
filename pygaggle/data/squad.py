from collections import OrderedDict, defaultdict
from typing import List
import json
import logging

from pydantic import BaseModel
import scipy.special as sp
import numpy as np

from .relevance import RelevanceExample, Cord19DocumentLoader
from pygaggle.model.tokenize import SpacySenticizer
from pygaggle.rerank.base import Query, Text


__all__ = ['MISSING_ID', 'SquadDataset']

           
MISSING_ID = '<missing>'


class SquadAnswer(BaseModel):
    answer_start = int
    text = str


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
            return cls(**json.load(f))

    def query_answer(self):
        return ((qas.question, answers, qas.id)
                for sqdata in self.data
                for paragraphs in sqdata.paragraphs
                for qas in paragraphs.qas
                for answers in qas.answers)

    def to_senticized_dataset(self,
                              index_path: str) -> List[RelevanceExample]:
        loader = Cord19DocumentLoader(index_path)
        tokenizer = SpacySenticizer()
        example_map = OrderedDict()
        rel_map = OrderedDict()

        for query, answer, id in self.query_answer():
            key = (query, answer)
            try:
                doc = loader.load_document(id)
                example_map.setdefault(key, tokenizer(doc.all_text))
            except ValueError as e:
                logging.warning(f'Skipping {id} ({e})')
                continue
            sents = example_map[key]
            print(sents)

            rel_map.setdefault(key, [False] * len(sents))
            for idx, s in enumerate(sents):
                if document.exact_answer in s:
                    rel_map[key][idx] = True
        
        return [RelevanceExample(Query(query), list(map(lambda s: Text(s,
                dict(docid=docid)), sents)), rels)
                for ((query, docid), sents), (_, rels) in
                zip(example_map.items(), rel_map.items())]

