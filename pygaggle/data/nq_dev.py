from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict, defaultdict
from typing import List
import json
import logging

import time

from torch._C import BoolType
from pydantic import BaseModel
import scipy.special as sp
import numpy as np

from .relevance import RelevanceExample
from pygaggle.model.tokenize import SpacySenticizer
from pygaggle.rerank.base import Query, Text

import re
import jsonlines


__all__ = ['MISSING_ID', 'ShortAnswers', 'LongAnswer',
            'Annotations', 'LongAnswers', 'NQDevData', 'NQDevDataset']

           
MISSING_ID = '<missing>'

def get_nq_tokens(simplified_nq_example):
      if "document_text" not in simplified_nq_example:
        raise ValueError("`get_nq_tokens` should be called on a simplified NQ"
                         "example that contains the `document_text` field.")

      return simplified_nq_example["document_text"].split(" ")


def simplify_nq_example(nq_example):

  def _clean_token(token):
    """Returns token in which blanks are replaced with underscores.
    HTML table cell openers may contain blanks if they span multiple columns.
    There are also a very few unicode characters that are prepended with blanks.
    Args:
      token: Dictionary representation of token in original NQ format.
    Returns:
      String token.
    """
    return re.sub(u" ", "_", token["token"])

  text = " ".join([_clean_token(t) for t in nq_example["document_tokens"]])

  def _remove_html_byte_offsets(span):
    if "start_byte" in span:
      del span["start_byte"]

    if "end_byte" in span:
      del span["end_byte"]

    return span

  def _clean_annotation(annotation):
    annotation["long_answer"] = _remove_html_byte_offsets(
        annotation["long_answer"])
    annotation["short_answers"] = [
        _remove_html_byte_offsets(sa) for sa in annotation["short_answers"]
    ]
    return annotation

  simplified_nq_example = {
      "question_text": nq_example["question_text"],
      "example_id": nq_example["example_id"],
      "document_url": nq_example["document_url"],
      "document_text": text,
      "long_answer_candidates": [
          _remove_html_byte_offsets(c)
          for c in nq_example["long_answer_candidates"]
      ],
      "annotations": [_clean_annotation(a) for a in nq_example["annotations"]]
  }

  if len(get_nq_tokens(simplified_nq_example)) != len(
      nq_example["document_tokens"]):
    raise ValueError("Incorrect number of tokens.")

  return simplified_nq_example


class ShortAnswers(BaseModel):
    end_token: int
    start_token: int


class LongAnswer(BaseModel):
    candidate_index: int
    end_token: int
    start_token: int


class Annotations(BaseModel):
    annotation_id: int
    long_answer: LongAnswer
    short_answers: List[ShortAnswers]
    yes_no_answer: str


class LongAnswers(BaseModel):
    end_token: int
    start_token: int
    top_level: bool


class NQDevData(BaseModel):
    question_text: str
    example_id: int
    document_url: str
    document_text: str
    long_answer_candidates: List[LongAnswers]
    annotations: List[Annotations]

class NQDevDataset(BaseModel):
    data: List[NQDevData]

    @classmethod
    def from_file(cls, filename: str, ndocs:int) -> 'NQDevDataset':
        data_str = ''
        count = 0
        with jsonlines.open(filename) as f:
            for line in f.iter():
                  data_aux = simplify_nq_example(line)
                  data_str = data_str + json.dumps(data_aux)
                  count += 1
                  if (count == ndocs):
                      break
                  else:
                    data_str = data_str + ","
        data_str = "{\"data\":["+data_str+"]}"
        data_str = json.loads(data_str)
        return cls(**(data_str))

    def query_answer(self):
        return ((nqdata.question_text, nqdata.annotations, nqdata.document_text, nqdata.example_id)
                for nqdata in self.data         
                )

    def to_senticized_dataset(self
                              ) -> List[RelevanceExample]:
        tokenizer = SpacySenticizer()
        example_map = OrderedDict()
        rel_map = OrderedDict()

        for query, annotations, context, id in self.query_answer():
            key = (query, id)
            context_split = context.split(' ')

            if (len(context_split) > 15000):
              continue

            try:
                example_map.setdefault(key, tokenizer(context))
            except ValueError as e:
                logging.warning(f'Skipping {id} ({e})')
                continue
            sents = example_map[key]
            rel_map.setdefault(key, [False] * len(sents))

            #answer_check = False
            for a in annotations:
              for short in a.short_answers:
                start = 0
                for idx,k in enumerate(context_split):
                  if idx == short.start_token:
                    break
                  start += len(k)

                total_len = 0
                for idx, s in enumerate(sents):
                    if (total_len < start and total_len + len(s) > start):
                        rel_map[key][idx] = True
                        #if answer_check == False:
                        #  answer_check = True
                    total_len += len(s)

            '''if answer_check == False or sents == []:
              rel_map.pop(key)
              example_map.pop(key)'''

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
            
        for k, v in mean_stats.items():
            logging.info(f'{k}: {np.mean(v)}')

        return [RelevanceExample(Query(query), list(map(lambda s: Text(s,
                dict(docid=id)), sents)), rels)
                for ((query, id), sents), (_, rels) in
                zip(example_map.items(), rel_map.items())]

