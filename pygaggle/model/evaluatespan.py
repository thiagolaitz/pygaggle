from collections import OrderedDict
from copy import deepcopy
from pygaggle.model.tokenize import SpacySenticizer
from typing import List, Optional, Dict
from pathlib import Path
import os
import abc

from sklearn.metrics import recall_score
from tqdm import tqdm
import numpy as np
import string
import regex as re

from pygaggle.data.kaggle import RelevanceExample
from pygaggle.data.retrieval import RetrievalExample
from pygaggle.rerank.base import Reranker
from pygaggle.qa.base import Reader
from pygaggle.model.writer import Writer, MsMarcoWriter
from pygaggle.data.segmentation import SegmentProcessor

__all__ = ['RerankerSpanEvaluator', 'metric_names']
METRIC_MAP = OrderedDict()


class MetricAccumulator:
    name: str = None

    def accumulate(self, scores: List[float], gold: List[RelevanceExample], threshold: float):
        return

    @abc.abstractmethod
    def value(self):
        return


class MeanAccumulator(MetricAccumulator):
    def __init__(self):
        self.scores = []

    @property
    def value(self):
        return np.mean(self.scores)


class TruncatingMixin:
    def truncated_rels(self, scores: List[float]) -> np.ndarray:
        return np.array(scores)


def register_metric(name):
    def wrap_fn(metric_cls):
        METRIC_MAP[name] = metric_cls
        metric_cls.name = name
        return metric_cls
    return wrap_fn


def metric_names():
    return list(METRIC_MAP.keys())


class TopkMixin(TruncatingMixin):
    top_k: int = None

    def truncated_rels(self, scores: List[float], threshold: float) -> np.ndarray:
        rel_idxs = sorted(list(enumerate(scores)),
                          key=lambda x: x[1], reverse=True)[self.top_k:]
        scores = np.array(scores)
        scores_backup = deepcopy(scores)
        scores[[x[0] for x in rel_idxs]] = -1
        for idx, x in enumerate(scores_backup):
            if x < threshold:
                scores[idx] = -1
        return scores


class DynamicThresholdingMixin(TruncatingMixin):
    threshold: float = 0.5

    def truncated_rels(self, scores: List[float]) -> np.ndarray:
        scores = np.array(scores)
        scores[scores < self.threshold * np.max(scores)] = 0
        return scores


class RecallAccumulator(TruncatingMixin, MeanAccumulator):
    def accumulate(self, rel_map: List[int], gold: RelevanceExample, threshold: float):
        gold_rels = np.array(gold.labels, dtype=int)
        rel_list = np.array(rel_map, dtype=int)
        score = recall_score(gold_rels, rel_list, zero_division=0)
        self.scores.append(score)


class PrecisionAccumulator(TruncatingMixin, MeanAccumulator):
    def accumulate(self, rel_map: List[int], gold: RelevanceExample, threshold: float):
        gold_rels = np.array(gold.labels, dtype=int)
        rel_list= np.array(rel_map, dtype=int)
        sum_score = rel_list.sum()
        if sum_score > 0:
            self.scores.append((rel_list & gold_rels).sum() / sum_score)

'''
@register_metric('precision@1')
class PrecisionAt1Metric(TopkMixin, PrecisionAccumulator):
    top_k = 1
'''

@register_metric('precision@2')
class PrecisionAt2Metric(TopkMixin, PrecisionAccumulator):
    top_k = 2


@register_metric('recall@2')
class RecallAt2Metric(TopkMixin, RecallAccumulator):
    top_k = 2

'''
@register_metric('recall@3')
class RecallAt3Metric(TopkMixin, RecallAccumulator):
    top_k = 3


@register_metric('recall@50')
class RecallAt50Metric(TopkMixin, RecallAccumulator):
    top_k = 50


@register_metric('recall@1000')
class RecallAt1000Metric(TopkMixin, RecallAccumulator):
    top_k = 1000'''


'''@register_metric('mrr')
class MrrMetric(MeanAccumulator):
    def accumulate(self, rel_map: List[int], gold: RelevanceExample, threshold: float):
        gold_rels = np.array(gold.labels, dtype=int)
        rel_list = np.array(rel_map, dtype=int)
        if ((rel_list & gold_rels).sum() != 0):
            rr = 1
        else:
            rr = 0
        
        self.scores.append(rr)'''

'''
@register_metric('mrr@10')
class MrrAt10Metric(MeanAccumulator):
    def accumulate(self, scores: List[float], gold: RelevanceExample, threshold: float):
        scores = sorted(list(enumerate(scores)), key=lambda x: x[1],
                        reverse=True)
        rr = next((1 / (rank_idx + 1) for rank_idx, (idx, s) in
                   enumerate(scores) if (gold.labels[idx] and rank_idx < 10 and s > threshold)),
                  0)
        self.scores.append(rr)'''


class ThresholdedRecallMetric(DynamicThresholdingMixin, RecallAccumulator):
    threshold = 0.5


class ThresholdedPrecisionMetric(DynamicThresholdingMixin,
                                 PrecisionAccumulator):
    threshold = 0.5


class RerankerSpanEvaluator:
    def __init__(self,
                 reranker: Reranker,
                 metric_names: List[str],
                 use_tqdm: bool = True,
                 writer: Optional[Writer] = None):
        self.reranker = reranker
        self.metrics = [METRIC_MAP[name] for name in metric_names]
        self.use_tqdm = use_tqdm
        self.writer = writer

    def evaluate(self,
                 examples: List[RelevanceExample], threshold: float) -> List[MetricAccumulator]:
        metrics = [cls() for cls in self.metrics]
        senticizer = SpacySenticizer()
        
        for example in tqdm(examples, disable=not self.use_tqdm):
            context_split = example.context.split(' ')

            if (len(context_split) > 15000):
              continue

            rel_map = self.reranker.rescore(example.query, example.context, threshold, top_n = 2)

            if len(example.labels) != len(rel_map):
                print("ruim")
                print(len(senticizer(example.context)))
                continue

            for metric in metrics:
                metric.accumulate(rel_map, example, threshold)

        return metrics