from collections import defaultdict
from copy import deepcopy
from itertools import combinations, permutations
from typing import List
from pygaggle.model.tokenize import SpacySenticizer

from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          PreTrainedModel,
                          PreTrainedTokenizer,
                          T5ForConditionalGeneration)
import torch
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from .base import Reranker, Query, Text
from .similarity import SimilarityMatrixProvider
from pygaggle.model import (BatchTokenizer,
                            LongBatchEncoder,
                            QueryDocumentBatch,
                            DuoQueryDocumentBatch,
                            QueryDocumentBatchTokenizer,
                            SpecialTokensCleaner,
                            T5BatchTokenizer,
                            T5DuoBatchTokenizer,
                            greedy_decode)


__all__ = ['MonoT5',
           'DuoT5',
           'UnsupervisedTransformerReranker',
           'MonoBERT',
           'QuestionAnsweringTransformerReranker',
           'SentenceTransformersReranker',
           'AlbertReranker']


class MonoT5(Reranker):
    def __init__(self,
                 model: T5ForConditionalGeneration = None,
                 tokenizer: QueryDocumentBatchTokenizer = None,
                 use_amp = False):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'castorini/monot5-base-msmarco',
                  *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path,
                                                          *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str = 't5-base',
                      *args, batch_size: int = 8, **kwargs) -> T5BatchTokenizer:
        return T5BatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs),
            batch_size=batch_size
        )

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentBatch(query=query, documents=texts)
        for batch in self.tokenizer.traverse_query_document(batch_input):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = batch.output['input_ids'].to(self.device)
                attn_mask = batch.output['attention_mask'].to(self.device)
                _, batch_scores = greedy_decode(self.model,
                                                input_ids,
                                                length=1,
                                                attention_mask=attn_mask,
                                                return_last_logits=True)

                # 6136 and 1176 are the indexes of the tokens false and true in T5.
                batch_scores = batch_scores[:, [6136, 1176]]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score

        return texts

class PTT5(Reranker):
    def __init__(self,
                 model: T5ForConditionalGeneration = None,
                 tokenizer: QueryDocumentBatchTokenizer = None,
                 use_amp = False):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'castorini/monot5-base-msmarco',
                  *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path,
                                                          *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path,
                      *args, batch_size: int = 8, **kwargs) -> T5BatchTokenizer:
        return T5BatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs),
            batch_size=batch_size
        )

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentBatch(query=query, documents=texts)
        for batch in self.tokenizer.traverse_query_document(batch_input):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = batch.output['input_ids'].to(self.device)
                attn_mask = batch.output['attention_mask'].to(self.device)
                _, batch_scores = greedy_decode(self.model,
                                                input_ids,
                                                length=1,
                                                attention_mask=attn_mask,
                                                return_last_logits=True)

                # 47 and 2593 are the indexes of the tokens no and yes in PTTT5.
                batch_scores = batch_scores[:, [47, 2593]]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score

        return texts

class MT5(Reranker):
    def __init__(self,
                 model: T5ForConditionalGeneration = None,
                 tokenizer: QueryDocumentBatchTokenizer = None,
                 use_amp = False):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'unicamp-dl/mt5-base-multi-msmarco',
                  *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path,
                                                          *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path,
                      *args, batch_size: int = 8, **kwargs) -> T5BatchTokenizer:
        return T5BatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs),
            batch_size=batch_size
        )

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentBatch(query=query, documents=texts)
        for batch in self.tokenizer.traverse_query_document(batch_input):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = batch.output['input_ids'].to(self.device)
                attn_mask = batch.output['attention_mask'].to(self.device)
                _, batch_scores = greedy_decode(self.model,
                                                input_ids,
                                                length=1,
                                                attention_mask=attn_mask,
                                                return_last_logits=True)

                # 259 and 6274 are the indexes of the tokens false and true in T5.
                batch_scores = batch_scores[:, [259, 6274]]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score

        return texts


class MT5_EN_PT(Reranker):
    def __init__(self,
                 model: T5ForConditionalGeneration = None,
                 tokenizer: QueryDocumentBatchTokenizer = None,
                 use_amp = False):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'unicamp-dl/mt5-base-en-pt-msmarco',
                  *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path,
                                                          *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path,
                      *args, batch_size: int = 8, **kwargs) -> T5BatchTokenizer:
        return T5BatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs),
            batch_size=batch_size
        )

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentBatch(query=query, documents=texts)
        for batch in self.tokenizer.traverse_query_document(batch_input):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = batch.output['input_ids'].to(self.device)
                attn_mask = batch.output['attention_mask'].to(self.device)
                _, batch_scores = greedy_decode(self.model,
                                                input_ids,
                                                length=1,
                                                attention_mask=attn_mask,
                                                return_last_logits=True)

                batch_scores = batch_scores[:, [375, 36339]]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score

        return texts


class DuoT5(Reranker):
    def __init__(self,
                 model: T5ForConditionalGeneration = None,
                 tokenizer: QueryDocumentBatchTokenizer = None,
                 use_amp = True):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'castorini/duot5-base-msmarco',
                  *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path,
                                                          *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str = 't5-base',
                      *args, batch_size: int = 8, **kwargs) -> T5DuoBatchTokenizer:
        return T5DuoBatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs),
            batch_size=batch_size
        )

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)#truncar em 50 ou 100 char
        for k in range(len(texts)):
            texts[k].text = texts[k].text[:150]
        doc_pairs = list(permutations(texts, 2))#ordenar por tamanho menor -> maior / mudar para arranjo
        scores = defaultdict(float)
        batch_input = DuoQueryDocumentBatch(query=query, doc_pairs=doc_pairs)
        for batch in self.tokenizer.traverse_duo_query_document(batch_input):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = batch.output['input_ids'].to(self.device)
                attn_mask = batch.output['attention_mask'].to(self.device)
                _, batch_scores = greedy_decode(self.model,
                                                input_ids,
                                                length=1,
                                                attention_mask=attn_mask,
                                                return_last_logits=True)

                # 6136 and 1176 are the indexes of the tokens false and true in T5.
                batch_scores = batch_scores[:, [6136, 1176]]
                batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
                batch_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.doc_pairs, batch_probs):
                scores[doc[0].metadata['docid']] += score
                scores[doc[1].metadata['docid']] += (1 - score)

        for text in texts:
            text.score = scores[text.metadata['docid']]

        return texts


class UnsupervisedTransformerReranker(Reranker):
    methods = dict(max=lambda x: x.max().item(),
                   mean=lambda x: x.mean().item(),
                   absmean=lambda x: x.abs().mean().item(),
                   absmax=lambda x: x.abs().max().item())

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: BatchTokenizer,
                 sim_matrix_provider: SimilarityMatrixProvider,
                 method: str = 'max',
                 clean_special: bool = True,
                 argmax_only: bool = False):
        assert method in self.methods, 'inappropriate scoring method'
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = LongBatchEncoder(model, tokenizer)
        self.sim_matrix_provider = sim_matrix_provider
        self.method = method
        self.clean_special = clean_special
        self.cleaner = SpecialTokensCleaner(tokenizer.tokenizer)
        self.device = next(self.model.parameters(), None).device
        self.argmax_only = argmax_only

    @torch.no_grad()
    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        encoded_query = self.encoder.encode_single(query)
        encoded_documents = self.encoder.encode(texts)
        texts = deepcopy(texts)
        max_score = None
        for enc_doc, text in zip(encoded_documents, texts):
            if self.clean_special:
                enc_doc = self.cleaner.clean(enc_doc)
            matrix = self.sim_matrix_provider.compute_matrix(encoded_query,
                                                             enc_doc)
            score = self.methods[self.method](matrix) if matrix.size(1) > 0 \
                else -10000
            text.score = score
            max_score = score if max_score is None else max(max_score, score)
        if self.argmax_only:
            for text in texts:
                if text.score != max_score:
                    text.score = max_score - 10000

        return texts


class MonoBERT(Reranker):
    def __init__(self,
                 model: PreTrainedModel = None,
                 tokenizer: PreTrainedTokenizer = None,
                 use_amp = False):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'castorini/monobert-large-msmarco',
                  *args, device: str = None, **kwargs) -> AutoModelForSequenceClassification:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                  *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str = 'bert-large-uncased',
                      *args, **kwargs) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs)

    @torch.no_grad()
    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        for text in texts:
            ret = self.tokenizer.encode_plus(query.text,
                                             text.text,
                                             max_length=512,
                                             truncation=True,
                                             return_token_type_ids=True,
                                             return_tensors='pt')
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = ret['input_ids'].to(self.device)
                tt_ids = ret['token_type_ids'].to(self.device)
                output, = self.model(input_ids, token_type_ids=tt_ids, return_dict=False)
                if output.size(1) > 1:
                    text.score = torch.nn.functional.log_softmax(
                        output, 1)[0, -1].item()
                else:
                    text.score = output.item()

        return texts


class QuestionAnsweringTransformerReranker(Reranker):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        for text in texts:
            ret = self.tokenizer.encode_plus(query.text,
                                             text.text,
                                             max_length=512,
                                             truncation=True,
                                             return_tensors='pt',
                                             return_token_type_ids=True)
            input_ids = ret['input_ids'].to(self.device)
            tt_ids = ret['token_type_ids'].to(self.device)
            start_scores, end_scores = self.model(input_ids,
                                                  token_type_ids=tt_ids,
                                                  return_dict=False)
            start_scores = start_scores[0]
            end_scores = end_scores[0]
            start_scores[(1 - tt_ids[0]).bool()] = -5000
            end_scores[(1 - tt_ids[0]).bool()] = -5000
            smax_val, smax_idx = start_scores.max(0)
            emax_val, emax_idx = end_scores.max(0)
            text.score = max(smax_val.item(), emax_val.item())

        return texts

class AlbertReranker(Reranker):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def rescore(self, query: Query, texts: Text, threshold: float, top_n: int) -> List[int]:
        context = deepcopy(texts)
        senticizer = SpacySenticizer()

        sents = senticizer(context)
        
        ret2 = self.tokenizer(
            query.text, 
            context,    
            truncation="only_second",        
            max_length=384,
            stride=192,
            return_tensors='pt',
            return_overflowing_tokens=True,
            padding="max_length"
        ).to(self.device)
        
        overflow_map = ret2.pop('overflow_to_sample_mapping')
        token_type = ret2.pop('token_type_ids')
        token_type = token_type.tolist()[0]
        output = self.model(**ret2)

        start_scores, end_scores = output.start_logits, output.end_logits
        start_scores = start_scores.tolist()
        end_scores = end_scores.tolist()
        start_scores = np.array(start_scores, dtype=float)
        end_scores = np.array(end_scores, dtype=float)
        
        print(ret2)
        count = 0#length of query
        for t in token_type:
            if (t == 0):
                count += 1
            else:
                break

        #Select max start/end scores
        start_max = [np.max(s[count:]) for s in start_scores]
        end_max = [np.max(s[count:]) for s in end_scores]      
        
        top1 = -500
        top2 = -500
        top1_s = 0
        top2_s = 0

        for idx,s in enumerate(start_max):
            if (s > top1):
                top1 = s
                top1_s = idx#index to window

        for idx,s in enumerate(start_max):
            if (s < top1 and s > top2):
                top2 = s
                top2_s = idx


        smax_idx = []#Position of max token in the top 2 score sentences
        smax_idx.append(np.argmax(start_scores[top1_s][count:])+count)
        smax_idx.append(np.argmax(start_scores[top2_s][count:])+count)

        top1 = -500
        top2 = -500
        top1_e = 0
        top2_e = 0

        for idx,s in enumerate(end_max):
            if (s > top1):
                top1 = s
                top1_e = idx#index to window

        for idx,s in enumerate(end_max):
            if (s < top1 and s > top2):
                top2 = s
                top2_e = idx
        
        emax_idx = []
        emax_idx.append(np.argmax(end_scores[top1_e][count:])+count-1)
        emax_idx.append(np.argmax(end_scores[top2_e][count:])+count-1)

        score = []#Scores to determine if the sentence has an answer
        score.append(start_scores[top1_s][0] + end_scores[top1_e][0] - start_scores[top1_s][smax_idx[0]] - end_scores[top1_e][emax_idx[0]])
        score.append(start_scores[top2_s][1] + end_scores[top2_e][1] - start_scores[top2_s][smax_idx[1]] - end_scores[top2_e][emax_idx[1]])
        
        #idx = [smax_idx, emax_idx]
        input_ids = ret2['input_ids'][top1_s].tolist()
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[::]))

        rel_map = [0 for _ in sents]
        len_start = []
        len_end = []

        #somar 192 por indice
        s1_over = top1_s * 192
        e1_over = top1_e * 192
        len_start.append(len(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[count+s1_over:smax_idx[0]]))))
        len_end.append(len(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[count+e1_over:emax_idx[0]]))))

        s2_over = top2_s * 192
        e2_over = top2_e * 192
        len_start.append(len(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[count+s2_over:smax_idx[1]]))))
        len_end.append(len(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[count+e2_over:emax_idx[1]]))))

        for n in range(top_n):    
            s_len = 0
            if score[n] <= threshold and emax_idx[n] > smax_idx[n]:
                start = -2
                for idx, s in enumerate(sents):
                    if s_len <= len_start[n] and (s_len+len(s) >= len_start[n]):
                        start = idx
                        rel_map[idx] = 1
                    elif s_len <= len_end[n] and (s_len+len(s) >= len_end[n]):
                        if (start != -2 and idx-start <= 1 and np.sum(rel_map)+1 <= top_n):
                            rel_map[start+1] = 1
                    s_len += len(s)
        
        return rel_map


class SentenceTransformersReranker(Reranker):
    def __init__(self,
                 pretrained_model_name_or_path='cross-encoder/ms-marco-MiniLM-L-2-v2',
                 max_length=512,
                 device=None,
                 use_amp=False):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp
        self.model = CrossEncoder(
            pretrained_model_name_or_path, max_length=max_length, device=device
        )

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            scores1 = self.model.predict(
                [(query.text, text.text) for text in texts],
                show_progress_bar=False,
            )

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            scores2 = self.model.predict(
                [(text.text, query.text) for text in texts],
                show_progress_bar=False,
            )

        assert len(texts) == len(scores1) == len(scores2)
        
        for (text, score, score2) in zip(texts, scores1, scores2):
            if (score.item() > score2.item()):
                text.score = score.item()
            else:
                text.score = score2.item()

        return texts

class SentenceTransformersBiEncoder(Reranker):
    def __init__(self,
                 pretrained_model_name_or_path,
                 device=None,
                 use_amp=False):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp
        self.model = SentenceTransformer(
            pretrained_model_name_or_path, device=device
        )

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            embeddings1 = self.model.encode([query.text], convert_to_tensor=True, show_progress_bar=False)
            embeddings2 = self.model.encode([text.text for text in texts], convert_to_tensor=True, show_progress_bar=False)
            scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        
        assert len(texts) == len(scores[0]), "Different shapes between texts and scores."

        for (text, score) in zip(texts, scores[0]):
            text.score = score.item()

        return texts
        
