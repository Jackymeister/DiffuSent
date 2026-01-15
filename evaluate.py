#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：insights-span-aste
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：hpuhzh@outlook.com
# @Date    ：03/08/2022 14:46 
# ====================================
import argparse
import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.collate import gold_labels, collate_fn
from models.metrics import SpanEvaluator
from models.model import SpanAsteModel
from utils.dataset import CustomDataset
from utils.processor import Res15DataProcessor
from utils.tager import SpanLabel, RelationLabel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate(model, metric, data_loader, device):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch_ix, batch in enumerate(data_loader):
            input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len,\
                 context2token_masks, token_masks, gt_spans, gt_types, gt_masks, raw_len, gt_rel, gt_rel_lables = batch
            input_ids = torch.tensor(input_ids, device=device)
            attention_mask = torch.tensor(attention_mask, device=device)
            token_type_ids = torch.tensor(token_type_ids, device=device)
            max_len=max(raw_len)
            pad_context2token_masks=[]
            pad_token_masks=[]
            for i,sent_mask in enumerate(context2token_masks):
                pad_mask=sent_mask.tolist()
                pad_len=max_len-len(pad_mask)
                pad_mask=pad_mask+[[False]*128]*pad_len
                pad_context2token_masks.append(pad_mask)
                pad_token_masks.append(token_masks[i]+[0]*(max_len-len(token_masks[i])))
            context2token_masks=torch.tensor(pad_context2token_masks, device=device)
            token_masks = torch.tensor(pad_token_masks,dtype=torch.bool, device=device)
            gt_spans1 = torch.tensor(gt_spans,dtype=torch.long, device=device)
            gt_types1 = torch.tensor(gt_types,dtype=torch.long, device=device)
            gt_masks1 = torch.tensor(gt_masks,dtype=torch.bool, device=device)
            
            # forward
            outputs = model(
                    encodings=input_ids, 
                    context_masks=attention_mask, 
                    seg_encoding = attention_mask, 
                    context2token_masks=context2token_masks, 
                    token_masks=token_masks)
            span_indices=[spans.tolist() for spans in outputs['pred_spans']]
            spans_probability=outputs['pred_logits']
            # candidate_indices=[spans.tolist() for spans in outputs['rel_ids']]
            # relations_probability=outputs['rel_prob']
            gold_span_indices, gold_span_labels = gold_labels(span_indices, gt_spans, gt_types)
            # gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, gt_rel, gt_rel_lables)

            # num_correct, num_infer, num_label = metric.compute(relations_probability.cpu(),
            #                                                    torch.tensor(gold_relation_labels))
            num_correct, num_infer, num_label = metric.compute(spans_probability.cpu(),
                                                               torch.tensor(gold_span_labels))
            metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    model.train()
    return precision, recall, f1

def do_eval():
    set_seed(1024)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"using device:{device}")
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # create processor
    processor = Res15DataProcessor(tokenizer, args.max_seq_len)

    print("Loading Train & Eval Dataset...")
    # Load dataset
    test_dataset = CustomDataset("dev", args.test_path, processor, tokenizer, args.max_seq_len)

    print("Construct Dataloader...")

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print("Building SPAN-ASTE model...")
    # get dimension of target and relation
    target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
    # build span-aste model
    model = SpanAsteModel(
        args.bert_model,
        target_dim,
        relation_dim,
        device=device
    )

    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location=torch.device(device)))
    model.to(device)

    metric = SpanEvaluator()

    precision, recall, f1 = evaluate(model, metric, test_dataloader, device)
    print("-----------------------------")
    print("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
          (precision, recall, f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", type=str, default=None, help="The name of bert.")
    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization.")
    args = parser.parse_args()

    do_eval()
