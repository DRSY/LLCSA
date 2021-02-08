import torch
import torch.nn as nn
import numpy as np
import transformers


class EpisodicMemory(object):
    """
    Memory module for sparse experience replay
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_vocab_id = self.tokenizer.pad_token_id
        # memory
        self.lm_input_ids = []
        self.lm_labels = []
        self.qa_input_ids = []
        self.qa_labels = []
        self.margin_ids = []
        self.margin_labels = []
        self.margin_cnts = []

    def make_batch(self, n_samples):
        ids = np.random.randint(len(self.lm_input_ids), size=min(
            [n_samples, len(self.lm_input_ids)]))
        _pool = set()
        lm_ids = []
        lm_labels = []
        qa_ids = []
        qa_labels = []
        margin_ids = []
        margin_labels = []
        margin_cnts = []
        for id in ids:
            if id in _pool:
                continue
            _pool.add(id)
            lm_ids.append(self.lm_input_ids[id])
            lm_labels.append(self.lm_labels[id])
            qa_ids.append(self.qa_input_ids[id])
            qa_labels.append(self.qa_labels[id])
            margin_ids.extend(self.margin_ids[id])
            margin_labels.extend(self.margin_labels[id])
            margin_cnts.extend(self.margin_cnts[id])
        lm_max_len = max([len(ids) for ids in lm_ids])
        qa_max_len = max([len(ids) for ids in qa_ids])
        margin_max_len = max([len(ids) for ids in margin_ids])
        lm_pt_ids = torch.tensor(
            [ids + (lm_max_len-len(ids)) * [self.pad_vocab_id] for ids in lm_ids], dtype=torch.long)
        lm_pt_mask = torch.ones_like(lm_pt_ids)
        lm_pt_mask[lm_pt_ids == self.pad_vocab_id] = 0
        lm_input_dict = {'input_ids': lm_pt_ids, 'attention_mask': lm_pt_mask}
        lm_input_dict = transformers.BatchEncoding(lm_input_dict)
        lm_pt_labels = torch.tensor(
            [ids + (lm_max_len-len(ids)) * [-100] for ids in lm_labels], dtype=torch.long)
        qa_pt_ids = torch.tensor(
            [ids + (qa_max_len-len(ids)) * [self.pad_vocab_id] for ids in qa_ids], dtype=torch.long)
        qa_pt_mask = torch.ones_like(qa_pt_ids)
        qa_pt_mask[qa_pt_ids == self.pad_vocab_id] = 0
        qa_input_dict = {'input_ids': qa_pt_ids, 'attention_mask': qa_pt_mask}
        qa_input_dict = transformers.BatchEncoding(qa_input_dict)
        qa_pt_labels = torch.tensor(
            [ids + (qa_max_len-len(ids)) * [-100] for ids in qa_labels], dtype=torch.long)
        margin_pt_ids = torch.tensor(
            [ids + (margin_max_len-len(ids)) * [self.pad_vocab_id] for ids in margin_ids], dtype=torch.long)
        margin_pt_mask = torch.ones_like(margin_pt_ids)
        margin_pt_mask[margin_pt_ids == self.pad_vocab_id] = 0
        margin_input_dict = {'input_ids': margin_pt_ids,
                             'attention_mask': margin_pt_mask}
        margin_input_dict = transformers.BatchEncoding(margin_input_dict)
        margin_pt_labels = torch.tensor(
            [ids + (margin_max_len-len(ids)) * [-100] for ids in margin_labels], dtype=torch.long)
        margin_pt_cnts = torch.tensor(margin_cnts, dtype=torch.float32)
        return lm_input_dict, lm_pt_labels, qa_input_dict, qa_pt_labels, margin_input_dict, margin_pt_labels, margin_pt_cnts

    def write(self, new_examples):
        for i, example in enumerate(new_examples):
            lm_input_id, lm_label, qa_input_id, qa_label, margin_ids, margin_labels, margin_cnts = example
            # remove additional padding for lm, qa, and marginal ranking
            lm_len = len(lm_input_id)
            while lm_input_id[lm_len-1] == self.pad_vocab_id:
                lm_len = lm_len - 1
            lm_input_id = lm_input_id[:lm_len]
            lm_label = lm_label[:lm_len]
            qa_len = len(qa_input_id)
            while qa_input_id[qa_len-1] == self.pad_vocab_id:
                qa_len = qa_len - 1
            qa_input_id = qa_input_id[:qa_len]
            qa_label = qa_label[:qa_len]
            _margin_ids = []
            _margin_labels = []
            _margin_cnts = []
            for margin_id, margin_label, margin_cnt in zip(margin_ids, margin_labels, margin_cnts):
                margin_len = len(margin_id)
                while margin_id[margin_len-1] == self.pad_vocab_id:
                    margin_len = margin_len - 1
                margin_id = margin_id[:margin_len]
                margin_label = margin_label[:margin_len]
                _margin_ids.append(margin_id)
                _margin_labels.append(margin_label)
                _margin_cnts.append(margin_cnt)
            # add to memory
            assert len(lm_input_id) == len(lm_label)
            assert len(qa_input_id) == len(qa_label)
            self.lm_input_ids.append(lm_input_id)
            self.lm_labels.append(lm_label)
            self.qa_input_ids.append(qa_input_id)
            self.qa_labels.append(qa_label)
            self.margin_ids.append(_margin_ids)
            self.margin_labels.append(_margin_labels)
            self.margin_cnts.append(margin_cnts)

    def sample(self, n_samples):
        batch = self.make_batch(n_samples)
        return batch

    def __len__(self):
        return len(self.lm_input_ids)

    def clear(self):
        self.lm_input_ids = []
        self.lm_labels = []
        self.qa_input_ids = []
        self.qa_labels = []
        self.margin_ids = []
        self.margin_labels = []
        self.margin_cnts = []
