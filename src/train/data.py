import jsonlines
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from config import GEN, ANS, EOS
import random


class CSDataset(Dataset):
    """
    """

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        file_obj = open(file_path, 'r')
        self.datas = list(jsonlines.Reader(file_obj))
        random.shuffle(self.datas)
        file_obj.close()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index: int):
        return self.datas[index]


class Collatefn(object):
    """
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ans_vocab_id = tokenizer.convert_tokens_to_ids([ANS])[0]
        self.gen_vocab_id = tokenizer.convert_tokens_to_ids([GEN])[0]
        self.pad_vocab_id = tokenizer.pad_token_id

    def __call__(self, batch):
        bs = len(batch)
        lm_input = []
        qa_input = []
        margin_ranking_input = []
        for i, item in enumerate(batch):
            context = item['context']
            candidates = item['candidates']
            answer = item['answer']
            correct_candidate = candidates[answer]
            qa_format = context + " " + ANS + " " + correct_candidate + " " + EOS
            lm_format = GEN + " " + context + " " + ANS + " " + correct_candidate + " " + EOS
            distractors = []
            for j in range(len(candidates)):
                if j == answer:
                    continue
                distractor_format = context + " " + ANS + " " + candidates[j] + " " + EOS
                distractors.append(distractor_format)
            distractors.append(qa_format)
            qa_input.append(qa_format)
            lm_input.append(lm_format)
            margin_ranking_input.append((distractors, -1))
        lm_input_dict = self.tokenizer(lm_input, add_special_tokens=False, padding='longest', return_tensors='pt')
        lm_input_labels = lm_input_dict['input_ids'].clone()  # same as input_ids
        lm_input_labels[lm_input_labels == self.pad_vocab_id] = -100
        qa_input_dict = self.tokenizer(qa_input, add_special_tokens=False, padding='longest', return_tensors='pt')
        qa_input_labels = []
        for i in range(bs):
            id_list = qa_input_dict['input_ids'][i].tolist()
            ans_idx = id_list.index(self.ans_vocab_id)
            new_id_list = id_list[:]
            new_id_list = torch.tensor(new_id_list, dtype=torch.long).unsqueeze(0)
            new_id_list[0, :ans_idx + 1] = -100
            new_id_list[new_id_list == self.pad_vocab_id] = -100
            qa_input_labels.append(new_id_list)
        qa_input_labels = torch.cat(qa_input_labels, dim=0)
        margin_input_dicts = []
        for i in range(bs):
            options, label = margin_ranking_input[i]
            options_input_dict = self.tokenizer(options, add_special_tokens=False, padding='longest',
                                                return_tensors='pt')
            id_list = options_input_dict['input_ids'][0].tolist()
            ans_idx = id_list.index(self.ans_vocab_id)
            options_labels = options_input_dict['input_ids'].clone()
            options_labels[:, :ans_idx + 1] = -100
            options_labels[options_labels == self.pad_vocab_id] = -100
            assert options_input_dict['input_ids'].shape == options_labels.shape
            label = torch.tensor(label, dtype=torch.long)
            margin_input_dicts.append((
                options_input_dict, options_labels, label
            ))
        assert len(margin_input_dicts) == bs
        assert lm_input_dict['input_ids'].shape == lm_input_labels.shape
        assert qa_input_dict['input_ids'].shape == qa_input_labels.shape
        return lm_input_dict, lm_input_labels, qa_input_dict, qa_input_labels, margin_input_dicts


class Collatefn_fast(object):
    """
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ans_vocab_id = tokenizer.convert_tokens_to_ids([ANS])[0]
        self.gen_vocab_id = tokenizer.convert_tokens_to_ids([GEN])[0]
        self.pad_vocab_id = tokenizer.pad_token_id

    def __call__(self, batch):
        bs = len(batch)
        lm_input = []
        qa_input = []
        margin_ranking_input = []
        for i, item in enumerate(batch):
            context = item['context']
            candidates = item['candidates']
            answer = item['answer']
            correct_candidate = candidates[answer]
            qa_format = context + " " + ANS + " " + correct_candidate + " " + EOS
            lm_format = GEN + " " + context + " " + ANS + " " + correct_candidate + " " + EOS
            distractors = []
            for j in range(len(candidates)):
                if j == answer:
                    continue
                distractor_format = context + " " + ANS + " " + candidates[j] + " " + EOS
                distractors.append(distractor_format)
            distractors.append(qa_format)
            qa_input.append(qa_format)
            lm_input.append(lm_format)
            margin_ranking_input.extend(distractors)
        lm_input_dict = self.tokenizer(lm_input, add_special_tokens=False, padding='longest', return_tensors='pt')
        lm_input_labels = lm_input_dict['input_ids'].clone()  # same as input_ids
        lm_input_labels[lm_input_labels == self.pad_vocab_id] = -100
        qa_input_dict = self.tokenizer(qa_input, add_special_tokens=False, padding='longest', return_tensors='pt')
        qa_input_labels = []
        for i in range(bs):
            id_list = qa_input_dict['input_ids'][i].tolist()
            ans_idx = id_list.index(self.ans_vocab_id)
            new_id_list = id_list[:]
            new_id_list = torch.tensor(new_id_list, dtype=torch.long).unsqueeze(0)
            new_id_list[0, :ans_idx + 1] = -100
            new_id_list[new_id_list == self.pad_vocab_id] = -100
            qa_input_labels.append(new_id_list)
        qa_input_labels = torch.cat(qa_input_labels, dim=0)
        margin_input_dict = self.tokenizer(margin_ranking_input, add_special_tokens=False, padding='longest',
                                           return_tensors='pt')
        margin_labels = margin_input_dict['input_ids'].clone()
        margin_cnt = []
        for i in range(bs * 3):
            _list = margin_labels[i].tolist()
            ans_idx = _list.index(self.ans_vocab_id)
            new_list = _list[:]
            new_list = torch.tensor(new_list, dtype=torch.long)
            new_list[:ans_idx + 1] = -100
            new_list[new_list == self.pad_vocab_id] = -100
            margin_labels[i] = new_list
            cnt = 0
            for j in range(new_list.shape[-1]):
                if new_list[j].item() != -100:
                    cnt += 1
            margin_cnt.append(cnt)
        margin_cnt = torch.tensor(margin_cnt, dtype=torch.float32)
        assert lm_input_dict['input_ids'].shape == lm_input_labels.shape
        assert qa_input_dict['input_ids'].shape == qa_input_labels.shape
        return lm_input_dict, lm_input_labels, qa_input_dict, qa_input_labels, margin_input_dict, margin_labels, margin_cnt
