import jsonlines
import json
import sys, os


def process_atomic():
    train_path = "../../data/ATOMIC/train_random.jsonl"
    train_datas = []
    dev_path = "../../data/ATOMIC/dev_random.jsonl"
    dev_datas = []
    for instance in jsonlines.Reader(open(train_path, 'r', encoding='utf-8')):
        context = instance['context']
        correct = instance['correct']
        candidates = instance['candidates']
        _tmp = dict()
        _tmp['context'] = context
        _tmp['candidates'] = candidates
        _tmp['answer'] = correct
        train_datas.append(_tmp)
    for instance in jsonlines.Reader(open(dev_path, 'r', encoding='utf-8')):
        context = instance['context']
        correct = instance['correct']
        candidates = instance['candidates']
        _tmp = dict()
        _tmp['context'] = context
        _tmp['candidates'] = candidates
        _tmp['answer'] = correct
        dev_datas.append(_tmp)
    return train_datas, dev_datas


def process_cwwv():
    train_path = "../../data/CWWV/train_random.jsonl"
    train_datas = []
    dev_path = "../../data/CWWV/dev_random.jsonl"
    dev_datas = []
    for instance in jsonlines.Reader(open(train_path, 'r', encoding='utf-8')):
        context = instance['question']['stem']
        splitted_context = context.split(" ")
        if not splitted_context[-1] == '[MASK]':
            continue
        new_context = " ".join(splitted_context[:-1])
        correct = instance['answerKey']
        candidates = instance['question']['choices']
        candidates_text = [c['text'] for c in candidates]
        for i, c in enumerate(candidates):
            if c['label'] == correct:
                correct = i
                break
        _tmp = dict()
        _tmp['context'] = new_context
        _tmp['candidates'] = candidates_text
        _tmp['answer'] = correct
        train_datas.append(_tmp)
    for instance in jsonlines.Reader(open(dev_path, 'r', encoding='utf-8')):
        context = instance['question']['stem']
        splitted_context = context.split(" ")
        if not splitted_context[-1] == '[MASK]':
            continue
        new_context = " ".join(splitted_context[:-1])
        correct = instance['answerKey']
        candidates = instance['question']['choices']
        candidates_text = [c['text'] for c in candidates]
        for i, c in enumerate(candidates):
            if c['label'] == correct:
                correct = i
                break
        _tmp = dict()
        _tmp['context'] = new_context
        _tmp['candidates'] = candidates_text
        _tmp['answer'] = correct
        dev_datas.append(_tmp)
    return train_datas, dev_datas


def main():
    atomic_train_datas, atomic_dev_datas = process_atomic()
    cwwv_train_datas, cwwv_dev_datas = process_cwwv()
    concat_train_datas = atomic_train_datas + cwwv_train_datas
    concat_dev_datas = atomic_dev_datas + cwwv_dev_datas
    with open('concat_train_random.jsonl', 'w') as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(concat_train_datas)
        # json.dump(concat_train_datas, f)
        print(f"size of concat train: {len(concat_train_datas)}")
    with open('concat_dev_random.jsonl', 'w') as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(concat_dev_datas)
        # json.dump(concat_dev_datas, f)
        print(f"size of concat dev: {len(concat_dev_datas)}")


if __name__ == '__main__':
    main()
