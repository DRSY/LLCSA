import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import jsonlines
import logging
from config import special_tokens, parse_args, MODEL_NAME, MODEL_META_CLASS, init_logging
from data import CSDataset, Collatefn, RandomSampler, DataLoader, Collatefn_fast, SequentialSampler
from memory import EpisodicMemory
from tqdm import trange, tqdm
import os

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(args, model, tokenizer, global_steps):
    """
    :param args:
    :param model:
    :param tokenizer:
    :return:
    """
    device = torch.device(args.device)
    model.eval()
    dev_dataset = CSDataset(args.dev_data)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.dev_batch_size,
                                collate_fn=Collatefn_fast(tokenizer), num_workers=2)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dev_dataset))
    logger.info("  Batch size = %d", args.dev_batch_size)
    loss_func = nn.MultiMarginLoss(margin=args.margin)
    cross_xnt = nn.CrossEntropyLoss(reduction='none')
    all_loss = 0.0
    all_cnt = 0
    all_right_cnt = 0
    for batch in tqdm(dev_dataloader, total=len(dev_dataloader)):
        lm_input_dict, lm_labels, qa_input_dict, qa_labels, margin_dict, margin_labels, margin_cnt = batch
        lm_input_dict = lm_input_dict.to(device)
        lm_labels = lm_labels.to(device)
        qa_input_dict = qa_input_dict.to(device)
        qa_labels = qa_labels.to(device)

        # lm loss
        lm_output = model(**lm_input_dict, labels=lm_labels)
        lm_loss = lm_output.loss

        # qa loss
        qa_output = model(**qa_input_dict, labels=qa_labels)
        qa_loss = qa_output.loss

        # margin ranking loss
        margin_dict = margin_dict.to(device)
        margin_labels = margin_labels.to(device)
        margin_cnt = margin_cnt.to(device)
        margin_output = model(**margin_dict)
        logits = margin_output.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = margin_labels[..., 1:].contiguous()
        loss = cross_xnt(shift_logits.transpose(
            1, 2).contiguous(), shift_labels)
        loss = loss.sum(dim=-1) / margin_cnt
        loss = (-1) * loss
        loss = loss.reshape(lm_labels.size(0), 3)
        multiplechoice_prediction = torch.argmax(loss, dim=-1)
        right_cnt = (multiplechoice_prediction == 2).sum().item()
        all_right_cnt += right_cnt
        margin_ranking_loss = loss_func(
            loss, torch.ones(loss.size(0)).long().to(device) * 2)

        total_loss = qa_loss + args.alpha * lm_loss + args.beta * margin_ranking_loss
        all_loss += total_loss.item()
        all_cnt += lm_labels.size(0)
    acc = all_right_cnt / all_cnt
    avg_loss = all_loss / all_cnt
    logger.info("global steps: {}, Acc: {}, Avg_loss:{}".format(
        global_steps, acc, avg_loss))
    return {'acc': acc, 'avg_loss': avg_loss}


def train(args, model, tokenizer):
    """
    :param args:
    :param dataset:
    :param model:
    :param tokenizer:
    :param eval_dataset:
    :return:
    """
    device = torch.device(args.device)
    model.to(device)
    train_dataset = CSDataset(args.train_data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=Collatefn_fast(tokenizer), num_workers=2)

    if args.max_steps is not None and args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.accumulate_grad_batches) + 1
    else:
        t_total = len(
            train_dataloader) // args.accumulate_grad_batches * args.max_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = int(args.warmup * t_total)
    logger.info("warm up steps = %d", warmup_steps)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr, betas=(0.9, 0.98))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # episodic memory
    if args.memory:
        memory = EpisodicMemory(tokenizer)
    else:
        memory = None

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.max_epochs)
    logger.info("  Gradient Accumulation steps = %d",
                args.accumulate_grad_batches)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Episode Memory = {}".format(memory))

    seed_everything(args.seed)
    train_iterator = trange(int(args.max_epochs), desc="Epoch")
    loss_func = nn.MultiMarginLoss(margin=args.margin, reduction='none')
    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_loss = 0.0
    best_acc = -1
    best_steps = 0
    total_n_input = 0
    cross_xnt = nn.CrossEntropyLoss(reduction='none')
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # lm_input_dict, lm_labels, qa_input_dict, qa_labels, margin_dicts = batch
            lm_input_dict, lm_labels, qa_input_dict, qa_labels, margin_dict, margin_labels, margin_cnt = batch
            total_n_input += lm_labels.size(0)
            lm_input_dict = lm_input_dict.to(device)
            lm_labels = lm_labels.to(device)
            qa_input_dict = qa_input_dict.to(device)
            qa_labels = qa_labels.to(device)

            # lm loss
            lm_output = model(**lm_input_dict, labels=lm_labels)
            lm_loss = lm_output.loss

            # qa loss
            qa_output = model(**qa_input_dict, labels=qa_labels)
            qa_loss = qa_output.loss

            # margin ranking loss
            margin_dict = margin_dict.to(device)
            margin_labels = margin_labels.to(device)
            margin_cnt = margin_cnt.to(device)
            margin_output = model(**margin_dict)
            logits = margin_output.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = margin_labels[..., 1:].contiguous()
            loss = cross_xnt(shift_logits.transpose(
                1, 2).contiguous(), shift_labels)
            loss = loss.sum(dim=-1) / margin_cnt
            loss = (-1) * loss
            loss = loss.reshape(lm_labels.size(0), 3)
            margin_ranking_loss = loss_func(
                loss, torch.ones(loss.size(0)).long().to(device) * 2)
            _id_max_margin_loss = torch.argmax(
                margin_ranking_loss, dim=-1).item()
            mean_margin_loss = margin_ranking_loss.mean(dim=-1)

            total_loss = qa_loss + args.alpha * lm_loss + args.beta * mean_margin_loss

            # add to memory
            if args.memory and memory is not None and (step + 1) % 50 == 0:
                added_ = []
                _tmp = (lm_input_dict['input_ids'][_id_max_margin_loss].tolist(), lm_labels[_id_max_margin_loss].tolist(), qa_input_dict['input_ids'][_id_max_margin_loss].tolist(), qa_labels[_id_max_margin_loss].tolist(), [
                        margin_dict['input_ids'][i].tolist() for i in range(_id_max_margin_loss, _id_max_margin_loss+3)], [margin_labels[i].tolist() for i in range(_id_max_margin_loss, _id_max_margin_loss+3)], [margin_cnt[i].item() for i in range(_id_max_margin_loss, _id_max_margin_loss+3)])
                added_.append(_tmp)
                memory.write(added_)
                logger.info("memory updated {}".format(len(memory)))

            # sparse experience replay
            if args.replay_interval > 0 and (step+1) % args.replay_interval == 0:
                lm_input_dict, lm_labels, qa_input_dict, qa_labels, margin_dict, margin_labels, margin_cnt = memory.sample(
                    total_n_input // (step + 1))
                lm_input_dict = lm_input_dict.to(device)
                lm_labels = lm_labels.to(device)
                qa_input_dict = qa_input_dict.to(device)
                qa_labels = qa_labels.to(device)

                # lm loss
                lm_output = model(**lm_input_dict, labels=lm_labels)
                _lm_loss = lm_output.loss

                # qa loss
                qa_output = model(**qa_input_dict, labels=qa_labels)
                _qa_loss = qa_output.loss

                # margin ranking loss
                margin_dict = margin_dict.to(device)
                margin_labels = margin_labels.to(device)
                margin_cnt = margin_cnt.to(device)
                margin_output = model(**margin_dict)
                logits = margin_output.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = margin_labels[..., 1:].contiguous()
                loss = cross_xnt(shift_logits.transpose(
                    1, 2).contiguous(), shift_labels)
                loss = loss.sum(dim=-1) / margin_cnt
                loss = (-1) * loss
                loss = loss.reshape(lm_labels.size(0), 3)
                _margin_ranking_loss = loss_func(
                    loss, torch.ones(loss.size(0)).long().to(device) * 2).mean(dim=-1)
                _total_loss = _qa_loss + args.alpha * _lm_loss + args.beta * _margin_ranking_loss
                logger.info("sparse experience replay done: {}".format(
                    _total_loss.item()))

            if args.accumulate_grad_batches > 0:
                total_loss = total_loss / args.accumulate_grad_batches
            total_loss.backward()
            tr_loss += total_loss.item()
            if (step + 1) % args.accumulate_grad_batches == 0:
                global_steps += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if args.logging_steps > 0 and global_steps % args.logging_steps == 0:
                    epoch_iterator.set_description(" global_step = {}, average loss = {}".format(global_steps, (
                        tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss
                if args.eval_steps > 0 and global_steps % args.eval_steps == 0:
                    torch.cuda.empty_cache()
                    # eval
                    eval_result = evaluate(
                        args, model, tokenizer, global_steps)
                    if eval_result['avg_loss'] < best_loss or eval_result['acc'] > best_acc:
                        best_loss = eval_result['avg_loss']
                        best_acc = eval_result['acc']
                        best_steps = global_steps
                        # save model checkpoint
                        model.save_pretrained(args.output_dir)
                        logger.info("model saved at {} global steps with acc {}".format(
                            global_steps, best_acc))


def main(args):
    if not args.model_type in MODEL_META_CLASS.keys():
        raise Exception("Invalid model type")
    tokenizer_class, model_class = MODEL_META_CLASS[args.model_type]
    model_name = MODEL_NAME[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(model_name)
    logger.info("original vocab size: {}".format(len(tokenizer)))
    model = model_class.from_pretrained(model_name)

    # add special tokens
    tokenizer.add_special_tokens(special_tokens())
    tokenizer.add_special_tokens({
        'pad_token': '<PAD>'
    })
    logger.info("new vocab size: {}".format(len(tokenizer)))
    model.resize_token_embeddings(len(tokenizer))

    train(args, model, tokenizer)


if __name__ == '__main__':
    args = parse_args()
    init_logging(os.path.join(args.output_dir, "log_train.txt"))
    logger.info("args = {}".format(str(args)))
    main(args)
