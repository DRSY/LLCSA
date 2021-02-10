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
from utils import generate_sample, generate_answer
from tqdm import trange, tqdm
from copy import deepcopy
import os

import optuna

logger = logging.getLogger(__name__)
KL_func = nn.KLDivLoss(log_target=True)
robust_kl_loss_fct = nn.KLDivLoss(reduction='batchmean')


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
    logger.info("global steps: {}, Dev accuracy: {}, Dev avg_loss:{}".format(
        global_steps, acc, avg_loss))
    return {'acc': acc, 'avg_loss': avg_loss}


def eval_loss_on_old_model(args, model, old_state_dict, batch, device, cross_xnt, loss_func):
    # reload model to previous parameters
    model.load_state_dict(old_state_dict)
    model.eval()
    with torch.no_grad():
        # normal continual training over stream of data
        lm_input_dict, lm_labels, qa_input_dict, qa_labels, margin_dict, margin_labels, margin_cnt = batch
        lm_input_dict = lm_input_dict.to(device)
        lm_labels = lm_labels.to(device)
        qa_input_dict = qa_input_dict.to(device)
        qa_labels = qa_labels.to(device)

        # lm loss
        lm_output = model(**lm_input_dict, labels=lm_labels)
        lm_loss = lm_output.loss
        lm_logits = lm_output.logits

        # qa loss
        qa_output = model(**qa_input_dict, labels=qa_labels)
        qa_loss = qa_output.loss
        qa_logits = qa_output.logits

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
        margin_logits = loss

        # total_loss = qa_loss + args.alpha * lm_loss + args.beta * mean_margin_loss
        total_loss = args.beta * mean_margin_loss
        if args.lm_loss:
            total_loss = total_loss + args.alpha * lm_loss
        if args.qa_loss:
            total_loss = total_loss + qa_loss
    model.train()
    return total_loss, (lm_logits, qa_logits, margin_logits)


def robust_kl_loss(tokenizer, args, old_logits, new_logits, lm_input_ids, qa_labels):
    temperature = 2.0
    old_lm_logits, old_qa_logits, old_margin_logits = old_logits
    vocab_size = old_lm_logits.size(-1)
    bs = old_lm_logits.size(0)
    new_lm_logits, new_qa_logits, new_margin_logits = new_logits
    lm_mask = (lm_input_ids != tokenizer.pad_token_id)
    selected_old_lm_logits = torch.softmax(torch.masked_select(
        old_lm_logits, lm_mask.unsqueeze(-1)).reshape(-1, vocab_size) / temperature, dim=-1)
    selected_new_lm_logits = torch.log_softmax(torch.masked_select(
        new_lm_logits, lm_mask.unsqueeze(-1)).reshape(-1, vocab_size) / temperature, dim=-1)
    qa_mask = (qa_labels != -100)
    for i in range(bs):
        for j in range(qa_mask.size(-1)):
            if qa_mask[i, j] == True and j > 0:
                qa_mask[i, j-1] = True
                break
    selected_old_qa_logits = torch.softmax(torch.masked_select(
        old_qa_logits, qa_mask.unsqueeze(-1)).reshape(-1, vocab_size) / temperature, dim=-1)
    selected_new_qa_logits = torch.log_softmax(torch.masked_select(
        new_qa_logits, qa_mask.unsqueeze(-1)).reshape(-1, vocab_size) / temperature, dim=-1)
    lm_kl_loss = robust_kl_loss_fct(
        selected_new_lm_logits, selected_old_lm_logits)
    qa_kl_loss = robust_kl_loss_fct(
        selected_new_qa_logits, selected_old_qa_logits)
    margin_target_logits = torch.softmax(old_margin_logits, dim=-1)
    margin_new_logits = torch.log_softmax(new_margin_logits, dim=-1)
    margin_kl_loss = robust_kl_loss_fct(
        margin_new_logits, margin_target_logits)
    _loss = args.beta * margin_kl_loss
    if args.lm_loss:
        _loss = _loss + args.alpha * lm_kl_loss
    if args.qa_loss:
        _loss = _loss + qa_kl_loss
    # _loss = qa_kl_loss + args.alpha * lm_kl_loss + args.beta * margin_kl_loss
    return _loss


def train(trial, args, model, tokenizer):
    suggested_kl_interval = trial.suggest_categorical(
        "kl_interval", [20, 30, 40, 60])
    suggested_replay_interval = trial.suggest_categorical(
        "replay_interval", [50, 100, 150, 200])
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
    if args.lm_loss:
        logger.info("Use LM loss")
    else:
        logger.info("Use no LM loss")
    if args.qa_loss:
        logger.info("Use QA loss")
    else:
        logger.info("Use no QA loss")
    if args.meta_replay:
        logger.info("Use meta replay")
    else:
        logger.info("Use no meta replay")
    if args.kl:
        logger.info("Use KL distillation")
    else:
        logger.info("Use no KL distillation")

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
    previous_state_dict = deepcopy(
        model.state_dict())  # previous model parameters
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            original_model_state_dict = deepcopy(model.state_dict())

            # normal continual training over stream of data
            lm_input_dict, lm_labels, qa_input_dict, qa_labels, margin_dict, margin_labels, margin_cnt = batch
            total_n_input += lm_labels.size(0)
            lm_input_dict = lm_input_dict.to(device)
            lm_labels = lm_labels.to(device)
            qa_input_dict = qa_input_dict.to(device)
            qa_labels = qa_labels.to(device)

            # language modeling loss
            lm_output = model(**lm_input_dict, labels=lm_labels)
            lm_loss = lm_output.loss
            lm_logits = lm_output.logits

            # qa loss
            qa_output = model(**qa_input_dict, labels=qa_labels)
            qa_loss = qa_output.loss
            qa_logits = qa_output.logits

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
            margin_logits = loss

            # total loss
            # total_loss = qa_loss + args.alpha * lm_loss + args.beta * mean_margin_loss
            total_loss = args.beta * mean_margin_loss
            if args.lm_loss:
                total_loss = total_loss + args.alpha * lm_loss
            if args.qa_loss:
                total_loss = total_loss + qa_loss

            # write to memory
            if args.memory and memory is not None and (step + 1) % args.write_interval == 0:
                added_ = []
                _tmp = (lm_input_dict['input_ids'][_id_max_margin_loss].tolist(), lm_labels[_id_max_margin_loss].tolist(), qa_input_dict['input_ids'][_id_max_margin_loss].tolist(), qa_labels[_id_max_margin_loss].tolist(), [
                        margin_dict['input_ids'][i].tolist() for i in range(_id_max_margin_loss, _id_max_margin_loss+3)], [margin_labels[i].tolist() for i in range(_id_max_margin_loss, _id_max_margin_loss+3)], [margin_cnt[i].item() for i in range(_id_max_margin_loss, _id_max_margin_loss+3)])
                added_.append(_tmp)
                memory.write(added_)

            if args.accumulate_grad_batches > 0:
                total_loss = total_loss / args.accumulate_grad_batches
            total_loss.backward(retain_graph=True)
            tr_loss += total_loss.item()
            if (step + 1) % args.accumulate_grad_batches == 0:
                global_steps += 1
                # eval loss on previous model parameters and encourage positive forward transfer
                # if args.kl and global_steps >= 1 and global_steps % args.kl_interval == 0:
                if args.kl and global_steps >= 1 and global_steps % suggested_kl_interval == 0:
                    _model = deepcopy(model)
                    _model.to(model.device)
                    loss_old_model, old_logits = eval_loss_on_old_model(args,
                                                                        _model, previous_state_dict, batch, device, cross_xnt, loss_func)
                    if loss_old_model >= (total_loss.item() * args.accumulate_grad_batches):
                        logger.info("move forward")
                    else:
                        kl_loss = robust_kl_loss(tokenizer, args, old_logits, (
                            lm_logits, qa_logits, margin_logits), lm_input_dict['input_ids'], qa_labels)
                        kl_loss.backward()
                        logger.info("KL loss: {}".format(kl_loss.item()))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                del total_loss, qa_loss, lm_loss, mean_margin_loss

                # sparse meta-replay using previously seen unconfident examples to alleviate catastrophic forgetting
                # if args.memory and memory is not None and args.replay_interval > 0 and (global_steps) % args.replay_interval == 0:
                if args.memory and memory is not None and suggested_replay_interval > 0 and (global_steps) % suggested_replay_interval == 0:
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
                    # _total_loss = _qa_loss + args.alpha * _lm_loss + args.beta * _margin_ranking_loss
                    _total_loss = args.beta * _margin_ranking_loss
                    if args.lm_loss:
                        _total_loss = _total_loss + args.alpha * _lm_loss
                    if args.qa_loss:
                        _total_loss = _total_loss + _qa_loss
                    _total_loss.backward()
                    # first-order MAML update upon gpt2
                    if args.meta_replay:
                        model.load_state_dict(original_model_state_dict)
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.info("sparse experience replay done, loss: {}, currrent memory size: {}".format(
                        _total_loss.item(), len(memory)))

                previous_state_dict = original_model_state_dict
                # experience replay using pesudo samples generated by model itself
                if args.pesudo_replay and (global_steps) % args.pesudo_replay_interval == 0:
                    pass

                # logging
                if args.logging_steps > 0 and global_steps % args.logging_steps == 0:
                    epoch_iterator.set_description(" global_step = {}, average loss = {}".format(global_steps, (
                        tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

                # eval
                if args.eval_steps > 0 and global_steps % args.eval_steps == 0:
                    torch.cuda.empty_cache()
                    eval_result = evaluate(
                        args, model, tokenizer, global_steps)
                    if eval_result['avg_loss'] < best_loss or eval_result['acc'] > best_acc:
                        best_loss = eval_result['avg_loss']
                        best_acc = eval_result['acc']
                        best_steps = global_steps
                        # save model checkpoint and tokenizer
                        model.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        logger.info("model saved at {} global steps with acc {}".format(
                            global_steps, best_acc))
    return best_acc


def main(trial):
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

    best_acc = train(trial, args, model, tokenizer)
    return best_acc


if __name__ == '__main__':
    args = parse_args()
    init_logging(os.path.join(args.output_dir, "log_train_right_kd.txt"))
    logger.info("args = {}".format(str(args)))
    study = optuna.create_study(direction='maximize')
    study.optimize(main, n_trials=16)
    # main(args)
