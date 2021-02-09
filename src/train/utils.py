import torch
import torch.nn as nn
from config import EOS, GEN, ANS
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def generate_sample(model, tokenizer, n_samples):
    """
    generate pseudo sample of previously seen examples
    """
    prefix = GEN
    ret = []
    input_ids = tokenizer(
        prefix, add_special_tokens=False, return_tensors='pt')['input_ids']
    outputs = model.generate(input_ids, max_length=25, min_length=6, do_sample=True,
                             top_p=0.9, top_k=0, temperature=0.7, repetition_penalty=1.0, num_return_sequences=n_samples)
    for i, sequence in enumerate(outputs):
        sequence = sequence.tolist()
        decoded_sequence = tokenizer.decode(sequence, clean_up_tokenization_spaces=True)
        if not ANS in decoded_sequence:
            continue
        if not EOS in decoded_sequence:
            continue
        decoded_sequence = decoded_sequence[:decoded_sequence.find(EOS)+1]
        ret.append(decoded_sequence)
    return ret


def generate_answer(model, tokenizer, n):
    """
    generate a textual continuation as answer for questions like 'context <ANS> '
    """
    pass


if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    generate_sample(model, tok, n_samples=2)
