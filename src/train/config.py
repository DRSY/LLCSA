import argparse
from pytorch_lightning import Trainer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging
import datetime

logger = logging.getLogger(__name__)

MODEL_NAME = {
    'gpt2': 'gpt2'
}

MODEL_META_CLASS = {
    'gpt2': (GPT2Tokenizer, GPT2LMHeadModel)
}

# special tokens
ANS = '<ANS>'
GEN = '<GEN>'
EOS = '<EOS>'


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = record.relativeCreated / 1000 - last / 1000
        record.relative = "{:.1f}".format(delta)
        record.uptime = str(datetime.timedelta(
            seconds=record.relativeCreated // 1000))
        self.last = record.relativeCreated
        return True


def init_logging(filename):
    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format,
                        filename=filename, filemode='a', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())


def parse_args():
    parser = argparse.ArgumentParser("Lifelong commonsense acquisition")
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--dev_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str,
                        default='gpt2', required=True)
    parser.add_argument("--model_name", type=str,
                        default='gpt2', required=True)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--dev_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--memory", action='store_true', default=False)
    parser.add_argument("--replay_interval", type=int, default=150)
    args = parser.parse_args()
    return args


def special_tokens():
    _special_tokens = {
        'additional_special_tokens': [ANS, GEN, EOS]
    }
    return _special_tokens
