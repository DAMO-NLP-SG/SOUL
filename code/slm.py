import argparse
import json
import logging
import os
import re
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import (AdamW, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

label2id = { "false": 0, "true": 1, "not-given": 2 }
id2label = {v: k for k, v in label2id.items()}
logger = logging.getLogger()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare_logger(args):
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    formatter = logFormatter = logging.Formatter(fmt='[%(asctime)s - %(name)s:%(lineno)d]: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    log_file = os.path.join(args.output_dir, "run.log")
    file_handler = logging.FileHandler(log_file, mode="w", encoding=None, delay=False)
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.handlers = [console_handler, file_handler]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="zero-shot", help="[zero-shot")
    parser.add_argument("--seed", type=int, default=42, help="[0, 1, 42]")
    parser.add_argument("--nrows", type=int, default=None, help="number of rows")
    parser.add_argument("--model", type=str, default="t5-base")
    parser.add_argument("--out_dir_name", type=str, default="slm", help="name of the output dir name")
    parser.add_argument("--name", type=str, default=None, help="name of output folder")
    parser.add_argument("--device", type=str, default="cuda", help="[cuda, cpu]")
    parser.add_argument("--use_justification", action="store_true", help="true - jg task; false - rc task")
    parser.add_argument("--do_eval", action="store_true", help="input with just")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--test_bs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=512)
    return parser.parse_args()


def postprocess_args(args):
    # create output folder
    model_name = args.model.split('/')[-1]
    if args.name:
        args.output_dir = f"./output/{args.out_dir_name}/{args.name}/seed{args.seed}"
    else:
        args.output_dir = f"./output/{args.out_dir_name}/{model_name}_seq{args.max_len}_e{args.epochs}_lr{args.lr}_bs{args.bs}/seed{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)

    prepare_logger(args)
    args.is_generative = True if "t5" in args.model else False
    if not args.is_generative and args.use_justification:
        raise ValueError("Justification can only be used in generative models.")
    if args.use_justification:
        args.test_bs = int(args.test_bs / 4)
        logger.info("Output justification needs to reduce test size to prevent CUDA OOM.")
        logger.info(f"Test batch size reduced to {args.test_bs}")

    logger.info(args)

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):

    def __init__(self, args, tokenizer, reviews, states, types, justs, split=None):
        self.args = args
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.split = split

        self.reviews, self.states, self.types, self.justs = reviews.tolist(), states.tolist(), types.tolist(), justs.tolist()
        assert len(self.reviews) == len(self.states) == len(self.types) == len(self.justs)
        if self.args.is_generative:
            self.inputs_tensor_list, self.targets_tensor_list = self.encode(self.reviews, self.states, self.types, self.justs)

        self.get_longest_seq()

    def get_longest_seq(self):
        self.longest_seq = 0
        self.exceed_num = 0
        for i in range(len(self.reviews)):
            seq_len = len(self.tokenizer(self.reviews[i])["input_ids"]) + len(self.tokenizer(self.states[i])["input_ids"])
            self.longest_seq = max(self.longest_seq, seq_len)
            if seq_len > 500:
                self.exceed_num += 1
        logger.info(f"Longest seq length is {self.longest_seq}")
        logger.info(f"Seq longer than 500: {self.exceed_num}")

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        if self.args.is_generative:
            source_ids = self.inputs_tensor_list[index]["input_ids"].squeeze()
            target_ids = self.targets_tensor_list[index]["input_ids"].squeeze()

            src_mask = self.inputs_tensor_list[index]["attention_mask"].squeeze()
            target_mask = self.targets_tensor_list[index]["attention_mask"].squeeze()

            return {"source_ids": source_ids, "source_mask": src_mask,
                    "target_ids": target_ids, "target_mask": target_mask}
        else:
            review = self.reviews[index]
            state = self.states[index]
            label = torch.tensor(label2id[self.types[index]])

            encoded_dict = self.tokenizer.encode_plus(
                review,
                state,  # Sentences to encode.
                add_special_tokens=True,  # Add the special tokens.
                max_length=self.max_len,  # Pad & truncate all sentences.
                truncation=True,
                padding='max_length',
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            padded_token_list = encoded_dict['input_ids'][0]
            att_mask = encoded_dict['attention_mask'][0]
            return {
                "source_ids": padded_token_list, "source_mask": att_mask, "target_ids": label
            }


    def encode(self, reviews, states, types, justs):

        inputs_tensor_list, targets_tensor_list = [], []
        input_text, target_text = [], []

        for i in range(len(reviews)):

            input_i = f"Review:\n{reviews[i]}\n\nStatement:\n{states[i]}"
            target_i = types[i]
            if self.args.use_justification:
                target_i = f"Label:\n{types[i]}\nJustification:\n{justs[i]}"

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input_i], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt",
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target_i], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt"
            )

            inputs_tensor_list.append(tokenized_input)
            targets_tensor_list.append(tokenized_target)
            input_text.append(input_i)
            target_text.append(target_i)

        df = pd.DataFrame({"input": input_text, "target": target_text})
        data_path = os.path.join(self.args.output_dir, f"{self.split}_text.csv")
        df.to_csv(data_path, index=True)
        logger.info(f"{self.split} text saved to {data_path}")

        return inputs_tensor_list, targets_tensor_list


def get_model(args):
    logger.info("Loading Model")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if args.is_generative:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(label2id), return_dict=False)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    return model, tokenizer, optimizer


def get_data(args, tokenizer, train_df, dev_df, test_df):
    train_dataset = TextDataset(args, tokenizer, train_df["review_text"], train_df["statement"], train_df["label"], train_df["justification"], "train")
    dev_dataset = TextDataset(args, tokenizer, dev_df["review_text"], dev_df["statement"], dev_df["label"], dev_df["justification"], "dev")
    test_dataset = TextDataset(args, tokenizer, test_df["review_text"], test_df["statement"], test_df["label"], test_df["justification"], "test")

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, drop_last=False, shuffle=True, num_workers=4)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.bs, drop_last=False, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_bs, drop_last=False, shuffle=False, num_workers=4)

    return train_dataloader, dev_dataloader, test_dataloader


def train(args, tokenizer, model, optimizer, train_dataloader, dev_dataloader):
    epoch_iterator = trange(int(args.epochs), dynamic_ncols=True, desc=f"Epoch")
    for n_epoch, _ in enumerate(epoch_iterator):
        epoch_train_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, dynamic_ncols=True, desc=f"Epoch {n_epoch} Training")):
            model.train()

            if n_epoch == step == 0:
                logger.info(tokenizer.decode(batch["source_ids"][0], skip_special_tokens=False))
                if args.is_generative:
                    logger.info(tokenizer.decode(batch["target_ids"][0], skip_special_tokens=True))
                else:
                    logger.info(batch["target_ids"][0])

            if args.is_generative:

                lm_labels = batch["target_ids"]
                lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

                outputs = model(
                    batch["source_ids"].to(args.device),
                    attention_mask=batch["source_mask"].to(args.device),
                    labels=lm_labels.to(args.device),
                    decoder_attention_mask=batch['target_mask'].to(args.device),
                    decoder_input_ids=None,
                )
            else:
                input_ids = batch["source_ids"].to(args.device)
                input_mask = batch["source_mask"].to(args.device)
                labels = batch["target_ids"].to(args.device)
                outputs = model(input_ids, attention_mask=input_mask, labels=labels)

            loss = outputs[0]
            loss.backward()
            epoch_train_loss += loss.item()

            optimizer.step()
            model.zero_grad()
        logger.info(f"Epoch {n_epoch} Avg epoch train loss: {epoch_train_loss / len(train_dataloader):.5f}")


def inference(args, model, dataloader, df, tokenizer):
    logger.info("Testing...")
    model.eval()
    with torch.no_grad():
        inputs, predictions, labels = [], [], []
        for batch in tqdm(dataloader):
            if args.is_generative:
                outs_dict = model.generate(
                    input_ids=batch['source_ids'].to(args.device),
                    attention_mask=batch['source_mask'].to(args.device),
                    max_length=args.max_len,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                outs = outs_dict["sequences"]
                batch_prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            else:
                outputs = model(
                    input_ids=batch["source_ids"].to(args.device),
                    attention_mask=batch["source_mask"].to(args.device),
                    labels=batch["target_ids"].to(args.device)
                )
                logits = outputs[1]
                outs = torch.argmax(logits, dim=1)
                batch_prediction = [id2label[int(i)] for i in outs]
            batch_input = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["source_ids"]]
            batch_label = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

            inputs.extend(batch_input)
            predictions.extend(batch_prediction)
            labels.extend(batch_label)

    return predictions


def get_prediction(x):
    match = re.search("(true|false|not-given|not given)", x.lower().strip())
    if match:
        return match.group().replace("not given", "not-given")
    else:
        return "error"


def save_result(args, outputs, df, split="test"):

    df["prediction"] = outputs
    output_path = os.path.join(args.output_dir, f"{split}.csv")
    df.to_csv(output_path, index=False)

    all_acc = accuracy_score(df["label"], df["prediction"])
    correct_num = len(df[df["label"] == df["prediction"]])
    with open(os.path.join(args.output_dir, f"{split}.txt"), 'w') as f:
        f.write(f"{split} all: {correct_num}/{len(df)} {all_acc:.4%}\n")
        logger.info(f"{split} all: {correct_num}/{len(df)} {all_acc:.4%}\n")


def process(args):
    train_df = pd.read_csv("./data/train.csv")[:args.nrows]
    dev_df = pd.read_csv("./data/dev.csv")[:args.nrows]
    test_df = pd.read_csv("./data/test.csv")[:args.nrows]

    model, tokenizer, optimizer = get_model(args)
    train_dataloader, dev_dataloader, test_dataloader = get_data(args, tokenizer, train_df, dev_df, test_df)
    logger.info(f"train: {len(train_dataloader.dataset)}")
    logger.info(f"dev: {len(dev_dataloader.dataset)}")
    logger.info(f"test: {len(test_dataloader.dataset)}")

    train(args, tokenizer, model, optimizer, train_dataloader, dev_dataloader)

    model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"Model saved to {model_dir}")
    logger.info(f"Tokenizer saved to {model_dir}")

    if args.do_eval:
        dev_outputs = inference(args, model, dev_dataloader, dev_df, tokenizer)
        save_result(args, dev_outputs, dev_df, split="dev")

    test_outputs = inference(args, model, test_dataloader, test_df, tokenizer)
    save_result(args, test_outputs, test_df, split="test")


def main():
    args = parse_args()
    args = postprocess_args(args)
    set_seed(args.seed)
    process(args)

if __name__ == "__main__":
    main()
