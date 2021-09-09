# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
import json
import math

from functools import partial
import numpy as np
import paddle
import paddle.nn.functional as F
import tqdm
from paddle.io import DataLoader
from args import parse_args
import paddle.nn as nn
from paddlenlp.data import Pad, Stack, Dict
from paddlenlp.transformers import BertForQuestionAnswering, BertTokenizer, ErnieForQuestionAnswering, ErnieTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup, CosineDecayWithWarmup
from paddlenlp.transformers import FunnelTokenizerFast, FunnelForQuestionAnswering
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.datasets import load_dataset
from visualdl import LogWriter
from paddle.distributed import fleet
freeze_basemodel=False

MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "ernie": (ErnieForQuestionAnswering, ErnieTokenizer),
    "funnel": (FunnelForQuestionAnswering, FunnelTokenizerFast)

}


def prepare_train_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=args.doc_stride, return_offsets_mapping=True, max_length=args.max_seq_length,
        return_overflowing_tokens=True, return_token_type_ids=True)

    # Let's label those examples!
    for i, tokenized_example in enumerate(tokenized_examples):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_example["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = tokenized_example['offset_mapping']

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        answers = examples[sample_index]['answers']
        answer_starts = examples[sample_index]['answer_starts']

        # If no answers are given, set the cls_index as answer.
        if len(answer_starts) == 0:
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
        else:
            # Start/end character index of the answer in the text.
            start_char = answer_starts[0]
            end_char = start_char + len(answers[0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Minus one more to reach actual text
            token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[
                    token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples[i]["start_positions"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples[i]["end_positions"] = token_end_index + 1

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=args.doc_stride, return_offsets_mapping=True, max_length=args.max_seq_length,
        return_overflowing_tokens=True, return_token_type_ids=True)

    # For validation, there is no need to compute start and end positions
    for i, tokenized_example in enumerate(tokenized_examples):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        tokenized_examples[i]["example_id"] = examples[sample_index]['id']

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples[i]["offset_mapping"] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_example["offset_mapping"])
        ]

    return tokenized_examples


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, data_loader, args, global_step, max_eval_examples=None):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()

    for batch in tqdm.tqdm(data_loader):
        input_ids, token_type_ids = batch
        start_logits_tensor, end_logits_tensor = model(input_ids,
                                                       token_type_ids=token_type_ids)

        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()

            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])
        if max_eval_examples is not None:
            if len(all_start_logits) > max_eval_examples:
                break

    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits), args.version_2_with_negative,
        args.n_best_size, args.max_answer_length,
        args.null_score_diff_threshold)

    # Can also write all_nbest_json and scores_diff_json files if needed
    with open(f'{str(global_step)}_prediction.json', "w", encoding='utf-8') as writer:
        writer.write(
            json.dumps(
                all_predictions, ensure_ascii=False, indent=4) + "\n")

    out_eval = squad_evaluate(
        examples=data_loader.dataset.data,
        preds=all_predictions,
        na_probs=scores_diff_json)

    model.train()
    return out_eval


class CrossEntropyLossForSQuAD(paddle.nn.Layer):
    def __init__(self, answerable_classifier=False, answerable_uses_start_logits=False):
        super(CrossEntropyLossForSQuAD, self).__init__()
        self.answerable_classifier = answerable_classifier
        self.answerable_uses_start_logits = answerable_uses_start_logits

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(
            input=start_logits, label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(
            input=end_logits, label=end_position)
        loss = (start_loss + end_loss) / 2

        return loss


def run(args):
    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)

    paddle.set_device(args.device)

    rank = paddle.distributed.get_rank()
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if os.path.isdir(args.model_name_or_path + "/tokenizer"):
        print("load model from local folder:", args.model_name_or_path + "/tokenizer")
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path + "/tokenizer")
    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    set_seed(args)
    if rank == 0:
        if os.path.exists(args.model_name_or_path):
            print("init checkpoint from %s" % args.model_name_or_path)
    if os.path.isdir(args.model_name_or_path):
        print("load model from local folder:", args.model_name_or_path + "/model_state.pdparams")
        model = model_class.from_pretrained(args.model_name_or_path + "/model_state.pdparams",
                                            config=args.model_name_or_path + "/model_config.json")
    else:
        model = model_class.from_pretrained(args.model_name_or_path)


    log_writer = LogWriter(logdir="./log/squad_v1.1")
    if args.do_predict:
        if args.predict_file:
            dev_ds = load_dataset('squad', data_files=args.predict_file)
            log_writer = LogWriter(logdir="./log/squad_pred")
        elif args.version_2_with_negative:
            dev_ds = load_dataset('squad', splits='dev_v2')
            log_writer = LogWriter(logdir="./log/squad_v2")
        else:
            dev_ds = load_dataset('squad', splits='dev_v1')
            log_writer = LogWriter(logdir="./log/squad_v1.1")

        dev_ds.map(partial(
            prepare_validation_features, tokenizer=tokenizer, args=args),
            batched=True)
        dev_batch_sampler = paddle.io.BatchSampler(
            dev_ds, batch_size=args.batch_size * 2, shuffle=False)

        dev_batchify_fn = lambda samples, fn=Dict({
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
        }): fn(samples)

        dev_data_loader = DataLoader(
            dataset=dev_ds,
            batch_sampler=dev_batch_sampler,
            collate_fn=dev_batchify_fn,
            num_workers=4,
            return_list=True)

    if args.do_train:
        # layer_lr for base
        ############################################################
        n_layers = 12
        depth_to_slice = {}
        start = 0
        for i in range(13):
            if i == 0:
                zengliang = 5
            if i == 1:
                zengliang = 23
            depth_to_slice[i] = (start, start + zengliang)
            start += zengliang
        depth_to_slice[14] = (start, -1)

        for depth, slice in depth_to_slice.items():
            layer_lr = args.layer_lr_decay ** (n_layers + 2 - depth)
            if slice[1] == -1:
                for p in model.parameters()[slice[0]:]:
                    p.optimize_attr["learning_rate"] *= layer_lr
            else:
                for p in model.parameters()[slice[0]:slice[1]]:
                    p.optimize_attr["learning_rate"] *= layer_lr
        ############################################################
        if args.train_file:
            train_ds = load_dataset('squad', data_files=args.train_file)
        elif args.version_2_with_negative:
            train_ds = load_dataset('squad', splits='train_v2')
        else:
            train_ds = load_dataset('squad', splits='train_v1')
        train_ds.map(partial(
            prepare_train_features, tokenizer=tokenizer, args=args),
            batched=True)
        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True)

        train_batchify_fn = lambda samples, fn=Dict({
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            "start_positions": Stack(dtype="int64"),
            "end_positions": Stack(dtype="int64")
        }): fn(samples)

        train_data_loader = paddle.io.DataLoader(
            dataset=train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=train_batchify_fn,
            num_workers=4,
            return_list=True)

        num_training_steps = args.max_steps if args.max_steps > 0 else len(
            train_data_loader) * args.num_train_epochs
        num_train_epochs = math.ceil(num_training_steps /
                                     len(train_data_loader))

        if args.scheduler_type == "linear":
            lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                                 args.warmup_proportion)
        elif args.scheduler_type == "cosine":
            lr_scheduler = CosineDecayWithWarmup(args.learning_rate, num_training_steps,
                                                 args.warmup_proportion)

            # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]


        if freeze_basemodel:

            optimizer = paddle.optimizer.AdamW(
                learning_rate=lr_scheduler,
                beta1=0.9,
                beta2=0.999,
                epsilon=args.adam_epsilon,
                parameters=model.qa_outputs.parameters(),
                weight_decay=args.weight_decay,
                apply_decay_param_fun=lambda x: x in decay_params)
        else:
            optimizer = paddle.optimizer.AdamW(
                learning_rate=lr_scheduler,
                beta1=0.9,
                beta2=0.999,
                epsilon=args.adam_epsilon,
                parameters=model.parameters(),
                weight_decay=args.weight_decay,
                apply_decay_param_fun=lambda x: x in decay_params)
        optimizer = fleet.distributed_optimizer(optimizer)
        model = fleet.distributed_model(model)
        criterion = CrossEntropyLossForSQuAD()

        global_step = 0
        tic_train = time.time()
        best_exact_score = 0
        for epoch in range(num_train_epochs):
            print("epoch:", epoch)
            descriptor = tqdm.tqdm(enumerate(train_data_loader))
            for step, batch in descriptor:
                global_step += 1
                input_ids, token_type_ids, start_positions, end_positions = batch

                logits = model(
                        input_ids=input_ids, token_type_ids=token_type_ids)

                loss = criterion(logits, (start_positions, end_positions))
                descriptor.set_description("loss:%.4f" % (float(loss)))

                if global_step % args.logging_steps == 0:
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch + 1, step + 1, loss,
                           args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()

                loss.backward()
                log_writer.add_scalar(tag="train/loss", step=global_step, value=loss)

                optimizer.step()

                lr_scheduler.step()
                optimizer.clear_grad()

                if global_step % args.save_steps == args.save_steps - 1 or global_step == num_training_steps:
                    if rank == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        old_dir = os.path.join(args.output_dir,
                                               "model_%d" % (global_step - args.save_steps))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        # model_to_save.save_pretrained(output_dir)
                        # tokenizer.save_pretrained(output_dir)
                        # print('Saving checkpoint to:', output_dir)
                        # os.system("rm -rf " + old_dir)
                    if global_step == num_training_steps:
                        break

                    if args.do_predict and rank == 0:
                        print("evaluate....")
                        out_eval = evaluate(model, dev_data_loader, args, global_step, max_eval_examples=1000)
                        out_eval['global_step'] = global_step
                        json.dump(out_eval, open(output_dir + "/eval.json", 'w'))
                        log_writer.add_scalar(tag="eval/exact_first1k", step=global_step, value=out_eval['exact'])
                        log_writer.add_scalar(tag="eval/f1_first1k", step=global_step, value=out_eval['f1'])
                        output_dir = os.path.join(args.output_dir,
                                                  "best")

                        print("out_eval:", out_eval)
                        try:
                            print("debug loading " + output_dir + "/eval.json")
                            best_eval = json.load(open(output_dir + "/eval.json"))
                            print("debug best_eval ", best_eval)
                            if best_exact_score < best_eval['exact']:
                                best_exact_score = best_eval['exact']
                                print("update local best score from other trainer!")

                        except Exception as Ex:
                            print(Ex)

                        if best_exact_score < out_eval['exact']:
                            best_exact_score = out_eval['exact']

                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            print('Saving best model to:', output_dir)
                            out_eval['global_step'] = global_step
                            json.dump(out_eval, open(output_dir + "/eval.json", 'w'))
                        else:
                            output_dir = os.path.join(args.output_dir,
                                                      "best")
                            import numpy as np
                            if np.random.rand() < 0.2:
                                os.system("ls -l " + output_dir)
                                motif = model_class.from_pretrained(
                                    output_dir)  # +"/motif_state.pdparams",config=output_dir+"/model_config.json")
                                print("reload the best model")

                        print("best_exact_score:", best_exact_score)
                        if best_exact_score > 88:
                            print("successful reach high exact score:", best_exact_score)
                            return

    if args.do_predict and rank == 0:
        out_eval = evaluate(model, dev_data_loader, args, "final")


if __name__ == "__main__":
    args = parse_args()
    run(args)