# -*- coding: utf-8 -*-
"""
Sentence Transformer training based on
https://towardsdatascience.com/a-complete-guide-to-transfer-learning-from-english-to-other-languages-using-sentence-embeddings-8c427f8804a9
"""


import argparse
from torch.utils.data import DataLoader
from sentence_transformers import(
        SentenceTransformer,
        models,
        losses,
        evaluation,
        readers,
        )
from sentence_transformers.datasets import ParallelSentencesDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path",
            help="Specify the path to the training data")
    parser.add_argument("out_path",
            help="Specify output directory")
    parser.add_argument("--teacher",
            default='bert-base-nli-stsb-mean-tokens',
            help="Specify teacher model name. "
            "Defaults to bert-base-nli-stsb-mean-tokens")
    parser.add_argument("--student",
            default="xlm-roberta-base",
            help="Specify student model name. "
            "Defaults to xlm-roberta-base")
    parser.add_argument("--max-seq-length",
            dest="max_seq_len", type=int,
            help="Specify the maximum sequence length of "
            "the student model. If None, it will be set "
            "to the same value as in the teacher model")
    parser.add_argument("--batch-size", dest="batch_size",
            type=int, default=32,
            help="Specify training batch size. "
            "Defaults to 32")
    parser.add_argument("--epochs",
            type=int, default=4,
            help="Specify the number of epochs. "
            "Defaults to 4")
    parser.add_argument("--checkpoint-path",
            dest="checkpoint_path",
            help="Specify the path to the "
            "checkpoint directory")
    parser.add_argument("--checkpoint-steps",
            dest="checkpoint_steps", type=int, default=1000,
            help="Specify checkpoint step size. "
            "Defaults to 1000")
    parser.add_argument("--checkpoint-limit",
            dest="checkpoint_limit", type=int, default=1,
            help="Specify the maximal number of "
            "checkpoints to be saved. Defaults to 1")
    parser.add_argument("--eval-data", dest="eval_data",
            type=argparse.FileType("r", encoding='utf-8'),
            help="Specify path to the evaluation data")
    parser.add_argument("--eval-steps", dest="eval_steps",
            type=int, default=1000,
            help="Specify evaluation step size. "
            "Defaults to 1000")
    args = parser.parse_args()
    return args


def group_args(args):
    load_model_args = {
            "teacher_name": args.teacher,
            "student_name": args.student,
            "max_seq_len": args.max_seq_len,
            }
    train_data_args = {
            "dataset_path": args.train_data_path,
            "train_batch_size": args.batch_size
            }
    train_model_args = {
            "eval_steps": args.eval_steps,
            "num_epochs": args.epochs,
            "output_path": args.out_path,
            "checkpoint_path": args.checkpoint_path,
            "checkpoint_steps": args.checkpoint_steps,
            "checkpoint_limit": args.checkpoint_limit
            } 
    return load_model_args, train_data_args, train_model_args


def load_models(teacher_name, student_name, max_seq_len):
    teacher_model = SentenceTransformer(teacher_name)
    if max_seq_len is None:
        max_seq_len = teacher_model.max_seq_length
    word_embedding_model = models.Transformer(student_name,
            max_seq_length=max_seq_len)
    pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
            )
    student_model = SentenceTransformer(modules=[word_embedding_model,
        pooling_model])
    return teacher_model, student_model 


def load_train_set(teacher_model, student_model,
        dataset_path, train_batch_size):
    train_reader = ParallelSentencesDataset(
            student_model=student_model,
            teacher_model=teacher_model
            )
    train_reader.load_data(dataset_path)
    train_dataloader = DataLoader(
            train_reader,
            shuffle=True,
            batch_size=train_batch_size
            )
    train_loss = losses.MSELoss(model=student_model)
    return train_dataloader, train_loss


def load_data_for_MSE_eval(eval_source, sep='\t'):
    source_sents = []
    target_sents = []
    for line in map(str.strip, eval_source):
        source_sent, target_sent = line.split(sep)
        source_sents.append(source_sent)
        target_sents.append(target_sent)
    return source_sents, target_sents


def get_MSE_evaluator(source_sents, target_sents,
        teacher_model, batch_size, name):
    evaluator = evaluation.MSEEvaluator(
            source_sentences=source_sents,
            target_sentences=target_sents,
            teacher_model=teacher_model,
            batch_size=batch_size,
            name=name
            )
    return evaluator


def train_model(student_model, train_dataloader, train_loss,
        evaluator, eval_steps, num_epochs, output_path,
        checkpoint_path, checkpoint_steps, checkpoint_limit):
    student_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=eval_steps,
        warmup_steps=10000,
        scheduler='warmupconstant',
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        checkpoint_save_steps=checkpoint_steps,
        checkpoint_save_total_limit=checkpoint_limit,
        optimizer_params= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
        )


def main():
    args = get_args()
    load_model_args, train_data_args, train_model_args = group_args(args)
    eval_data = args.eval_data
    eval_batch_size = args.batch_size
    del args
    teacher_model, student_model = load_models(**load_model_args)
    train_dataloader, train_loss = load_train_set(teacher_model,
            student_model, **train_data_args)
    if eval_data is not None:
        eval_sources, eval_targets = load_data_for_MSE_eval(eval_data)
        evaluator = get_MSE_evaluator(eval_sources, eval_targets,
                teacher_model, eval_batch_size, "MSE_eval")
    else:
        evaluator = None
    print("Models and datasets loaded")
    train_model(student_model=student_model,
            train_dataloader=train_dataloader,
            train_loss=train_loss,
            evaluator=evaluator,
            **train_model_args)


if __name__ == "__main__":
    main()

