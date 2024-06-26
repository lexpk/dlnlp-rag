"""Evaluation script for RAG models."""

import argparse
import ast
import logging
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
import json

from transformers import BartForConditionalGeneration, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
from transformers import logging as transformers_logging


sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip
from rag.utils_rag import exact_match_score, exact_match_prefix_score, f1_score  # noqa: E402 # isort:skip


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

transformers_logging.set_verbosity_info()


def infer_model_type(model_name_or_path):
    if "token" in model_name_or_path:
        return "rag_token"
    if "sequence" in model_name_or_path:
        return "rag_sequence"
    if "bart" in model_name_or_path:
        return "bart"
    return None


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def get_scores(preds_path, gold_data_path, gold_data_mode="qa", **kwargs):
    hypos = [line.strip() for line in open(preds_path, "r").readlines()]
    answers = []

    if gold_data_mode == "qa":
        data = pd.read_csv(gold_data_path, sep="\t", header=None)
        for answer_list in data[1]:
            ground_truths = ast.literal_eval(answer_list)
            answers.append(ground_truths)
    else:
        references = [line.strip() for line in open(gold_data_path, "r").readlines()]
        answers = [[reference] for reference in references]

    f1 = em = total = 0
    for prediction, ground_truths in zip(hypos, answers):
        total += 1
        em += metric_max_over_ground_truths(exact_match_prefix_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    em = 100.0 * em / total
    f1 = 100.0 * f1 / total

    logger.info(f"F1: {f1:.2f}")
    logger.info(f"EM: {em:.2f}")

    return em, f1


def get_precision_at_k(preds_path, gold_data_path, k=1, **kwargs):
    hypos = [line.strip() for line in open(preds_path, "r").readlines()]
    references = [line.strip() for line in open(gold_data_path, "r").readlines()]

    em = total = 0
    for hypo, reference in zip(hypos, references):
        hypo_provenance = set(hypo.split("\t")[:k])
        ref_provenance = set(reference.split("\t"))
        total += 1
        em += len(hypo_provenance & ref_provenance) / k

    em = 100.0 * em / total
    logger.info(f"Precision@{k}: {em: .2f}")

    return em


def retrieve_batch_docs(rag_model, questions, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), n_docs=None, **kwargs):
    retriever_input_ids = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )["input_ids"].to(device)

    question_enc_outputs = rag_model.rag.question_encoder(retriever_input_ids)
    question_enc_pool_output = question_enc_outputs[0]

    result = rag_model.retriever(
        retriever_input_ids,
        question_enc_pool_output.cpu().detach().to(torch.float32).numpy(),
        prefix=rag_model.rag.generator.config.prefix,
        n_docs=n_docs or rag_model.config.n_docs,
        return_tensors="pt",
    )
    return rag_model.retriever.index.get_doc_dicts(result.doc_ids)


def evaluate_batch_retrieval(rag_model, questions, **kwargs):
    def strip_title(title):
        if title.startswith('"'):
            title = title[1:]
        if title.endswith('"'):
            title = title[:-1]
        return title
    
    all_docs = retrieve_batch_docs(rag_model, questions, **kwargs)
    provenance_strings = []
    for docs in all_docs:
        provenance = [strip_title(title) for title in docs["title"]]
        provenance_strings.append("\t".join(provenance))
    return provenance_strings


def evaluate_batch_e2e(rag_model, questions, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_beams=4, min_length=1, max_length=50, print_predictions=False, **kwargs):
    with torch.no_grad():
        inputs_dict = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
            questions, return_tensors="pt", padding=True, truncation=True
        )

        input_ids = inputs_dict.input_ids.to(device)
        attention_mask = inputs_dict.attention_mask.to(device)
        outputs = rag_model.generate(  # rag_model overwrites generate
            input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            min_length=min_length,
            max_length=max_length,
            early_stopping=False,
            num_return_sequences=1,
            bad_words_ids=[[0, 0]],  # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = rag_model.retriever.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if print_predictions:
            for q, a in zip(questions, answers):
                logger.info("Q: {} - A: {}".format(q, a))

        return answers
    

def evaluate_batch_generation(tokenizer_model, questions, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_beams=4, min_length=1, max_length=50, **kwargs):
    tokenizer, model = tokenizer_model

    with torch.no_grad():
        inputs_dict = tokenizer.batch_encode_plus(
            questions, return_tensors="pt", padding=True, truncation=True
        )

        input_ids = inputs_dict.input_ids.to(device)
        attention_mask = inputs_dict.attention_mask.to(device)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            min_length=min_length,
            max_length=max_length,
            early_stopping=False,
            num_return_sequences=1,
            bad_words_ids=[[0, 0]],  # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return answers


def eval_rag(score_fn, evaluate_batch_fn, model, evaluation_set, gold_path, predictions_path, eval_batch_size=8, **kwargs):
    with open(evaluation_set, "r") as eval_file, open(predictions_path, "w") as preds_file:
        questions = []
        for line in tqdm(eval_file):
            questions.append(line.strip())
            if len(questions) == eval_batch_size:
                answers = evaluate_batch_fn(model, questions, **kwargs)
                preds_file.write("\n".join(answers) + "\n")
                preds_file.flush()
                questions = []
        if len(questions) > 0:
            answers = evaluate_batch_fn(model, questions, **kwargs)
            preds_file.write("\n".join(answers))
            preds_file.flush()

    return score_fn(predictions_path, gold_path, **kwargs)


def retrieve_docs(rag_model, evaluation_set, documents_path, write_batch_size=8, **kwargs):
    def drop_embeddings(doc):
        return {k: v for k, v in doc.items() if k != "embeddings"}

    with open(evaluation_set, "r") as eval_file, open(documents_path, "w") as docs_file:
        questions = []
        for line in tqdm(eval_file):
            questions.append(line.strip())
            if len(questions) == write_batch_size:
                docs = retrieve_batch_docs(rag_model, questions, **kwargs)
                docs_file.write("\n".join(map(json.dumps, map(drop_embeddings, docs))) + "\n")
                docs_file.flush()
                questions = []
        if len(questions) > 0:
            docs = retrieve_batch_docs(rag_model, questions, **kwargs)
            docs_file.write("\n".join(map(json.dumps, map(drop_embeddings, docs))))
            docs_file.flush()


def eval_rag_docs(score_fn, evaluate_batch_docs_fn, model, evaluation_set, documents_path, gold_path, predictions_path, eval_batch_size=8, n_docs=5, offset=0, **kwargs):
    def docs_list(docs):
        return [{"title": title, "text": text} for title, text in zip(docs["title"], docs["text"])]
    
    with open(evaluation_set, "r") as eval_file, open(documents_path, "r") as docs_file, open(predictions_path, "w") as preds_file:
        for _ in range(offset):
            next(eval_file)
            next(docs_file)

        questions = []
        documents = []
        for line, docs in tqdm(zip(eval_file, docs_file)):
            questions.append(line.strip())
            documents.append(docs_list(json.loads(docs))[:n_docs])
            if len(questions) == eval_batch_size:
                answers = evaluate_batch_docs_fn(model, questions, documents, **kwargs)
                preds_file.write("\n".join(answers) + "\n")
                preds_file.flush()
                questions = []
                documents = []
        if len(questions) > 0:
            answers = evaluate_batch_docs_fn(model, questions, documents, **kwargs)
            preds_file.write("\n".join(answers))
            preds_file.flush()

    return score_fn(predictions_path, gold_path, **kwargs)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["rag_sequence", "rag_token", "bart"],
        type=str,
        help=(
            "RAG model type: rag_sequence, rag_token or bart, if none specified, the type is inferred from the"
            " model_name_or_path"
        ),
    )
    parser.add_argument(
        "--index_name",
        default=None,
        choices=["exact", "compressed", "legacy"],
        type=str,
        help="RAG model retriever type",
    )
    parser.add_argument(
        "--index_path",
        default=None,
        type=str,
        help="Path to the retrieval index",
    )
    parser.add_argument("--n_docs", default=5, type=int, help="Number of retrieved docs")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained checkpoints or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["e2e", "retrieval"],
        default="e2e",
        type=str,
        help=(
            "Evaluation mode, e2e calculates exact match and F1 of the downstream task, retrieval calculates"
            " precision@k."
        ),
    )
    parser.add_argument("--k", default=1, type=int, help="k for the precision@k calculation")
    parser.add_argument(
        "--evaluation_set",
        default=None,
        type=str,
        required=True,
        help="Path to a file containing evaluation samples",
    )
    parser.add_argument(
        "--gold_data_path",
        default=None,
        type=str,
        required=True,
        help="Path to a tab-separated file with gold samples",
    )
    parser.add_argument(
        "--gold_data_mode",
        default="qa",
        type=str,
        choices=["qa", "ans"],
        help=(
            "Format of the gold data file"
            "qa - a single line in the following format: question [tab] answer_list"
            "ans - a single line of the gold file contains the expected answer string"
        ),
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="predictions.txt",
        help="Name of the predictions file, to be stored in the checkpoints directory",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--recalculate",
        help="Recalculate predictions even if the prediction file exists",
        action="store_true",
    )
    parser.add_argument(
        "--num_beams",
        default=4,
        type=int,
        help="Number of beams to be used when generating answers",
    )
    parser.add_argument("--min_length", default=1, type=int, help="Min length of the generated answers")
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the generated answers")

    parser.add_argument(
        "--print_predictions",
        action="store_true",
        help="If True, prints predictions while evaluating.",
    )
    parser.add_argument(
        "--print_docs",
        action="store_true",
        help="If True, prints docs retried while generating.",
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def main(args):
    model_kwargs = {}
    if args.model_type is None:
        args.model_type = infer_model_type(args.model_name_or_path)
        assert args.model_type is not None
    if args.model_type.startswith("rag"):
        model_class = RagTokenForGeneration if args.model_type == "rag_token" else RagSequenceForGeneration
        model_kwargs["n_docs"] = args.n_docs
        if args.index_name is not None:
            model_kwargs["index_name"] = args.index_name
        if args.index_path is not None:
            model_kwargs["index_path"] = args.index_path
    else:
        model_class = BartForConditionalGeneration

    checkpoints = (
        [f.path for f in os.scandir(args.model_name_or_path) if f.is_dir()]
        if args.eval_all_checkpoints
        else [args.model_name_or_path]
    )

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    score_fn = get_scores if args.eval_mode == "e2e" else get_precision_at_k
    evaluate_batch_fn = evaluate_batch_e2e if args.eval_mode == "e2e" else evaluate_batch_retrieval

    for checkpoint in checkpoints:
        if os.path.exists(args.predictions_path) and (not args.recalculate):
            logger.info("Calculating metrics based on an existing predictions file: {}".format(args.predictions_path))
            score_fn(args, args.predictions_path, args.gold_data_path)
            continue

        logger.info("***** Running evaluation for {} *****".format(checkpoint))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Predictions will be stored under {}".format(args.predictions_path))

        if args.model_type.startswith("rag"):
            retriever = RagRetriever.from_pretrained(checkpoint, **model_kwargs)
            model = model_class.from_pretrained(checkpoint, retriever=retriever, **model_kwargs)
            model.retriever.init_retrieval()
        else:
            model = model_class.from_pretrained(checkpoint, **model_kwargs)
        model.to(args.device)

        eval_rag(
            score_fn,
            evaluate_batch_fn,
            model,
            args.evaluation_set,
            args.predictions_path,
            args.gold_data_path,
            **vars(args),
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
