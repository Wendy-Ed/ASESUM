import glob
import json
import os
import pandas as pd
import numpy as np
import torch

from argparse import ArgumentParser
from rouge_score import rouge_scorer

from utils.evaluation_utils import (
    obtain_embeddings_stf,
    cluster_aspects,
    pagerank_most_entail_sentence,
)
from loguru import logger
from summac.model_summac import SummaCZS, SummaCConv
from sklearn.cluster import DBSCAN


def get_jackknife_rouge(predictions, references, stemming=True):
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=stemming)
    score_list = []

    for refs, pred in zip(
        references,
        predictions,
    ):
        # ROUGE multi-ref jackknifing
        if len(refs) > 1:
            cur_scores = [rouge.score(ref, pred) for ref in refs]

            # get best score for all leave-one-out sets
            best_scores = []
            for leave in range(len(refs)):
                cur_scores_leave_one = [
                    cur_scores[s] for s in range(len(refs)) if s != leave
                ]
                best_scores.append(
                    {
                        rouge_type: max(
                            [s[rouge_type] for s in cur_scores_leave_one],
                            key=lambda s: s.fmeasure,
                        )
                        for rouge_type in rouge_types
                    }
                )

            # average the leave-one-out bests to produce the final score
            score = {
                rouge_type: scoring.Score(
                    np.mean([b[rouge_type].precision for b in best_scores]),
                    np.mean([b[rouge_type].recall for b in best_scores]),
                    np.mean([b[rouge_type].fmeasure for b in best_scores]),
                )
                for rouge_type in rouge_types
            }
        else:
            score = rouge.score(refs[0], pred)

        # convert the named tuples to plain nested dicts
        score = {
            rouge_type: {
                "precision": score[rouge_type].precision,
                "recall": score[rouge_type].recall,
                "fmeasure": score[rouge_type].fmeasure,
            }
            for rouge_type in rouge_types
        }
        score_list.append(score)

    l1_keys = list(score_list[0].keys())
    # l2_keys = score_list[0][l1_keys[0]].keys()
    return {
        key1: round(np.mean([score[key1]["fmeasure"] for score in score_list]) * 100, 5)
        for key1 in l1_keys
    }


def generate_ground_truth(file_path, on="reference"):
    logger.debug(os.getcwd())
    logger.debug(os.path.exists(file_path))
    logger.debug(os.path.abspath(file_path))
    with open(file_path) as file:
        original_data = json.load(file)
    logger.debug(f"Now we are comparing against {on}")

    all_ground_truth = []
    if on == "reference":
        for wb_sum in original_data["website_summaries"]:
            ground_truth = ""
            ground_truth += wb_sum["verdict"]
            ground_truth += " " + ". ".join(wb_sum["pros"])
            ground_truth += " " + ". ".join(wb_sum["cons"])
            ground_truth += ".\n"
            all_ground_truth.append(ground_truth)
        return all_ground_truth
    elif on == "review":
        for r in original_data["customer_reviews"]:
            all_ground_truth.append(r["text"])
        return ["\n".join(all_ground_truth)]
        # return all_ground_truth
    # logger.debug(ground_truth)


def cluster_detection(input_summary, cluster_method="dbscan", device=None):
    summary_sentences = input_summary[0].split(".")
    logger.debug(len(summary_sentences))
    sebd = obtain_embeddings_stf(summary_sentences, device)
    if cluster_method == "dbscan":
        clustering = DBSCAN(eps=0.42, min_samples=1, metric="cosine").fit(sebd)
        # clustering = clustering.labels_
        return max(clustering.labels_)


def sentiment_score(sentiments):
    score_result = []
    for s in sentiments:
        if s == "good" or s == "positive":
            score_result.append(1)
        elif s == "bad" or s == "negative":
            score_result.append(-1)
        else:
            score_result.append(0)
    # sentiment_score = [1 if s == 'good' else -1 for s in sentiments]
    return sum(score_result)


def argument_framework_score(data_df, total_sentence, device=None):
    """Extracting the final summary sentences by score
    The score is calculated with the relations between arguments.
    """
    # Get the cluster labels for all the evidence sentences in the data
    logger.debug("Here we are clustering evidence")
    evidence_clusters = cluster_aspects(
        data_df["evidence"].values.tolist(), "dbscan", device, 0.21
    )
    logger.debug(f"There are {data_df.shape[0]} evidences.")
    data_df["cluster_label"] = evidence_clusters

    # Find the cluster aspect relations
    summary_df_group = data_df.groupby(
        ["pure_aspects", "cluster_label"], as_index=False
    ).agg(list)
    summary_df_group["size"] = summary_df_group["evidence"].apply(lambda x: len(x))
    logger.debug(f"There are {summary_df_group.shape[0]} aspect cluster pairs.")
    summary_df_group.drop(
        summary_df_group[
            summary_df_group["size"] <= summary_df_group["size"].quantile(0.25)
        ].index,
        inplace=True,
    )
    logger.debug(
        f"Drop the clusters no larger than {summary_df_group['size'].quantile(0.25)}"
    )
    logger.debug(
        f"Now the resulting aspect cluster pairs are {summary_df_group.shape[0]}"
    )

    # Calculate the score per aspect cluster pair
    summary_df_group["score"] = summary_df_group["sentiment"].apply(
        lambda x: sentiment_score(x)
    )
    summary_df_group["general_sentiment"] = summary_df_group["score"].apply(
        lambda x: "positive" if x > 0 else "negative"
    )
    summary_df_group["score"] = summary_df_group["sentiment"].apply(
        lambda x: np.abs(sentiment_score(x))
    )
    summary_df_group = summary_df_group.sort_values("score", ascending=False)
    top_clusters = summary_df_group.iloc[
        : min(2 * total_sentence, summary_df_group.shape[0])
    ]["cluster_label"]

    # Select representative sentence per cluster
    cluster_df = data_df.groupby("cluster_label", as_index=False)["evidence"].agg(list)
    cluster_df = cluster_df[cluster_df["cluster_label"].isin(top_clusters)]
    cluster_df["pick_evidence"] = cluster_df["evidence"].apply(
        lambda x: pagerank_most_entail_sentence(x, device=device)
    )

    summary_df_group = summary_df_group.merge(
        cluster_df, "inner", "cluster_label"
    ).sort_values("score", ascending=False)
    logger.debug(summary_df_group["score"].median())
    logger.debug(
        summary_df_group[
            [
                "pure_aspects",
                "cluster_label",
                "score",
                "general_sentiment",
                "pick_evidence",
            ]
        ]
        # .head(10)
        .to_string()
    )

    result_list = []
    for sid, s in summary_df_group.iterrows():
        if len(result_list) > total_sentence:
            break
        # if sid > total_sentence:
        #     break
        if s["pick_evidence"] not in result_list:
            logger.debug(f"The aspect {s['pure_aspects']}")
            logger.debug(s["pick_evidence"])
            logger.debug(s["general_sentiment"])
            result_list.append(s["pick_evidence"])

    return result_list


def evidence_from_top_cluster(data_df, top_n, eps, existing_evidence, device=None):
    """Cluster the evidence inside data_df wrt to the semantic similarity
    Return the top_n evidence
    """
    evidence_list = []
    logger.debug(data_df.shape)
    clustered_evidence = cluster_aspects(
        data_df["evidence"].values.tolist(), "dbscan", device=device, eps=eps
    )
    cluster_size = [
        len(np.where(clustered_evidence == i)[0]) for i in np.unique(clustered_evidence)
    ]
    cluster_size_id = {id: cluster_size[id] for id in range(len(cluster_size))}
    cluster_size_id = dict(
        sorted(cluster_size_id.items(), key=lambda i: i[1], reverse=True)
    )
    # cluster_id = list(cluster_size_id.keys())[:top_n]
    for i in list(cluster_size_id.keys()):
        logger.debug(i)
        ed_i = data_df.iloc[np.where(clustered_evidence == i)[0]][
            "evidence"
        ].values.tolist()
        s_i = pagerank_most_entail_sentence(ed_i, device=device)
        if s_i not in existing_evidence:
            logger.debug(s_i)
            evidence_list.append(s_i)
        if len(evidence_list) >= top_n:
            break
    return evidence_list


def compact_summaries(summary_df):
    """Generate the aspect based summary"""
    summary_df["evidence"] = summary_df["evidence"].apply(
        lambda x: x.strip("because").strip(" ")
    )
    summary_per_aspect = summary_df.groupby(["pure_aspects", "sentiment"])[
        "evidence"
    ].apply(list)
    # logger.debug(summary_per_aspect)
    generate_summary = ""
    for idx, row in summary_per_aspect.iterrows():
        evidences = ", ".join(row)
        generate_summary += (
            f'This product has {idx[1]} "{idx[0]}", because ' + evidences
        )
        generate_summary += "\n"
        logger.debug(generate_summary)
    return [generated_summary]


def unpack_evidence(summary_df, uniform=False, device=None):
    if "pure_aspects" not in summary_df.columns:
        summary_df = summary_df.explode("aspects").reset_index(drop=True)
        summary_df.drop(summary_df[summary_df["aspects"].isna()].index, inplace=True)
        summary_df["pure_aspects"] = summary_df["aspects"].apply(
            lambda x: x["aspect"] if "aspect" in x else x
        )
        summary_df["evidence"] = summary_df["aspects"].apply(
            lambda x: x["evidence"] if "evidence" in x else ""
        )
        summary_df["sentiment"] = summary_df["aspects"].apply(
            lambda x: x["sentiment"] if "sentiment" in x else ""
        )
    if uniform:
        aspect_clusters = cluster_aspects(
            summary_df["pure_aspects"].unique(), "dbscan", device=device, eps=0.5
        )
        mapping_dict = {}
        for i in range(len(aspect_clusters)):
            key_c = np.where(aspect_clusters == aspect_clusters[i])[0]
            mapping_dict[summary_df["pure_aspects"].unique()[i]] = summary_df[
                "pure_aspects"
            ].unique()[key_c[0]]
        summary_df["pure_aspects"] = summary_df["pure_aspects"].apply(
            lambda x: mapping_dict[x] if x in mapping_dict else x
        )
    return summary_df


def final_summary(summary_df, strategy, params):
    """Generate the final summary out of the pipeline wrt different strategies.
    params: different pramemter dict related to different strategies
    """
    generated_summary = ""
    evidence_list = []
    if "evidence" in summary_df.columns:
        summary_df.drop(
            summary_df[summary_df["evidence"] == "None"].index, inplace=True
        )

    if strategy == "top_freq":
        summary_count = summary_df["evidence"].value_counts()
        # summary_count.drop(index="None", inplace=True)
        top_n = params["top_n"]
        logger.debug("Here are the most mentioned evidence")
        logger.debug(summary_count[:top_n].index)
        evidence_list = summary_count[:top_n].index.to_list()
    elif strategy == "ratio":
        # Pick the claims with respect to the ratio of the aspects
        evidence_list = set()
        aspect_count = summary_df["pure_aspects"].value_counts(normalize=True)
        logger.debug(aspect_count.index)
        # aspect_count.drop(index='None', inplace=True)
        aspect_count = aspect_count * params["total_s"]
        for asp in aspect_count.index:
            if aspect_count[asp] >= 1:
                aspect_df = summary_df[summary_df["pure_aspects"] == asp][
                    "evidence"
                ].value_counts()
                c = 0
                logger.debug(int(np.floor(aspect_count[asp])))
                for e in aspect_df.index:
                    if c >= int(np.floor(aspect_count[asp])):
                        break
                    if e not in evidence_list:
                        logger.debug(e)
                        logger.debug(c)
                        evidence_list.add(e)
                        c += 1
    elif strategy == "cluster":
        # evidence_list = evidence_from_top_cluster(
        #     summary_df, params["top_n"], eps=0.21, existing_evidence=evidence_list
        # )
        clustered_evidence = cluster_aspects(
            summary_df["evidence"].values.tolist(), "dbscan", device=params["device"], eps=0.21
        )
        logger.debug("cluster complete")
        cluster_size = [
            len(np.where(clustered_evidence == i)[0])
            for i in np.unique(clustered_evidence)
        ]
        cluster_size_id = {id: cluster_size[id] for id in range(len(cluster_size))}
        cluster_size_id = dict(
            sorted(cluster_size_id.items(), key=lambda i: i[1], reverse=True)
        )
        cluster_id = list(cluster_size_id.keys())[: params["top_n"]]
        for i in cluster_id:
            ed_i = summary_df.iloc[np.where(clustered_evidence == i)[0]][
                "evidence"
            ].values.tolist()
            s_i = pagerank_most_entail_sentence(ed_i, device=params["device"])
            logger.debug(s_i)
            evidence_list.append(s_i)
    elif strategy == "aspect_cluster":
        aspect_count = summary_df["pure_aspects"].value_counts(normalize=True)
        logger.debug(aspect_count.index)
        logger.debug(summary_df.shape)
        aspect_count = aspect_count * params["total_s"]
        for asp in aspect_count.index:
            logger.debug(f"The aspect is {asp}, the number is {aspect_count[asp]}")
            if aspect_count[asp] >= 1:
                evidence_list.extend(
                    evidence_from_top_cluster(
                        summary_df[summary_df["pure_aspects"] == asp],
                        aspect_count[asp],
                        eps=0.21,
                        existing_evidence=evidence_list,
                    ),
                )
    elif strategy == "argument_frame":
        evidence_list = argument_framework_score(summary_df, params["total_s"], device=params["device"])
        logger.debug(evidence_list)

    evidence_list = [x.strip(". ") for x in evidence_list]
    for x in evidence_list:
        if x.startswith("because"):
            x = x[len("because") :]
    logger.debug(evidence_list)
    generated_summary = ". ".join(evidence_list)
    logger.debug(f"Here are the final summary from the pipeline\n{generated_summary}")
    return [generated_summary]


def summac_score(summary, ground_truth, model, device):
    logger.info(f"The generated summaries are {summary}")
    # logger.info(f"The reference summaries are {ground_truth}")
    summac_model = None
    if model == "zsc":
        summac_model = SummaCZS(granularity="sentence", model_name="vitc", device=device)
    elif model == "conv":
        summac_model = SummaCConv(
            models=["vitc"],
            bins="percentile",
            granularity="sentence",
            nli_labels="e",
            device=device,
            start_file="default",
            agg="mean",
        )

    score = summac_model.score(ground_truth, summary)
    # del summac_model
    logger.info("Now the Summac evaluation")
    logger.info(score)
    return score


def  score_calculation(metric, summary, ground_truth, device):
    """Calculate the evaluation score
    summary and ground_truth should be list object
    """
    score = 0
    if metric == "rouge":
        # rouge = evaluate.load("rouge")
        # results = rouge.compute(predictions=summary, references=ground_truth)
        score = get_jackknife_rouge(summary, ground_truth)
        logger.info("The ROUGE evaluation")
        logger.debug(score)
    elif metric == "summac":
        score = summac_score(summary, ground_truth, "conv", device)["scores"]
        score = np.average(score)
    elif metric == "cluster":
        logger.debug(
            f"All the sentences in summary can be clustered in {cluster_detection(summary, device=device)}"
        )
    return score


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--res_p", type=str, default=None)
    parser.add_argument("--res_dir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu" )
    args = parser.parse_args()

    if args.res_p is None and args.res_dir is None:
        logger.exception("No defined generated results....")
        exit()
    avg_score = []
    if args.res_dir is not None:
        result = []
        for file in glob.glob(os.path.join(args.res_dir, "*qwen_scheme.json")):
            if args.device == "mps": 
                torch.mps.empty_cache()
            elif args.device == "cuda":
                torch.cuda.empty_cache()
            
            res_per_file = {}
            logger.debug(file)
            data_df = unpack_evidence(pd.read_json(file, lines=True), uniform=True, device=args.device)
            # if os.path.exists(os.path.join(args.res_dir, os.path.basename(file).strip(".json") + "_merge_evidence.json")):
            #     continue
            # summary_df = merge_evidence(summary_df)
            assert "evidence" in data_df.columns
            # summary_df.to_json(os.path.join(args.res_dir, os.path.basename(file).strip(".json") + "_merge_evidence.json"))
            # continue
            file_prefix = os.path.basename(file).split("_")[0]
            logger.debug(file_prefix)
            # if file_prefix in ['B000I1X3W8', 'B07GFP7H8C', 'B071HLJL4Q', 'B00RE1UL52', 'B07774L6TT', 'B00LY8JVW0', 'B00DMS0LCO', 'B01N9RM9N3']:
            logger.info(f"Now calculating the file {os.path.basename(file)}")
            all_reviews = generate_ground_truth(os.path.join(args.gt_dir, f"{file_prefix}.json"), on="review")
            res_per_file["entity_id"] = file_prefix
            # logger.debug(f"Ground truth are {ground_truth}")
            res_per_file["all_reviews"] = all_reviews
            reference = generate_ground_truth(os.path.join(args.gt_dir, f"{file_prefix}.json"), on="reference")
            res_per_file["reference"] = reference

            generated_summary = final_summary(data_df, "argument_frame", {"total_s": 8, "device": args.device})

            res_per_file["generated_summary"] = generated_summary

            if len(all_reviews) > 1:
                generated_summary = generated_summary * len(all_reviews)
            summac_s = score_calculation(args.metric, generated_summary, all_reviews, args.device)

            avg_score.append(summac_s)
            res_per_file["summac_score"] = summac_s
            result.append(res_per_file)

        with open(
            os.path.join(
                args.res_dir,
                f"arg_scheme_{os.path.basename(os.path.dirname(args.res_dir))}_qwen_results.json",
            ),
            "w",
        ) as f:
            json.dump(result, f)

    elif args.res_p is not None and not args.batch:
        logger.info(os.path.exists(args.res_p))
        with open(args.res_p) as file:
            file_data = []
            for line in file:
                file_data.append(json.loads(line))
        logger.debug(len(file_data))
        logger.debug(len(file_data[0]))
        for fd in file_data[0]:
            avg_score.append(
                score_calculation(args.metric, fd["summaries"], fd["all_reviews"], device=args.device)
            )
    elif args.res_p is not None and args.batch:
        logger.info(f"Now evaluating file {args.res_p}")
        generate_summaries = pd.read_json(args.res_p)

    logger.debug(
        f"The average {args.metric} is {sum(avg_score)/len(avg_score) if len(avg_score) != 1 else avg_score[0]}"
    )
