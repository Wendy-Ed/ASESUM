from loguru import logger
import networkx as nx
import numpy as np
import os
import sys
import yaml

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompter import GPTPrompter


def pick_with_LLM(result_df, llm_name, level):
    if llm_name.startwith("gpt"):
        prompter = GPTPrompter()

    input_string = ""
    for asp in result_df["pure_aspects"].unique():
        aspect_df = result_df[result_df["pure_aspects"] == asp]
        all_evidence = aspect_df["evidence_y"].explode().values
        input_string += f"{asp}: " + str(all_evidence)

    response = prompter.prompting(
        input_string, "select_evidence.yaml", level, message_type="system"
    )
    logger.debug(response)

    if level == "aspect":
        logger.debug(response)
    elif level == "doc":
        logger.debug(response)


def pick_top_n(result_df, top_n):
    logger.debug(result_df.index)
    logger.debug(range(0, top_n + 1))
    return range(0, min(top_n, result_df.shape[0]))


def pick_with_curvature(result_df):
    logger.debug(f"The scores are {result_df['score'].values}")
    result_df_no = result_df.drop_duplicates(subset=["cluster_label", "pure_aspects"])
    score_list = result_df_no["score"].values
    x_values = np.arange(len(score_list))
    idx_list = result_df_no.index.values
    logger.debug(f"The score list is {score_list}")
    logger.debug(f"The index of score list is {idx_list}")
    first_d = np.gradient(score_list, x_values)
    second_d = np.gradient(first_d, x_values)
    curvature = second_d / (1.0 + first_d**2) ** 1.5

    avg_first_d = np.mean(first_d[1:-1])
    cur_tuples = []
    for i in range(len(curvature)):
        if i == 0 or i == len(curvature) - 1:
            continue
        if score_list[i] == 0:
            break
        cur_tuples.append((curvature[i], first_d[i], i))
    cur_tuples = sorted(cur_tuples, key=lambda x: x[0], reverse=True)[:10]
    index = cur_tuples[0][2]
    point_score = score_list[index]
    logger.debug(curvature)
    logger.debug(f"The score from curvature is {point_score}, the index is {index}")
    logger.debug(f"The length is {index >= 4} larger than 5.")
    return range(0, index + 1)


def pick_with_gradient(result_df):
    logger.debug(result_df["score"].values)
    score_list = result_df["score"].values
    second_d = score_list[2:] - 2 * score_list[1:-1] + score_list[:-2]
    points = []
    for i in range(len(second_d) - 1):
        if second_d[i] * second_d[i + 1] <= 0:
            points.append(i + 2)
    logger.debug(result_df["score"].values[points])

    # if max_grad_id == 0:
    #     max_grad_id += 1
    logger.debug(f"Now the maxmium gradient point is {points[0]}")
    return range(0, points[0] + 1)


def pagerank_most_entail_sentence(list_of_sentences, max_sentence=1, device=None):
    if len(list_of_sentences) <= max_sentence:
        return list_of_sentences

    setence_embedding = obtain_embeddings_stf(list_of_sentences, device)
    setence_similarity = cosine_similarity(setence_embedding)
    nx_graph = nx.from_numpy_array(setence_similarity)
    scores = nx.pagerank(nx_graph)
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return [list_of_sentences[si] for si in list(scores.keys())[:max_sentence]]


def obtain_embeddings_stf(list_of_words, device=None, similarity=False):
    model = SentenceTransformer("./all-mpnet-base-v2")
    if device is not None:
        model.to(device)
    preprocess = []
    for l in list_of_words:
        if isinstance(l, str):
            preprocess.append(l)
        else:
            logger.warning(f"This is not string {l}")
            exit()
    embeddings = model.encode(preprocess)
    if similarity:
        similarities = model.similarity(embeddings[:-1], [embeddings[-1]])
        return similarities
    del model
    return embeddings

def unpack_evidence(summary_df, uniform=False, device=None, main_aspects=None):
    if "pure_aspects" not in summary_df.columns:
        summary_df["aspects"] = summary_df["aspects"].apply(
            lambda x: x["review"] if "review" in x else x
        )
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
        logger.debug(
            f"Before cluster, the aspects are {summary_df['pure_aspects'].unique()}"
        )
        aspect_clusters = cluster_aspects(
            summary_df["pure_aspects"].unique(), "dbscan", device=device, eps=0.45
        )
        mapping_dict = {}
        # cluster_check_new(summary_df['pure_aspects'].unique(), main_aspects, mapping_dict, device=device)
        for i in range(len(aspect_clusters)):
            key_c = np.where(aspect_clusters == aspect_clusters[i])[0]
            mapping_dict[summary_df["pure_aspects"].unique()[i]] = summary_df[
                "pure_aspects"
            ].unique()[key_c[0]]
        logger.debug(mapping_dict)
        summary_df["pure_aspects"] = summary_df["pure_aspects"].apply(
            lambda x: mapping_dict[x] if x in mapping_dict else x
        )
    return summary_df


def cluster_aspects(aspect_list, method, device, eps=0.42):
    """The aspect list is sorted by the count of the aspect
    Return: the clustering of all the aspects
    """
    aspect_embedding = obtain_embeddings_stf(aspect_list, device=device)
    clustering = None
    if method == "dbscan":
        clustering = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit(
            aspect_embedding
        )
        clustering = clustering.labels_
    if method == "hdbscan":
        hdb = HDBSCAN(min_cluster_size=2, min_samples=1, metric="cosine").fit(
            aspect_embedding
        )
        clustering = hdb.labels_
        probs = hdb.probabilities_
        logger.debug(clustering)
        logger.debug(probs)
    if method == "hac":
        distance = pdist(aspect_embedding, metric="cosine")
        return_matrix = linkage(distance, method="average")
        # return return_matrix
        # plt.figure(figsize=(22, 8))
        # d = dendrogram(return_matrix, labels=aspect_list)
        # plt.show()
        return return_matrix

    return clustering

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

def find_pairs(similarities, against_index):
    result_pair = {}
    np.fill_diagonal(similarities, 0)
    for i in range(len(against_index) - 1):
        for j in range(i + 1, len(against_index)):
            similarities[i][j] = 0
            similarities[j][i] = 0
    for i in range(len(similarities)):
        j = np.argmax(similarities[i])
        if max(similarities[i]) > 0.45 and np.argmax(similarities[j]) == i:
            if j not in result_pair.values():
                result_pair[j] = i
        else:
            result_pair[i] = i
    logger.debug(result_pair)
    return result_pair


def cluster_check_new(list_of_words, gold_cluster, clusters, device="cpu"):
    aspect_clusters = cluster_aspects(list_of_words, "hdbscan", device=device)
    logger.debug(len(np.unique(aspect_clusters)))
    if len(np.unique(aspect_clusters)) != 1:
        for c in np.unique(aspect_clusters):
            keys = np.where(aspect_clusters == c)[0]
            logger.debug(keys)
            logger.debug(list_of_words[keys])
            if c == -1:
                clusters.update({k: k for k in list_of_words[keys]})
            else:
                cluster_check_new(list_of_words[keys], gold_cluster, clusters)
    # elif gold_cluster:
    #     comm_list = list(set(list_of_words).intersection(set(gold_cluster)))
    else:
        comm_list = []
        if gold_cluster:
            comm_list = list(set(list_of_words).intersection(set(gold_cluster)))

        if len(comm_list) >= 2:
            idxs = []
            for i in comm_list:
                idxs.append(np.where(list_of_words == i)[0][0])
            # logger.debug(f"The rest of list is {rest_list}")
            logger.debug(idxs)
            embeddings_y = obtain_embeddings_stf(list_of_words)
            # embeddings_x = obtain_embeddings_stf(comm_list)
            similarities = cosine_similarity(embeddings_y, embeddings_y)
            pairs = find_pairs(similarities, idxs)
            for k, v in pairs.items():
                clusters.update({list_of_words[k]: list_of_words[v]})
            logger.debug(similarities)
        elif len(comm_list) == 1:
            clusters.update({k: comm_list[0] for k in list_of_words})
        else:
            clusters.update({k: list_of_words[0] for k in list_of_words})
