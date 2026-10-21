from argparse import ArgumentParser
import glob
from loguru import logger
import json
import os
import pandas as pd
import numpy as np

from scipy.special import beta as beta_function
from scipy.integrate import quad

from prompter import GPTPrompter
from uncertainpy import gradual as grad
from utils.evaluation_utils import sentiment_score, unpack_evidence, cluster_aspects, pagerank_most_entail_sentence

class AspectNode:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
    
    def to_dict(self):
        if not self.children:
            return self.name
        return {self.name: [child.to_dict() for child in self.children]}
    
    def check(self, node_name):
        if self.name == node_name:
            return self
        for child in self.children:
            child_check = child.check(node_name)
            if child_check is not None:
                return child_check
        return None
    
def construct_aspect_pair(aspect_list):
    aspect_pairs = []
    for i in range(len(aspect_list)):
        for j in range(i+1, len(aspect_list)):
            aspect_pairs.append((aspect_list[i], aspect_list[j]))
    # pair_string = "\n".join([f"{pair[0]} - {pair[1]}" for pair in aspect_pairs[:10]])

    return aspect_pairs

def result_to_schema(asp_pair, result, old_schema):
    if len(asp_pair) != len(result):
        logger.warning("The length of aspect pairs and result do not match.")
        return None
    
    for i in range(len(result)):
        if result[i] == 'A':
            if asp_pair[i][1] not in old_schema:
                old_schema[asp_pair[i][1]] = [asp_pair[i][0]]



def LLM_hierachy(list_of_aspects, model_name):
    prompter = None
    if model_name.startswith("gpt"):
        prompter = GPTPrompter(model_name, {})
    else:
        raise NotImplementedError(f"The model {model_name} is not implemented yet.")
    # aspect_paired = construct_aspect_pair(list_of_aspects)
    logger.debug(list_of_aspects)
    schema_tree = AspectNode("hotel")
    if prompter is not None:
        # response = prompter.prompting(str(list_of_aspects), "hierarchy_prompt.txt", None, None, message_type="system")
        # for i in range(0, len(aspect_paired), 10):
            # input_str = "\n".join([f"{pair[0]} - {pair[1]}" for pair in aspect_paired[i:i+10]])
        for asp in list_of_aspects:
            para_dict = {"schema_tree": json.dumps(schema_tree.to_dict())}
            logger.debug(para_dict)
            logger.debug(f"The aspect to be processed is {asp}")
            # logger.debug(f"The input string for LLM is {input_str}")
            response = prompter.prompting(str(asp), "hierarchy_prompt_tree.txt", None, parameters=para_dict, message_type="system")
            try:
                result_aspects = json.loads(response['result'])

            except json.JSONDecodeError:
                logger.warning("Failed to decode LLM response as JSON.")
                result_aspects = None
            logger.debug(f"The LLM response is {result_aspects}")
            sub_tree = schema_tree.check(result_aspects['node'])
            sub_tree.add_child(AspectNode(asp))
            # exit()
    return schema_tree.to_dict()


def unpack_matrix(return_matrix, aspect_list):
    cluster_dict = {3*len(aspect_list): []}
    logger.debug(return_matrix[0][2])
    logger.debug(return_matrix[-1][2])
    logger.debug(np.percentile(return_matrix[:, 2], 25))
    for i in range(len(return_matrix)):
        if return_matrix[i][2] > 0.5:
            for j in range(2):
                if return_matrix[i][j] < len(aspect_list):
                    logger.debug(f"The aspect {aspect_list[int(return_matrix[i][j])]} is a single cluster, the id is {return_matrix[i][j]}, the cluster is {len(aspect_list) + i}")
                    cluster_dict[3*len(aspect_list)].append(int(return_matrix[i][j]))
            continue
        
        if return_matrix[i][3] == 2:
            cluster_dict[len(aspect_list) + i] = [int(return_matrix[i][0]), int(return_matrix[i][1])]
        else:
            result_cluster = []
            for j in range(2):
                if return_matrix[i][j] >= len(aspect_list):
                    result_cluster.extend(cluster_dict[int(return_matrix[i][j])])
                    cluster_dict.pop(int(return_matrix[i][j]))
                else:
                    result_cluster.append(int(return_matrix[i][j])) 
            cluster_dict[len(aspect_list) + i] = result_cluster
    return cluster_dict


def check_validity(aspect_schema, word_set):
    for root, children in aspect_schema.items():
        logger.debug(word_set)
        if root in word_set:
            return False
        word_set.add(root)
        if isinstance(children, list):
            for c in children:
                if isinstance(c, dict):
                    return check_validity(c, word_set)
                elif isinstance(c, str) and c in word_set:
                    return False
                elif isinstance(c, str):
                     word_set.add(c)
        elif isinstance(children, dict):
            return check_validity(children, word_set)
    return True

def graph_tuple(aspect_list, aspect_score, tuple_result, key_node=None):
    for root in aspect_list:
        
        if isinstance(root, str) and key_node is None:
            tuple_result.append((root, root, 'n'))
            continue
        elif isinstance(root, str) and key_node is not None:
            t = build_tuple(key_node, root, aspect_score)
            if t is not None:
                tuple_result.append(t)
        elif isinstance(root, dict):
            k = list(root.keys())[0]
            if key_node is not None:
                t = build_tuple(key_node, k, aspect_score)
                if t is not None:
                    tuple_result.append(t)
            graph_tuple(root[k], aspect_score, tuple_result, key_node=k)

            # for c in children:
            #     if isinstance(c, str):
            #         t = build_tuple(root, c, aspect_score)
            #         if t is not None:
            #             tuple_result.append(t)
            #     elif isinstance(c, dict):
            #         for k in c.keys():
            #             logger.debug(f"Processing the aspect {k} with parent aspect {root}")
            #             t = build_tuple(root, k, aspect_score)
            #             if t is not None:
            #                 tuple_result.append(t)
            #             graph_tuple(c, aspect_score, tuple_result)

def build_tuple(root, c, scores):
    if root not in scores or c not in scores:
        logger.warning(f"Either {root} or {c} is not in the scores dictionary. Skipping this pair.")
        return None
    if scores[c] * scores[root] < 0:
        return (c, root, 'a')
    else:
        return (c, root, 's')

def integration_score(pos_score, neg_score):
    norm = beta_function(pos_score + 1, neg_score + 1)
    def f(x):
        return (x ** pos_score) * ((1-x) ** neg_score) / norm
    value, _ = quad(lambda x: abs(f(x) - 1.0), 0, 1, limit=200)

    return value * 0.5

def uncertatinty_score(summary_df, method="easy_score"):
    result_dict = {}
    group_summary_df = summary_df.groupby(["pure_aspects", "sentiment"], as_index=False).agg(list)
    group_summary_df["uc_s"] = group_summary_df["evidence"].apply(lambda x: len(x))
    for asp in group_summary_df["pure_aspects"].unique():
        asp_df = group_summary_df[group_summary_df["pure_aspects"] == asp]
        pos_num, neg_num = 0, 0
        if "positive" in asp_df["sentiment"].values:
            pos_num = asp_df[asp_df["sentiment"] == "positive"]["uc_s"].values[0]
        if "negative" in asp_df["sentiment"].values:
            neg_num = asp_df[asp_df["sentiment"] == "negative"]["uc_s"].values[0]
        if pos_num == 0 or neg_num == 0:
            score = 1

        elif method == "easy_score":
            score = integration_score(pos_num, neg_num)

        pos_score = asp_df[asp_df["sentiment"] == "positive"]["uc_s"].values
        
        result_dict[asp] = 1 - score
        logger.debug(f"For the aspect {asp}, the positive evidence number is {pos_num}, the negative evidence number is {neg_num}, and the uncertainty score is {1 - score}")
    return result_dict


def arg_score_sentiment(summary_df, llm_result, normalize="all"):
    # Calculate overall sentiment of aspects
    aspect_sentiment = summary_df.groupby('pure_aspects')['sentiment'].apply(list).reset_index()
    if normalize == "all":
        aspect_sentiment['sentiment_score'] = aspect_sentiment['sentiment'].apply(lambda x: sentiment_score(x)/summary_df.shape[0])
    elif normalize == "aspect":
        aspect_sentiment['sentiment_score'] = aspect_sentiment['sentiment'].apply(lambda x: sentiment_score(x)/len(x))
    else:
        logger.warning(f"The normalization method {normalize} is not recognized. No normalization will be applied.")
    
    base_score = {}
    for index, row in aspect_sentiment.iterrows():
        base_score[row['pure_aspects']] = row['sentiment_score']
    logger.debug(base_score)
    result_t = []
    graph_tuple(llm_result, base_score, result_t)
    for k ,v in base_score.items():
        base_score[k] = 0
    logger.debug(f"The aspects base scores are {base_score}")
    
    # Calculate the scores of arguments by clustering the evidence
    evidence_clusters = cluster_aspects(
        summary_df["evidence"].values.tolist(), "dbscan", eps=0.20, device="cpu"
    )
    summary_df["evidence_cluster"] = evidence_clusters
    logger.debug(f"The pure aspects for this file is {summary_df['pure_aspects'].unique()}")
    summary_df_group = summary_df.groupby(
        ["evidence_cluster", "sentiment", "pure_aspects"], as_index=False
    ).agg(list)
    logger.debug(f"The columns of the summary df group is {summary_df_group.columns}")
    summary_df_group["scores"] = summary_df_group["evidence"].apply(lambda x: len(x)/summary_df_group.shape[0])
    summary_df_group["evidence"] = summary_df_group["evidence"].apply(lambda x: pagerank_most_entail_sentence(x,max_sentence=1) if len(x) > 1 else x)
    # summary_df_group["aspects"] = summary_df_group.apply(lambda x: [{"aspect": x["pure_aspects"], "evidence": x["evidence"][0], "sentiment": x["sentiment"]}] if len(x["aspects"]) > 1 else x["aspects"], axis=1)
    logger.debug(f"The number of pairs of aspects and evidence is {summary_df_group.shape[0]}")
    for index, row in summary_df_group.iterrows():
        result_t.append((row["evidence"][0], row["pure_aspects"], 's' if row["sentiment"] == "positive" else 'a'))
        base_score.update({row["evidence"][0]: row["scores"] if row["sentiment"] == "positive" else -row["scores"]})
    logger.debug(result_t)
    return [result_t, base_score]


def unpack_matrix(return_matrix, aspect_list):
    cluster_dict = {3*len(aspect_list): []}
    logger.debug(return_matrix[0][2])
    logger.debug(return_matrix[-1][2])
    logger.debug(np.percentile(return_matrix[:, 2], 25))
    for i in range(len(return_matrix)):
        if return_matrix[i][2] > 0.5:
            for j in range(2):
                if return_matrix[i][j] < len(aspect_list):
                    logger.debug(f"The aspect {aspect_list[int(return_matrix[i][j])]} is a single cluster, the id is {return_matrix[i][j]}, the cluster is {len(aspect_list) + i}")
                    cluster_dict[3*len(aspect_list)].append(int(return_matrix[i][j]))
            continue
        
        if return_matrix[i][3] == 2:
            cluster_dict[len(aspect_list) + i] = [int(return_matrix[i][0]), int(return_matrix[i][1])]
        else:
            result_cluster = []
            for j in range(2):
                if return_matrix[i][j] >= len(aspect_list):
                    result_cluster.extend(cluster_dict[int(return_matrix[i][j])])
                    cluster_dict.pop(int(return_matrix[i][j]))
                else:
                    result_cluster.append(int(return_matrix[i][j])) 
            cluster_dict[len(aspect_list) + i] = result_cluster
    return cluster_dict

def find_highest_score_child(graph, arg):
    max_score = float('-inf')
    child_arg = None
    for s in arg.supports:
        # logger.debug(f"s source is {s.source}, s target is {s.target}")
        s = s.source
        if s.strength > max_score and len(s.name.split(" ")) != 1:
            max_score = s.strength
            child_arg = s
            # logger.debug(f"s source is {s.source}, s target is {s.target}")
            logger.debug(f"The supporter {s.name} has strength {s.strength}")
    for a in arg.attacks:
        # logger.debug(f"a source is {a.source}, a target is {a.target}")
        a = a.source
        if a.strength > max_score and len(a.name.split(" ")) != 1:
            max_score = a.strength
            child_arg = a
            # logger.debug(f"a source is {a.source}, a target is {a.target}")
            logger.debug(f"The attacker {a.name} has strength {a.strength}")
    return child_arg.name if child_arg is not None else None


def argument_selection(graph, arg_score, arg_uc, method="top_k", **kwargs):
    arg_score = {k: v for k, v in sorted(arg_score.items(), key=lambda item: item[1], reverse=True)}
    summary = ""
    logger.debug(f"The kwargs for argument selection are {kwargs}")
    if method == "top_k" and "k" in kwargs:
        k = kwargs["k"]
        for arg in list(arg_score.keys())[:k]:
            main_arg = graph.arguments[arg]
            child_arg = find_highest_score_child(graph, main_arg)
            if child_arg is not None:
                summary += child_arg.strip(".") + ". "
        logger.debug(f"The selected arguments are {list(arg_score.keys())[:k]} and the summary is \n{summary}")
    return summary


def graph_construction(graph_node_list, node_score):
    graph = grad.BAG()
    for links in graph_node_list:
        if links[0] not in graph.arguments:
            arg_0 = graph.add_argument(links[0], node_score[links[0]])
        else:
            arg_0 = graph.arguments[links[0]]
        if links[1] not in graph.arguments:
            arg_1 = graph.add_argument(links[1], node_score[links[1]])
        else:
            arg_1 = graph.arguments[links[1]]

        if links[2] == 'a':
            graph.add_attack(arg_0, arg_1)
        elif links[2] == 's':
            graph.add_support(arg_0, arg_1)
    return graph

def DF_QUAD(graph, pure_aspects):
    agg = grad.semantics.modular.ProductAggregation()
    inf = grad.semantics.modular.LinearInfluence(conservativeness=1)
    stregths = grad.algorithms.computeStrengthValues(graph, agg, inf)

    sorted_args = sorted(graph.arguments.values(), key=lambda x: x.strength, reverse=True)
    arg_final_strength = {}
    print_length = 10
    for arg in sorted_args:
        if print_length > 0:
            logger.info(f"Argument: {arg.name}, Strength: {arg.strength}")
            print_length -= 1
        if arg.name in pure_aspects:
            arg_final_strength[arg.name] = arg.strength
    return arg_final_strength

if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument("--input_dir", type=str, default="../../data/space/test_data/test_result/")
    parse.add_argument("--output_dir", type=str, default="../../data/space/test_data/test_result/")
    parse.add_argument("--model_name", type=str, default="gpt-5-mini")
    args = parse.parse_args()

    count = 0
    result = []
    # gold_summary = pd.read_json("../../data/space/test_data/test_result/argscheme_space_cur_gpt_1208.json", lines=True)
    # logger.debug(f"The columns of the gold summary dataframe are {gold_summary.columns}")
    for file in glob.glob(os.path.join(args.input_dir, "*gpt-4o-mini.json")):
        result_per_file = {}
        logger.debug(f"Processing the file {file}")
        file_id = int(os.path.basename(file).split("_")[0])
        # gold_reference = gold_summary[gold_summary["entity_id"] == file_id]["reference"].values[0][0]
        # old_summary = gold_summary[gold_summary["entity_id"] == file_id]["generated_summary"].values[0][0]
        result_per_file["file_id"] = file_id
        summary_df = pd.read_json(file, lines=True)
        
        summary_df["aspects"] = summary_df["aspects"].apply(lambda x: x["review"] if "review" in x else x)
        summary_df = unpack_evidence(summary_df, uniform=False, device="cpu")
        aspect_list = summary_df['pure_aspects'].unique()
        logger.debug(summary_df['pure_aspects'].unique())

        aspect_counts = summary_df['pure_aspects'].value_counts().to_dict()
        logger.info(f"The aspect counts are {aspect_counts}")
        cluster = cluster_aspects(summary_df['pure_aspects'].unique(), 'hac', device='cpu')
        cluster_dict = unpack_matrix(cluster, aspect_list)
        processed_cluster = {}
        for k, v in cluster_dict.items():
            counts = []
            aspects = aspect_list[v]
            # logger.debug(f"The cluster {k} has the aspects {aspects}")
            if k == 3 * len(aspect_list):
                continue
            for aspect in aspects:
                counts.append(aspect_counts[aspect])
            cluster_dict[k] = aspects[np.argmax(counts)]
            for i in range(len(aspects)):
                processed_cluster[aspects[i]] = aspects[np.argmax(counts)]
        logger.debug(f"The processed cluster dict is {processed_cluster}")
        summary_df['pure_aspects'] = summary_df['pure_aspects'].apply(lambda x: processed_cluster[x] if x in processed_cluster else x)
        
        if len(summary_df['pure_aspects'].unique()) > 0 and not os.path.exists(os.path.join(args.input_dir, f"aspect_schema/hierarchy_{file_id}.json")): 
            ac = summary_df['pure_aspects'].value_counts().to_dict()
            llm_result = LLM_hierachy(list(ac.keys()), "gpt-5-mini")
            with open(os.path.join(args.input_dir, f"aspect_schema/hierarchy_{file_id}.json"), "w") as f:
                json.dump(llm_result, f, indent=2)
        elif len(summary_df['pure_aspects'].unique()) > 0:
            with open(os.path.join(args.input_dir, f"aspect_schema/hierarchy_{file_id}.json"), "r") as f:
                llm_result = json.load(f)

        [g_tuples, base_score] = arg_score_sentiment(summary_df, llm_result["hotel"], normalize="aspect")
        
        us = uncertatinty_score(summary_df, method="easy_score")
        # args = [('convention center', 'building', 's'), ('accessibility', 'building', 's'), ('towels', 'rooms', 's'), ('cleanliness', 'rooms', 's'), ('noise', 'rooms', 'a'), ('wifi', 'rooms', 's'), ('pet policy', 'rooms', 's'), ('gym', 'service', 's'), ('spa', 'service', 's'), ('food', 'service', 's')]

        # arg_strengths = {'accessibility': 0.002232142857142857, 'building': 0.026785714285714284, 'cleanliness': 0.006696428571428571, 'convention center': 0.002232142857142857, 'food': 0.04017857142857143, 'gym': 0.004464285714285714, 'location': 0.171875, 'noise': -0.006696428571428571, 'pet policy': 0.004464285714285714, 'price': 0.004464285714285714, 'rooms': 0.17857142857142858, 'service': 0.16294642857142858, 'spa': 0.008928571428571428, 'towels': 0.0, 'wifi': 0.006696428571428571}

        graph = graph_construction(g_tuples, base_score)
        arg_stre_final = DF_QUAD(graph, summary_df['pure_aspects'].unique())
        for key, value in us.items():
            if key in arg_stre_final:
                arg_stre_final[key] *= (1 - value)
                # arg_stre_final[key] *= value

        generated_summary = argument_selection(graph, arg_stre_final, us, method="top_k", **{"k": 8})
        # logger.info("The gold reference summary is \n" + gold_reference)
        # logger.info("The old generated summary is \n" + old_summary)
        result_per_file["generated_summary"] = generated_summary
        result.append(result_per_file)
    with open(os.path.join(args.output_dir, "argumentation_summary_all_results.json"), "w") as f:
        for r in result:
            f.write(json.dumps(r) + "\n")
        
