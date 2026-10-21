import networkx as nx

from loguru import logger

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity



def pagerank_most_entail_sentence(list_of_sentences, device=None):
    setence_embedding = obtain_embeddings_stf(list_of_sentences, device)
    setence_similarity = cosine_similarity(setence_embedding)
    nx_graph = nx.from_numpy_array(setence_similarity)
    scores = nx.pagerank(nx_graph)
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    logger.debug(f"The highest score evidences for this cluster are: \n{list(scores.values())[:4]}")
    return list_of_sentences[list(scores.keys())[0]]

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
    # preprocess = [l.lower() for l in list_of_words if isinstance(l,str) else ]
    embeddings = model.encode(preprocess)
    if similarity:
        similarities = model.similarity(embeddings[:-1], [embeddings[-1]])
        return similarities
    del model
    return embeddings

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
    return clustering