import torch
from src.bertkeywords.src.similarities import Embedding, Similarities
from src.bertkeywords.src.keywords import Keyword, Keywords
from src.coherencegraph.coherence_graph import CoherenceNode, CoherenceGraph
from src.coherencegraph.utils import check_similarity, get_similarity
from src.dataset.utils import dedupe_list, flatten, truncate_string, truncate_by_token
import networkx as nx
import time
import numpy as np

supported_models = [
    "sentence-transformers/LaBSE",
    "bert-base-uncased",  # default
    "roberta-base",
    "sentence-transformers/all-MiniLM-L6-v2",
]


class Coherence:
    def __init__(
        self,
        max_words_per_step=4,
        coherence_threshold=0.5,
        model_string="bert-base-uncased",
        keyword_diversity=0.0,
        diverse_keywords=False,
        similar_keywords=True,
        ablation=False,  # remove the coherence if set to True.
        include_numeric_keywords=True,
        scoring_factor=2,
    ):
        self.max_words_per_step = max_words_per_step
        self.coherence_threshold = coherence_threshold
        self.model_string = model_string
        self.keyword_diversity = keyword_diversity

        self.diverse_keywords = diverse_keywords
        self.similar_keywords = similar_keywords
        self.scoring_factor = scoring_factor

        if model_string not in supported_models:
            self.model_string = "bert-base-uncased"

        similarities_lib = Similarities(self.model_string)

        self.keywords_lib = Keywords(similarities_lib.model, similarities_lib.tokenizer, self.model_string)
        self.embedding_lib = Embedding(
            similarities_lib.model, similarities_lib.tokenizer
        )

        self.ablation = ablation
        self.include_numeric_keywords = include_numeric_keywords

        self.prev_sentence = None

    def predict_by_chain_count(
        self,
        text_data,
        max_tokens=128,
        max_graph_depth=7,
        batch_size=1,
    ):
        predictions = []
        scores = []
        prev_sentence = None

        print(f"TOTAL BATCHES: {len(text_data) // batch_size}\n")

        G = CoherenceGraph(coherence_threshold=self.coherence_threshold)
        temp_prev_graph = nx.Graph()
        prev_num_chains = 0

        # set up batching
        for j, batch_num in enumerate(range(0, len(text_data) // batch_size)):
            # create the current batch to iterate over.
            # this method relies on previous sentence as it always keeps track
            curr_batch = text_data[
                batch_num * batch_size : batch_num * batch_size + batch_size
            ]

            curr_batch = [truncate_by_token(x, max_tokens) for x in curr_batch]

            embedding_technique = (
                self.keywords_lib.get_batch_keywords_with_kb_embeddings
            )
            # get all the keywords per sentence and truncate at max number of words
            batch_keywords = [
                x[: self.max_words_per_step]
                for x in embedding_technique(
                    curr_batch,
                    diversity=self.keyword_diversity,
                    diverse_keywords=self.diverse_keywords,
                    similar_keywords=self.similar_keywords,
                    include_numeric_keywords=self.include_numeric_keywords,
                )
            ]

            batch_keywords = [[Keyword(z[0], z[1], z[2]) for z in x] for x in batch_keywords]
            
            # start iterating over the current batch
            for i, sentence in enumerate(batch_keywords):
                # add 1 distance to each word
                G.balance_graph()
                G.prune_max_depth(max_depth=max_graph_depth)
                num_chains = 0
                for word in sentence:
                    node = CoherenceNode(word.text, word.embedding, word.importance)
                    # for n in G.get_nodes_at_distance(distance=1):
                    #     if check_similarity(torch.Tensor(n.vector), torch.Tensor(node.vector), coherence_threshold=self.coherence_threshold):
                    #         temp_graph.add_edge(node, n, weight=get_similarity(torch.Tensor(n.vector), torch.Tensor(node.vector)))
                                                
                    # G = nx.compose(G,temp_graph)
                    # temp_prev_graph.add_node(node)
                    # add the node. If it was already added, it won't be added again
                    G.add_node(node)
                    # create the unique chains and memoize
                    node.process_unique_chains(G)

                    # prediction by chain count ----
                    if j != 0:
                        chains = node.get_unique_chains()
                        num_chains += len(chains)
                if i == 0 and j == 0:
                    prediction = 1
                    predictions.append(prediction)
                    scores.append(-1) # beginning of scores
                    print(".", end="")
                else:
                    prediction = 1 if num_chains < (prev_num_chains//(self.scoring_factor)) else 0
                    # if num_chains == 0:
                    #     prediction = 1
                    if prediction == 1: 
                        # G.prune_max_depth(max_depth=1)
                        prev_num_chains = num_chains
                    else:
                        prev_num_chains = num_chains

                    predictions.append(prediction)
                    scores.append((prev_num_chains//(self.scoring_factor)))
                    print(".", end="")
                        
            print(f" {batch_num+1} ", end="")

        return predictions, scores

    def predict_by_weighted_count(
        self,
        text_data,
        max_tokens=128,
        max_graph_depth=7,
        batch_size=1,
    ):
        predictions = []
        scores = []

        print(f"TOTAL BATCHES: {len(text_data) // batch_size}\n")

        G = CoherenceGraph(coherence_threshold=self.coherence_threshold)
        prev_similarity = 0

        # set up batching
        for j, batch_num in enumerate(range(0, len(text_data) // batch_size)):
            # create the current batch to iterate over.
            # this method relies on previous sentence as it always keeps track
            curr_batch = text_data[
                batch_num * batch_size : batch_num * batch_size + batch_size
            ]

            curr_batch = [truncate_by_token(x, max_tokens) for x in curr_batch]

            embedding_technique = (
                self.keywords_lib.get_batch_keywords_with_kb_embeddings
            )
            # get all the keywords per sentence and truncate at max number of words
            batch_keywords = [
                x[: self.max_words_per_step]
                for x in embedding_technique(
                    curr_batch,
                    diversity=self.keyword_diversity,
                    diverse_keywords=self.diverse_keywords,
                    similar_keywords=self.similar_keywords,
                    include_numeric_keywords=self.include_numeric_keywords,
                )
            ]

            batch_keywords = [[Keyword(z[0], z[1], z[2]) for z in x] for x in batch_keywords]
            
            # start iterating over the current batch
            for i, sentence in enumerate(batch_keywords):
                # add 1 distance to each word
                G.balance_graph()
                G.prune_max_depth(max_depth=max_graph_depth)
                num_chains = 0
                similarities = []
                for word in sentence:
                    node = CoherenceNode(word.text, word.embedding, word.importance)
                    # for n in G.get_nodes_at_distance(distance=1):
                    #     if check_similarity(torch.Tensor(n.vector), torch.Tensor(node.vector), coherence_threshold=self.coherence_threshold):
                    #         temp_graph.add_edge(node, n, weight=get_similarity(torch.Tensor(n.vector), torch.Tensor(node.vector)))
                                                
                    # G = nx.compose(G,temp_graph)
                    # temp_prev_graph.add_node(node)
                    # add the node. If it was already added, it won't be added again
                    G.add_node(node)
                    # create the unique chains and memoize
                    node.process_unique_chains(G)
                    weighted_count = 0

                    # prediction by weighted similarity ----
                    if j != 0 or i != 0:
                        for prev_node in G.get_nodes_at_distance(distance=1):
                            # # don't consider this previous node if it isn't highly related to the current node.
                            # if not check_similarity(torch.Tensor(n.vector), torch.Tensor(node.vector), coherence_threshold=G.coherence_threshold):
                            #     continue
                            # get the similarity between the current node and the previous node. 
                            # multiply by the importance of current node
                            similarity = node.importance * get_similarity(torch.Tensor(prev_node.vector), torch.Tensor(node.vector))
                            # multiply by the number of chains into the previous node
                            # emphasizing the importance of that chain (theme).
                            weighted_count = (len(prev_node.get_unique_chains())+1) * similarity
                            similarities.append(weighted_count)

                if j == 0 and i == 0:
                    prediction = 1
                    predictions.append(prediction)
                    scores.append(-1) # beginning of the score
                    print(".", end="")
                else:
                    # print(similarities)
                    if len(similarities) == 0:
                        prediction = 0
                        predictions.append(prediction)
                        scores.append(0)
                    else:
                        total_similarity = torch.sum(torch.Tensor(similarities))
                        if total_similarity < (prev_similarity*(self.scoring_factor)):
                            prediction = 1
                        else:
                            prediction = 0
                        prev_similarity = total_similarity
                        predictions.append(prediction)
                        scores.append((prev_similarity*(self.scoring_factor)))
                        if prediction == 1:
                            G.prune_max_depth(max_depth=1)
                    print(".", end="")
                        
            print(f" {batch_num+1} ", end="")

        return predictions, scores
