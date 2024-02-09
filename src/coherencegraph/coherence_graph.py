import networkx as nx
import matplotlib.pyplot as plt
import uuid

class CoherenceNode():
    def __init__(self, word, vector, importance, distance=0):
        self.id = uuid.uuid1()
        self.word = word
        self.vector = vector
        self.importance = importance
        self.distance = distance # distance from current sentence (starts at 0)
        # self.level = level
    
    def __repr__(self):
        return f'Node({self.id}, \'{self.word}\', {self.vector})'
        
    def __str__(self):
        return f'Node(\'{self.word}\')'

class CoherenceGraph(nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)

    # overriden function to add more functionality to the traditional add_edge function
    def add_edge(self, *args, **kwargs):
        super().add_edge(*args, **kwargs)

    def empty_graph(self):
        nx.Graph.__init__(self)

    # get a node based on an id
    def get_node_by_id(self, id: str):
        for n, nbrs in self.adj.items():
            if n.id == id:
                return (n, nbrs)
        return None

    # get a node based on a word
    def get_node(self, word: str):
        for n, nbrs in self.adj.items():
            if n.word == word:
                return (n, nbrs)
        return None

    # get an edge between two nodes, which holds its weight
    def get_edge(self, node_one, node_two):
        return self.get_edge_data(node_one, node_two)

    # get all the nodes at a certain distance
    def get_nodes_at_distance(self, distance=0):
        nodes = []
        for n, nbrs in self.adj.items():
            if n.distance == distance:
                nodes.append(n)
        return nodes

    # remove all nodes that don't have any edges
    def prune_isolated_nodes(self):
        isolated_nodes = list(nx.isolates(self))
        self.remove_nodes_from(list(nx.isolates(self)))
        return isolated_nodes

    # remove all nodes beyond the max_depth
    def prune_max_depth(self, max_depth: int):
        pruned_nodes = []
        for n, nbrs in self.adj.items():
            if n.distance >= max_depth:
                pruned_nodes.append(n)

        self.remove_nodes_from(list(pruned_nodes))
        return pruned_nodes

    # add 1 distance to all the nodes in the graph
    # we call this function when we iterate to the next sentence in the segment
    def balance_graph(self):
        for n, _ in self.adj.items():
            n.distance += 1

    # remove all the nodes in the graph beyond a certain point
    # e.g., if nodes get too far away (i.e., are too far from the current sentence)
    def prune_max_distance(self, max_distance: int):
        pruned_nodes = []
        for n, _ in self.adj.items():
            try:
                if n.distance > max_distance:
                    pruned_nodes.append(n)
            except:
                # no shortest path could be found - node is isolated
                pass

        self.remove_nodes_from(list(pruned_nodes))
        return pruned_nodes

    # we want to get all the nodes that have some pathway to the current node
    # so we can gather all the cohesive words and make a word compilation vector.
    def get_all_paths_to_node(self, current_node: CoherenceNode):
        paths = []
        for n in self.nodes():
            for path in nx.all_simple_paths(self, source=current_node, target=n):
                paths.append(path)
        return paths

    def get_chain_vector(self, chain):
        chain_vector = 1
        for node in chain:
            chain_vector *= node.vector * node.importance * (1/(node.distance+1))
        return len(chain)*chain_vector
    
    def get_chain_importance(self, chain):
        importance = 0
        for node in chain:
            importance += node.importance
        return importance

    # get all the paths from the current node backward. 
    def get_chains_to_node(self, current_node: CoherenceNode):
        paths = []
        all_paths = self.get_all_paths_to_node(current_node)

        for path in all_paths:
            current_distance = 0
            path_valid = True
            for node in path:
                if node.distance < current_distance:
                    path_valid = False
                current_distance += 1
            if path_valid:
                paths.append(path)

        return paths

    def get_unique_chains_to_node(self, current_node: CoherenceNode):
        unique_chains = []
        chains = self.get_chains_to_node(current_node)
        for chain in chains:
            for inner_chain in chains:
                different = True
                if chain == inner_chain:
                    different = False
                    continue
                if len(inner_chain) > len(chain):
                    # if inner_chain not in unique_chains:
                    #     unique_chains.append(inner_chain)
                    continue
                for i in reversed(range(0, len(inner_chain))):
                    # print(str(chain[i]), str(inner_chain[i]))
                    if chain[i] != inner_chain[i]:
                        different = True
                        break
                    if chain[i] == inner_chain[i] and i == 0:
                        different = False
                        break
                if not different:
                    continue
                if inner_chain not in unique_chains:
                    unique_chains.append(inner_chain)
        
        # fine all the chains to remove
        chains_to_remove = []
        for j, chain in enumerate(unique_chains):
            for k, inner_chain in enumerate(unique_chains):
                if j == k:
                    # same chain, move forward.
                    break
                if len(chain) <= len(inner_chain):
                    found = False
                    # comparison phase
                    for i in range(0, len(chain)):
                        # check each iteration if each node is the same until the last node
                        # in the original chain.
                        if chain[i] == inner_chain[i] and i == (len(chain)-1):
                            # stopping case
                            chains_to_remove.append(chain)
                        if chain[i] != inner_chain[i]:
                            break
        
                    if found:
                        # move on to next chain
                        break

        unique_chains = [x for x in unique_chains if x not in chains_to_remove]
        return unique_chains
                    

    # debug by printing an easy to read representation of the chain
    # TODO: create new class for chain and override repr
    def print_chain(self, chain, with_weights=True):
        prev_node = None
        for node in chain:
            if prev_node is not None:
                if with_weights:
                    print(str(prev_node), "--", self.get_edge(prev_node, node)["weight"], "--> ", end="")
                else:
                    print(str(prev_node), "--> ", end="")
            prev_node = node
        print(str(node))