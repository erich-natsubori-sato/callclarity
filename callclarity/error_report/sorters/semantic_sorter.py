import numpy as np

from tqdm import tqdm
from loguru import logger

from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

class SemanticSorter:

    def __init__(self, embeddings = None, nn_graph = None, embedding_distance_step: float = 0.01, max_embedding_distance: float = 0.08):
        
        if not ((nn_graph != None) ^ (embeddings != None)):
            raise ValueError("One and only one of the following arguments must be provided: nn_graph, embeddings")

        self.embeddings = embeddings
        self.embedding_distance_step = embedding_distance_step
        self.max_embedding_distance = max_embedding_distance
        self.nn_graph = nn_graph if nn_graph != None else self._init_nn_graph()
        self.sorted_indices = None

    def _init_nn_graph(self):
        logger.info("Instantiating radius neighbors graph")
        nn_graph = radius_neighbors_graph(
            self.embeddings,radius = self.max_embedding_distance, mode = "distance", metric = "cosine", n_jobs = -1
            )
        
        return nn_graph

    def get_sort_indices(self):

        nn_graph = self.nn_graph
        coo_graph = nn_graph.tocoo()
        non_zero_cond = (coo_graph.data > 0)

        max_embedding_distance = self.max_embedding_distance
        sort_steps = np.arange(
            self.embedding_distance_step,
            max_embedding_distance + self.embedding_distance_step, 
            self.embedding_distance_step
            )

        sort_matrix = np.zeros(shape = (nn_graph.shape[0],len(sort_steps)))
        for index,sort_step in enumerate(tqdm(sort_steps)):
            
            # filter only connections that matter
            #   1. it has to be connected (non zero)
            #   2. it has to be sorted
            mask = non_zero_cond & (coo_graph.data <= sort_step)
            filtered_nn_graph = csr_matrix((coo_graph.data[mask], (coo_graph.row[mask], coo_graph.col[mask])), shape=nn_graph.shape)
            
            # calculate calculated components
            n_components, labels = connected_components(filtered_nn_graph)
            
            # sort and reindex, assign smaller indices for larger components
            #   1. create count dictionary
            #   2. reindex components according to size
            #   3. reassign components indices
            element_counts = dict(zip(*np.unique(labels, return_counts=True)))
            element_counts = dict(sorted(element_counts.items(), key = lambda item: item[1], reverse = True))
            index_map = {
                old_index:new_index
                for new_index, old_index in enumerate(element_counts.keys())
            }
            labels = np.array([index_map[old_index] for old_index in labels])
            
            # store indices in sort matrix
            sort_matrix[:,index] = labels

        # create indices sorted by multiple columns, hierarchically. 
        sorted_indices = np.lexsort(sort_matrix.T, axis = 0)

        # store the sorted indices in the class
        self.sorted_indices = sorted_indices

        return sorted_indices
