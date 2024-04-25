import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
    
class DecisionTree:
    
    def __init__(self, a_dict: dict):
        self.a_dict = a_dict
        self.a_map = self.__build_feature_map__()
        
        self.x = pd.DataFrame()
        self.y = np.ndarray([])
        
        self._root = None 
        self._root_name = None
        self.depth = 0
        
        self.p_counts = {}
        self.a_counts = {}
        self.gains = {}
        
        self.p = 0
        self.data_entropy = 0
        
    def __build_feature_map__(self):
        a_map = {}
        for a_name, a_values in self.a_dict.items():
            a_map[a_name] = {v : v_idx for v_idx, v in enumerate(a_values)}
        return a_map
    
    def get_common_label(self, y = None):
        if y is None:
            y = self.y
        p = len(np.where(y)[0])
        p_ = len(np.where(~y)[0])
        
        return self.labels[p > p_]
    
    # Returns the positive proportion of labels. p = # positive labels / # labels
    def get_binary_p(self):
        return self.y.sum()/len(self.y)

    # Returns the dataset Entropy given the proportion of positive labels for a binary classification.
    def get_entropy(self, p):
        eps = 1e-10
        if type(p) is np.ndarray:
            p[p == 0.0] = eps
            p[p == 1.0] = 1.0 - eps
        else:
            p = 1.0 if p > 1.0 else p
            p = p + eps if p < 1.0 else p - eps
        return -p*np.log2(p) - (1-p)*np.log2(1-p)

    # Returns the information gain for an attribute 'a_i'.
    def get_info_gain(self, a_name, y):
        S_length = len(y)
        
        a_i = self.a_dict[a_name]
        
        Sv_lengths = np.array([self.a_counts.get((a_name, v), 0) for v in a_i])
        p_counts = np.array([self.p_counts.get((a_name, v), 0) for v in a_i])
        
        nz_idx = Sv_lengths.nonzero()
                
        length_ratios = Sv_lengths[nz_idx] / (np.ones_like(Sv_lengths[nz_idx])*S_length)
        Sv_entropies = self.get_entropy(p_counts[nz_idx]/Sv_lengths[nz_idx])
        return self.data_entropy - np.sum(length_ratios*Sv_entropies)

    # Calculates and returns the information gains (Gain(S,A)) for all attributes a_i in A in a python list.
    def calculate_gains(self, features: pd.DataFrame, y: np.ndarray) -> dict:
        
        for a_name, a_features in features.iteritems():
            for v_idx, v in a_features.items():
                self.p_counts[(a_name, v)] = self.p_counts.get((a_name, v), 0) + int(y[v_idx])
                self.a_counts[(a_name, v)] = self.a_counts.get((a_name, v), 0) + 1
 
        return {a_i : self.get_info_gain(a_i, y) for a_i in self.a_dict.keys()}
    
    def get_max_gain(self, gains:dict = None):
        if(gains is None):
            gains = self.gains
        return max(gains, key=gains.get), max(gains.values())
    
    def _add_child(self, node, parent, child, child_idx):
        node[parent][child_idx] = {child : [None]*len(self.a_dict[child])}
    
    def _pop_max_gain(self, gains):
        if(gains is self.gains):
            return self.get_max_gain()
        
        max_gain_key, _ = self.get_max_gain(gains=gains)
        return max_gain_key, gains.pop(max_gain_key)
    
    def _ID3_iterative(self, max_height=None):
        
        if self.y.all() or (~self.y).all():
            self.root = self.y[0]
            return
        
        param_stack = []
        root_node = {}
        
        a_names = {a_name : i for i, a_name in enumerate(list(self.x.columns))}
        features = self.x.copy().to_numpy()
        
        gains = dict(self.gains)
        
        root, _ = self._pop_max_gain(gains=gains)
        root_node[root] = [None]*len(self.a_dict[root])
        max_depth = 0
        param_stack.append((root, root_node, (features, self.y.copy()), gains, 0))
        while len(param_stack) > 0:
            
            a_max, node, s, gains, depth = param_stack.pop()
            x, y = s
                
            if y.all() or (~y).all():
                depth+=1
                max_depth = max(max_depth, depth)
                
                node[a_max] = bool(y[0])
                continue
            
            new_gains = dict(gains)
            new_a_max, _ = self._pop_max_gain(gains=new_gains)
            for v_idx, v in enumerate(self.a_dict[a_max]):
                
                sv_idxs = np.where(x[:, a_names[a_max]] == v)
                
                new_x = x[sv_idxs].copy()
                new_y = y[sv_idxs].copy()
                if len(new_x) == 0:
                    max_depth = max(max_depth, depth+1)
                    
                    node[a_max][v_idx] = self.common
                    continue
                
                if(max_height is not None and depth+1 >=  max_height):
                    max_depth = max(max_depth, depth+1)

                    node[a_max][v_idx] = self.common
                    continue
                    
                self._add_child(node, a_max, new_a_max, v_idx)
                
                param_stack.append((new_a_max, node[a_max][v_idx], (new_x, new_y), new_gains, depth + 1))
                
        return root_node, root, max_depth
    
    def _ID3(self, max_height=None):
            
        pass
    
    def get_tree(self) -> dict:
        return dict(self._root)
    
    def get_root_name(self):
        return self._root_name
    
    # Returns Decision Tree depth via Level Order Traversal
    def calculate_tree_depth(self):
        depth = 0
    
        q = []
        root_node = dict(self._root)

        q.append((self._root_name, root_node))
        q.append((None, None))
        while(len(q) > 0):
            
            node_name, node = q[0]; q = q[1:]
            if(node is None):
                depth += 1
            if(node is not None and type(node[node_name]) is not bool):     
                for new_node in node[node_name]:
                    if(new_node is None):
                        continue    
                    if(type(new_node) is bool):
                        q.append((0,[True])) 
                        continue 
                    new_node_name = list(new_node.keys())[0]
                    q.append((new_node_name, new_node))
                    
            elif(node is not None and type(node[node_name]) is bool):
                continue
            elif(len(q) > 0):
                q.append((None, None))
        return depth
                
    def train(self, x: pd.DataFrame, y:np.ndarray, labels: dict, max_height=None):
        self.labels = labels
        self._labels = {v : k for k, v in labels.items()}
        self.x = x
        self.y = y
        
        self.common = self._labels[self.get_common_label()]
        
        self.p = self.get_binary_p()
        self.data_entropy = self.get_entropy(self.p)
        
        self.gains = self.calculate_gains(x, y)
        self._root, self._root_name, self.depth = self._ID3_iterative(max_height=max_height)
    
    def _predict_example(self, features: pd.DataFrame):
        
        node = self.get_tree()
        while type(node) is not bool:
            node_name = list(node.keys())[0]
            v = features[node_name]
            
            node_children = node[node_name]
            
            new_node = node_children[self.a_map[node_name][v]] if type(node_children) is list else node_children
            node = new_node
            
        return node
    
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(shape=(x.shape[0],))
        for idx, row in x.iterrows():
            predictions[idx] = self._predict_example(features=row)
        
        return predictions
