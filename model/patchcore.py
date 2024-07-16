import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss

from .sampler import GreedyCoresetSampler, ApproximateGreedyCoresetSampler
from .scorer import NearestNeighbourScorer, FaissNN

class PatchCore:
    def __init__(self, args):
        self.percentage = args.percentage
        self.device = torch.device(args.device)
        self.dim_coreset_feat = args.dim_coreset_feat
        self.num_nn = args.num_nn
        # self.extractor = extractor

        if self.device.type != 'cuda':
            self.on_gpu = False
        else:
            self.on_gpu = True
        
        # self.sampler = ApproximateGreedyCoresetSampler(self.percentage, self.device, 
        #                                                dimension_to_project_features_to=self.dim_coreset_feat)
        self.sampler = GreedyCoresetSampler(self.percentage, self.device, 
                                           dimension_to_project_features_to=self.dim_coreset_feat)

        self.scorer = NearestNeighbourScorer(self.num_nn, FaissNN(self.on_gpu))

    def fit(self, features):
        features = self.sampler.run(features)
        # if features is not empty
        if features.shape[0] > 0:
            self.scorer.fit(detection_features = [features])
        return features

    def predict(self, features):
        scores = self.scorer.predict([features])[0]
        return scores
    
    def retrieve_features(self):
        index = self.scorer.nn_method.search_index
        features = np.array([index.reconstruct(i) for i in range(index.ntotal)])
        return features
