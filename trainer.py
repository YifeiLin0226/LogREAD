import logging
import functools
import pickle
import sys
import time

import faiss
import torch
import torch.optim as optim
from torch.nn.functional import adaptive_avg_pool1d
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding

from model.simcse import BertForCL, RobertaForCL
from model.patchcore import PatchCore
from data.dataset import LogDataset

# how to let the logger print as well as save the log file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('trainer.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Trainer:
    def __init__(self, args):
        for layer in args.layers_extract:
            assert layer in range(0, args.num_hidden_layers), "Layer index out of range"

        logger.info("args: %s", args)
        
        
        if args.model_name == 'bert':
            config = AutoConfig.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
            config.num_hidden_layers = args.num_hidden_layers
            config.layers_extract = args.layers_extract
            self.model = BertForCL.from_pretrained('google/bert_uncased_L-4_H-256_A-4', config = config)
            # self.model = BertForCL(config)
            self.tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
        
        else:
            raise NotImplementedError("Model not implemented")


        
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.patchcore = PatchCore(args)
        
    
    def pretrain(self):
        logger.info("Pretraining the model")
        self.model.train()
        self.model.to(self.device)

        dataset = LogDataset(self.args.dataset_name, self.args.dir, mode = 'train', train_ratio = self.args.train_ratio)
        collate_fn = functools.partial(dataset.collate_fn, tokenizer = self.tokenizer, augmentation = True)
        dataloader = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True, collate_fn = collate_fn, num_workers = 4, pin_memory = True)
        num_train_optimization_steps = int(len(dataset) / self.args.batch_size / self.args.gradient_accumulation_steps) * self.args.pretrain_epochs

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        warmup_steps = int(self.args.warmup_proportion * num_train_optimization_steps)
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps = self.args.adam_epsilon)
        # optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.args.lr, eps = self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = warmup_steps, 
                                                    num_training_steps = num_train_optimization_steps)
        
        

        for epoch in range(self.args.pretrain_epochs):
            self.model.train()
            running_loss = 0.0
            num_samples = 0
            for step, batch in enumerate(tqdm(dataloader, desc = f'Epoch {epoch}')):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids = batch

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = outputs['loss']

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()
                num_samples += input_ids.size(0) / 2
                running_loss += loss.item() * input_ids.size(0) / 2
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
            
            logger.info(f"Epoch {epoch} loss: {running_loss / num_samples}")
            print(f"Epoch {epoch} loss: {running_loss / num_samples}")
        
        self.model.save_pretrained(f'{self.args.dataset_name}_pretrained_model')
        self.tokenizer.save_pretrained(f'{self.args.dataset_name}_pretrained_tokenizer')


    def fill_memory(self):
        logger.info("Filling memory")

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f'{self.args.dataset_name}_pretrained_tokenizer')
        self.model = BertForCL.from_pretrained(f'{self.args.dataset_name}_pretrained_model')

        self.model.eval()
        self.model.to(self.device)

        dataset = LogDataset(self.args.dataset_name, self.args.dir, mode = 'train', train_ratio = self.args.train_ratio)
        collate_fn = functools.partial(dataset.collate_fn, tokenizer = self.tokenizer, augmentation = False)
        dataloader = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = False, collate_fn = collate_fn, num_workers = 4, pin_memory = True)

        total_features = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader, desc = 'Filling memory')):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids = batch
                outputs = self.model(input_ids, attention_mask, token_type_ids, sent_emb = True)
                hidden_features = outputs['hidden_states']
                hidden_size = hidden_features[0].size(-1)
                features = torch.cat([hidden_features[layer + 1][:, 0] for layer in self.args.layers_extract], dim = -1)
                features = adaptive_avg_pool1d(features, hidden_size)
                # print(step, features.shape)
                total_features.append(features.cpu().numpy())
        # save patchcore
        # with open(f'{self.args.dataset_name}_memory', 'wb') as f:
        #     pickle.dump(self.patchcore, f)
        total_features = np.concatenate(total_features, axis = 0)
        self.saved_features = self.patchcore.fit(total_features)
        
       
        

    def validate(self):
        logger.info("Validating")
        self.model.eval()
        self.model.to(self.device)

        dataset = LogDataset(self.args.dataset_name, self.args.dir, mode = 'valid', train_ratio = self.args.train_ratio)
        collate_fn = functools.partial(dataset.collate_fn, tokenizer = self.tokenizer, augmentation = False)
        dataloader = DataLoader(dataset, batch_size = 256, shuffle = False, collate_fn = collate_fn, num_workers = 4, pin_memory = True)

    
        anomaly_scores = []
        gt = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch
                outputs = self.model(input_ids, attention_mask, token_type_ids, sent_emb = True)
                features = outputs['pooler_output']
                scores = self.patchcore.predict(features.cpu().numpy())
                anomaly_scores.extend(scores)
                gt.extend(labels.cpu().numpy())
                

        
        assert len(anomaly_scores) == len(dataset)
        result = calc_score(gt, anomaly_scores)
        logger.info("K-nearest neighbour based anomaly detection")
        print("K-nearest neighbour based anomaly detection")
        logger.info(result)
        print(result)
        self.threshold = result['threshold']
        
    
    def predict(self):
        logger.info("Predicting")
        

        # load tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(f'{self.args.dataset_name}_pretrained_tokenizer')
        # self.model = BertForCL.from_pretrained(f'{self.args.dataset_name}_pretrained_model')
        self.model.eval()
        self.model.to(self.device)

        # load memory
        # with open(f'{self.args.dataset_name}_memory', 'rb') as f:
        #     self.patchcore = pickle.load(f)

        dataset = LogDataset(self.args.dataset_name, self.args.dir, mode = 'test', train_ratio = self.args.train_ratio)
        collate_fn = functools.partial(dataset.collate_fn, tokenizer = self.tokenizer, augmentation = False)
        dataloader = DataLoader(dataset, batch_size = 256, shuffle = False, collate_fn = collate_fn, num_workers = 4, pin_memory = True)

        
        anomaly_scores = []
        gt = []
        avg_time = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch
                start_time = time.time()
                outputs = self.model(input_ids, attention_mask, token_type_ids, sent_emb = True)
                features = outputs['pooler_output']
                scores = self.patchcore.predict(features.cpu().numpy())
                time_interval = time.time() - start_time
                avg_time.append(time_interval / len(input_ids))
                # if len(avg_time) % 1000 == 0:
                #     print("Average time:", np.mean(avg_time))
                anomaly_scores.extend(scores)
                gt.extend(labels.cpu().numpy())
                

        
        assert len(anomaly_scores) == len(dataset)
        result = calc_score(gt, anomaly_scores)
        logger.info("Optimal threshold based anomaly detection")
        print("Optimal threshold based anomaly detection")
        logger.info(result)
        print(result)

        result = calc_score(gt, anomaly_scores, thre = self.threshold)
        logger.info("Validation threshold based anomaly detection")
        print("Validation threshold based anomaly detection")
        logger.info(result)
        print(result)
        result['avg_time'] = np.mean(avg_time)


        # index = self.patchcore.scorer.nn_method.search_index
        # stored_features = np.array([index.reconstruct(i) for i in range(index.ntotal)])
        # print("Size of memory:", features.shape)

        # anomalous_features = []

        
        # with torch.no_grad():
        #     for batch in tqdm(dataloader):
        #         batch = tuple(t.to(self.device) for t in batch)
        #         input_ids, attention_mask, token_type_ids, labels = batch
        #         if torch.any(labels == 1):
        #             outputs = self.model(input_ids, attention_mask, token_type_ids, sent_emb = True)
        #             features = outputs['pooler_output'].cpu().numpy()
        #             anomalous_features.append(features[labels.cpu().numpy() == 1])
                
        
        # anomalous_features = np.concatenate(anomalous_features)
        # print(f"Length of anomalous features: {anomalous_features.shape[0]}")
        # # sample 1000
        # anomalous_features = anomalous_features[np.random.choice(anomalous_features.shape[0], 1000, replace = False)]

        # all_features = np.concatenate([stored_features, anomalous_features])
        # print(f"Length of all features: {all_features.shape[0]}")
        # labels = [0] * stored_features.shape[0] + [1] * anomalous_features.shape[0]
        # labels = np.array(labels)

        # print("Visualizing using LLE...")
        # lle = LocallyLinearEmbedding(n_components=2, n_neighbors=30, random_state=42)
        # lle_results = lle.fit_transform(all_features)
        # plt.figure(figsize=(10, 7))
        # plt.scatter(lle_results[labels == 0, 0], lle_results[labels == 0, 1], c='blue', label='Memory Bank (Normal Logs)', alpha=0.7)
        # plt.scatter(lle_results[labels == 1, 0], lle_results[labels == 1, 1], c='red', label='Anomalous Logs', alpha=0.7)
        # plt.legend()
        # plt.title('LLE Visualization of Memory Bank and Anomalous Log Embeddings')
        # plt.xlabel('LLE Component 1')
        # plt.ylabel('LLE Component 2')
        # plt.savefig(f'{self.args.dataset_name}_lle.png')

        # # clear canvas
        # plt.clf()

        # print("Visualizing using t-SNE...")
        # tsne = TSNE(n_components=2, random_state=42)
        # tsne_results = tsne.fit_transform(all_features)
        # plt.figure(figsize=(10, 7))
        # plt.scatter(tsne_results[labels == 0, 0], tsne_results[labels == 0, 1], c='blue', label='Memory Bank (Normal Logs)', alpha=0.7)
        # plt.scatter(tsne_results[labels == 1, 0], tsne_results[labels == 1, 1], c='red', label='Anomalous Logs', alpha=0.7)
        # plt.legend()
        # plt.title('t-SNE Visualization of Memory Bank and Anomalous Log Embeddings')
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')
        # plt.savefig(f'{self.args.dataset_name}_tsne.png')
        return result, anomaly_scores, gt
        


    def online(self):
        logger.info("Online mode")
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(f'{self.args.dataset_name}_pretrained_tokenizer')
        self.model = BertForCL.from_pretrained(f'{self.args.dataset_name}_pretrained_model')
        self.model.eval()
        self.model.to(self.device) 

        dataset = LogDataset(self.args.dataset_name, self.args.dir, mode = 'test', train_ratio = self.args.train_ratio)
        collate_fn = functools.partial(dataset.collate_fn, tokenizer = self.tokenizer, augmentation = False)
        dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, collate_fn = collate_fn, num_workers = 4, pin_memory = True)

        
        TP = 0
        FP = 0
        TN = 0
        FN = 0


        window_pred = []
        window_gt = []
        avg_time = 0
        num_samples = 0
        cur_prec = []
        cur_rec = []
        cur_f1 = []
        with torch.no_grad():
            num_added = 0
            log_count = 0
            cur_gt = 0
            cur_pred = 0
            for step, batch in enumerate(tqdm(dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch
                start_time = time.time()
                outputs = self.model(input_ids, attention_mask, token_type_ids, sent_emb = True)
                features = outputs['pooler_output'].cpu().numpy()
                scores = self.patchcore.predict(features)
                avg_time += time.time() - start_time
                num_samples += 1

                flag_fn = False
                flag_fp = False
                if scores >= self.threshold:
                    if labels == 1:
                        TP += 1
                    else:
                        FP += 1
                        flag_fp = True
                    
                    cur_pred = 1
                
                else:
                    if labels == 0:
                        TN += 1
                    else:
                        FN += 1
                        flag_fn = True
                    
                if labels == 1:
                    cur_gt = 1
                
                if flag_fp:
                    self.patchcore.scorer.fit(detection_features = [features])
                    num_added += 1

                    if num_added == 32:
                        index = self.patchcore.scorer.nn_method.search_index
                        print("Size of memory:", index.ntotal)
                        # retrieve 32 features
                        features = np.array([index.reconstruct(i) for i in range(index.ntotal)])
                        # reset the index
                        self.patchcore.scorer.nn_method.reset_index()
                        self.patchcore.scorer.fit(detection_features = [features[:-32]])
                        self.patchcore.fit(features[-32:])
                        index = self.patchcore.scorer.nn_method.search_index
                        print("Size of memory after reducing:", index.ntotal)
                        num_added = 0

                elif flag_fn:
                    print("Retraining")
                    logger.info("Retraining")

                    prev_features = torch.tensor(features).to(self.device)
                    prev_params = [param.clone() for param in self.model.parameters()]
                    prev_params = torch.cat([param.detach().view(-1) for param in prev_params])

                    param_optimizer = list(self.model.named_parameters())
                    no_decay = ['bias', 'LayerNorm.weight']
                    optimizer_grouped_parameters = [
                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                    ]
                    
                    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.retrain_lr, eps = self.args.adam_epsilon)
                    # optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.args.lr, eps = self.args.adam_epsilon)
                    

                    for epoch in range(self.args.retrain_epochs):
                        self.model.eval()
                        features = self.model(input_ids, attention_mask, token_type_ids, sent_emb = True)['pooler_output']
                        query_distances, query_indices = self.patchcore.scorer.nn_method.run(self.patchcore.num_nn, features.cpu().numpy())
                        if np.mean(query_distances) >= self.threshold:
                            print("Early stopping")
                            logger.info("Early stopping")
                            break
                        
                        self.model.train()
                        with torch.enable_grad():

                            features = self.model(input_ids, attention_mask, token_type_ids, sent_emb = True)['pooler_output']
                            query_distances, query_indices = self.patchcore.scorer.nn_method.run(self.patchcore.num_nn, features.detach().cpu().numpy())
                            loss = 0
                           
                            for index in query_indices.squeeze():
                                target_vector = self.patchcore.scorer.nn_method.search_index.reconstruct(int(index))
                                target_vector = torch.tensor(target_vector).to(self.device).squeeze()
                                loss += torch.linalg.norm(features - target_vector)
                            
                            loss = torch.mean(loss)
                            loss = max(torch.tensor(self.threshold) * 2 - loss, 0)

                            # add regularization
                            cur_params = torch.cat([param.view(-1) for param in self.model.parameters()])
            
                            loss += torch.linalg.norm(cur_params - prev_params) * self.args.retrain_lambda
                            loss += torch.linalg.norm(features - prev_features) * self.args.retrain_lambda

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        print(f"Epoch {epoch} loss: {loss.item()}")
                        logger.info(f"Epoch {epoch} loss: {loss.item()}")

                    self.model.eval()
                
                log_count += 1
                # if log_count == self.args.window_size:
                #     window_pred.append(cur_pred)
                #     window_gt.append(cur_gt)
                #     log_count = 0
                #     cur_gt = 0
                #     cur_pred = 0

                if log_count % 1000 == 0:
                    try:
                        prec = TP / (TP + FP)
                    except ZeroDivisionError:
                        prec = 0
                    try:
                        rec = TP / (TP + FN)
                    except ZeroDivisionError:
                        rec = 0
                    try:
                        f1 = 2 * prec * rec / (prec + rec)
                    except ZeroDivisionError:
                        f1 = 0
                    cur_prec.append(prec)
                    cur_rec.append(rec)
                    cur_f1.append(f1)

            # if log_count != 0:
            #     window_pred.append(cur_pred)
            #     window_gt.append(cur_gt)






        # plt.plot(np.arange(len(cur_prec)), cur_prec, label = 'Precision')
        # plt.plot(np.arange(len(cur_rec)), cur_rec, label = 'Recall')
        # plt.plot(np.arange(len(cur_f1)), cur_f1, label = 'F1')
        # plt.legend()
        # plt.savefig(f'online_{self.args.dataset_name}.png')
                
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        logger.info("Online mode result")
        print("Online mode result")
        logger.info({"precision": precision, "recall": recall, "f1": f1})
        print({"precision": precision, "recall": recall, "f1": f1})

        # window_precision = precision_score(window_gt, window_pred)
        # window_recall = recall_score(window_gt, window_pred)
        # window_f1 = f1_score(window_gt, window_pred)


        # index = self.patchcore.scorer.nn_method.search_index
        # stored_features = np.array([index.reconstruct(i) for i in range(index.ntotal)])
        # print("Size of memory:", features.shape)

        # anomalous_features = []
        # with torch.no_grad():
        #     for batch in dataloader:
        #         batch = tuple(t.to(self.device) for t in batch)
        #         input_ids, attention_mask, token_type_ids, labels = batch
        #         if labels == 1:
        #             outputs = self.model(input_ids, attention_mask, token_type_ids, sent_emb = True)
        #             features = outputs['pooler_output'].cpu().numpy()
        #             anomalous_features.append(features.squeeze())
        
        # anomalous_features = np.array(anomalous_features)
        # all_features = np.concatenate([stored_features, anomalous_features])
        # labels = [0] * stored_features.shape[0] + [1] * anomalous_features.shape[0]

        # print("Visualizing using LLE...")
        # lle = LocallyLinearEmbedding(n_components=2, n_neighbors=30, random_state=42)
        # lle_results = lle.fit_transform(all_features)
        # plt.figure(figsize=(10, 7))
        # plt.scatter(lle_results[labels == 0, 0], lle_results[labels == 0, 1], c='blue', label='Memory Bank (Normal Logs)', alpha=0.7)
        # plt.scatter(lle_results[labels == 1, 0], lle_results[labels == 1, 1], c='red', label='Anomalous Logs', alpha=0.7)
        # plt.legend()
        # plt.title('LLE Visualization of Memory Bank and Anomalous Log Embeddings')
        # plt.xlabel('LLE Component 1')
        # plt.ylabel('LLE Component 2')
        # plt.savefig(f'{self.args.dataset_name}_lle.png')

        # # clear canvas
        # plt.clf()

        # print("Visualizing using t-SNE...")
        # tsne = TSNE(n_components=2, random_state=42)
        # tsne_results = tsne.fit_transform(all_features)
        # plt.figure(figsize=(10, 7))
        # plt.scatter(tsne_results[labels == 0, 0], tsne_results[labels == 0, 1], c='blue', label='Memory Bank (Normal Logs)', alpha=0.7)
        # plt.scatter(tsne_results[labels == 1, 0], tsne_results[labels == 1, 1], c='red', label='Anomalous Logs', alpha=0.7)
        # plt.legend()
        # plt.title('t-SNE Visualization of Memory Bank and Anomalous Log Embeddings')
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')
        # plt.savefig(f'{self.args.dataset_name}_tsne.png')




        # return {"precision": precision, "recall": recall, "f1": f1, "window_precision": window_precision, "window_recall": window_recall, "window_f1": window_f1, "avg_time": avg_time / num_samples}
        return {"precision": precision, "recall": recall, "f1": f1, "avg_time": avg_time / num_samples}, cur_prec, cur_rec, cur_f1

                    
        
        



        
def calc_score(gt, pred, thre = None):
    window_gt = [max(gt[i:i + 128]) for i in range(0, len(gt), 128)]
    window_pred = [max(pred[i:i + 128]) for i in range(0, len(pred), 128)]
    auc = roc_auc_score(gt, pred)
    ap = average_precision_score(gt, pred)
    # window_auc = roc_auc_score(window_gt, window_pred)
    # window_ap = average_precision_score(window_gt, window_pred)
    prs, recs, thres = precision_recall_curve(gt, pred)
    w_prs, w_recs, w_thres = precision_recall_curve(window_gt, window_pred)

    if thre is not None:
        f1 = f1_score(gt, pred > thre)
        precision = precision_score(gt, pred > thre)
        recall = recall_score(gt, pred > thre)
        window_f1 = f1_score(window_gt, window_pred > thre)
        window_precision = precision_score(window_gt, window_pred > thre)
        window_recall = recall_score(window_gt, window_pred > thre)
        return {'auroc': auc, 'ap': ap, 'f1': f1, 'precision': precision, 'recall': recall, 'window_f1': window_f1, 'window_precision': window_precision, 'window_recall': window_recall, 'threshold': thre}
    
    f1 = np.divide(2 * prs * recs, (prs + recs), out=np.zeros_like(prs), where=(prs + recs) != 0)
    valid_indices = np.where(~np.isnan(f1) & (f1 > 0))[0]
    max_f1_idx = valid_indices[f1[valid_indices].argmax()]
    threshold = thres[max_f1_idx]
    precision = prs[max_f1_idx]
    recall = recs[max_f1_idx]
    f1 = f1[max_f1_idx]

    window_f1 = np.divide(2 * w_prs * w_recs, (w_prs + w_recs), out=np.zeros_like(w_prs), where=(w_prs + w_recs) != 0)
    valid_indices = np.where(~np.isnan(window_f1) & (window_f1 > 0))[0]
    max_f1_idx = valid_indices[window_f1[valid_indices].argmax()]
    window_threshold = w_thres[max_f1_idx]
    window_precision = w_prs[max_f1_idx]
    window_recall = w_recs[max_f1_idx]
    return {'auroc': auc, 'ap': ap, 'f1': f1, 'precision': precision, 'recall': recall, 'threshold': threshold, 'window_f1': window_f1[max_f1_idx], 'window_precision': window_precision, 'window_recall': window_recall, 'window_threshold': window_threshold}

    
    