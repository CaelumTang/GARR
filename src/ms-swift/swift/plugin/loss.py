# Copyright (c) Alibaba, Inc. and its affiliates.
# Modified for GARR in 2026; see src/ms-swift/GARR_MODIFICATIONS.md.
import os
from enum import Enum
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import gather_object
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.utils import strtobool


def _parse_pair_sentence(outputs):
    if isinstance(outputs, dict):
        last_hidden_state = outputs['last_hidden_state']
    else:
        last_hidden_state = outputs
    batch_size = last_hidden_state.shape[0]
    shape_len = len(last_hidden_state.shape)
    first_sentence = list(range(0, batch_size, 2))
    second_sentence = list(range(1, batch_size, 2))
    if shape_len == 3:
        sentence1 = last_hidden_state[first_sentence][:, 0].squeeze(dim=1)
        sentence2 = last_hidden_state[second_sentence][:, 0].squeeze(dim=1)
    else:
        sentence1 = last_hidden_state[first_sentence]
        sentence2 = last_hidden_state[second_sentence]
    return sentence1, sentence2


# Code borrowed from sentence_transformers
class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)  # noqa
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)  # noqa
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)  # noqa


def cosine_similarity_func(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    cos_score_transformation = nn.Identity()
    loss_fct = MSELoss()
    sentence1, sentence2 = _parse_pair_sentence(outputs)
    output = cos_score_transformation(torch.cosine_similarity(sentence1, sentence2))
    return loss_fct(output, labels.to(output.dtype).view(-1))


def contrastive_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    sentence1, sentence2 = _parse_pair_sentence(outputs)
    distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
    distances = distance_metric(sentence1, sentence2)
    margin = 0.5
    labels = labels.to(sentence1.dtype)
    losses = 0.5 * (labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2))
    return losses.mean()


def calculate_paired_metrics(embeddings, labels):
    from sklearn.metrics.pairwise import (paired_cosine_distances, paired_euclidean_distances,
                                          paired_manhattan_distances)
    from scipy.stats import pearsonr, spearmanr

    embeddings1, embeddings2 = _parse_pair_sentence(embeddings)
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
    eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

    eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
    eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

    eval_pearson_dot, _ = pearsonr(labels, dot_products)
    eval_spearman_dot, _ = spearmanr(labels, dot_products)

    return {
        'pearson_cosine': eval_pearson_cosine,
        'pearson_euclidean': eval_pearson_euclidean,
        'pearson_manhattan': eval_pearson_manhattan,
        'pearson_dot_product': eval_pearson_dot,
        'spearman_cosine': eval_spearman_cosine,
        'spearman_euclidean': eval_spearman_euclidean,
        'spearman_manhattan': eval_spearman_manhattan,
        'spearman_dot_product': eval_spearman_dot,
    }


def calculate_infonce_metrics(embeddings, labels):
    hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)
    use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
    if hard_negatives is not None:
        hard_negatives = int(hard_negatives)
    split_tensors = _parse_multi_negative_sentences(torch.tensor(embeddings), torch.tensor(labels), hard_negatives)
    split_tensors = [t.numpy() for t in split_tensors]
    can_batched = hard_negatives is not None
    if hard_negatives is None and len(set([s.shape[0] for s in split_tensors])) == 1:
        can_batched = True
    all_similarity_matrix = []
    all_labels = []
    pos_neg_margins = []
    if not use_batch:
        if can_batched:
            sentences = np.stack(split_tensors, axis=0)
            similarity_matrix = np.matmul(sentences[:, 0:1], sentences[:, 1:].transpose((0, 2, 1))).squeeze(1)
            all_similarity_matrix.append(similarity_matrix)
            labels = np.zeros_like(similarity_matrix)
            labels[:, 0] = 1
            all_labels.append(labels)
        else:
            for tensor in split_tensors:
                similarity_matrix = np.matmul(tensor[0], tensor[1:].T)
                all_similarity_matrix.append(similarity_matrix)
                labels = np.zeros_like(similarity_matrix)
                labels[0] = 1
                all_labels.append(labels)
                max_neg_scores = np.max(similarity_matrix[labels == 0], axis=-1)
                pos_neg_margins.append(np.mean(similarity_matrix[labels == 1] - max_neg_scores).item())
    else:
        if can_batched:
            sentences = np.stack(split_tensors, axis=0)
            similarity_matrix = np.matmul(sentences[:, 0], sentences[:, 1:].reshape(-1, sentences.shape[2]).T)
            all_similarity_matrix.append(similarity_matrix)
            labels = np.zeros_like(similarity_matrix)
            for row, col in enumerate(range(0, sentences.shape[0] * (sentences.shape[1] - 1), sentences.shape[1] - 1)):
                labels[row, col] = 1
            all_labels.append(labels)
        else:
            all_tensors = []
            for tensor in split_tensors:
                all_tensors.append(tensor[1:])
            sentences = np.concatenate(all_tensors, axis=0)
            length = 0
            for idx, tensor in enumerate(split_tensors):
                similarity_matrix = np.matmul(tensor[0], sentences.T)
                all_similarity_matrix.append(similarity_matrix)
                labels = np.zeros_like(similarity_matrix)
                labels[length] = 1
                all_labels.append(labels)
                length += tensor.shape[0] - 1
                max_neg_scores = np.max(similarity_matrix[labels == 0], axis=-1)
                pos_neg_margins.append(np.mean(similarity_matrix[labels == 1] - max_neg_scores).item())

    similarity_matrix = np.concatenate(all_similarity_matrix, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if can_batched:
        pos_scores = similarity_matrix[labels == 1].reshape(similarity_matrix.shape[0], -1)
        neg_scores = similarity_matrix[labels == 0].reshape(similarity_matrix.shape[0], -1)
        max_neg_scores = np.max(neg_scores, axis=-1)
        pos_neg_margin = np.mean(pos_scores - max_neg_scores).item()
    else:
        pos_scores = similarity_matrix[labels == 1]
        neg_scores = similarity_matrix[labels == 0]
        pos_neg_margin = np.mean(pos_neg_margins)

    mean_neg = np.mean(neg_scores)
    mean_pos = np.mean(pos_scores)
    return {'margin': pos_neg_margin, 'mean_neg': mean_neg, 'mean_pos': mean_pos}


def calculate_reranker_metrics(logits, labels):
    """
    Calculate MRR and NDCG metrics for reranker.

    This function first groups the data based on query boundaries (identified by
    positive samples), then calculates MRR and NDCG for each group independently,
    and finally returns the mean across all queries.

    Data format:
    - Each query group starts with a positive sample (label=1) followed by negatives (label=0)
    - Example: [1,0,0,1,0,0,0] represents 2 queries: query1=[1,0,0], query2=[1,0,0,0]

    Args:
        logits: Model output scores [batch_size] (numpy array or can be converted to numpy)
        labels: Binary labels (1 for positive, 0 for negative) [batch_size]

    Returns:
        dict: Dictionary containing MRR and NDCG metrics averaged across all queries
    """
    import numpy as np

    # Convert to numpy if needed
    if hasattr(logits, 'numpy'):
        logits = logits.numpy()
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()

    logits = np.array(logits).flatten()
    labels = np.array(labels).flatten()

    # Step 1: Find all positive sample indices (query boundaries)
    positive_indices = np.where(labels == 1)[0]

    if len(positive_indices) == 0:
        return {'mrr': 0.0, 'ndcg': 0.0}

    # Step 2: Split into groups (queries)
    query_groups = []
    for i, pos_idx in enumerate(positive_indices):
        # Each group starts at a positive index
        group_start = pos_idx

        # Group ends at the next positive index or end of data
        if i + 1 < len(positive_indices):
            group_end = positive_indices[i + 1]
        else:
            group_end = len(labels)

        # Extract this query's data
        query_logits = logits[group_start:group_end]
        query_labels = labels[group_start:group_end]

        query_groups.append((query_logits, query_labels))

    # Step 3: Calculate metrics for each query independently
    mrr_scores = []
    ndcg_scores = []

    for query_idx, (query_logits, query_labels) in enumerate(query_groups):
        # Skip groups that are too small (need at least 1 positive + 1 negative)
        if len(query_logits) < 2:
            print(f'Query {query_idx}: Skipped (too small: {len(query_logits)} items)')
            continue

        # Verify that the first sample is positive (data format validation)
        if query_labels[0] != 1:
            print(f'Query {query_idx}: Skipped (first sample not positive)')
            continue

        # Step 3a: Calculate ranking within this query
        ranking = np.argsort(-query_logits)  # Sort by logits descending

        # Step 3b: Find position of positive document (should be at index 0 in query)
        pos_rank = np.where(ranking == 0)[0][0] + 1  # +1 for 1-based ranking

        # Step 3c: Calculate MRR for this query
        mrr = 1.0 / pos_rank
        mrr_scores.append(mrr)

        # Step 3d: Calculate NDCG for this query
        def calculate_ndcg_single_query(relevance_scores, ranking):
            """Calculate NDCG for a single query"""
            # Calculate DCG (Discounted Cumulative Gain)
            dcg = 0.0
            for rank_pos, doc_idx in enumerate(ranking):
                relevance = relevance_scores[doc_idx]
                dcg += (2**relevance - 1) / np.log2(rank_pos + 2)  # rank_pos+2 because log2(1) undefined

            # Calculate IDCG (Ideal DCG)
            ideal_relevance = np.sort(relevance_scores)[::-1]  # Sort relevance descending
            idcg = 0.0
            for rank_pos, relevance in enumerate(ideal_relevance):
                idcg += (2**relevance - 1) / np.log2(rank_pos + 2)

            # NDCG = DCG / IDCG
            if idcg == 0:
                return 0.0
            return dcg / idcg

        # Create relevance scores (1 for positive, 0 for negative)
        relevance_scores = query_labels.astype(float)
        ndcg = calculate_ndcg_single_query(relevance_scores, ranking)
        ndcg_scores.append(ndcg)

    # Step 4: Calculate mean metrics across all valid queries
    if len(mrr_scores) == 0:
        print('No valid queries found for metric calculation')
        return {'mrr': 0.0, 'ndcg': 0.0}

    mean_mrr = np.mean(mrr_scores)
    mean_ndcg = np.mean(ndcg_scores)

    return {
        'mrr': mean_mrr,
        'ndcg': mean_ndcg,
    }


def _parse_multi_negative_sentences(sentences, labels, hard_negatives=None):
    split_indices = torch.nonzero(labels, as_tuple=False).squeeze().tolist()
    if isinstance(split_indices, int):
        split_indices = [split_indices]
    split_indices.append(len(labels))
    split_indices = np.array(split_indices) + np.array(list(range(len(split_indices))))
    split_tensors = []

    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        split_part = sentences[start:end]
        if hard_negatives is not None:
            negatives = len(split_part) - 2
            assert negatives > 0
            if negatives > hard_negatives:
                split_part = split_part[:hard_negatives + 2]
            elif negatives < hard_negatives:
                selected = np.random.choice(list(range(negatives)), size=hard_negatives - negatives, replace=True)
                selected += 1  # skip positive
                split_part = torch.cat((split_part, split_part[selected]), dim=0)
        split_tensors.append(split_part)
    return split_tensors


def infonce_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    temperature = float(os.environ.get('INFONCE_TEMPERATURE', '0.01'))  # temperature
    # calculate CE across the batch, meaning all samples will be negative except the matching positive
    use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
    hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)  # how many negative prompts kept in one sample
    # mask out fake negatives
    infonce_mask_fake_negative = strtobool(os.environ.get('INFONCE_MASK_FAKE_NEGATIVE', 'False'))
    if hard_negatives is not None:
        hard_negatives = int(hard_negatives)
    from swift.utils import get_dist_setting
    rank, _, world_size, _ = get_dist_setting()
    # repeat of anchor(1)+positive(1)+negatives(n)
    sentences = outputs['last_hidden_state']

    if world_size > 1 and use_batch:
        # gather all the sentences and labels across the gpus when calculate loss across all batches of all gpus
        all_sentences = gather_object(sentences.unsqueeze(0))
        labels = gather_object(labels)
        # override the gathered one
        all_sentences[rank] = sentences
        for idx in range(len(all_sentences)):
            if idx == rank:
                continue
            # we don't calculate grad from other gpus
            all_sentences[idx] = all_sentences[idx].detach().to(sentences.device)
        sentences = torch.cat(all_sentences, dim=0)
        labels = [tensor.to(sentences.device) for tensor in labels]
        labels = torch.stack(labels, dim=0)

    # split tensors into single sample
    # for example: batch_size=2 with tensor anchor(1)+positive(1)+negatives(3) + anchor(1)+positive(1)+negatives(2)
    # labels will be [1,0,0,0,1,0,0], meaning 1 positive, 3 negatives, 1 positive, 2 negatives
    split_tensors = _parse_multi_negative_sentences(sentences, labels, hard_negatives)
    loss = 0
    can_batched = hard_negatives is not None
    if hard_negatives is None and len(set([s.shape[0] for s in split_tensors])) == 1:
        # all tensors have the same batch size
        can_batched = True
    if not use_batch:
        # only calculate loss inside one sample
        if can_batched:
            # negative numbers are equal
            # [B, neg+2, D]
            sentences = torch.stack(split_tensors, dim=0)
            # [B, 1, D] * [B, neg+1, D]
            similarity_matrix = torch.matmul(sentences[:, 0:1], sentences[:, 1:].transpose(1, 2)) / temperature
            # The positive one is the first element
            labels = torch.zeros(len(split_tensors), dtype=torch.int64).to(sentences.device)
            loss = nn.CrossEntropyLoss()(similarity_matrix.squeeze(1), labels)
        else:
            # the negative numbers may be different, use for loop
            for tensor in split_tensors:
                # [D] * [neg+1, D]
                similarity_matrix = torch.matmul(tensor[0], tensor[1:].T) / temperature
                # The positive one is the first element
                labels = torch.tensor(0).to(tensor.device)
                loss += nn.CrossEntropyLoss()(similarity_matrix, labels)
            # avg between all batches in one gpu
            loss /= len(split_tensors)
    else:

        def mask_fake_negative(sim_matrix, sim_labels):
            thresholds = sim_matrix[torch.arange(sim_matrix.size(0)), sim_labels].view(-1, 1) + 0.1
            thresholds = thresholds.detach()
            mask = sim_matrix > thresholds
            sim_matrix[mask] = float('-inf')

        if can_batched:
            # [B, neg+2, D]
            sentences = torch.stack(split_tensors, dim=0)
            # [B, D] * [B*(neg+1), D]
            similarity_matrix = torch.matmul(sentences[:, 0].squeeze(1), sentences[:,
                                                                                   1:].reshape(-1, sentences.size(2)).T)
            labels = torch.tensor(range(0,
                                        sentences.size(0) * (sentences.size(1) - 1),
                                        sentences.size(1) - 1)).view(-1).to(sentences.device)
            if infonce_mask_fake_negative:
                mask_fake_negative(similarity_matrix, labels)
            similarity_matrix = similarity_matrix / temperature
            # every neg+1 is positive start from 0
            loss = nn.CrossEntropyLoss()(similarity_matrix, labels) / world_size  # avoid duplicate
        else:
            all_tensors = []
            for tensor in split_tensors:
                all_tensors.append(tensor[1:])
            # cat all neg+1 tensors
            sentences = torch.cat(all_tensors, dim=0)
            length = 0
            for idx, tensor in enumerate(split_tensors):
                # [D] * [B*(neg+1), D], neg numbers are different
                similarity_matrix = torch.matmul(tensor[0], sentences.T) / temperature
                labels = torch.tensor(length).to(tensor.device)
                loss += nn.CrossEntropyLoss()(similarity_matrix, labels)
                # next positive is neg+1
                length += tensor.size(0) - 1
            loss /= len(split_tensors)
            loss /= world_size  # avoid duplicate
    return loss


def online_contrastive_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    sentence1, sentence2 = _parse_pair_sentence(outputs)
    distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
    distance_matrix = distance_metric(sentence1, sentence2)
    negs = distance_matrix[labels == 0]
    poss = distance_matrix[labels == 1]

    # select hard positive and hard negative pairs
    negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
    positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

    positive_loss = positive_pairs.pow(2).sum()
    margin = 0.5
    negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss


def channel_loss_func(outputs,
                      labels,
                      num_items_in_batch=None,
                      sample_channels=None,
                      trainer=None,
                      position_ids=None,
                      **kwargs) -> torch.Tensor:
    from swift.trainers import per_token_loss_func
    channels = trainer.args.channels
    assert channels is not None, 'Please pass --channels as a hyperparameter.'
    assert sample_channels is not None, 'Data does not have channel field.'

    if outputs.loss is None:
        outputs.loss = per_token_loss_func(outputs, labels)
    token_loss = outputs.loss
    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
    if num_items_in_batch is None:
        num_items_in_batch = masks.sum()
    loss = token_loss.sum() / num_items_in_batch

    if position_ids is not None and trainer.template.padding_free:
        start_idx_mask = position_ids.view(-1).eq(0).int()
        sample_idx = (torch.cumsum(start_idx_mask, dim=0) - 1).tolist()
        token_channels = [sample_channels[i] for i in sample_idx]
    else:
        bs, seq = labels.shape
        token_channels = []
        for i in range(bs):
            token_channels.extend([sample_channels[i]] * seq)

    mode = 'train' if trainer.model.training else 'eval'
    metrics = trainer.custom_metrics[mode]
    for ch in set(sample_channels):
        indices = [i for i, c in enumerate(token_channels) if c == ch]
        if not indices:
            continue
        ch_loss = token_loss[indices][masks[indices]]
        metrics[f'loss_{ch}'].update(ch_loss)

    return loss


def reranker_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    logits = outputs.logits
    logits = logits.squeeze(1)
    labels = labels.to(logits.dtype)
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, labels)
    return loss


def generative_reranker_loss(outputs,
                             labels,
                             loss_scale=None,
                             num_items_in_batch=None,
                             trainer=None,
                             **kwargs) -> torch.Tensor:
    """
    Generative reranker loss function.

    This loss function is designed for generative rerankers that use token probabilities
    (e.g., "yes"/"no") to determine relevance scores. It only computes loss on the
    last token position for specific tokens.

    Args:
        outputs: Model outputs containing logits
        labels: Binary labels (0/1) for irrelevant/relevant pairs
        loss_scale: Not used for generative reranker
        num_items_in_batch: Not used for generative reranker
        trainer: Trainer instance to access tokenizer

    Returns:
        torch.Tensor: Cross entropy loss for yes/no classification
    """
    if trainer is None:
        raise ValueError('trainer is required for generative_reranker_loss to access tokenizer')

    logits = outputs.logits
    tokenizer = trainer.processing_class

    # Get token IDs for positive and negative tokens
    # Default to "yes"/"no", but can be configured via environment variables
    positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
    negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')

    try:
        positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
        negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
    except Exception as e:
        raise ValueError(f"Failed to convert tokens '{positive_token}'/'{negative_token}' to IDs. "
                         f'Please check if these tokens exist in the tokenizer vocabulary. Error: {e}')

    # Extract logits for positive and negative tokens directly from last position
    # This avoids creating the large intermediate tensor last_logits
    positive_logits = logits[:, -1, positive_token_id]  # [batch_size]
    negative_logits = logits[:, -1, negative_token_id]  # [batch_size]

    # Stack to create binary classification logits
    # Shape: [batch_size, 2] where dim=1 represents [negative, positive]
    binary_logits = torch.stack([negative_logits, positive_logits], dim=1)

    # Convert labels to the correct device and type
    binary_labels = labels.to(binary_logits.device).long()

    # Compute cross entropy loss
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(binary_logits, binary_labels)

    return loss


def listwise_reranker_loss(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs) -> torch.Tensor:
    """
    List-wise reranker loss function.

    This loss function groups samples by query based on the pattern where each group
    consists of 1 positive document followed by n negative documents. It treats the
    ranking task as a classification problem within each group, using cross-entropy
    loss to identify the positive document among all candidates.

    Data format expected:
    - labels: [1, 0, 0, 0, 1, 0, 0, ...] where 1 indicates positive, 0 indicates negative
    - Each 1 is followed by its corresponding negative documents until the next 1

    Environment variables for configuration:
    - LISTWISE_RERANKER_TEMPERATURE: Temperature for softmax (default: 1.0)
    - LISTWISE_RERANKER_MIN_GROUP_SIZE: Minimum group size to include (default: 2)

    Args:
        outputs: Model outputs containing logits [batch_size, 1]
        labels: Binary labels (1 for positive, 0 for negative) [batch_size]
        loss_scale: Not used for listwise reranker
        num_items_in_batch: Not used for listwise reranker

    Returns:
        torch.Tensor: Cross entropy loss for ranking classification
    """
    logits = outputs.logits.squeeze(-1)  # [batch_size]
    labels = labels.float()

    # Configuration from environment variables
    temperature = float(os.environ.get('LISTWISE_RERANKER_TEMPERATURE', '1.0'))
    min_group_size = int(os.environ.get('LISTWISE_RERANKER_MIN_GROUP_SIZE', '2'))

    # Find positive sample indices to determine group boundaries
    positive_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)

    if len(positive_indices) == 0:
        # No positive samples in this batch, return zero loss
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Ensure positive_indices is 1D
    if positive_indices.dim() == 0:
        positive_indices = positive_indices.unsqueeze(0)

    total_loss = 0.0
    num_groups = 0

    for i, pos_idx in enumerate(positive_indices):
        # Determine group boundaries
        group_start = pos_idx.item()

        # Find the end of current group (start of next group or end of batch)
        if i + 1 < len(positive_indices):
            group_end = positive_indices[i + 1].item()
        else:
            group_end = len(labels)

        # Extract group logits and labels
        group_logits = logits[group_start:group_end]  # [group_size]
        group_labels = labels[group_start:group_end]  # [group_size]

        # Skip groups that are too small
        if len(group_logits) < min_group_size:
            continue

        # Verify that the first sample in the group is positive
        if group_labels[0] != 1:
            continue  # Skip malformed groups

        # Apply temperature scaling for better training dynamics
        scaled_logits = group_logits / temperature

        # The positive document is always at index 0 within the group
        target = torch.tensor(0, dtype=torch.long, device=logits.device)

        # Apply cross-entropy loss: positive document should have highest score
        loss_fct = CrossEntropyLoss()
        group_loss = loss_fct(scaled_logits.unsqueeze(0), target.unsqueeze(0))

        total_loss += group_loss
        num_groups += 1

    if num_groups == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Return average loss across all groups
    return total_loss / num_groups


def listwise_generative_reranker_loss(outputs,
                                      labels,
                                      loss_scale=None,
                                      num_items_in_batch=None,
                                      trainer=None,
                                      **kwargs) -> torch.Tensor:
    """
    List-wise generative reranker loss function.

    This loss function combines the generative reranker approach (using token probabilities)
    with list-wise ranking. It groups samples by query based on the pattern where each group
    consists of 1 positive document followed by n negative documents, then uses the
    probabilities of specific tokens (e.g., "yes"/"no") to perform ranking within each group.

    Data format expected:
    - labels: [1, 0, 0, 0, 1, 0, 0, ...] where 1 indicates positive, 0 indicates negative
    - Each 1 is followed by its corresponding negative documents until the next 1

    Environment variables for configuration:
    - GENERATIVE_RERANKER_POSITIVE_TOKEN: Token for positive relevance (default: "yes")
    - GENERATIVE_RERANKER_NEGATIVE_TOKEN: Token for negative relevance (default: "no")
    - LISTWISE_RERANKER_TEMPERATURE: Temperature for softmax (default: 1.0)
    - LISTWISE_RERANKER_MIN_GROUP_SIZE: Minimum group size to include (default: 2)

    Args:
        outputs: Model outputs containing logits [batch_size, seq_len, vocab_size]
        labels: Binary labels (1 for positive, 0 for negative) [batch_size]
        loss_scale: Not used for listwise generative reranker
        num_items_in_batch: Not used for listwise generative reranker
        trainer: Trainer instance to access tokenizer

    Returns:
        torch.Tensor: Cross entropy loss for ranking classification based on token probabilities
    """
    if trainer is None:
        raise ValueError('trainer is required for listwise_generative_reranker_loss to access tokenizer')

    logits = outputs.logits
    tokenizer = trainer.processing_class
    labels = labels.float()

    # Configuration from environment variables
    positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
    negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
    temperature = float(os.environ.get('LISTWISE_RERANKER_TEMPERATURE', '1.0'))
    min_group_size = int(os.environ.get('LISTWISE_RERANKER_MIN_GROUP_SIZE', '2'))

    # Get token IDs for positive and negative tokens
    try:
        positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
        negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
    except Exception as e:
        raise ValueError(f"Failed to convert tokens '{positive_token}'/'{negative_token}' to IDs. "
                         f'Please check if these tokens exist in the tokenizer vocabulary. Error: {e}')

    # Extract logits for positive and negative tokens from last position
    positive_logits = logits[:, -1, positive_token_id]  # [batch_size]
    negative_logits = logits[:, -1, negative_token_id]  # [batch_size]

    # Create binary classification logits for each sample
    # Shape: [batch_size, 2] where dim=1 represents [negative, positive]
    binary_logits = torch.stack([negative_logits, positive_logits], dim=1)

    # Convert to relevance scores using softmax (probability of positive class)
    binary_probs = torch.softmax(binary_logits, dim=1)
    relevance_scores = binary_probs[:, 1]  # Probability of positive class [batch_size]

    # Find positive sample indices to determine group boundaries
    positive_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)

    if len(positive_indices) == 0:
        # No positive samples in this batch, return zero loss
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Ensure positive_indices is 1D
    if positive_indices.dim() == 0:
        positive_indices = positive_indices.unsqueeze(0)

    total_loss = 0.0
    num_groups = 0

    for i, pos_idx in enumerate(positive_indices):
        # Determine group boundaries
        group_start = pos_idx.item()

        # Find the end of current group (start of next group or end of batch)
        if i + 1 < len(positive_indices):
            group_end = positive_indices[i + 1].item()
        else:
            group_end = len(labels)

        # Extract group relevance scores and labels
        group_scores = relevance_scores[group_start:group_end]  # [group_size]
        group_labels = labels[group_start:group_end]  # [group_size]

        # Skip groups that are too small
        if len(group_scores) < min_group_size:
            continue

        # Verify that the first sample in the group is positive
        if group_labels[0] != 1:
            continue  # Skip malformed groups

        # Convert relevance scores to logits for cross-entropy loss
        # We use log to convert probabilities back to logits, then apply temperature
        group_logits = torch.log(group_scores + 1e-8) / temperature  # Add small epsilon for numerical stability

        # The positive document is always at index 0 within the group
        target = torch.tensor(0, dtype=torch.long, device=logits.device)

        # Apply cross-entropy loss: positive document should have highest relevance score
        loss_fct = CrossEntropyLoss()
        group_loss = loss_fct(group_logits.unsqueeze(0), target.unsqueeze(0))

        total_loss += group_loss
        num_groups += 1

    if num_groups == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Return average loss across all groups
    return total_loss / num_groups


loss_mapping = {
    'channel_loss': channel_loss_func,
    # embedding
    'cosine_similarity': cosine_similarity_func,
    'contrastive': contrastive_loss,
    'online_contrastive': online_contrastive_loss,
    'infonce': infonce_loss,
    # reranker
    'reranker': reranker_loss,
    'generative_reranker': generative_reranker_loss,
    'listwise_reranker': listwise_reranker_loss,
    'listwise_generative_reranker': listwise_generative_reranker_loss,
    'garr_loss': None,
}


def get_loss_func(loss_type: Optional[str]) -> Optional[Callable]:
    if loss_type is None:
        return None
    return loss_mapping[loss_type]


def garr_loss(
        outputs,
        labels,
        loss_scale=None,
        num_items_in_batch=None,
        trainer=None,
        **kwargs) -> torch.Tensor:
    """Combine score-token prediction and multimodal contrastive alignment."""
    if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
        last = outputs.last_hidden_state
    elif isinstance(outputs, dict) and 'last_hidden_state' in outputs:
        last = outputs['last_hidden_state']
    else:
        raise RuntimeError('last_hidden_state is required by garr_loss. Ensure trainer patches forward.')

    use_ce = int(os.environ.get('GARR_USE_CE', '1')) == 1
    if use_ce:
        from swift.trainers import per_token_loss_func
        if outputs.loss is None:
            outputs.loss = per_token_loss_func(outputs, labels)
        token_loss = outputs.loss
        masks_flat = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
        if num_items_in_batch is None:
            num_items_in_batch = masks_flat.sum()
        ce_loss = token_loss.sum() / num_items_in_batch
    else:
        ce_loss = torch.tensor(0.0, device=last.device, dtype=last.dtype)

    input_ids = kwargs.get('input_ids', None)
    if input_ids is None:
        if hasattr(outputs, 'input_ids'):
            input_ids = getattr(outputs, 'input_ids')
        elif isinstance(outputs, dict) and 'input_ids' in outputs:
            input_ids = outputs['input_ids']
    if input_ids is None:
        raise RuntimeError('garr_loss requires input_ids to build vision/text masks.')
    if trainer is None or getattr(trainer, 'processing_class', None) is None:
        raise RuntimeError('garr_loss requires trainer.processing_class to build masks.')

    dataset_name = os.environ.get('GARR_DATASET_NAME', '') or getattr(getattr(trainer, 'args', None), 'dataset_name', None)
    if not dataset_name:
        raise RuntimeError("garr_loss requires dataset name for mask building. Set env: GARR_DATASET_NAME=MicroLens|TopicVid_douyin")

    tok = trainer.processing_class
    cache = getattr(garr_loss, '_cache', None)
    if cache is None:
        cache = {}
        garr_loss._cache = cache

    if 'img_id' not in cache:
        img_id = tok.convert_tokens_to_ids('<IMG_CONTEXT>')
        unk_id = getattr(tok, 'unk_token_id', None)
        if img_id is None or (unk_id is not None and img_id == unk_id):
            img_id = tok.convert_tokens_to_ids('<|image_pad|>')
        if img_id is None or (unk_id is not None and img_id == unk_id):
            img_id = getattr(tok, 'image_token_id', None)
        if img_id is None or (unk_id is not None and img_id == unk_id):
            img_id = getattr(getattr(tok, 'tokenizer', None), 'image_token_id', None)
        if img_id is None:
            raise RuntimeError("garr_loss requires image token id: expected <IMG_CONTEXT> or <|image_pad|>.")
        cache['img_id'] = img_id
    if 'pred_ids' not in cache:
        cache['pred_ids'] = tok('Please predict', add_special_tokens=False).input_ids
    if 'markers' not in cache:
        cache['markers'] = {}
    if dataset_name not in cache['markers']:
        if dataset_name == 'MicroLens':
            cache['markers'][dataset_name] = {
                'title_ids': tok('Video-Title:', add_special_tokens=False).input_ids,
                'cat_ids': tok('Video-Category:', add_special_tokens=False).input_ids,
            }
        elif dataset_name == 'TopicVid_douyin':
            cache['markers'][dataset_name] = {
                'topic_ids': tok('Video-Topic:', add_special_tokens=False).input_ids,
                'title_ids': tok('Video-Title:', add_special_tokens=False).input_ids,
                'desc_ids': tok('Video-Description:', add_special_tokens=False).input_ids,
            }
        else:
            raise RuntimeError(f'Unsupported dataset_name={dataset_name!r} for GARR mask building.')

    def _find(seq, pat, start=0):
        if not pat:
            return -1
        L = len(seq)
        P = len(pat)
        i = max(0, start)
        while i + P <= L:
            if seq[i:i + P] == pat:
                return i
            i += 1
        return -1

    ids = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids
    vm_list = []
    tm_list = []
    img_cnt_total = 0
    text_mark_total = 0
    pred_ids = cache['pred_ids']
    m = cache['markers'][dataset_name]

    for seq in ids:
        Tlen = len(seq)
        vm = [1.0 if t == cache['img_id'] else 0.0 for t in seq]
        img_cnt_total += int(sum(vm))

        tm = [0.0] * Tlen
        if dataset_name == 'MicroLens':
            title_ids = m['title_ids']
            cat_ids = m['cat_ids']
            s_title = _find(seq, title_ids, 0)
            s_cat = _find(seq, cat_ids, max(0, s_title + len(title_ids) if s_title != -1 else 0))
            s_pred = _find(seq, pred_ids, max(0, s_cat + len(cat_ids) if s_cat != -1 else 0))
            if s_title != -1 and s_cat != -1 and s_cat > s_title:
                for i in range(s_title + len(title_ids), s_cat):
                    if 0 <= i < Tlen:
                        tm[i] = 1.0
            if s_cat != -1 and s_pred != -1 and s_pred > s_cat:
                for i in range(s_cat + len(cat_ids), s_pred):
                    if 0 <= i < Tlen:
                        tm[i] = 1.0
        else:
            topic_ids = m['topic_ids']
            title_ids = m['title_ids']
            desc_ids = m['desc_ids']
            s_topic = _find(seq, topic_ids, 0)
            s_title = _find(seq, title_ids, max(0, s_topic + len(topic_ids) if s_topic != -1 else 0))
            s_desc = _find(seq, desc_ids, max(0, s_title + len(title_ids) if s_title != -1 else 0))
            s_pred = _find(seq, pred_ids, max(0, s_desc + len(desc_ids) if s_desc != -1 else 0))
            if s_topic != -1 and s_title != -1 and s_title > s_topic:
                for i in range(s_topic + len(topic_ids), s_title):
                    if 0 <= i < Tlen:
                        tm[i] = 1.0
            if s_title != -1 and s_desc != -1 and s_desc > s_title:
                for i in range(s_title + len(title_ids), s_desc):
                    if 0 <= i < Tlen:
                        tm[i] = 1.0
            if s_desc != -1 and s_pred != -1 and s_pred > s_desc:
                for i in range(s_desc + len(desc_ids), s_pred):
                    if 0 <= i < Tlen:
                        tm[i] = 1.0

        text_mark_total += int(sum(tm))
        vm_list.append(vm)
        tm_list.append(tm)

    if not (img_cnt_total > 0 and text_mark_total > 0):
        raise RuntimeError('garr_loss mask building failed: template markers not found in input_ids.')

    vision_mask = torch.tensor(vm_list, device=last.device, dtype=last.dtype)
    text_mask = torch.tensor(tm_list, device=last.device, dtype=last.dtype)

    vm = vision_mask.to(last.dtype)
    tm = text_mask.to(last.dtype)
    v_cnt = vm.sum(dim=1, keepdim=True).clamp_min(1.0)
    t_cnt = tm.sum(dim=1, keepdim=True).clamp_min(1.0)

    z_v = (last * vm.unsqueeze(-1)).sum(dim=1) / v_cnt
    z_t = (last * tm.unsqueeze(-1)).sum(dim=1) / t_cnt

    use_con = int(os.environ.get('GARR_USE_CON', '1')) == 1
    con_term = torch.tensor(0.0, device=last.device, dtype=last.dtype)
    con_active = False
    inv_tau_f32 = torch.tensor(0.0, dtype=torch.float32, device=last.device)

    if use_con:
        epoch_now = (trainer.state.epoch if trainer is not None
                     and getattr(trainer, 'state', None) is not None else 0.0)
        con_epoch_stop = float(os.environ.get('GARR_CON_EPOCH_STOP', '1'))
        reached_cutoff = epoch_now is not None and epoch_now >= con_epoch_stop

        z_v = F.normalize(z_v, p=2, dim=-1)
        z_t = F.normalize(z_t, p=2, dim=-1)

        base_model = trainer.model.module if (trainer is not None and hasattr(trainer.model, 'module')) else (trainer.model if trainer is not None else None)
        if base_model is not None and hasattr(base_model, 'garr_con_logit_scale'):
            inv_tau = base_model.garr_con_logit_scale.exp()
            inv_max = base_model.garr_con_inv_t_max
        else:
            tau_fixed = float(os.environ.get('GARR_CON_T', '0.15'))
            inv_tau = torch.tensor(1.0 / tau_fixed, device=last.device, dtype=last.dtype)
            inv_max = torch.tensor(float(os.environ.get('GARR_CON_INV_T_MAX', '100.0')), device=last.device, dtype=last.dtype)

        inv_tau = inv_tau.to(device=last.device, dtype=last.dtype)
        inv_max = inv_max.to(device=last.device, dtype=last.dtype)
        inv_tau = torch.clamp(inv_tau, max=inv_max)
        inv_tau_f32 = inv_tau.detach().float()

        use_gather_env = os.environ.get('GARR_CON_GATHER', 'True')
        use_gather = bool(isinstance(use_gather_env, str) and use_gather_env.lower() in ('1', 'true', 'yes'))
        if use_gather and trainer is not None:
            from swift.utils import get_dist_setting
            rank, _, world_size, _ = get_dist_setting()
            if world_size and world_size > 1:
                all_v = gather_object(z_v)
                all_t = gather_object(z_t)
                if not (isinstance(all_v, list) and len(all_v) == world_size):
                    raise RuntimeError('gather_object returned unexpected data for GARR contrastive loss.')
                if not (isinstance(all_t, list) and len(all_t) == world_size):
                    raise RuntimeError('gather_object returned unexpected data for GARR contrastive loss.')

                all_v[rank] = z_v
                all_t[rank] = z_t

                def _coerce_2d(x, ref: torch.Tensor, name: str) -> torch.Tensor:
                    if not torch.is_tensor(x):
                        x = torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
                    else:
                        x = x.to(device=ref.device, dtype=ref.dtype)
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    if x.dim() != 2:
                        raise RuntimeError(f'garr_loss gather {name} must be 2D [B,D], got shape={list(x.shape)}')
                    return x

                for idx in range(world_size):
                    all_v[idx] = _coerce_2d(all_v[idx], z_v, 'z_v')
                    all_t[idx] = _coerce_2d(all_t[idx], z_t, 'z_t')
                    if idx != rank:
                        all_v[idx] = all_v[idx].detach()
                        all_t[idx] = all_t[idx].detach()

                z_v = torch.cat(all_v, dim=0)
                z_t = torch.cat(all_t, dim=0)

        q_max = int(os.environ.get('GARR_QUEUE_SIZE', '4096'))
        if not hasattr(garr_loss, '_queue_v') or garr_loss._queue_v is None:
            garr_loss._queue_v = torch.empty(0, z_v.size(-1), device=z_v.device, dtype=z_v.dtype)
            garr_loss._queue_t = torch.empty(0, z_t.size(-1), device=z_t.device, dtype=z_t.dtype)

        q_ready = (garr_loss._queue_v.size(0) >= q_max) and (garr_loss._queue_t.size(0) >= q_max)
        if q_ready:
            pos_v = inv_tau * (z_v * z_t).sum(dim=-1, keepdim=True)
            neg_v = inv_tau * torch.matmul(z_v, garr_loss._queue_t.t())
            logits_v = torch.cat([pos_v, neg_v], dim=1)
            labels_v = torch.zeros(z_v.size(0), dtype=torch.long, device=z_v.device)
            loss_v = CrossEntropyLoss()(logits_v, labels_v)

            pos_t = inv_tau * (z_t * z_v).sum(dim=-1, keepdim=True)
            neg_t = inv_tau * torch.matmul(z_t, garr_loss._queue_v.t())
            logits_t = torch.cat([pos_t, neg_t], dim=1)
            labels_t = torch.zeros(z_t.size(0), dtype=torch.long, device=z_t.device)
            loss_t = CrossEntropyLoss()(logits_t, labels_t)

            con_term = 0.5 * (loss_v + loss_t)

        with torch.no_grad():
            qv = torch.cat([garr_loss._queue_v, z_v.detach()], dim=0)
            qt = torch.cat([garr_loss._queue_t, z_t.detach()], dim=0)
            if qv.size(0) > q_max:
                qv = qv[-q_max:]
            if qt.size(0) > q_max:
                qt = qt[-q_max:]
            garr_loss._queue_v = qv
            garr_loss._queue_t = qt

        con_active = bool(q_ready and not reached_cutoff)
        w_con = float(os.environ.get('GARR_CON_WEIGHT', '0.002')) if con_active else 0.0
    else:
        w_con = 0.0

    total_loss = ce_loss + w_con * con_term

    # Keep the loss connected to the graph when all configured terms are inactive.
    if not total_loss.requires_grad:
        total_loss = total_loss + (last.sum() * 0.0)

    if trainer is not None:
        mode = 'train' if trainer.model.training else 'eval'
        metrics = trainer.custom_metrics[mode]
        metrics['loss_ce'].update(ce_loss.detach())
        metrics['loss_contrastive'].update(con_term.detach())
        metrics['con_active'].update(torch.tensor(1.0 if con_active else 0.0, device=last.device))
        metrics['loss_garr'].update(total_loss.detach())
        if hasattr(garr_loss, '_queue_v') and garr_loss._queue_v is not None:
            metrics['queue_v'].update(torch.tensor(float(garr_loss._queue_v.size(0)), device=last.device))
            metrics['queue_t'].update(torch.tensor(float(garr_loss._queue_t.size(0)), device=last.device))
        metrics['temp_inv'].update(inv_tau_f32.to(device=last.device))
        metrics['temp'].update((1.0 / (inv_tau_f32 + 1e-12)).to(device=last.device))

    return total_loss


loss_mapping['garr_loss'] = garr_loss
