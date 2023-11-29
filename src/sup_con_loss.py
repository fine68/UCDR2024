from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

def sup_con_loss(features, temperature=0.07, contrast_mode='all', base_temperature=0.07, labels=None, mask=None, device = None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device) # 对角线全1的矩阵
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1] # 4 batch = 10
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 40,50
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature) # 40 40
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # 40,1
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count) # 40, 40
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # 40

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean() # 4,10 -> mean

    return loss

def soft_sup_con_loss(features, softlabels, hard_labels, temperature=0.07, base_temperature=0.07, device = None):
    """Compute loss for model. 
    Args:
        features: hidden vector of shape [bsz, hide_dim].
        soft_labels : hidden vector of shape [bsz, hide_dim].
        labels: ground truth of shape [bsz].
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    features_dot_softlabels = torch.div(torch.matmul(features, softlabels.T), temperature) #
    loss = torch.nn.functional.cross_entropy(features_dot_softlabels, hard_labels)

    return loss
def Euclidean_MSE(semantic_emb):
    def mse_loss(enc_out, cls_id):
        gt_cls_sim = torch.cdist(semantic_emb[cls_id], semantic_emb, p=2.0)
        eucdist_logits = torch.cdist(enc_out, semantic_emb, p=2.0)
        cls_euc_scaled = gt_cls_sim/(torch.max(gt_cls_sim, dim=1).values.unsqueeze(-1))
        cls_wts = torch.exp(-1.5*cls_euc_scaled)
        emb_mse = nn.MSELoss(reduction='none')(eucdist_logits, gt_cls_sim)
        emb_mse *= cls_wts
        return emb_mse.mean()
    return mse_loss

def triplet_loss_random(features, queues, hard_labels, margin=0.2, device = None):
    """
    Compute the triplet loss.
    Parameters:
    - anchor: the feature vector of the current sample, shape (D,)
    - queue: the feature vectors of samples in the queue, shape (N, D)
    - labels: the labels of samples in the queue, shape (N,)
    - margin: the margin for triplet loss
    Returns:
    - loss: triplet loss
    """
    # pdb.set_trace()
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    for i, (image, image_class) in enumerate(zip(features, hard_labels)):
        positive_samples = []
        negative_samples = []
        for cls, queue in queues.items():
            if cls == image_class:
                positive_samples.extend(queue)
            else:
                negative_samples.extend(queue)
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            continue

        random_index_p1 = torch.randint(0, len(positive_samples), (1,)).item()
        random_index_n1 = torch.randint(0, len(negative_samples), (1,)).item()
        random_index_p2 = torch.randint(0, len(positive_samples), (1,)).item()
        random_index_n2 = torch.randint(0, len(negative_samples), (1,)).item()
            
        positive_samples = torch.stack(positive_samples).to(device)

        random_positive_sample1 = positive_samples[random_index_p1]
        random_positive_sample2 = positive_samples[random_index_p2]

        negative_samples = torch.stack(negative_samples).to(device)

        random_negative_sample1 = negative_samples[random_index_n1]
        random_negative_sample2 = negative_samples[random_index_n2]
        
        positive_distances1 = torch.cdist(image.unsqueeze(0), random_positive_sample1.unsqueeze(0), p=2.0)
        positive_distances2 = torch.cdist(image.unsqueeze(0), random_positive_sample2.unsqueeze(0), p=2.0)

        # hardest_positive_distance = positive_distances.max()
        negative_distances1 = torch.cdist(image.unsqueeze(0), random_negative_sample1.unsqueeze(0), p=2.0) 
        negative_distances2 = torch.cdist(image.unsqueeze(0), random_negative_sample2.unsqueeze(0), p=2.0) 

        # hardest_negative_distance = negative_distances.min()
        loss1 = F.relu(positive_distances1 - negative_distances1 + margin)
        loss2 = F.relu(positive_distances2 - negative_distances2 + margin)

        loss = 0.5 * loss1 + 0.5 * loss2

        losses.append(loss)   
    # Average the losses for the batch
    return torch.stack(losses).mean()
def triplet_loss_hard_sample(features, queues, hard_labels, margin=0.2, device = None):
    """
    Compute the triplet loss.
    Parameters:
    - anchor: the feature vector of the current sample, shape (D,)
    - queue: the feature vectors of samples in the queue, shape (N, D)
    - labels: the labels of samples in the queue, shape (N,)
    - margin: the margin for triplet loss
    Returns:
    - loss: triplet loss
    """
    # pdb.set_trace()
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    for i, (image, image_class) in enumerate(zip(features, hard_labels)):
        positive_samples = []
        negative_samples = []
        for cls, queue in queues.items():
            # print(f"cls:{cls}")
            # print(f"image_class:{image_class}")
            if cls == image_class:
                positive_samples.extend(queue)
            else:
                negative_samples.extend(queue)
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            continue
        positive_samples = torch.stack(positive_samples).to(device)
        # random_index_p = torch.randint(0, len(positive_samples), (1,)).item()
        # random_positive_sample = positive_samples[random_index_p]
        negative_samples = torch.stack(negative_samples).to(device)
        # random_index_n = torch.randint(0, len(negative_samples), (1,)).item()
        # random_negative_sample = negative_samples[random_index_n]
        positive_distances = torch.cdist(image.unsqueeze(0), positive_samples.unsqueeze(0), p=2.0)
        # hardest_positive_distance = positive_distances.max()
        negative_distances = torch.cdist(image.unsqueeze(0), negative_samples.unsqueeze(0), p=2.0) 
        hardest_positive_distance = positive_distances.max()
        hardest_negative_distance = negative_distances.min()
        # hardest_negative_distance = negative_distances.min()
        loss = F.relu(hardest_positive_distance - hardest_negative_distance + margin)

        losses.append(loss)   
    # Average the losses for the batch
    return torch.stack(losses).mean()



def itcs_m(features, softlabels, hard_labels, dom_labels, queues, temperature=0.07, base_temperature=0.07, device=None):

 

    """Compute loss for model.

    Args:

        features: hidden vector of shape [bsz, hide_dim].

        soft_labels : hidden vector of shape [bsz, hide_dim].

        labels: ground truth of shape [bsz].

    Returns:

        A loss scalar.

    """
    losses = []
    sca = 0.1
    text_samples = softlabels
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_similarities = {}
    for i, (image, image_class, image_domain) in enumerate(zip(features, hard_labels, dom_labels)): 
        positive_samples = []
        negative_samples = []
        for cls, domain_queues in queues.items():
            if cls == image_class:
                positive_samples.extend(domain_queues[image_domain])
            else:
                negative_samples.extend(domain_queues[image_domain])
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            continue
        positive_samples = torch.stack(positive_samples).to(device)
        negative_samples = torch.stack(negative_samples).to(device)
        text_sim = torch.div(torch.matmul(image.unsqueeze(0),text_samples.T), temperature)
        negative_similarities = torch.div(torch.matmul(image.unsqueeze(0), negative_samples.T), temperature)
        all_similarities = torch.cat([text_sim, negative_similarities], dim=1)
        all_similarities = all_similarities.squeeze(0)
        loss = nn.functional.cross_entropy(all_similarities,image_class)
        losses.append(loss)
    return torch.mean(torch.stack(losses))


