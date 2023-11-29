import sys
import os
from tqdm import tqdm
from src.models.UCDR_Adapter import UCDR_Adapter
import torch
import math
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.DomainNet import domainnet
from src.data.Sketchy import sketchy_extended
from src.data.TUBerlin import tuberlin_extended
import numpy as np
import torch.backends.cudnn as cudnn
from src.data.dataloaders import CuMixloader, BaselineDataset
from src.data.sampler import BalancedSampler
from src.utils import utils, GPUmanager
from src.utils.logger import AverageMeter
from src.utils.metrics import compute_retrieval_metrics
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from src.sup_con_loss import soft_sup_con_loss
from torch import optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
gm = GPUmanager.GPUManager()
gpu_index = gm.auto_choice()
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
import torch
# -*- coding: utf-8 -*-
# !/usr/bin/python
from datetime import datetime

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from src.alogs.UCDR_Adapter.trainer import Trainer
from src.options.options import Options

import pdb
import os
from shutil import copyfile
def main(args):
    trainer = Trainer(args)
    trainer.test()
os.makedirs('retrieval', exist_ok=True)

class Trainer:

    def __init__(self, args):
        self.args = args
        print('\nLoading data...')
        if args.dataset == 'Sketchy':
            data_input = sketchy_extended.create_trvalte_splits(args)
        if args.dataset == 'DomainNet':
            data_input = domainnet.create_trvalte_splits(args)
        if args.dataset == 'TUBerlin':
            data_input = tuberlin_extended.create_trvalte_splits(args)

        self.tr_classes = data_input['tr_classes']
        self.va_classes = data_input['va_classes']
        self.te_classes = data_input['te_classes']
        self.data_splits = data_input['splits']
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        use_gpu = torch.cuda.is_available()
        self.weight_path = args.weight

        if use_gpu:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)

        # Imagenet standards
        im_mean = [0.485, 0.456, 0.406]
        im_std = [0.229, 0.224, 0.225]
        # Image transformations
        self.image_transforms = {
            'train':
                transforms.Compose([
                    transforms.RandomResizedCrop((args.image_size, args.image_size), (0.8, 1.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(im_mean, im_std)
                ]),
            'eval': transforms.Compose([
                transforms.Resize(args.image_size, interpolation=BICUBIC),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                # lambda image: image.convert("RGB"),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        }

        # class dictionary
        self.dict_clss = utils.create_dict_texts(self.tr_classes)
        self.te_dict_class = utils.create_dict_texts(self.tr_classes + self.va_classes + self.te_classes)

        fls_tr = self.data_splits['tr']
        cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
        dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
        tr_domains_unique = np.unique(dom_tr)

        # doamin dictionary
        self.dict_doms = utils.create_dict_texts(tr_domains_unique)
        print(self.dict_doms)
        domain_ids = utils.numeric_classes(dom_tr, self.dict_doms)
        data_train = CuMixloader(fls_tr, cls_tr, dom_tr, self.dict_doms, transforms=self.image_transforms['train'])
        train_sampler = BalancedSampler(domain_ids, args.batch_size // len(tr_domains_unique),
                                        domains_per_batch=len(tr_domains_unique))
        self.train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)
        self.train_loader_for_SP = DataLoader(dataset=data_train, batch_size=400, sampler=train_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

        print('Loading Done\n')


        self.model = UCDR_Adapter(self.args, self.dict_clss, self.dict_doms, device)
        weight = torch.load(
            self.weight_path)[
            "model_state_dict"]
        self.model.load_state_dict(weight)
        self.model = self.model.to(device)

        if args.dataset == 'DomainNet':
            self.save_folder_name = 'seen-' + args.seen_domain + '_unseen-' + args.holdout_domain + '_x_' + args.gallery_domain
            if not args.include_auxillary_domains:
                self.save_folder_name += '_noaux'
        elif args.dataset == 'Sketchy':
            if args.is_eccv_split:
                self.save_folder_name = 'eccv_split'
            else:
                self.save_folder_name = 'random_split'
        else:
            self.save_folder_name = ''

        if args.dataset == 'DomainNet' or (args.dataset == 'Sketchy' and args.is_eccv_split):
            self.map_metric = 'mAP@200'
            self.prec_metric = 'prec@200'
        else:
            self.map_metric = 'mAP@all'
            self.prec_metric = 'prec@100'

        self.suffix = 'e-' + str(args.epochs) + '_es-' + str(args.early_stop) + '_opt-' + args.optimizer + \
                      '_bs-' + str(args.batch_size) + '_lr-' + str(args.lr)

        # exit(0)
        path_log = os.path.join(args.root_path, 'logs', args.dataset, self.save_folder_name, self.suffix)
        self.path_cp = os.path.join(args.root_path, 'src/alogs/UCDR_Adapter/saved_models', args.dataset, self.save_folder_name)

        # Logger
        print('Setting logger...', end='')
        self.logger = SummaryWriter(path_log)
        print('Done\n')

        self.start_epoch = 0
        self.best_map = 0
        self.early_stop_counter = 0
        self.last_chkpt_name = 'init'

        print("================Start Testing=================")
        print("==================================================")

        # self.resume_from_checkpoint(args.resume_dict)

    def test(self):
        if self.args.dataset == 'DomainNet':
            if self.args.ucddr == 0:
                te_data = []
                # for domain in [self.args.seen_domain, self.args.holdout_domain]:
                for domain in [ self.args.holdout_domain]:
                    for includeSeenClassinTestGallery in [0, 1]:
                        test_head_str = 'Query:' + domain + '; Gallery:' + self.args.gallery_domain + '; Generalized:' + str(
                            includeSeenClassinTestGallery)
                        print(test_head_str)
                        # pdb.set_trace()

                        splits_query = domainnet.trvalte_per_domain(self.args, domain, 0, self.tr_classes, self.va_classes,
                                                                    self.te_classes)
                        splits_gallery = domainnet.trvalte_per_domain(self.args, self.args.gallery_domain,
                                                                      includeSeenClassinTestGallery, self.tr_classes,
                                                                      self.va_classes, self.te_classes)

                        data_te_query = BaselineDataset(np.array(splits_query['te']),
                                                        transforms=self.image_transforms['eval'])
                        data_te_gallery = BaselineDataset(np.array(splits_gallery['te']),
                                                          transforms=self.image_transforms['eval'])

                        # PyTorch test loader for query
                        te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 10,
                                                     shuffle=False,
                                                     num_workers=self.args.num_workers, pin_memory=True)
                        # PyTorch test loader for gallery
                        te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 10,
                                                       shuffle=False,
                                                       num_workers=self.args.num_workers, pin_memory=True)

                        # print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.')
                        result = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class,
                                          self.dict_doms, 4, self.args)
                        te_data.append(result)

                        out = f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n" % (
                        result[self.map_metric], result[self.prec_metric])

                        print(out)
                map_ = te_data[3][self.map_metric]
                prec = te_data[3][self.prec_metric]
            else:
                if self.args.holdout_domain == 'quickdraw':
                    p = 0.1
                else:
                    p = 0.25
                splits_query = domainnet.seen_cls_te_samples(self.args, self.args.holdout_domain, self.tr_classes, p)
                splits_gallery = domainnet.seen_cls_te_samples(self.args, self.args.gallery_domain, self.tr_classes, p)
                data_te_query = BaselineDataset(np.array(splits_query), transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(np.array(splits_gallery), transforms=self.image_transforms['eval'])
                # PyTorch test loader for query
                te_loader_query = DataLoader(dataset=data_te_query, batch_size=2048, shuffle=False,
                                             num_workers=self.args.num_workers, pin_memory=True)
                # PyTorch test loader for gallery
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=2048, shuffle=False,
                                               num_workers=self.args.num_workers, pin_memory=True)
                te_data = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4,
                                   self.args)
                map_ = te_data[self.map_metric]
                prec = te_data[self.prec_metric]
                out = "mAP@200 = %.4f, Prec@200 = %.4f\n" % (map_, prec)
                print(out)
        else:
            data_te_query = BaselineDataset(self.data_splits['query_te'], transforms=self.image_transforms['eval'])
            data_te_gallery = BaselineDataset(self.data_splits['gallery_te'], transforms=self.image_transforms['eval'])

            te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 5, shuffle=False,
                                         num_workers=self.args.num_workers, pin_memory=True)
            te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 5, shuffle=False,
                                           num_workers=self.args.num_workers, pin_memory=True)

            print(
                f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

            te_data = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4,
                               self.args)
            out = f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n" % (
            te_data[self.map_metric], te_data[self.prec_metric])
            map_ = te_data[self.map_metric]
            prec = te_data[self.prec_metric]
            out = f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n" % (map_, prec)
            print(out)




@torch.no_grad()

def evaluate(loader_sketch, loader_image, model, dict_clss, dict_doms, stage, args):
    sketchEmbeddings = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in tqdm(enumerate(loader_sketch), desc='Extrac query feature', total=len(loader_sketch)):

        sk = sk.float().to(device)
        cls_id = utils.numeric_classes(cls_sk, dict_clss)
        # pdb.set_trace()
        dom_id = utils.numeric_classes(dom, dict_doms)
        sk_em = model.image_encoder(sk, dom_id, cls_id, stage)
        sketchEmbeddings.append(sk_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        sketchLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings = list()
    realLabels = list()

    for i, (im, cls_im, dom) in tqdm(enumerate(loader_image), desc='Extrac gallery feature', total=len(loader_image)):

        im = im.float().to(device)
        cls_id = utils.numeric_classes(cls_im, dict_clss)
        # pdb.set_trace()
        dom_id = utils.numeric_classes(dom, dict_doms)
        im_em= model.image_encoder(im, dom_id, cls_id, stage)
        realEmbeddings.append(im_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        realLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    realEmbeddings = torch.cat(realEmbeddings, 0)
    realLabels = torch.cat(realLabels, 0)

    print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
    eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)
    return eval_data



def evaluate_tsne(loader_sketch, loader_image, model, dict_clss, dict_doms, stage, args):
    sketchEmbeddings = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in tqdm(enumerate(loader_sketch), desc='Extrac query feature', total=len(loader_sketch)):

        sk = sk.float().to(device)
        cls_id = utils.numeric_classes(cls_sk, dict_clss)
        # pdb.set_trace()
        dom_id = utils.numeric_classes(dom, dict_doms)
        sk_em = model.image_encoder(sk, dom_id, cls_id, stage)
        sketchEmbeddings.append(sk_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        sketchLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings = list()
    realLabels = list()

    for i, (im, cls_im, dom) in tqdm(enumerate(loader_image), desc='Extrac gallery feature', total=len(loader_image)):

        im = im.float().to(device)
        cls_id = utils.numeric_classes(cls_im, dict_clss)
        # pdb.set_trace()
        dom_id = utils.numeric_classes(dom, dict_doms)
        im_em= model.image_encoder(im, dom_id, cls_id, stage)
        realEmbeddings.append(im_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        realLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    realEmbeddings = torch.cat(realEmbeddings, 0)
    realLabels = torch.cat(realLabels, 0)

    print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
    eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)

    embeddings = torch.cat([sketchEmbeddings, realEmbeddings], 0)
    labels = torch.cat([sketchLabels, realLabels], 0)
    domains = torch.cat([torch.zeros(sketchEmbeddings.size(0)), torch.ones(realEmbeddings.size(0))], 0)

    chosen_labels = [316, 319, 326, 304, 324, 325, 304, 327, 330, 301]

    # Convert your chosen list to a tensor
    selected_labels = torch.tensor(chosen_labels)

    mask = torch.zeros_like(labels, dtype=torch.bool)
    for label in selected_labels:
        mask |= labels == label
    
    filtered_embeddings = embeddings[mask]
    filtered_labels = labels[mask]
    filtered_domains = domains[mask]

    max_samples_per_class = 100
    sampled_embeddings = []
    sampled_labels = []
    sampled_domains = []
    
    for label in selected_labels:

        label_mask = filtered_labels == label
        label_embeddings = filtered_embeddings[label_mask]
        label_domains = filtered_domains[label_mask]

        if len(label_embeddings) > max_samples_per_class:
            sampled_indices = torch.randperm(len(label_embeddings))[:max_samples_per_class]
        else:
            sampled_indices = torch.arange(len(label_embeddings))
    
        sampled_embeddings.append(label_embeddings[sampled_indices])
        sampled_labels.append(filtered_labels[label_mask][:max_samples_per_class])
        sampled_domains.append(label_domains[sampled_indices])

    filtered_embeddings = torch.cat(sampled_embeddings, dim=0)
    filtered_labels = torch.cat(sampled_labels, dim=0)
    filtered_domains = torch.cat(sampled_domains, dim=0)

    filtered_embeddings = filtered_embeddings.to(device)
    filtered_labels = filtered_labels.to(device)
    filtered_domains = filtered_domains.to(device)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(filtered_embeddings.cpu().numpy())

    plt.figure(figsize=(8, 6))
    class_colors = ['#e57373', '#64b5f6', '#81c784', '#ba68c8', '#ffb74d', '#fff176', '#a1887f', '#f06292', '#90a4ae', '#4db6ac']
    domain_markers = {'real': '^', 'sketch': '*'}
    # Plot each class and domain combination
    for i, label in enumerate(selected_labels):
        idx = filtered_labels == label
        for domain in [0, 1]:  # Assuming 0 for sketch, 1 for real
            domain_name = 'sketch' if domain == 0 else 'real'
            domain_idx = idx & (filtered_domains == domain)
            domain_idx_cpu = domain_idx.cpu()
            plt.scatter(tsne_results[domain_idx_cpu, 0], tsne_results[domain_idx_cpu, 1],
                        color=class_colors[i], marker=domain_markers[domain_name], label=f'{label.item()} ({domain_name})')

    # Create custom legends
    class_legend_elements = [Line2D([0], [0], marker='o', color=color, label=f'Class {label.item()}',
                                    markerfacecolor=color, markersize=10) for label, color in zip(selected_labels, class_colors)]
    domain_legend_elements = [Line2D([0], [0], marker=marker, color='w', label=domain.title(),
                                     markerfacecolor='black', markersize=15) for domain, marker in domain_markers.items()]

    # Combine both legends
    legend_elements = class_legend_elements + domain_legend_elements

    # Add the combined custom legend to the plot
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=len(legend_elements) // 2)

    # Set title and labels
    # plt.title(f"T-SNE Visualization of Sketch and Real Domains ({'Training' if stage == 'train' else 'Validation'})")
    # plt.xlabel('TSNE-1')
    # plt.ylabel('TSNE-2')

    # Save the figure with the current time as the name
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"tsne_plot_{current_time}.png", bbox_inches='tight')

    # Show the plot
    plt.show()
    markers = ['o', 'x']
    for i, label in enumerate(selected_labels):
        idx = filtered_labels == label
        for j, domain in enumerate([0, 1]):
            domain_idx = idx & (filtered_domains == domain)
            domain_idx_cpu = domain_idx.cpu()
            plt.scatter(tsne_results[domain_idx_cpu, 0], tsne_results[domain_idx_cpu, 1],
                        color=class_colors[i], marker=markers[j], label=f'Class {label.item()}, Domain {domain}')

    plt.title(f"T-SNE Visualization of Sketch and Real Domains ({'Training' if stage == 'train' else 'Validation'})")
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    plt.legend()
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"tsne_plot_{current_time}.png")
    plt.show()
    return eval_data



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)
