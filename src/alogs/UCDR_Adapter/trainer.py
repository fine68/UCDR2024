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
from src.sup_con_loss import soft_sup_con_loss, Euclidean_MSE, triplet_loss_random, triplet_loss_hard_sample, itcs_m
from torch import optim
import collections


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
gm = GPUmanager.GPUManager()
gpu_index = gm.auto_choice()
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')


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
                #lambda image: image.convert("RGB"),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
        }

        # class dictionary
        self.dict_clss = utils.create_dict_texts(self.tr_classes) 
        self.te_dict_class = utils.create_dict_texts(self.tr_classes+self.va_classes+self.te_classes)

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
        self.train_loader_for_SP = DataLoader(dataset=data_train, batch_size= 400, sampler=train_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)
        
        print('Loading Done\n')

        self.model = UCDR_Adapter(self.args, self.dict_clss, self.dict_doms, device)
        self.model = self.model.to(device)

        if args.dataset=='DomainNet':
            self.save_folder_name = 'seen-'+args.seen_domain+'_unseen-'+args.holdout_domain+'_x_'+args.gallery_domain
            if not args.include_auxillary_domains:
                self.save_folder_name += '_noaux'
        elif args.dataset=='Sketchy':
            if args.is_eccv_split:
                self.save_folder_name = 'eccv_split'
            else:
                self.save_folder_name = 'random_split'
        else:
            self.save_folder_name = ''

        if args.dataset=='DomainNet' or (args.dataset=='Sketchy' and args.is_eccv_split):
            self.map_metric = 'mAP@200'
            self.prec_metric = 'prec@200'
        else:
            self.map_metric = 'mAP@all'
            self.prec_metric = 'prec@100'

        self.suffix =  'e-'+str(args.epochs)+'_es-'+str(args.early_stop)+'_opt-'+args.optimizer+\
                        '_bs-'+str(args.batch_size)+'_lr-'+str(args.lr)

        # exit(0)
        path_log = os.path.join(args.root_path,'src/alogs/UCDR_Adapter/logs', args.dataset, self.save_folder_name, self.suffix)
        self.path_cp = os.path.join(args.root_path, 'src/alogs/UCDR_Adapter/saved_models', args.dataset, self.save_folder_name)
        
    
        # Logger
        print('Setting logger...', end='')
        self.logger = SummaryWriter(path_log)
        print('Done\n')

        self.start_epoch = 0
        self.best_map = 0
        self.early_stop_counter = 0
        self.last_chkpt_name='init'

        print("================Training Settings=================")
        print(f"lr = {self.args.lr}")
        print(f"batch_size = {self.args.batch_size}")
        print(f"generation domain prompt numbers = {self.args.GP_DOM_NUM_TOKENS}")
        print(f"generation class prompt numbers = {self.args.GP_CLS_NUM_TOKENS}")
        print(f"text prompt numbers= {self.args.tp_N_CTX}")
        print("==================================================")

        self.resume_from_checkpoint(args.resume_dict)
    
    def training_set(self, stage):
        
        lr = self.args.lr
        if stage == 1:
            lr = 0.0001
            print("========Training Source Prompt Learning phase========")
            train_parameters = ['text_prompt_learner.ctx', 'image_encoder.specific_domain_prompts', 'image_encoder.specific_class_prompts']
        elif stage == 2 :
            lr = self.args.lr
            lr = lr/2
            print("========Training Target Prompt Generation phase========")
            train_parameters = ['image_encoder.layer_norm2', 'image_encoder.prompt_proj','image_encoder.W_image', 'image_encoder.W_prompt', 'image_encoder.feature_template','image_encoder.feature_proj']
        for name, param in self.model.named_parameters():
            for str in train_parameters:
                flag = 0
                if name.startswith(str) == True:
                    param.requires_grad_(True)
                    print(name)
                    flag = 1
                    break
            if flag == 0:
                param.requires_grad_(False)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

        print(f"Total trainable parameters: {trainable_params}")
        print(f"Total non-trainable parameters: {non_trainable_params}")

        # NOTE: only give prompt_learner to the optimizer
        if self.args.optimizer=='sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), weight_decay=self.args.l2_reg, momentum=self.args.momentum, nesterov=False, lr=lr)
        elif self.args.optimizer=='adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.l2_reg)
        print("===============================================")

    def do_epoch(self, stage, epoch):
        
        self.model.train()
    
        batch_time = AverageMeter()
        total_loss = AverageMeter()

        # Start counting time
        time_start = time.time()
        if stage == 1:
            train_loader = self.train_loader_for_SP
        else :
            train_loader = self.train_loader
        for i, (im, cls, dom) in enumerate(train_loader):
          
            im = im.float().to(device, non_blocking=True)
            cls_numeric = torch.from_numpy(utils.numeric_classes(cls, self.dict_clss)).long().to(device)
            # print(cls_numeric.shape)
            self.optimizer.zero_grad()
            feature, soft_label, cls_id, queues= self.model(im, dom, cls, stage) 
            if self.args.tp_N_CTX == -1: # no text prompt tuning 
                 hard_labels = cls_numeric.contiguous().view(-1, 1)
                 hard_labels = torch.eq(hard_labels, hard_labels.T).float().to(device) # batch, batch
                 hard_labels = hard_labels / hard_labels.sum(-1, keepdim=True)
            else :
                hard_labels = cls_numeric
            self.embedding_loss = Euclidean_MSE(soft_label)
            if epoch == 0:
                loss = soft_sup_con_loss(feature, soft_label, hard_labels, device=device) + triplet_loss_random(feature, queues, cls_id, device=device)
            else:
                loss = soft_sup_con_loss(feature, soft_label, hard_labels, device=device)
            loss.backward()
            self.optimizer.step()
            total_loss.update(loss.item(), im.size(0))
        
            # time
            time_end = time.time()
            batch_time.update(time_end - time_start)
            time_start = time_end

            if (i + 1) % self.args.log_interval == 0:
                print('[Train] Epoch: [{0}/{1}][{2}/{3}]\t'
                    # 'lr:{3:.6f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'net {net.val:.4f} ({net.avg:.4f})\t'
                    .format(self.current_epoch+1, self.args.epochs, i+1, len(train_loader), batch_time=batch_time, net=total_loss))
                # with open("train_loss.txt", 'a') as f:
                #     f.write(str(total_loss.val) +''+str(total_loss.avg) + "\n")
                if self.args.debug_mode == 1:
                    break
        return {'net':total_loss.avg}

    def do_training(self):

        print('***Start Train***')
        
        self.training_set(1)      

        self.current_epoch = 0
        loss = self.do_epoch(1,self.current_epoch)
        self.training_set(2)
        for self.current_epoch in range(self.start_epoch, self.args.epochs):

            start = time.time()

            self.adjust_learning_rate()
            loss = self.do_epoch(2,self.current_epoch+1)
            print('\n***Validation***')
            if self.args.dataset=='DomainNet':
                if self.args.ucddr == 0:
                    te_data = []
                    for domain in [self.args.seen_domain, self.args.holdout_domain]:
                        for includeSeenClassinTestGallery in [0, 1]:
                            test_head_str = 'Query:' + domain + '; Gallery:' + self.args.gallery_domain + '; Generalized:' + str(includeSeenClassinTestGallery)
                            print(test_head_str)
                            
                            splits_query = domainnet.trvalte_per_domain(self.args, domain, 0, self.tr_classes, self.va_classes, self.te_classes)
                            splits_gallery = domainnet.trvalte_per_domain(self.args, self.args.gallery_domain, includeSeenClassinTestGallery, self.tr_classes, self.va_classes, self.te_classes)

                            data_te_query = BaselineDataset(np.array(splits_query['te']), transforms=self.image_transforms['eval'])
                            data_te_gallery = BaselineDataset(np.array(splits_gallery['te']), transforms=self.image_transforms['eval'])

                            # PyTorch test loader for query
                            te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 10, shuffle=False,
                                                            num_workers=self.args.num_workers, pin_memory=True)
                            # PyTorch test loader for gallery
                            te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 10, shuffle=False,
                                                            num_workers=self.args.num_workers, pin_memory=True)

                            # print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.')
                            result = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4, self.args)
                            te_data.append(result)
                            
                            out = f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(result[self.map_metric], result[self.prec_metric])
                            print(out)
                    map_ = te_data[3][self.map_metric]
                    prec = te_data[3][self.prec_metric]
                else :
                    if self.args.holdout_domain == 'quickdraw':
                        p = 0.1
                    else :
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
                    te_data = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4, self.args)
                    map_ = te_data[self.map_metric]
                    prec = te_data[self.prec_metric]
                    out ="mAP@200 = %.4f, Prec@200 = %.4f\n"%(map_, prec)
                    print(out)
            else :

                data_te_query = BaselineDataset(self.data_splits['query_te'], transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(self.data_splits['gallery_te'], transforms=self.image_transforms['eval'])

                te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size*5, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size*5, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

                print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

                te_data = evaluate(te_loader_query, te_loader_gallery, self.model,self.te_dict_class, self.dict_doms, 4, self.args)
                out =f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(te_data[self.map_metric], te_data[self.prec_metric])
                map_ = te_data[self.map_metric]
                prec = te_data[self.prec_metric]
                out =f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(map_, prec)
                print(out)
            end = time.time()
            elapsed = end-start

            print(f"Epoch Time:{elapsed//60:.0f}m{elapsed%60:.0f}s lr:{utils.get_lr(self.optimizer):.7f} mAP:{map_:.4f} prec:{prec:.4f}\n")

            if map_ > self.best_map:

                self.best_map = map_
                self.early_stop_counter = 0

                model_save_name = 'val_map-'+'{0:.4f}'.format(map_)+'_prec-'+'{0:.4f}'.format(prec)+'_ep-'+str(self.current_epoch+1)+self.suffix
                utils.save_checkpoint({
                                        'epoch':self.current_epoch+1,
                                        'model_state_dict':self.model.state_dict(),
                                        'optimizer_state_dict':self.optimizer.state_dict(),
                                        'best_map':self.best_map,
                                        'corr_prec':prec
                                        }, directory=self.path_cp, save_name=model_save_name, last_chkpt=self.last_chkpt_name)
                self.last_chkpt_name = model_save_name

            else:
                self.early_stop_counter += 1
                if self.args.early_stop==self.early_stop_counter:
                    print(f"Validation Performance did not improve for {self.args.early_stop} epochs."
                            f"Early stopping by {self.args.epochs-self.current_epoch-1} epochs.")
                    break

                print(f"Val mAP hasn't improved from {self.best_map:.4f} for {self.early_stop_counter} epoch(s)!\n")

            # Logger step
            self.logger.add_scalar('Train/total loss', loss['net'], self.current_epoch)
            self.logger.add_scalar('Train/Learning rate', utils.get_lr(self.optimizer), self.current_epoch)
            self.logger.add_scalar('Val/map', map_, self.current_epoch)
            self.logger.add_scalar('Val/prec', prec, self.current_epoch)
            

        self.logger.close()

        print('\n***Training and Validation complete***')

    
    def adjust_learning_rate(self, min_lr=1e-6):
        lr = self.args.lr * math.pow(1e-3, float(self.current_epoch)/20)
        lr = max(lr, min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def resume_from_checkpoint(self, resume_dict):

        if resume_dict is not None:
            print('==> Resuming from checkpoint: ',resume_dict)
            model_path = os.path.join(self.path_cp, resume_dict+'.pth')
            checkpoint = torch.load(model_path, map_location=device)
            self.start_epoch = checkpoint['epoch']+1
            self.last_chkpt_name = resume_dict
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.best_map = checkpoint['best_map']
            

@torch.no_grad()
def evaluate(loader_sketch, loader_image, model:UCDR_Adapter, dict_clss, dict_doms, stage, args):

    # Switch to test mode
    model.eval()

    sketchEmbeddings = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in tqdm(enumerate(loader_sketch), desc='Extrac query feature', total=len(loader_sketch)):

        sk = sk.float().to(device)
        cls_id = utils.numeric_classes(cls_sk, dict_clss)
        dom_id = utils.numeric_classes(dom, dict_doms)
        sk_em = model.image_encoder(sk, dom_id, cls_id, stage)
        sketchEmbeddings.append(sk_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        sketchLabels.append(cls_numeric)
        if  args.debug_mode == 1 and i == 2:
            break
    sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings = list()
    realLabels = list()

    for i, (im, cls_im, dom) in tqdm(enumerate(loader_image), desc='Extrac gallery feature', total=len(loader_image)):

        im = im.float().to(device)
        cls_id = utils.numeric_classes(cls_im, dict_clss)
        dom_id = utils.numeric_classes(dom, dict_doms)  
        im_em = model.image_encoder(im, dom_id, cls_id, stage)
        realEmbeddings.append(im_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        realLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2 :
            break
    realEmbeddings = torch.cat(realEmbeddings, 0)
    realLabels = torch.cat(realLabels, 0)

    print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
    eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)

    return eval_data
