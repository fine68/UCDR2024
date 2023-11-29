"""
    Parse input arguments
"""
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='SPG for UCDR/ZS-SBIR')

        parser = argparse.ArgumentParser(description='SnMpNet for UCDR/ZS-SBIR')
        parser.add_argument('-debug_mode', '--debug_mode',default=1, type=int, help='use debug mode')
        parser.add_argument('-ucddr', '--ucddr',default=0, type=int, help='evaluate ucddr?')
        parser.add_argument('-root', '--root_path',default="", type=str, help='file path of this repository')
        parser.add_argument('-data_path', '--dataset_path',default="", type=str, help='file path of your datasets')
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')
        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['Sketchy', 'DomainNet', 'TUBerlin'])
        parser.add_argument('-weight', '--weight', default="", type=str, help='file path')
        parser.add_argument('-eccv', '--is_eccv_split', choices=[0, 1], default=1, type=int, help='whether or not to use eccv18 split\
                            if dataset="Sketchy"')

        # CLIP
        parser.add_argument('-clip_bb', '--clip_backbone', type=str, choices=['RN50x4', 'RN50x16', 'ViT-B/16', 'ViT-B/32'], default='ViT-B/32', help='choose clip backbone')
        parser.add_argument('-backbone', '--backbone', type=str, choices=['deit_tiny_patch16_224', 'dino_vits16', 'dino_vitb16', 'deit_small_patch16_224', 'deit_base_patch16_224'], default='deit_small_patch16_224', help='choose backbone')
        parser.add_argument('-CLS_NUM_TOKENS', '--CLS_NUM_TOKENS',default=300, type=int, help='number of class visual prompt tokens')
        parser.add_argument('-DOM_NUM_TOKENS', '--DOM_NUM_TOKENS',default=5, type=int, help='number of domain visual prompt tokens')
        parser.add_argument('-DOM_PROJECT', '--DOM_PROJECT',default=-1, type=int, help='projection for domain visual prompt')
        parser.add_argument('-CLS_PROJECT', '--CLS_PROJECT',default=-1, type=int, help='projection for class visual prompt')
        parser.add_argument('-VP_INITIATION', '--VP_INITIATION',default='random', type=str, help='initiation for visual prompt')
        parser.add_argument('-tp_N_CTX', '--tp_N_CTX',default=-1, type=int, help='initiation for visual prompt')
        parser.add_argument('-dropout', '--dropout',default=0.0, type=float, help='use debug model')
        parser.add_argument('-ratio_soft_dom', '--ratio_soft_dom',default=0.75, type=float, help='domian softmax ratio')
        parser.add_argument('-ratio_soft_cls', '--ratio_soft_cls',default=0.75, type=float, help='class softmax ratio')
        parser.add_argument('-ratio_prompt', '--ratio_prompt',default=33, type=float, help='prompt ratio')
        parser.add_argument('-prompt', '--prompt',default=1, type=float, help='prompt length')

        # DomainNet specific arguments
        parser.add_argument('-sd', '--seen_domain', default='painting', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-hd', '--holdout_domain', default='infograph', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-gd', '--gallery_domain', default='real', choices=['clipart', 'infograph', 'photo', 'painting', 'real'])
        parser.add_argument('-aux', '--include_auxillary_domains', choices=[0, 1], default=1, type=int, help='whether(1) or not(0) to include\
                            domains other than seen domain and gallery')

        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='adam')

        # Loss weight & reg. parameters
        parser.add_argument('-l2', '--l2_reg', default=0.0, type=float, help='L2 Weight Decay for optimizer')

        # Size parameters
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')

        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=180)
        parser.add_argument('-bs', '--batch_size', default=50, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of workers in data loader')

        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N', help='Number of epochs to train')
        parser.add_argument('-lr', '--lr', type=float, default=0.001, metavar='LR', help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=2, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N', help='How many batches to wait before logging training status')

        self.parser = parser


    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
