import os
from tqdm import tqdm
import json
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import pprint
import re
import shutil
from datetime import datetime


from src.models import shallow_conv, resnet_12, wide_resnet, dense_net
from src.algorithm_trainer.algorithm_trainer import Meta_algorithm_trainer, Init_algorithm_trainer
from src.algorithms.algorithm import SVM, ProtoNet, Ridge, InitBasedAlgorithm
from src.optimizers import modified_sgd
from src.data.dataset_managers import MetaDataLoader
from src.data.datasets import MetaDataset, ClassImagesSet


def ensure_path(path):
    assert os.path.exists(path), "Output folder must already exist"
    return path


def str2bool(arg):
    return arg.lower() == 'true'


def main(args):

    ####################################################
    #                LOGGING AND SAVING                #
    ####################################################
    args.output_folder = ensure_path('./runs/{0}'.format(args.output_folder))
    if str2bool(args.eot_model):
        eval_results = f'{args.output_folder}/evaleot_results.txt'
    else:
        eval_results = f'{args.output_folder}/eval_results.txt'
    with open(eval_results, 'w') as f:
        f.write("--"*20 + "EVALUATION RESULTS" + "--"*20 + '\n')


    ####################################################
    #         DATASET AND DATALOADER CREATION          #
    ####################################################

    # json paths
    dataset_name = args.dataset_path.split('/')[-1]
    image_size = args.img_side_len
    dataset_name = args.dataset_path.split('/')[-1]
    # Following is needed when same train config is used for both 5w5s and 5w1s evaluations. 
    # This is the case in the case of SVM when 5w15s5q is used for both 5w5s and 5w1s evaluations. 
    all_n_shot_vals = [args.n_shot_val, 1] if str2bool(args.do_one_shot_eval_too) else [args.n_shot_val]
    base_class_generalization = dataset_name.lower() in ['miniimagenet', 'fc100-base', 'cifar-fs-base']
    train_file = os.path.join(args.dataset_path, 'base.json')
    val_file = os.path.join(args.dataset_path, 'val.json')
    test_file = os.path.join(args.dataset_path, 'novel.json')
    if base_class_generalization:
        base_test_file = os.path.join(args.dataset_path, 'base_test.json')
    print("Dataset name", dataset_name, "image_size", image_size, "all_n_shot_vals", all_n_shot_vals)
    print("base_class_generalization:", base_class_generalization)
    
    """
    1. Create ClassImagesSet object, which handles preloading of images
    2. Pass ClassImagesSet to MetaDataset which handles nshot, nquery and fixSupport
    3. Create Dataloader object from MetaDataset
    """

    print("\n", "--"*20, "BASE", "--"*20)
    train_classes = ClassImagesSet(train_file, preload=str2bool(args.preload_train))
    
    # create a dataloader that has no fixed support
    no_fixS_train_meta_dataset = MetaDataset(
                                    dataset_name=dataset_name,
                                    support_class_images_set=train_classes,
                                    query_class_images_set=train_classes,
                                    image_size=image_size,
                                    support_aug=False,
                                    query_aug=False,
                                    fix_support=0, # no fixed support
                                    save_folder='',
                                    verbose=False)

    no_fixS_train_loader = MetaDataLoader(
                                dataset=no_fixS_train_meta_dataset,
                                n_batches=args.n_iterations_val,
                                batch_size=args.batch_size_val,
                                n_way=args.n_way_val,
                                n_shot=args.n_shot_val,
                                n_query=args.n_query_val, 
                                randomize_query=False)

    print("\n", "--"*20, "VAL", "--"*20)
    val_classes = ClassImagesSet(val_file, preload=str2bool(args.preload_train))    
    val_meta_datasets = {}
    val_loaders = {}
    for ns_val in all_n_shot_vals:
        print("====", f"n_shots_val {ns_val}", "====")
        val_meta_datasets[ns_val] = MetaDataset(
                                        dataset_name=dataset_name,
                                        support_class_images_set=val_classes,
                                        query_class_images_set=val_classes,
                                        image_size=image_size,
                                        support_aug=False,
                                        query_aug=False,
                                        fix_support=0,
                                        save_folder='')

        val_loaders[ns_val] = MetaDataLoader(
                                dataset=val_meta_datasets[ns_val],
                                n_batches=args.n_iterations_val,
                                batch_size=args.batch_size_val,
                                n_way=args.n_way_val,
                                n_shot=ns_val,
                                n_query=args.n_query_val, 
                                randomize_query=False)

    print("\n", "--"*20, "NOVEL", "--"*20)
    test_classes = ClassImagesSet(test_file, preload=str2bool(args.preload_train))
    test_meta_datasets = {}
    test_loaders = {}
    for ns_val in all_n_shot_vals:
        print("====", f"n_shots_val {ns_val}", "====")    
        test_meta_datasets[ns_val] = MetaDataset(
                                        dataset_name=dataset_name,
                                        support_class_images_set=test_classes,
                                        query_class_images_set=test_classes,
                                        image_size=image_size,
                                        support_aug=False,
                                        query_aug=False,
                                        fix_support=0,
                                        save_folder='')

        test_loaders[ns_val] = MetaDataLoader(
                                    dataset=test_meta_datasets[ns_val],
                                    n_batches=args.n_iterations_val,
                                    batch_size=args.batch_size_val,
                                    n_way=args.n_way_val,
                                    n_shot=ns_val,
                                    n_query=args.n_query_val,
                                    randomize_query=False,)

    if base_class_generalization:
        # can only do this if there is only one type of evaluation
        print("\n", "--"*20, "BASE TEST", "--"*20)
        base_test_classes = ClassImagesSet(base_test_file, preload=str2bool(args.preload_train))
        
        if args.fix_support > 0:
            base_test_meta_dataset_using_fixS = MetaDataset(
                                                    dataset_name=dataset_name,
                                                    support_class_images_set=train_classes, query_class_images_set=base_test_classes,
                                                    image_size=image_size,
                                                    support_aug=False,
                                                    query_aug=False,
                                                    fix_support=0,
                                                    save_folder='', 
                                                    fix_support_path=os.path.join(args.output_folder, "fixed_support_pool.pkl"))

            base_test_loader_using_fixS = MetaDataLoader(
                                            dataset=base_test_meta_dataset_using_fixS,
                                            n_batches=args.n_iterations_val,
                                            batch_size=args.batch_size_val,
                                            n_way=args.n_way_val,
                                            n_shot=args.n_shot_val,
                                            n_query=args.n_query_val, 
                                            randomize_query=False,)

        print("\n", "--"*20, "BASE + NOVEL TEST", "--"*20)
        assert len(set(base_test_classes.keys()).intersection(set(test_classes.keys()))) == 0,\
            f"the base and novel classes must have different ids, base:{set(base_test_classes.keys())}, novel: f{set(test_classes.keys())}"
        # combine both base and novel classes
        base_novel_test_classes = ClassImagesSet(base_test_file, test_file)
        base_novel_test_meta_dataset = MetaDataset(
                                    dataset_name=dataset_name,
                                    support_class_images_set=base_novel_test_classes,
                                    query_class_images_set=base_novel_test_classes,
                                    image_size=image_size,
                                    support_aug=False,
                                    query_aug=False,
                                    fix_support=0,
                                    save_folder='')

        # sample classes from base and novel with mix prob. given by lambd
        base_novel_test_loaders_dict = {}
        for lambd in np.arange(0., 1.1, 0.1):
            base_novel_test_loaders_dict[lambd] = MetaDataLoader(
                dataset=base_novel_test_meta_dataset,
                n_batches=args.n_iterations_val,
                batch_size=args.batch_size_val, 
                n_way=args.n_way_val,
                n_shot=args.n_shot_val,
                n_query=args.n_query_val, 
                randomize_query=False,
                p_dict={
                    k: ((1-lambd) / len(base_test_classes) if k in base_test_classes else lambd / len(test_classes))
                        for k in list(base_test_classes) + list(test_classes)
                }
            )


    ####################################################
    #             MODEL/BACKBONE CREATION              #
    ####################################################
    
    print("\n", "--"*20, "MODEL", "--"*20)

    if args.model_type == 'resnet_12':
        if 'miniImagenet' in dataset_name or 'CUB' in dataset_name:
            model = resnet_12.resnet12(avg_pool=str2bool(args.avg_pool), drop_rate=0.1, dropblock_size=5,
                num_classes=args.num_classes_train, classifier_type=args.classifier_type,
                projection=str2bool(args.projection))
        else:
            model = resnet_12.resnet12(avg_pool=str2bool(args.avg_pool), drop_rate=0.1, dropblock_size=2,
                num_classes=args.num_classes_train, classifier_type=args.classifier_type,
                projection=str2bool(args.projection))
    elif args.model_type in ['conv64', 'conv48', 'conv32']:
        dim = int(args.model_type[-2:])
        model = shallow_conv.ShallowConv(z_dim=dim, h_dim=dim, num_classes=args.num_classes_train, 
            classifier_type=args.classifier_type, projection=str2bool(args.projection))
    elif args.model_type == 'wide_resnet28_10':
        model = wide_resnet.wrn28_10(
            projection=str2bool(args.projection), classifier_type=args.classifier_type)
    elif args.model_type == 'wide_resnet16_10':
        model = wide_resnet.wrn16_10(
            projection=str2bool(args.projection), classifier_type=args.classifier_type)
    else:
        raise ValueError(
            'Unrecognized model type {}'.format(args.model_type))
    print("Model\n" + "=="*27)    
    print(model)   


    ####################################################
    #                LOAD FROM CHECKPOINT              #
    ####################################################

    assert args.checkpoint != '', "Must provide checkpoint"
    print(f"loading model from {args.checkpoint}")
    model_dict = model.state_dict()
    chkpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    chkpt_state_dict = chkpt['model']
    chkpt_state_dict_cpy = chkpt_state_dict.copy()
    # remove "module." from key, possibly present as it was dumped by data-parallel
    for key in chkpt_state_dict_cpy.keys():
        if 'module.' in key:
            new_key = re.sub('module\.', '',  key)
            chkpt_state_dict[new_key] = chkpt_state_dict.pop(key)
    chkpt_state_dict = {k: v for k, v in chkpt_state_dict.items() if k in model_dict}
    model_dict.update(chkpt_state_dict)
    updated_keys = set(model_dict).intersection(set(chkpt_state_dict))
    print(f"Updated {len(updated_keys)} keys using chkpt")
    print("Following keys updated :", "\n".join(sorted(updated_keys)))
    missed_keys = set(model_dict).difference(set(chkpt_state_dict))
    print(f"Missed {len(missed_keys)} keys")
    print("Following keys missed :", "\n".join(sorted(missed_keys)))
    model.load_state_dict(model_dict)
                    
        
    # Multi-gpu support and device setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('Using GPUs: ', os.environ["CUDA_VISIBLE_DEVICES"])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()


    ####################################################
    #        ALGORITHM AND ALGORITHM TRAINER           #
    ####################################################

    # algorithm
    if args.algorithm == 'InitBasedAlgorithm':
        algorithm = InitBasedAlgorithm(
            model=model,
            loss_func=torch.nn.CrossEntropyLoss(),
            method=args.init_meta_algorithm,
            alpha=args.alpha,
            inner_loop_grad_clip=args.grad_clip_inner,
            inner_update_method=args.inner_update_method,
            device='cuda')
    elif args.algorithm == 'ProtoNet':
        algorithm = ProtoNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            device='cuda',
            scale=args.scale_factor,
            metric=args.classifier_metric)
    elif args.algorithm == 'SVM':
        algorithm = SVM(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            scale=args.scale_factor,
            device='cuda')
    elif args.algorithm == 'Ridge':
        algorithm = Ridge(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            scale=args.scale_factor,
            device='cuda')
    else:
        raise ValueError(
            'Unrecognized algorithm {}'.format(args.algorithm))


    if args.algorithm != 'InitBasedAlgorithm':
        trainer = Meta_algorithm_trainer(
            algorithm=algorithm,
            optimizer=None,
            writer=None,
            log_interval=args.log_interval, 
            save_folder='', 
            grad_clip=None,
            init_global_iteration=None)
    else:
        trainer = Init_algorithm_trainer(
            algorithm=algorithm,
            optimizer=None,
            writer=None,
            log_interval=args.log_interval, 
            save_folder='', 
            grad_clip=None,
            num_updates_inner_train=args.num_updates_inner_train,
            num_updates_inner_val=args.num_updates_inner_val,
            init_global_iteration=None)


    ####################################################
    #                    EVALUATION                    #
    ####################################################

    print("\n", "--"*20, "BEGIN EVALUATION", "--"*20)
    
    # On ML train objective
    print("Train Loss on ML objective")
    results = trainer.run(
        mt_loader=no_fixS_train_loader, is_training=False)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)
    with open(eval_results, 'a') as f:
        f.write(f"TrainLossOnML{args.n_way_val}w{ns_val}s: Loss {round(results['test_loss_after']['loss'], 3)} Acc {round(results['test_loss_after']['accu'], 3)}"+'\n')

    # validation/test
    val_accus = {}
    novel_test_losses = {}
    for ns_val in all_n_shot_vals:
        print("Validation ", f"n_shots_val {ns_val}")
        results = trainer.run(
            mt_loader=val_loaders[ns_val], is_training=False)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(results)
        with open(eval_results, 'a') as f:
            f.write(f"Val{args.n_way_val}w{ns_val}s: Loss {round(results['test_loss_after']['loss'], 3)} Acc {round(results['test_loss_after']['accu'], 3)}"+'\n')

        print("Test ", f"n_shots_val {ns_val}")
        results = trainer.run(
            mt_loader=test_loaders[ns_val], is_training=False)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(results)
        with open(eval_results, 'a') as f:
            f.write(f"Test{args.n_way_val}w{ns_val}s: Loss {round(results['test_loss_after']['loss'], 3)} Acc {round(results['test_loss_after']['accu'], 3)}"+'\n')

        
    # base class generalization
    if base_class_generalization:
        # can only do this if there is only one type of evaluation
        
        print("Base Test")
        
        if args.fix_support > 0:
            print("Base Test using FixSupport, matching train and test for fixml")
            results = trainer.run(
                mt_loader=base_test_loader_using_fixS, is_training=False)
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(results)
            with open(eval_results, 'a') as f:
                f.write(f"BaseTestUsingFixSupport: Loss {round(results['test_loss_after']['loss'], 3)} Acc {round(results['test_loss_after']['accu'], 3)}"+'\n')

        for lambd, base_novel_test_loader in base_novel_test_loaders_dict.items():
            print(f"Base + Novel Test lambda={round(lambd, 2)} Novel {round(1-lambd, 2)} Base")
            results = trainer.run(
                mt_loader=base_novel_test_loader, is_training=False)
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(results)
            with open(eval_results, 'a') as f:
                f.write(f"Base+NovelTestLambda={round(lambd, 2)}Novel{round(1-lambd, 2)}Base: Loss {round(results['test_loss_after']['loss'], 3)} Acc {round(results['test_loss_after']['accu'], 3)}"+'\n')




if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(
        description='Training the feature backbone on all classes from all tasks.')
    
    parser.add_argument('--random-seed', type=int, default=0,
        help='')

    # Algorithm
    parser.add_argument('--algorithm', type=str, help='type of algorithm')


    # Model
    parser.add_argument('--model-type', type=str, default='gatedconv',
        help='type of the model')
    parser.add_argument('--classifier-type', type=str, default='no-classifier',
        help='classifier type [distance based, linear, GDA]')
    parser.add_argument('--loss-names', type=str, nargs='+', default='cross_ent',
        help='names of various loss functions that are part fo overall objective')
    parser.add_argument('--scale-factor', type=float, default=1.,
        help='scalar factor multiplied with logits')


    # Initialization-based methods
    parser.add_argument('--alpha', type=float, default=0.0,
        help='inner learning rate for init based methods')
    parser.add_argument('--init-meta-algorithm', type=str, default='MAML',
        help='MAML/Reptile/FOMAML')
    parser.add_argument('--grad-clip-inner', type=float, default=0.0,
        help='gradient clip value in inner loop')
    parser.add_argument('--num-updates-inner-train', type=int, default=1,
        help='number of updates in inner loop')
    parser.add_argument('--num-updates-inner-val', type=int, default=1,
        help='number of updates in inner loop val')
    parser.add_argument('--inner-update-method', type=str, default='sgd',
        help='inner update method can be sgd or adam')


    # Dataset
    parser.add_argument('--fix-support', type=int, default=0,
        help='fix support set')
    parser.add_argument('--fix-support-path', type=str, default='',
        help='path to fix support')    
    parser.add_argument('--dataset-path', type=str,
        help='which dataset to use')
    parser.add_argument('--img-side-len', type=int, default=84,
        help='width and height of the input images')
    parser.add_argument('--num-classes-train', type=int, default=0,
        help='no of train classes')
    parser.add_argument('--num-classes-val', type=int, default=0,
        help='no of novel (val) classes')
    parser.add_argument('--batch-size-val', type=int, default=10,
        help='batch size for validation')
    parser.add_argument('--n-query-val', type=int, default=15,
        help='how many samples per class for validation (meta val)')
    parser.add_argument('--n-shot-val', type=int, default=5,
        help='how many samples per class for train (meta val)')
    parser.add_argument('--do-one-shot-eval-too', type=str, default="False",
        help='do one shot eval too, especially for SVM expts\
         where same train config is used for 5w1s, 5w5s')
    parser.add_argument('--n-way-val', type=int, default=5,
        help='how classes per task for train (meta val)')
    parser.add_argument('--label-offset', type=int, default=0,
        help='offset for label values during fine tuning stage')
    parser.add_argument('--n-query-pool', type=int, default=0,
        help='pool for query samples')
    parser.add_argument('--eps', type=float, default=0.0,
        help='epsilon of label smoothing')
    parser.add_argument('--support-aug', type=str, default="False",
        help='perform data augmentation on support set')
    parser.add_argument('--query-aug', type=str, default="False",
        help='perform data augmentation for query')
    parser.add_argument('--n-iters-per-epoch', type=int, default=1000,
        help='number of iters in epoch')
    parser.add_argument('--n-iterations-val', type=int, default=100,
        help='no. of iterations validation.') 
    parser.add_argument('--preload-train', type=str, default="True")
    parser.add_argument('--eot-model', type=str, default="False")


    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--device-number', type=str, default='0',
        help='gpu device number')
    parser.add_argument('--log-interval', type=int, default=100,
        help='number of batches between tensorboard writes')
    parser.add_argument('--checkpoint', type=str, default='',
        help='path to saved parameters.')
    parser.add_argument('--classifier-metric', type=str, default='',
        help='')
    parser.add_argument('--projection', type=str, default='',
        help='')
    parser.add_argument('--avg-pool', type=str, default='True',
        help='')
    
    
    args = parser.parse_args()

    # set random seed. only set for numpy, uncomment the below lines for torch and CuDNN.
    if args.random_seed != 0:
        np.random.seed(args.random_seed)
    
    # main function call
    main(args)
