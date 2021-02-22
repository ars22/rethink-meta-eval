import os
from tqdm import tqdm
import json
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import pprint
from tensorboardX import SummaryWriter
import re
import shutil
from datetime import datetime


from src.models import shallow_conv, resnet_12, wide_resnet, dense_net
from src.algorithm_trainer.algorithm_trainer import Meta_algorithm_trainer, Init_algorithm_trainer, TL_algorithm_trainer
from src.algorithms.algorithm import SVM, ProtoNet, Ridge, InitBasedAlgorithm
from src.optimizers import modified_sgd
from src.data.dataset_managers import MetaDataLoader
from src.data.datasets import MetaDataset, ClassImagesSet, SimpleDataset


def ensure_path(path):
    if os.path.exists(path):
        print("Path Exists", path, "Appending timestamp")
        path = path + "_" + datetime.now().strftime("%d:%b:%Y:%H:%M:%S")
        print("New path is", path)
        os.makedirs(path)
    else:
        os.makedirs(path)
    return path


def str2bool(arg):
    return arg.lower() == 'true'


def main(args):


    ####################################################
    #                LOGGING AND SAVING                #
    ####################################################
    args.output_folder = ensure_path('./runs/{0}'.format(args.output_folder))
    writer = SummaryWriter(args.output_folder)
    with open(f'{args.output_folder}/config.txt', 'w') as config_txt:
        for k, v in sorted(vars(args).items()):
            config_txt.write(f'{k}: {v}\n')
    save_folder = args.output_folder


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

    print("\n", "--"*20, "TRAIN", "--"*20)
    train_classes = ClassImagesSet(train_file, preload=str2bool(args.preload_train))
    if args.algorithm == 'TransferLearning':
        """
        For Transfer Learning we create a SimpleDataset.
        The augmentation is decided by query_aug flag.
        """
        train_dataset = SimpleDataset(
                            dataset_name=dataset_name,
                            class_images_set=train_classes,
                            image_size=image_size,
                            aug=str2bool(args.query_aug))

        train_loader = torch.utils.data.DataLoader(
                            train_dataset, 
                            batch_size=args.batch_size_train, 
                            shuffle=True,
                            num_workers=6)

    else:
        train_meta_dataset = MetaDataset(
                                dataset_name=dataset_name,
                                support_class_images_set=train_classes,
                                query_class_images_set=train_classes, 
                                image_size=image_size,
                                support_aug=str2bool(args.support_aug),
                                query_aug=str2bool(args.query_aug),
                                fix_support=args.fix_support,
                                save_folder=save_folder,
                                fix_support_path=args.fix_support_path)

        train_loader = MetaDataLoader(
                            dataset=train_meta_dataset,
                            batch_size=args.batch_size_train,
                            n_batches=args.n_iters_per_epoch,
                            n_way=args.n_way_train,
                            n_shot=args.n_shot_train,
                            n_query=args.n_query_train,
                            randomize_query=str2bool(args.randomize_query))

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
    val_classes = ClassImagesSet(val_file, preload=False)    
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

    print("\n", "--"*20, "TEST", "--"*20)
    test_classes = ClassImagesSet(test_file)
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
        base_test_classes = ClassImagesSet(base_test_file)
        base_test_meta_dataset = MetaDataset(
                                    dataset_name=dataset_name,
                                    support_class_images_set=base_test_classes,
                                    query_class_images_set=base_test_classes,
                                    image_size=image_size,
                                    support_aug=False,
                                    query_aug=False,
                                    fix_support=0,
                                    save_folder=save_folder)
        base_test_loader = MetaDataLoader(
                                dataset=base_test_meta_dataset,
                                n_batches=args.n_iterations_val,
                                batch_size=args.batch_size_val,
                                n_way=args.n_way_val,
                                n_shot=args.n_shot_val,
                                n_query=args.n_query_val, 
                                randomize_query=False)


        if args.fix_support > 0:
            base_test_meta_dataset_using_fixS = MetaDataset(
                                                    dataset_name=dataset_name,
                                                    support_class_images_set=train_classes, query_class_images_set=base_test_classes,
                                                    image_size=image_size,
                                                    support_aug=False,
                                                    query_aug=False,
                                                    fix_support=0,
                                                    save_folder=save_folder, 
                                                    fix_support_path=os.path.join(save_folder, "fixed_support_pool.pkl"))

            base_test_loader_using_fixS = MetaDataLoader(
                                            dataset=base_test_meta_dataset_using_fixS,
                                            n_batches=args.n_iterations_val,
                                            batch_size=args.batch_size_val,
                                            n_way=args.n_way_val,
                                            n_shot=args.n_shot_val,
                                            n_query=args.n_query_val, 
                                            randomize_query=False,)
                                            

    ####################################################
    #             MODEL/BACKBONE CREATION              #
    ####################################################
    
    print("\n", "--"*20, "MODEL", "--"*20)

    if args.model_type == 'resnet_12':
        if 'miniImagenet' in dataset_name or 'CUB' in dataset_name:
            model = resnet_12.resnet12(avg_pool=str2bool(args.avg_pool), drop_rate=0.1, dropblock_size=5,
                num_classes=args.num_classes_train, classifier_type=args.classifier_type,
                projection=str2bool(args.projection), learnable_scale=str2bool(args.learnable_scale))
        else:
            model = resnet_12.resnet12(avg_pool=str2bool(args.avg_pool), drop_rate=0.1, dropblock_size=2,
                num_classes=args.num_classes_train, classifier_type=args.classifier_type,
                projection=str2bool(args.projection), learnable_scale=str2bool(args.learnable_scale))
    elif args.model_type in ['conv64', 'conv48', 'conv32']:
        dim = int(args.model_type[-2:])
        model = shallow_conv.ShallowConv(z_dim=dim, h_dim=dim, num_classes=args.num_classes_train, 
            classifier_type=args.classifier_type, projection=str2bool(args.projection), learnable_scale=str2bool(args.learnable_scale))
    elif args.model_type == 'wide_resnet28_10':
        model = wide_resnet.wrn28_10(
            projection=str2bool(args.projection), classifier_type=args.classifier_type, learnable_scale=str2bool(args.learnable_scale))
    elif args.model_type == 'wide_resnet16_10':
        model = wide_resnet.wrn16_10(
            projection=str2bool(args.projection), classifier_type=args.classifier_type, learnable_scale=str2bool(args.learnable_scale))
    else:
        raise ValueError(
            'Unrecognized model type {}'.format(args.model_type))
    print("Model\n" + "=="*27)    
    print(model)   



    ####################################################
    #                OPTIMIZER CREATION                #
    ####################################################

    # optimizer construction
    print("\n", "--"*20, "OPTIMIZER", "--"*20)
    print("Optimzer", args.optimizer_type)
    if args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
        ])
    else:
        optimizer = modified_sgd.SGD([
            {'params': model.parameters(), 'lr': args.lr,
            'weight_decay': args.weight_decay, 'momentum': 0.9, 'nesterov': True},
        ])
    print("Total n_epochs: ", args.n_epochs)   

    # learning rate scheduler creation
    if args.lr_scheduler_type == 'deterministic':
        drop_eps = [int(x) for x in args.drop_lr_epoch.split(',')]
        if args.drop_factors != '':
            drop_factors = [float(x) for x in args.drop_factors.split(',')]
        else:
            drop_factors = [0.06, 0.012, 0.0024]
        assert len(drop_factors) >= len(drop_eps), "No ennough drop factors"
        print("Drop lr at epochs", drop_eps)
        print("Drop factors", drop_factors[:len(drop_eps)])
        assert len(drop_eps) <= 3, "Must give less than or equal to three epochs to drop lr"
        if len(drop_eps) == 3:
            lambda_epoch = lambda e: 1.0 if e < drop_eps[0] else (drop_factors[0] if e < drop_eps[1] else drop_factors[1] if e < drop_eps[2] else (drop_factors[2]))
        elif len(drop_eps) == 3:
            lambda_epoch = lambda e: 1.0 if e < drop_eps[0] else (drop_factors[0] if e < drop_eps[1] else drop_factors[1])
        else:
            lambda_epoch = lambda e: 1.0 if e < drop_eps[0] else drop_factors[0]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
             optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
        for _ in range(args.restart_iter):
            lr_scheduler.step()

    elif args.lr_scheduler_type == 'val_based':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            mode='max', patience=5, factor=0.1, min_lr=5e-6, threshold=0.5)    

    else:
        raise ValueError("Unimplemented lr scheduler")
    
    print("LR scheduler ", args.lr_scheduler_type)  


    ####################################################
    #                LOAD FROM CHECKPOINT              #
    ####################################################

    if args.checkpoint != '':
        print(f"loading model from {args.checkpoint}")
        model_dict = model.state_dict()
        chkpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        try:
            print(f"loading optimizer from {args.checkpoint}")
            optimizer.state = chkpt['optimizer'].state
            print("Successfully loaded optimizer")
        except:
            print("Failed to load optimizer")
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

    # start tboard from restart iter
    init_global_iteration = 0
    if args.restart_iter:
        init_global_iteration = args.restart_iter * args.n_iters_per_epoch 

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
    elif args.algorithm == 'TransferLearning':
        """
        We use the ProtoNet algorithm at test time.
        """
        algorithm = ProtoNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            device='cuda',
            scale=args.scale_factor,
            metric=args.classifier_metric)
    else:
        raise ValueError(
            'Unrecognized algorithm {}'.format(args.algorithm))


    if args.algorithm == 'InitBasedAlgorithm':
        trainer = Init_algorithm_trainer(
            algorithm=algorithm,
            optimizer=optimizer,
            writer=writer,
            log_interval=args.log_interval, 
            save_folder=save_folder, 
            grad_clip=args.grad_clip,
            num_updates_inner_train=args.num_updates_inner_train,
            num_updates_inner_val=args.num_updates_inner_val,
            init_global_iteration=init_global_iteration)
    elif args.algorithm == 'TransferLearning':
        trainer = TL_algorithm_trainer(
            algorithm=algorithm,
            optimizer=optimizer,
            writer=writer,
            log_interval=args.log_interval, 
            save_folder=save_folder, 
            grad_clip=args.grad_clip,
            init_global_iteration=init_global_iteration
        )
    else:        
        trainer = Meta_algorithm_trainer(
            algorithm=algorithm,
            optimizer=optimizer,
            writer=writer,
            log_interval=args.log_interval, 
            save_folder=save_folder, 
            grad_clip=args.grad_clip,
            init_global_iteration=init_global_iteration)
        

    ####################################################
    #                  TRAINER LOOP                    #
    ####################################################

    print("\n", "--"*20, "BEGIN TRAINING", "--"*20)
    
    # iterate over training epochs
    for iter_start in range(args.restart_iter, args.n_epochs):

        # training
        for param_group in optimizer.param_groups:
            print('\n\nlearning rate:', param_group['lr'])

        trainer.run(
            mt_loader=train_loader,
            is_training=True,
            epoch=iter_start + 1)

        if iter_start % args.val_frequency == 0:
            # On ML train objective
            print("Train Loss on ML objective")
            results = trainer.run(
                mt_loader=no_fixS_train_loader, is_training=False)
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(results)
            writer.add_scalar(
                "train_acc_on_ml", results['test_loss_after']['accu'], iter_start + 1)
            writer.add_scalar(
                "train_loss_on_ml", results['test_loss_after']['loss'], iter_start + 1)
            base_train_loss = results['test_loss_after']['loss']

            # validation/test
            val_accus = {}
            novel_test_losses = {}
            for ns_val in all_n_shot_vals:
                print("Validation ", f"n_shots_val {ns_val}")
                results = trainer.run(
                    mt_loader=val_loaders[ns_val], is_training=False)
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(results)
                writer.add_scalar(
                    f"val_acc_{args.n_way_val}w{ns_val}s", results['test_loss_after']['accu'], iter_start + 1)
                writer.add_scalar(
                    f"val_loss_{args.n_way_val}w{ns_val}s", results['test_loss_after']['loss'], iter_start + 1)
                val_accus[ns_val] = results['test_loss_after']['accu']
                
                print("Test ", f"n_shots_val {ns_val}")
                results = trainer.run(
                    mt_loader=test_loaders[ns_val], is_training=False)
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(results)
                writer.add_scalar(
                    f"test_acc_{args.n_way_val}w{ns_val}s", results['test_loss_after']['accu'], iter_start + 1)
                writer.add_scalar(
                    f"test_loss_{args.n_way_val}w{ns_val}s", results['test_loss_after']['loss'], iter_start + 1)
                novel_test_losses[ns_val] = results['test_loss_after']['loss']

            val_accu = val_accus[args.n_shot_val] # stick with 5w5s for model selection
            novel_test_loss = novel_test_losses[args.n_shot_val] # stick with 5w5s for model selection
            
            # base class generalization
            if base_class_generalization:
                # can only do this if there is only one type of evaluation
                
                print("Base Test")
                results = trainer.run(
                    mt_loader=base_test_loader, is_training=False)
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(results)
                writer.add_scalar(
                    "base_test_acc", results['test_loss_after']['accu'], iter_start + 1)
                writer.add_scalar(
                    "base_test_loss", results['test_loss_after']['loss'], iter_start + 1)
                base_test_loss = results['test_loss_after']['loss']
                writer.add_scalar(
                    "base_gen_gap", base_test_loss - base_train_loss, iter_start + 1)
                writer.add_scalar(
                    "novel_gen_gap", novel_test_loss - base_train_loss, iter_start + 1)
            
                if args.fix_support > 0:
                    print("Base Test using FixSupport, matching train and test for fixml")
                    results = trainer.run(
                        mt_loader=base_test_loader_using_fixS, is_training=False)
                    pp = pprint.PrettyPrinter(indent=4)
                    pp.pprint(results)
                    writer.add_scalar(
                        "base_test_acc_usingFixS", results['test_loss_after']['accu'], iter_start + 1)
                    writer.add_scalar(
                        "base_test_loss_usingFixS", results['test_loss_after']['loss'], iter_start + 1)
    

        # scheduler step
        if args.lr_scheduler_type == 'val_based':
            assert args.val_frequency == 1, "eval after every epoch is mandatory for val based lr scheduler"
            lr_scheduler.step(val_accu)
        else:
            lr_scheduler.step()


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(
        description='Training the feature backbone on all classes from all tasks.')
    
    parser.add_argument('--random-seed', type=int, default=0,
        help='')

    # Algorithm
    parser.add_argument('--algorithm', type=str, help='type of algorithm')


    # Model
    parser.add_argument('--model-type', type=str, default='resnet_12',
        help='type of the model')
    parser.add_argument('--classifier-type', type=str, default='no-classifier',
        help='classifier type [distance based, linear, GDA]')
    parser.add_argument('--loss-names', type=str, nargs='+', default='cross_ent',
        help='names of various loss functions that are part fo overall objective')
    parser.add_argument('--scale-factor', type=float, default=1.,
        help='scalar factor multiplied with logits')
    parser.add_argument('--learnable-scale', type=str, default="False",
        help='scalar receives grads')

    # Optimization
    parser.add_argument('--optimizer-type', type=str, default='SGDM',
        help='SGDM/Adam')
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate for the global update')
    parser.add_argument('--grad-clip', type=float, default=0.0,
        help='gradient clipping')
    parser.add_argument('--n-epochs', type=int, default=60000,
        help='number of model training epochs')
    parser.add_argument('--weight-decay', type=float, default=0.,
        help='weight decay')
    parser.add_argument('--drop-lr-epoch', type=str, default='20,40,50')
    parser.add_argument('--drop-factors', type=str, default='')
    parser.add_argument('--lr-scheduler-type', type=str, default='deterministic')


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
    parser.add_argument('--batch-size-train', type=int, default=20,
        help='batch size for training')
    parser.add_argument('--num-classes-train', type=int, default=0,
        help='no of train classes')
    parser.add_argument('--num-classes-val', type=int, default=0,
        help='no of novel (val) classes')
    parser.add_argument('--batch-size-val', type=int, default=10,
        help='batch size for validation')
    parser.add_argument('--n-query-train', type=int, default=15,
        help='how many samples per class for validation (meta train)')
    parser.add_argument('--n-query-val', type=int, default=15,
        help='how many samples per class for validation (meta val)')
    parser.add_argument('--n-shot-train', type=int, default=5,
        help='how many samples per class for train (meta train)')
    parser.add_argument('--n-shot-val', type=int, default=5,
        help='how many samples per class for train (meta val)')
    parser.add_argument('--do-one-shot-eval-too', type=str, default="False",
        help='do one shot eval too, especially for SVM expts\
         where same train config is used for 5w1s, 5w5s')
    parser.add_argument('--n-way-train', type=int, default=5,
        help='how classes per task for train (meta train)')
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
    parser.add_argument('--randomize-query', type=str, default="False",
        help='random query pts per class')
    parser.add_argument('--preload-train', type=str, default="False")


    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--device-number', type=str, default='0',
        help='gpu device number')
    parser.add_argument('--log-interval', type=int, default=100,
        help='number of batches between tensorboard writes')
    parser.add_argument('--checkpoint', type=str, default='',
        help='path to saved parameters.')
    parser.add_argument('--restart-iter', type=int, default=0,
        help='iteration at restart') 
    parser.add_argument('--classifier-metric', type=str, default='',
        help='')
    parser.add_argument('--projection', type=str, default='',
        help='')
    parser.add_argument('--avg-pool', type=str, default='True',
        help='')
    parser.add_argument('--val-frequency', type=int, default=1,
        help='') 
    
    
    args = parser.parse_args()

    # set random seed. only set for numpy, uncomment the below lines for torch and CuDNN.
    if args.random_seed != 0:
        np.random.seed(args.random_seed)
    
    # main function call
    main(args)
