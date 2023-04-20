import sys
sys.path.append("/coc/pskynet4/ksarangmath3/lsr/robust-ssl")

from src.utils import AllReduce
import src.deit as deit
import src.resnet50 as resnet
from src.classifier import LinearClassifier, distLinear
from src.data_manager import init_data as init_inet_data
from src.wilds_loader import init_data as init_wilds_data
from src.utils import init_model as init_pt_model
from clip.model import VisionTransformer, ModifiedResNet


import argparse
import os
# import wget
import torch
# import clip
import os
import json
import operator
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA
from utils import get_model_from_sd, test_model_on_dataset
from finetune import build_transform



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="val dataset",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('/srv/share4/ksarangmath3/lsr/model-soups/soups/1imgs_class_soup_try2/'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--download-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--eval-individual-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--uniform-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--greedy-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        '--input_size', 
        default=224, 
        type=int,
        help='images input size'
    )
    parser.add_argument(
        "--forward-blocks",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--normalize", action="store_true", default=False,
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="clip_vitb16"
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        default="lineval"
    )
    parser.add_argument(
        "--soup-size",
        type=int,
        default=9
    )
    parser.add_argument(
        "--greedy-soup-key",
        type=str,
        default="id_val"
    )
    return parser.parse_args()


def init_model(
    device,
    num_classes,
    num_blocks=1,
    normalize=True,
    finetuning=False,
    eval_type='lineval',
    model_name='deit_base',
):
    # -- init model and freeze parameters based on finetuning type
    if 'deit' in model_name:
        encoder = deit.__dict__[model_name]()
        emb_dim = 384 if 'small' in model_name else 768 if 'base' in model_name else 1024
        # emb_dim *= num_blocks
        encoder.fc = None
        encoder.norm = None
        if finetuning == 'block':
            for n, p in encoder.named_parameters():
                if 'blocks' not in n:
                    p.requires_grad_(False); continue
                n_blk = int(n.split('.')[1])
                if len(encoder.blocks) - n_blk > num_blocks: p.requires_grad_(False)
                else: p.requires_grad_(True)
        else:
            grad_flag = True if finetuning else False
            for n, p in encoder.named_parameters(): p.requires_grad_(grad_flag)
    elif 'resnet' in model_name:
        encoder = resnet.__dict__[model_name](output_dim=0, eval_mode=False)
        emb_dim = 2048 if model_name == 'resnet50' else 4096
        grad_flag = True if finetuning else False
        for n, p in encoder.named_parameters(): p.requires_grad_(grad_flag)
    elif 'clip' in model_name:
        # NOTE: -- CLIP import can somehow lead to DDP issues
        if 'vitb16' in model_name:
            emb_dim = 512
            encoder = VisionTransformer(input_resolution=224, patch_size=16, \
                width=768, layers=12, heads=12, output_dim=emb_dim)
        elif 'rn50' in model_name:
            emb_dim = 1024
            encoder = ModifiedResNet(input_resolution=224, layers=(3, 4, 6, 3), \
                heads=32, width=64, output_dim=emb_dim)
        grad_flag = True if finetuning else False
        for n, p in encoder.named_parameters(): p.requires_grad_(grad_flag) 
    else:
        raise Exception(f"Model {model_name} is not supported.")
        exit(0)

    # -- different linear classifiers based on eval type
    if eval_type == 'lineval':  
        linear_classifier = LinearClassifier(emb_dim, num_classes, normalize).to(device)
    elif eval_type == 'bslplpl':
        linear_classifier = distLinear(emb_dim, num_classes, normalize).to(device)
    elif eval_type == 'zeroshot':  
        linear_classifier = LinearClassifier(emb_dim, num_classes, True, False).to(device)
    else:
        raise Exception(f"Evaluation type {eval_type} is not supported.")
        exit(0)

    return encoder, linear_classifier

def val_step(encoder, linear_classifier, val_data_loader, num_classes, val_projection_fn, device="cuda:0"):
    encoder.eval()
    top1_correct, avg_acc, total = 0, 0, 0
    conf_mat = torch.zeros(num_classes, num_classes)
    print(f'Length of val set: {len(val_data_loader)}')
    for i, data in enumerate(val_data_loader):
        if i % 20 == 0:
            print(f'iteration {i}')
        with torch.cuda.amp.autocast(enabled=True):
            inputs, labels = data[0].to(device), data[1].to(device)
            # outputs = encoder_wo_ddp.forward_blocks(inputs, num_blocks)
            outputs = encoder(inputs)
            outputs = linear_classifier(outputs)
        if val_projection_fn:
            outputs = val_projection_fn(outputs, device)
        total += inputs.shape[0]
        top1_correct += outputs.max(dim=1).indices.eq(labels).sum()
        top1_acc = 100. * top1_correct / total
        preds = outputs.max(dim=1).indices.detach().clone()
        for l, p in zip(labels, preds): conf_mat[l, p] += 1
    top1_acc = AllReduce.apply(top1_acc)
    # -- get per-class accuracies from confusion matrix and average
    tot_per_cls, corr_per_cls = conf_mat.sum(axis=1), conf_mat.diagonal()
    per_cls_acc = corr_per_cls[tot_per_cls != 0] / tot_per_cls[tot_per_cls != 0]
    avg_acc = 100. * per_cls_acc.mean()
    return top1_acc, avg_acc


if __name__ == '__main__':
    args = parse_arguments()
    NUM_MODELS = 9
    EXP_NAME = args.model_location.strip('/').split('/')[-1]
    INDIVIDUAL_MODEL_RESULTS_FILE = f'{EXP_NAME}_individual_model_results.jsonl'
    UNIFORM_SOUP_RESULTS_FILE = f'{EXP_NAME}_uniform_soup_results.jsonl'
    GREEDY_SOUP_RESULTS_FILE = f'{EXP_NAME}_greedy_soup_results.jsonl'

    # model_paths = [os.path.join(args.model_location, f'model_{i}.pt') for i in range(NUM_MODELS)]
    model_paths = [join(args.model_location, f) for f in listdir(args.model_location) if isfile(join(args.model_location, f))]
    NUM_MODELS = len(model_paths)
    print(model_paths, len(model_paths))

    # Step 2: Evaluate individual models.
    # if args.eval_individual_models or args.uniform_soup or args.greedy_soup:
    #     pass
        # base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)
        # preprocess = build_transform(False, args)
        # base_model = deit.__dict__['deit_base']()
        # linear_classifier = LinearClassifier(768, 1000, False)
        # base_model = VisionTransformer(input_resolution=224, patch_size=16, \
        #         width=768, layers=12, heads=12, output_dim=512)
        # linear_classifier = LinearClassifier(512, 1000, False)

        # base_model, linear_classifier = init_model(
        #     device='cuda:0',
        #     num_classes=args.num_classes,
        #     normalize=args.normalize,
        #     eval_type=args.eval_type,
        #     model_name=args.model_name
        # )

    
    if args.eval_individual_models:

        init_data = init_inet_data if 'imagenet' in args.dataset else init_wilds_data
        image_folder_test = 'camelyon17_v1.0/' if 'camelyon' in args.dataset else ('iwildcam_v2.0/' if 'iwildcam' in args.dataset else 'ImageNet/imagenet/') 

        val_data_loader, _ = init_data(
            transform=None,
            batch_size=args.batch_size,        
            image_folder=image_folder_test, 
            training=False,
            drop_last=False,
            eval_type=args.eval_type,
            model_name=args.model_name,
            num_workers=args.workers
        )
        val_projection_fn = getattr(val_data_loader, 'project_logits', None)

        if os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
            print("INDIVIDUAL MODEL RESULTS ALREADY EXIST")
            exit(0)
            # os.remove(INDIVIDUAL_MODEL_RESULTS_FILE)
        for j in range(0, args.soup_size):
            model_path = join(args.model_location, f'{EXP_NAME}_{j}.pth.tar')
            print(model_path)
            assert os.path.exists(model_path)

            encoder, linear_classifier, _, _ = init_pt_model(
                device="cuda:0",
                num_classes=args.num_classes,
                normalize=args.normalize,
                training=False,        
                r_enc_path=model_path,
                world_size=1,
                model_name=args.model_name,
                finetuning=False,
                eval_type=args.eval_type,
                ref_lr=0,
                num_epochs=1,
                its_per_epoch=100)

            results = {'model_name' : f'{EXP_NAME}_{j}.pth.tar'}

            with torch.no_grad():
                top1_acc, avg_acc = val_step(encoder, linear_classifier, val_data_loader, args.num_classes, val_projection_fn)

            results['id_val'] = float(top1_acc)
            results['id_val_top1'] = float(top1_acc)
            results['id_val_avg'] = float(avg_acc)
            print(f'top1 acc: {float(top1_acc)}')
            print(f'avg acc: {float(avg_acc)}')

            with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
                f.write(json.dumps(results) + '\n')

    # Step 3: Uniform Soup.
    if args.uniform_soup:
        if os.path.exists(UNIFORM_SOUP_RESULTS_FILE):
            os.remove(UNIFORM_SOUP_RESULTS_FILE)

        base_model, linear_classifier = init_model(
            device='cuda:0',
            num_classes=args.num_classes,
            normalize=args.normalize,
            eval_type=args.eval_type,
            model_name=args.model_name
        )

        skip_paths = []
        for j, model_path in enumerate(model_paths):

            print(f'Checking validity of model {j} of {NUM_MODELS - 1} for uniform soup.')
            print(model_path)
            assert os.path.exists(model_path)

            try:
                checkpoint = torch.load(model_path, map_location='cpu')
            except:
                print(f'NOT VALID: {model_path} ')
                skip_paths.append(model_path)

        NUM_MODELS -= len(skip_paths)
        # create the uniform soup sequentially to not overload memory
        uniform_soup_model = None
        uniform_soup_lin_clf = None
        for j, model_path in enumerate(model_paths):

            if model_path in skip_paths:
                print('************************')
                print(f'skipping {model_path}')
                print('************************')
                continue

            print(f'Adding model {model_path} to uniform soup.')
            assert os.path.exists(model_path)

            try:
                checkpoint = torch.load(model_path, map_location='cpu')
            except:
                print('couldnt load model')
                continue
            
            state_dict_model = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
            state_dict_lin_clf = {k.replace('module.', ''): v for k, v in checkpoint['linear_classifier'].items()}

            if not uniform_soup_model:
                uniform_soup_model = {k : v * (1./NUM_MODELS) for k, v in state_dict_model.items()}
            else:
                uniform_soup_model = {k : v * (1./NUM_MODELS) + uniform_soup_model[k] for k, v in state_dict_model.items()}
            
            if not uniform_soup_lin_clf:
                uniform_soup_lin_clf = {k : v * (1./NUM_MODELS) for k, v in state_dict_lin_clf.items()}
            else:
                uniform_soup_lin_clf = {k : v * (1./NUM_MODELS) + uniform_soup_lin_clf[k] for k, v in state_dict_lin_clf.items()}

        for k, v in base_model.state_dict().items():
            if k not in uniform_soup_model:
                print(f'key "{k}" could not be found in loaded state dict')
            elif uniform_soup_model[k].shape != v.shape:
                print(f'key "{k}" is of different shape in model and loaded state dict')
                uniform_soup_model[k] = v
    
        for k, v in linear_classifier.state_dict().items():
            if k not in uniform_soup_lin_clf:
                print(f'key "{k}" could not be found in loaded state dict')
            elif uniform_soup_lin_clf[k].shape != v.shape:
                print(f'key "{k}" is of different shape in model and loaded state dict')
                uniform_soup_lin_clf[k] = v

        msg1 = base_model.load_state_dict(uniform_soup_model, strict=False)
        msg2 = linear_classifier.load_state_dict(uniform_soup_lin_clf, strict=False)

        print(msg1)
        print(msg2)

        base_model.to('cuda:0')
        linear_classifier.to('cuda:0')

        save_dict = {
                'target_encoder': base_model.state_dict(),
                'linear_classifier': linear_classifier.state_dict(),
            }
        torch.save(save_dict, f'{args.model_location}/uniform_soup.pth.tar')

        # results = {'model_name' : f'uniform_soup'}
        # for dataset_cls in [ImageNetSketch, ImageNetR, ImageNetA, ImageNetV2, ObjectNet]:

        #     print(f'Evaluating on {dataset_cls.__name__}.')

        #     dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
        #     accuracy = test_model_on_dataset(base_model, linear_classifier, dataset)
        #     results[dataset_cls.__name__] = accuracy
        #     print(accuracy)
       
        # with open(UNIFORM_SOUP_RESULTS_FILE, 'a+') as f:
        #     f.write(json.dumps(results) + '\n')


    # Step 4: Greedy Soup.
    if args.greedy_soup:
        if os.path.exists(GREEDY_SOUP_RESULTS_FILE):
            print("GREEDY SOUP RESULTS FILE ALREADY EXISTS")
            exit(0)

        encoder, linear_classifier = init_model(
            device='cuda:0',
            num_classes=args.num_classes,
            normalize=args.normalize,
            eval_type=args.eval_type,
            model_name=args.model_name
        )

        init_data = init_inet_data if 'imagenet' in args.dataset else init_wilds_data
        image_folder_test = 'camelyon17_v1.0/' if 'camelyon' in args.dataset else ('iwildcam_v2.0/' if 'iwildcam' in args.dataset else 'ImageNet/imagenet/') 
        val_data_loader, _ = init_data(
            transform=None,
            batch_size=args.batch_size,        
            image_folder=image_folder_test, 
            training=False,
            drop_last=False,
            eval_type=args.eval_type,
            model_name=args.model_name,
            num_workers=args.workers
        )
        val_projection_fn = getattr(val_data_loader, 'project_logits', None)

        # Sort models by decreasing accuracy on the held-out validation set ImageNet2p
        # (We call the held out-val set ImageNet2p because it is 2 percent of ImageNet train)
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_val_accs = {}
        for _, row in individual_model_db.iterrows():
            individual_model_val_accs[row['model_name']] = row[args.greedy_soup_key]
        individual_model_val_accs = sorted(individual_model_val_accs.items(), key=operator.itemgetter(1))
        individual_model_val_accs.reverse()
        sorted_models = [x[0] for x in individual_model_val_accs]
        
        # Start the soup by using the first ingredient.
        greedy_soup_ingredients = [sorted_models[0]]
        checkpoint = torch.load(os.path.join(args.model_location, sorted_models[0]))
        greedy_soup_params_enc = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
        greedy_soup_params_clf = {k.replace('module.', ''): v for k, v in checkpoint['linear_classifier'].items()}
        best_val_acc_so_far = individual_model_val_accs[0][1]

        # Now, iterate through all models and consider adding them to the greedy soup.
        for i in range(1, len(sorted_models)):
            print(f'Testing model {i} of {len(sorted_models)}')

            # Get the potential greedy soup, which consists of the greedy soup with the new model added.

            new_checkpoint = torch.load(os.path.join(args.model_location, sorted_models[i]))
            new_state_dict_enc = {k.replace('module.', ''): v for k, v in new_checkpoint['target_encoder'].items()}
            new_state_dict_clf = {k.replace('module.', ''): v for k, v in new_checkpoint['linear_classifier'].items()}
            num_ingredients = len(greedy_soup_ingredients)

            potential_greedy_soup_params_enc = {
                k : greedy_soup_params_enc[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                    new_state_dict_enc[k].clone() * (1. / (num_ingredients + 1))
                for k in new_state_dict_enc
            }
            potential_greedy_soup_params_clf = {
                k : greedy_soup_params_clf[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                    new_state_dict_clf[k].clone() * (1. / (num_ingredients + 1))
                for k in new_state_dict_clf
            }

            for k, v in encoder.state_dict().items():
                if k not in potential_greedy_soup_params_enc:
                    print(f'key "{k}" could not be found in loaded state dict')
                elif potential_greedy_soup_params_enc[k].shape != v.shape:
                    print(f'key "{k}" is of different shape in model and loaded state dict')
                    potential_greedy_soup_params_enc[k] = v
        
            for k, v in linear_classifier.state_dict().items():
                if k not in potential_greedy_soup_params_clf:
                    print(f'key "{k}" could not be found in loaded state dict')
                elif potential_greedy_soup_params_clf[k].shape != v.shape:
                    print(f'key "{k}" is of different shape in model and loaded state dict')
                    potential_greedy_soup_params_clf[k] = v

            msg1 = encoder.load_state_dict(potential_greedy_soup_params_enc, strict=False)
            msg2 = linear_classifier.load_state_dict(potential_greedy_soup_params_clf, strict=False)

            print(msg1)
            print(msg2)

            # Run the potential greedy soup on the held-out val set.
            encoder.to('cuda:0')
            linear_classifier.to('cuda:0')
            with torch.no_grad():
                top1_acc, avg_acc = val_step(encoder, linear_classifier, val_data_loader, args.num_classes, val_projection_fn)
                metrics = {"id_val": float(top1_acc), "id_val_top1": float(top1_acc), "id_val_avg": float(avg_acc)}
            held_out_val_accuracy = metrics[args.greedy_soup_key]

            # If accuracy on the held-out val set increases, add the new model to the greedy soup.
            print(f'Potential greedy soup val acc {held_out_val_accuracy}, best so far {best_val_acc_so_far}.')
            if held_out_val_accuracy > best_val_acc_so_far:
                greedy_soup_ingredients.append(sorted_models[i])
                best_val_acc_so_far = held_out_val_accuracy
                greedy_soup_params_enc = potential_greedy_soup_params_enc
                greedy_soup_params_clf = potential_greedy_soup_params_clf
                print(f'Adding to soup. New soup is {greedy_soup_ingredients}')

        # Finally, evaluate the greedy soup.

        for k, v in encoder.state_dict().items():
            if k not in greedy_soup_params_enc:
                print(f'key "{k}" could not be found in loaded state dict')
            elif greedy_soup_params_enc[k].shape != v.shape:
                print(f'key "{k}" is of different shape in model and loaded state dict')
                greedy_soup_params_enc[k] = v
    
        for k, v in linear_classifier.state_dict().items():
            if k not in greedy_soup_params_clf:
                print(f'key "{k}" could not be found in loaded state dict')
            elif greedy_soup_params_clf[k].shape != v.shape:
                print(f'key "{k}" is of different shape in model and loaded state dict')
                greedy_soup_params_clf[k] = v

        msg1 = encoder.load_state_dict(greedy_soup_params_enc, strict=False)
        msg2 = linear_classifier.load_state_dict(greedy_soup_params_clf, strict=False)

        print(msg1)
        print(msg2)

        save_dict = {
                'target_encoder': encoder.state_dict(),
                'linear_classifier': linear_classifier.state_dict(),
            }

        torch.save(save_dict, f'{args.model_location}/greedy_soup.pth.tar')
        results = {'model_name' : f'greedy_soup'}
        encoder.to('cuda:0')
        linear_classifier.to('cuda:0')
        with torch.no_grad():
            top1_acc, avg_acc = val_step(encoder, linear_classifier, val_data_loader, args.num_classes, val_projection_fn)
        results['id_val'] = float(top1_acc)
        results['id_val_top1'] = float(top1_acc)
        results['id_val_avg'] = float(avg_acc)
        print(f'top1 acc: {float(top1_acc)}')
        print(f'avg acc: {float(avg_acc)}')
        

        with open(GREEDY_SOUP_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')
