
from src.utils import init_model
from src.data_manager import init_data as init_inet_data
from src.wilds_loader import init_data as init_wilds_data

import argparse
import os
import torch
from tqdm import tqdm
import time

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from utils import maybe_dictionarize_batch, cosine_lr
import torchvision.transforms as transforms

import PIL

from argparse import Namespace



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_location",
        type=str,
        default=os.path.expanduser('/srv/share/datasets/ImageNet/'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default=os.path.expanduser('/srv/share/datasets/'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default=os.path.expanduser('ImageNet/'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--subset_file",
        type=str,
        default=os.path.expanduser('/srv/share4/ksarangmath3/lsr/robust-ssl/subsets/imagenet_subsets1/1imgs_class.txt'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=os.path.expanduser('/srv/share4/asingh866/msn/pretrained/msn/LPFT_subsets1_1imgs_class-lin-eval.pth.tar'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model_location",
        type=str,
        default=os.path.expanduser('/srv/share4/ksarangmath3/lsr/model-soups/soups/'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--name",
        default='finetune_cp_msn',
        help='Filename for the checkpoints.'
    )
    parser.add_argument(
        "--timm_aug", action="store_true", default=False,
    )
    parser.add_argument(
        "--aug",
        type=str,
        default=None,
        help="randaug hparams",
    )
    parser.add_argument(
        "--mix",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--nb_classes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--soup_num",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--soup_size",
        type=int,
        default=72,
    )

    parser.add_argument(
        '--input_size', 
        default=224, 
        type=int,
        help='images input size'
    )

    parser.add_argument(
        '--num_iter', 
        default=None, 
        type=int,
        help='images input size'
    )

    parser.add_argument(
        "--forward-blocks",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--eval_type",
        type=str,
        default='lineval',
        help="randaug hparams",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default='deit_base',
        help="randaug hparams",
    )

    return parser.parse_args()

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        input_size = 224 if 'clip' in args.model_name else args.input_size
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            auto_augment=args.aug,
            interpolation='bicubic',
            mean=mean,
            std=std,
        )
        if 'clip' in args.model_name:
            transform = transforms.Compose([transforms.Resize(size=(224, 224)),transform])
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def main(submitit_args=None):

    args = Namespace(**submitit_args)
    print(args)

    DEVICE = 'cuda:0'

    mixup_fn = None
    mixup_active = args.mix > 0

    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mix,
            label_smoothing=args.smoothing, 
            num_classes=args.nb_classes
        )

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    val_criterion = torch.nn.CrossEntropyLoss()

    train_aug = build_transform(True, args)

    init_data = init_wilds_data if 'wilds' in args.root_path else init_inet_data

    train_loader, _ = init_data(
            transform=train_aug,
            batch_size=args.batch_size,
            num_workers=args.workers,
            root_path=args.root_path,
            image_folder=args.image_folder,
            training=True,
            drop_last=True,
            subset_file=args.subset_file,
            model_name=args.model_name
        )

    test_loader, _ = init_data(
            transform=None,
            batch_size=args.batch_size,
            num_workers=args.workers,
            root_path=args.root_path,
            image_folder=args.image_folder,
            training=False,
            drop_last=False,
            subset_file=None,
            model_name=args.model_name
        )

    model, linear_classifier, _, _ = init_model(
        eval_type=args.eval_type,
        model_name=args.model_name,
        device=DEVICE,
        num_classes=args.nb_classes,
        training=True,
        r_enc_path=args.pretrained_path,
        world_size=1,
        ref_lr=1,
        num_epochs=1,
        its_per_epoch=1,
        num_blocks=1,
        normalize=True,
        finetuning=False,
        warmup_epochs=0,
        weight_decay=0,
        nesterov=True,
        dampening=0.0,
        start_lr=None,
        final_lr=None,
    )

    devices = [x for x in range(torch.cuda.device_count())]

    
    if args.forward_blocks:
        print('Using forward blocks')
    else:
        model = torch.nn.DataParallel(model,  device_ids=devices)
        linear_classifier = torch.nn.DataParallel(linear_classifier,  device_ids=devices)
    
    model = model.cuda()
    linear_classifier = linear_classifier.cuda()

    model_parameters = [p for p in model.parameters() if p.requires_grad] + [p for p in linear_classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.wd)

    num_batches = len(train_loader)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    model_path = os.path.join(args.model_location, f'{args.name}_{args.soup_num}.pth.tar')
    if not os.path.exists(args.model_location):
        os.makedirs(args.model_location)
    print('Saving model to', model_path)
    save_dict = {
        'target_encoder': model.state_dict(),
        'linear_classifier': linear_classifier.state_dict()
    }
    torch.save(save_dict, model_path)

    if args.num_iter:
        args.epochs = (args.num_iter // len(train_loader)) * args.epochs
        print(f"Number of adjusted epochs: {args.epochs}")
        
    save_freq = args.epochs // 4

    for epoch in range(args.epochs):
        
        # Train
        model.train()
        linear_classifier.train()
        end = time.time()
        for i, batch in enumerate(train_loader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)


            if mixup_fn is not None:
                inputs, labels = mixup_fn(inputs, labels)
            
            data_time = time.time() - end

            if args.forward_blocks:
                outputs = model.forward_blocks(inputs, args.forward_blocks)
            else:
                outputs = model(inputs)
            outputs = linear_classifier(outputs)

            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(linear_classifier.parameters(), 1.0)

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            if i % 20 == 0:
                percent_complete = 100.0 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        if epoch % save_freq == 0:
            model_path = os.path.join(args.model_location, f'{args.name}_{args.soup_num}.pth.tar')
            save_dict = {
                'target_encoder': model.state_dict(),
                'linear_classifier': linear_classifier.state_dict()
            }
            torch.save(save_dict, model_path)

    # #Evaluate
    model.eval()
    linear_classifier.eval()
    with torch.no_grad():
        print('*'*80)
        print('Starting eval')
        correct, count = 0.0, 0.0
        pbar = tqdm(test_loader)
        for batch in pbar:
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)

            if args.forward_blocks:
                outputs = model.forward_blocks(inputs, args.forward_blocks)
            else:
                outputs = model(inputs)
            outputs = linear_classifier(outputs)

            loss = val_criterion(outputs, labels)

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            count += len(outputs)
            pbar.set_description(
                f"Val loss: {loss.item():.4f}   Acc: {100*correct/count:.2f}")
        top1 = correct / count
    print(f'Val acc at epoch {args.epochs}: {100*top1:.2f}')

    model_path = os.path.join(args.model_location, f'{args.name}_{args.soup_num}.pth.tar')
    print('Saving model to', model_path)
    save_dict = {
        'target_encoder': model.state_dict(),
        'linear_classifier': linear_classifier.state_dict()
    }
    torch.save(save_dict, model_path)

if __name__ == '__main__':
    main()