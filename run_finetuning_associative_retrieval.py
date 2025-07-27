import logging
import os
from pathlib import Path

import torch
import numpy as np
import transformers
from torch.utils.data import DataLoader

from lm_experiments_tools.data import ar_collate_fn, ARDataset
from lm_experiments_tools import Trainer, TrainerArgs
from functools import partial

import accelerate

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# import transformers  # noqa: E402
from transformers import AutoConfig, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402
import lm_experiments_tools.optimizers as optimizers  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
# torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_name', type=str, help='Scrolls task name: "gov_report", "summ_screen_fd", "qmsum", '
                                                  '"narrative_qa", "qasper", "quality", "contract_nli"')

parser.add_argument('--report_to', type=str, default='wandb', help='')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')

parser.add_argument('--wrap_pos', action='store_true', default=False,
                    help='Wrap positional encoding for memory tokens (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
# parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
# parser.add_argument('--target_seq_len', type=int, default=16, help='target sequnce length, should be set to '
                                                                #    'max(len(target))+1 for EOS (default: 16).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attention mask, '
                    'eval on last segment only', default=False)

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')
parser.add_argument('--layers_attr', type=str, default=None, help='attribute of model, which contains layers')

# Dataset args
parser.add_argument('--key_size', type=int, default=None, help='number of digits in keys')
parser.add_argument('--value_size', type=int, default=None, help='number of digits in values')
parser.add_argument('--num_pairs', type=int, default=None, help='number of key-value pairs in sample')
parser.add_argument('--num_test_pairs', type=int, default=None, help='number of key-value pairs in test sample')
parser.add_argument('--dataset_path', type=str, default="/home/jovyan/rmt/datasets/associative_retrieval/", help="path to saved datasets")
parser.add_argument('--train_size', type=int, default=10000, help='number of samples in train split')
parser.add_argument('--valid_size', type=int, default=1000, help='number of samples in validation split')
parser.add_argument('--test_size', type=int, default=2000, help='number of samples in test split')
parser.add_argument('--segment_size', type=int, default=128, help='number of useful tokens in a segment')
parser.add_argument('--d_mem', type=int, default=32, help='number of rows in associative matrix')
parser.add_argument('--rewrite_setting', action='store_true', default=False,
                    help='keys can occur several times')
parser.add_argument('--no_correction', action='store_true', default=False,
                    help='ARMT shmidhuber correction for rewriting')
parser.add_argument('--desired_metric', type=float, default=1.0, help='metric to stop training')
# Aydar # RMT args 
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--segment_alignment', type=str, default=None, help="How to align segments when splitting input")
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')
parser.add_argument('--res_mem_count', type=int, default=-1, help='max number of memory segments to keep after dropout')
parser.add_argument('--aggr_type', type=str, default='mem_attn', help='aggregation type for memory retrieval')
parser.add_argument('--aggr_pos_embed', type=str, default='rope', help='positional embeddings for memory tokens')


# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')


NUM_SYMBOLS = 16
if __name__ == '__main__':
    args = parser.parse_args()
    if args.num_test_pairs is None:
        args.num_test_pairs = args.num_pairs
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    from accelerate.logging import get_logger
    logger = get_logger('')

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    rewrite_setting = args.rewrite_setting
    prepare_run(args, logger, logger_fmt)

    block_size = args.segment_size
    sep_token, gen_token, eos_token = 100, 101, 102

    collate_fn = partial(ar_collate_fn, vary_n_segments=args.vary_n_segments, rewrite_setting=args.rewrite_setting,
                            sep_token=sep_token, gen_token=gen_token, eos_token=eos_token, value_size=args.value_size)

    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    # get train dataset
    logger.info(f'preparing dataset for: {args.task_name}')
    
    dataset_name = f"AR_k{args.key_size}_v{args.value_size}_p{args.num_pairs}_valp{args.num_test_pairs}"
    
    if rewrite_setting:
        dataset_name += '_rewrite'
    if args.vary_n_segments:
        dataset_name += '_rnd'
    if args.validate_only:
        dataset_name += '_for_testing'
    else:
        dataset_name += '_for_training'
    path = os.path.join(args.dataset_path, dataset_name)
    with accelerator.main_process_first():
        if os.path.exists(path):
            print(f"Loading {dataset_name} from disk.")
            train_dataset = torch.load(os.path.join(path, 'train'), weights_only=False)
            valid_dataset = torch.load(os.path.join(path, 'valid'), weights_only=False)
            test_dataset = torch.load(os.path.join(path, 'test'), weights_only=False)
        else:
            os.system(f"mkdir {path}")
            train_dataset = ARDataset(args.key_size, args.value_size, sample_len=args.num_pairs, num_samples=args.train_size, rewrite_setting=args.rewrite_setting, num_symbols=NUM_SYMBOLS)
            valid_dataset = ARDataset(args.key_size, args.value_size, sample_len=args.num_test_pairs, num_samples=args.valid_size, rewrite_setting=args.rewrite_setting, num_symbols=NUM_SYMBOLS)
            test_dataset = ARDataset(args.key_size, args.value_size, sample_len=args.num_test_pairs, num_samples=args.test_size, rewrite_setting=args.rewrite_setting, num_symbols=NUM_SYMBOLS)
            torch.save(train_dataset, os.path.join(path, 'train'))
            torch.save(valid_dataset, os.path.join(path, 'valid'))
            torch.save(test_dataset,  os.path.join(path, 'test'))

    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size,  generator=train_rnd_generator,
                                  collate_fn=collate_fn, **kwargs)
    valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size,
                                  collate_fn=collate_fn, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size,
                                  collate_fn=collate_fn, **kwargs)
    

    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model
    model_cls = get_cls_by_name(args.model_cls)

    logger.info(f'Using model class: {model_cls}')
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        model = model_cls(config=model_cfg)
    else:
        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        model = model_cls.from_pretrained(args.from_pretrained)

    # ## add [GEN] token
    # model.resize_token_embeddings(len(tokenizer))
    
    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'])
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

        
    memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
    recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
    logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
    
    
    mem_cell_args = dict(
        base_model=model,
    )
    if args.d_mem is not None:
        mem_cell_args['d_mem'] = args.d_mem

    if args.num_mem_tokens is not None:
        mem_cell_args['num_mem_tokens'] = args.num_mem_tokens
        # mem_cell_args['wrap_pos'] = args.wrap_pos
    if args.layers_attr is not None:
        mem_cell_args['layers_attr'] = args.layers_attr
        mem_cell_args["aggr_type"] = args.aggr_type
    if args.aggr_pos_embed is not None:
        mem_cell_args["aggr_pos_embed"] = args.aggr_pos_embed
    if args.res_mem_count is not None:
        mem_cell_args["res_mem_count"] = args.res_mem_count

    if args.no_correction:
        mem_cell_args['correction'] = False

    cell = memory_cell_cls(**mem_cell_args)
    model = recurrent_wrapper_cls(
        cell, 
        segment_size=block_size,
        max_n_segments=args.max_n_segments, 
        k2=args.k2,
        segment_alignment=args.segment_alignment,
        res_mem_count=args.res_mem_count
    )
                                    

        ## load cpt of rmt
    if args.model_cpt and args.model_cpt != 'None':
        model_cpt = os.path.join(args.model_cpt, "model_best/model.pth")
        cpt = torch.load(model_cpt, map_location='cpu')
        model.load_state_dict(cpt, strict=False)
        logger.info(f'Loaded RMT state dict from: {args.model_cpt}')

    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            if 'memory' not in n and 'lora' not in n and 'aggr' not in n:
                p.requires_grad = False
        logger.info(f'Frozen moodel weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    
    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        # https://github.com/huggingface/transformers/pull/9751/files -> transformers 4.3.0
        optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                                  scale_parameter=args.scale_parameter,
                                  relative_step=args.relative_step,
                                  warmup_init=args.warmup_init,
                                  weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        data['labels'] = batch['labels']
        data['labels_mask'] = batch['labels_mask']

        if 'generation_outputs' in output:
            data['generation_outputs'] = output['generation_outputs']
        
        if 'logits' in output:
            data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
            data['predicted_labels'] = [p[m] for p, m in zip(data['predictions'], batch['labels_mask'])]

        for key in batch.keys():
            if 'loss' in key: 
                data[key] = batch[key]

        return data

    # HF datasets can compute metrics on each gpu process and then aggregate them on process with rank 0
    # synchronization is done by using temporay files on a shared filesystem
    # rank and number of workers is set by num_process and process_id params
    # BUT our Trainer aggregates all prediction from all gpus!
    #   this will lead to computing metrics for predictions repeated xN_GPUS times
    # need to try:
    # - keep_in_memory=True, may lead to OOM for large validation sets, after sync predictions and targets for the full
    #       validation set would be stored on each GPU -> xN_GPUs RAM
    #   - implemented currently
    # - compute metrics on batch lvl
    # - add support of HF metrics and turn off aggregation in case if metric has .add_batch method

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = None, None
        if 'generation_outputs' in data:
            y, p = data['labels'], data['generation_outputs']

            metrics['exact_match'] = np.mean([(len(p_) >= args.value_size + 1) and torch.all(torch.tensor(y_)[-args.value_size - 1:] == torch.tensor(p_[-args.value_size - 1:])) \
                                              for p_, y_ in zip (p, y)])

            if args.show_valid_examples > 0:
                for i in range(min(args.show_valid_examples, len(y))):
                    logger.info(f"labels: {data['labels'][i]}")
                    logger.info(f"gen: {data['generation_outputs'][i]}")
                    logger.info(f'y: {y[i][-args.value_size - 1:]}')
                    logger.info(f'p: {p[i][-args.value_size - 1:]}')
                    logger.info(f'full gen p: {p[i]}')

                    logger.info('-' * 50)
        
        elif 'predictions' in data:
            y, p = [torch.Tensor(i) for i in data['labels']], [torch.Tensor(i) for i in data['predictions']]
            # y, p = data['labels'], data['predictions']
            masks = data['labels_mask']

            for i in range(len(y)):
                # logger.info(y[i])
                # if not isinstance(y, torch.Tensor):
                #     y[i] = torch.tensor(y[i])
                y[i] = y[i][masks[i]]
                p[i] = p[i][:-1][masks[i][1:]]

            metrics['exact_match'] = np.mean([(len(p_) >= args.value_size + 1) and torch.all(torch.tensor(y_)[-args.value_size - 1:] == torch.tensor(p_[-args.value_size - 1:])) \
                                              for p_, y_ in zip(p, y)])
            if args.show_valid_examples > 0:
                for i in range(min(args.show_valid_examples, len(y))):
                    logger.info(f'y: {y[i]}')
                    logger.info(f'p: {p[i]}')

                    logger.info('-' * 50)

        return metrics

    # accelerate
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, None)

    ### booydar
    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    trainer = Trainer(args, accelerator, model, optimizer, train_dataloader, valid_dataloader,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      ###booydar
                      batch_metrics_fn=batch_metrics_fn,
                      generate_kwargs={
                          'max_new_tokens': int(args.value_size * 2),
                          'pad_token_id': 102
                      },
                      )

    # try:
    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        accelerator.wait_for_everyone()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best')
            logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:
            logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False, split='valid')
        # if test_dataloader is not None:
        #     logger.info('Runnning validation on test data:')
            # trainer.validate(test_dataloader, write_tb=True, split='test')
        trainer.save_metrics(save_path=args.model_path)
    else:
        # run validation, do not write to tensorboard
        # logger.info('Running validation on train set:')
        # trainer.validate(train_dataloader, split='train', write_tb=True)
        if valid_dataloader is not None:
            logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=True, split='valid')
        else:
            raise "No valid dataset"
        # if test_dataloader is not None:
        #     logger.info('Runnning validation on test data:')
        #     trainer.validate(test_dataloader, write_tb=True, split='test')
    # except Exception as e:
    #     print(f"Got exception: {e}")
    print('Done!')