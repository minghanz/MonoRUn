import argparse
import os
import warnings

import mmcv
from numpy.lib.shape_base import split
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from monorun.apis import single_gpu_test, single_gpu_eval
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input', help='input result file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--val-set',
        action='store_true',
        help='whether to test validation set instead of test set')
    parser.add_argument(
        '--show-extra', action='store_true',
        help='whether to draw extra results (covariance and reconstruction)')
    parser.add_argument(
        '--show-cov-scale', type=float, default=5.0,
        help='covariance scaling factor for visualization')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if args.val_set:
        cfg.data.test = cfg.data.val
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        raise NotImplementedError
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        raise NotImplementedError
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if (args.show or args.show_dir) and args.show_extra:
        model.test_cfg['rcnn']['debug'] = True
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    outputs = read_input_result(args.input)

    if args.show_dir:
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_eval(outputs, model, data_loader, args.show, args.show_dir,
                                    args.show_score_thr, args.show_cov_scale)
        else:
            raise NotImplementedError
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                    args.gpu_collect)

    # froot = "/home/minghan.zhu/repos/local3d/preds"
    else:
        outputs = sort_input_result(outputs, dataset)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


def read_input_result(froot):
    d_all = dict()
    fs = os.listdir(froot)
    for fpath in fs:
        img_name = fpath.split('.')[0]
        assert img_name.isdigit(), img_name
        img_n = int(img_name)

        d = default_d()

        txt_path = os.path.join(froot, fpath)
        with open(txt_path) as f:
            lines = f.read().splitlines()
        if len(lines) > 0:
            bboxes_2d = [[], [], []]
            bboxes_3d = [[], [], []]
            for line in lines:
                assert not any(math.isnan(float(x)) for x in line.split()), txt_path
                list_id = int(line.split()[0])
                props = [int(float(x)) if int(float(x)) == float(x) else float(x) for x in line.split()]
                bbox2d = np.array([*(props[1:5]), props[-1]], dtype=np.float32)
                bbox3d = np.array(props[5:], dtype=np.float32)
                bboxes_2d[list_id].append(bbox2d)
                bboxes_3d[list_id].append(bbox3d)
            for list_id in range(3):
                if len(bboxes_2d[list_id]) > 0:
                    bboxes_2d[list_id] = np.stack(bboxes_2d[list_id], axis=0)
                    bboxes_3d[list_id] = np.stack(bboxes_3d[list_id], axis=0)
            
                    d['bbox_results'][list_id] = bboxes_2d[list_id]
                    d['bbox_3d_results'][list_id] = bboxes_3d[list_id]
        d_all[img_n] = d
    return d_all

def default_d():
    d = dict()
    d['bbox_results'] = [np.empty((0, 5), dtype=np.float32)]*3
    d['bbox_3d_results'] = [np.empty((0, 8), dtype=np.float32)]*3
    return d

def sort_input_result(d_all, dataset):
    list_dets = []
    for i in range(len(dataset)):
        img_n = dataset.img_idxs[i]
        d = d_all[img_n] if img_n in d_all else default_d()
        list_dets.append(d)
    return list_dets

if __name__ == '__main__':
    main()
