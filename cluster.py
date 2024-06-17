import argparse
import mmcv
import numpy as np
import os
import os.path as osp
import pickle
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from escape.datasets import build_dataset
from escape.models.utils.utils import KNN_cluster
from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.core import wrap_fp16_model
from mmpose.datasets import build_dataloader
from mmpose.models import build_posenet


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--meta', help='meta file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--threshold', type=float, default=0.8, help='clustering threshold')
    parser.add_argument(
        '--extract-features',
        action='store_true',
        default=False,
        help='whether to extract keypoints features')
    parser.add_argument(
        '--checkpoint-with-superkeypoints',
        help='override superkeypoints in checkpoint state_dict')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def cluster_keypoints(outputs, keypoints, threshold, meta):
    if outputs is not None:
        keypoints = [[] for k in range(meta['keypoint_num'])]
        for output in outputs:
            image_paths = output['image_paths']
            keypoint_index = output['keypoint_index']
            keypoint_features = output['keypoint_features']
            target_weight = output['target_weight']

            batch_size = len(image_paths)
            for i in range(batch_size):
                for j, k in enumerate(keypoint_index[i]):
                    if target_weight[i, j].item() > 0:
                        keypoints[k].append(keypoint_features[i, j])

        keypoints = [np.stack(_).mean(0) for _ in keypoints]
        keypoints = np.stack(keypoints)

    if 'superkeypoint_num' in meta:
        superkeypoint_num = meta['superkeypoint_num']
        _keypoint_to_superkeypoint = meta['_keypoint_to_superkeypoint']
        superkeypoints = [[] for _ in range(superkeypoint_num)]
        for i, j in enumerate(_keypoint_to_superkeypoint):
            superkeypoints[j].append(keypoints[i])

        superkeypoints = [np.stack(_, 0).mean(0) for _ in superkeypoints]
        superkeypoints = torch.from_numpy(np.stack(superkeypoints, 0))
    else:
        superkeypoints, _keypoint_to_superkeypoint, superkeypoint_num = KNN_cluster(
            keypoints, meta, threshold=threshold)

    return superkeypoints, _keypoint_to_superkeypoint, superkeypoint_num


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.model.test_cfg.fewshot_testing = False

    args.work_dir = osp.join('./work_dirs',
                             osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.extract_features, dict(test_mode=True))
    dataloader_setting = dict(
        samples_per_gpu=cfg.data.get('samples_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False,
        drop_last=False)

    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    checkpoint = torch.load(args.checkpoint)

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if args.extract_features:
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)
    else:
        outputs = None

    rank, _ = get_dist_info()

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        if args.meta is not None:
            meta = torch.load(args.meta)
            keypoints = None
        else:
            meta = {}
            meta['keypoint_num'] = dataset.keypoint_num
            meta['keypoint_index'] = dataset.keypoint_index
            meta['_ind_to_class'] = dataset._ind_to_class
            keypoints = checkpoint['state_dict'][
                'keypoint_head.keypoints'].numpy()

        superkeypoints, _keypoint_to_superkeypoint, superkeypoint_num = cluster_keypoints(
            outputs, keypoints, args.threshold, meta)

        meta['_keypoint_to_superkeypoint'] = _keypoint_to_superkeypoint
        meta['superkeypoint_num'] = superkeypoint_num
        meta['superkeypoints'] = superkeypoints

        meta_file_dir = os.path.join(args.work_dir, 'superkeypoints.pth')
        torch.save(meta, meta_file_dir)

        if args.checkpoint_with_superkeypoints is not None:
            checkpoint['state_dict'][
                'keypoint_head.superkeypoints'] = torch.from_numpy(
                    superkeypoints)
            torch.save(checkpoint, args.checkpoint_with_superkeypoints)


if __name__ == '__main__':
    main()
