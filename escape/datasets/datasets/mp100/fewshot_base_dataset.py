import copy
import json_tricks as json
import mmcv
import numpy as np
import torch
from abc import ABCMeta, abstractmethod
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_pck_accuracy)
from mmpose.datasets import DATASETS
from mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class FewshotBaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        self.image_info = {}
        self.ann_info = {}

        self.annotations_path = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['flip_pairs'] = None

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.db = []
        self.num_shots = 1
        self.paired_samples = []
        self.episodes = []
        self.pipeline = Compose(self.pipeline)

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    @abstractmethod
    def _select_kpt(self, obj, kpt_id):
        """Select kpt."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """Evaluate keypoint results."""
        raise NotImplementedError

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self,
                       res_file,
                       metrics,
                       pck_thr=0.2,
                       pckh_thr=0.7,
                       auc_nor=30):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.paired_samples)
        outputs = []
        gts = []
        masks = []
        threshold_bbox = []
        threshold_head_box = []

        for pred, pair in zip(preds, self.paired_samples):
            item = self.db[pair[-1]]
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])

            mask_query = ((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            mask_sample = ((np.array(self.db[pair[0]]['joints_3d_visible'])[:,
                                                                            0])
                           > 0)
            for id_s in pair[:-1]:
                mask_sample = np.bitwise_and(
                    mask_sample,
                    ((np.array(self.db[id_s]['joints_3d_visible'])[:, 0]) > 0))
            masks.append(np.bitwise_and(mask_query, mask_sample))

            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            if 'PCKh' in metrics:
                head_box_thr = item['head_size']
                threshold_head_box.append(
                    np.array([head_box_thr, head_box_thr]))

        if 'PCK' in metrics:
            pck_avg = []
            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks,
                                                    threshold_bbox):
                _, pck, _ = keypoint_pck_accuracy(
                    np.expand_dims(output, 0), np.expand_dims(gt, 0),
                    np.expand_dims(mask, 0), pck_thr,
                    np.expand_dims(thr_bbox, 0))
                pck_avg.append(pck)
            info_str.append(('PCK', np.mean(pck_avg)))

        return info_str

    def _merge_obj(self, Xs_list, Xq_list, idx):
        """ merge Xs_list and Xq.

        :param Xs_list: N-shot samples X
        :param Xq_list: list of queries
        :param idx: id of episode
        :return: Xall
        """
        Xall = dict()
        Xall['img'] = torch.stack([Xs['img'] for Xs in Xs_list], 0)
        Xall['target'] = np.stack([Xs['target'] for Xs in Xs_list], 0)
        Xall['target_weight'] = np.stack(
            [Xs['target_weight'] for Xs in Xs_list], 0)

        Xall['img_q'] = torch.stack([Xq['img'] for Xq in Xq_list], 0)
        xq_img_metas = [Xq['img_metas'].data for Xq in Xq_list]

        Xall['img_metas'] = []
        for query_id, xq_img_meta in enumerate(xq_img_metas):
            img_metas = {
                'query_' + key: value
                for key, value in xq_img_meta.items()
            }
            img_metas['bbox_id'] = idx * self.num_queries + query_id
            img_metas = DC(img_metas, cpu_only=True)
            Xall['img_metas'].append(img_metas)

        return Xall

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.episodes)

    def __getitem__(self, idx):
        """Get the sample given index."""

        pair_ids = self.episodes[idx]
        assert len(pair_ids) == self.num_shots + self.num_queries
        sample_id_list = pair_ids[:self.num_shots]
        query_id_list = pair_ids[self.num_shots:]

        sample_obj_list = []
        for sample_id in sample_id_list:
            sample_obj = copy.deepcopy(self.db[sample_id])
            sample_obj['ann_info'] = copy.deepcopy(self.ann_info)
            sample_obj_list.append(sample_obj)

        Xs_list = []
        for sample_obj in sample_obj_list:
            Xs = self.pipeline(sample_obj)
            Xs_list.append(Xs)

        query_obj_list = []
        for query_id in query_id_list:
            query_obj = copy.deepcopy(self.db[query_id])
            query_obj['ann_info'] = copy.deepcopy(self.ann_info)
            query_obj_list.append(query_obj)

        Xq_list = []
        for query_obj in query_obj_list:
            Xq = self.pipeline(query_obj)
            Xq_list.append(Xq)

        Xall = self._merge_obj(Xs_list, Xq_list, idx)
        return Xall

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
