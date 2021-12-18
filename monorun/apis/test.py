import os.path as osp
import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import encode_mask_results
import numpy as np

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    cov_scale=5):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    data['cam_intrinsic'][0].data[0][0].cpu().numpy(),
                    result,
                    score_thr=show_score_thr,
                    cov_scale=cov_scale,
                    show=show,
                    out_file=out_file)

        # encode mask results
        if isinstance(result, tuple):
            bbox_results, mask_results = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        results.append(result[0])   
        ### result is a list with a single element which is a dict of 'bbox_results' and 'bbox_3d_results'
        ### each item is a list of 3 arrays corresponding to 3 categories 'Car', 'Pedestrian', 'Cyclist', each array is a n*5 or n*8
        ### for "bbox_results": x_min, y_min, x_max, y_max, conf
        ### for "bbox_3d_results": l,h,w,x,y,z,yaw,conf

        # print("result:", result[0])
        # if i > 5:
        #     break

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def default_d():
    d = dict()
    d['bbox_results'] = [np.empty((0, 5), dtype=np.float32)]*3
    d['bbox_3d_results'] = [np.empty((0, 8), dtype=np.float32)]*3
    return d

def single_gpu_eval(results_d,
                    model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    cov_scale=5, 
                    ):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        img_n = dataset.img_idxs[i]
        result = results_d[img_n] if img_n in results_d else default_d()
        result = [result]
        # with torch.no_grad():
        #     result = model(return_loss=False, rescale=True, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    data['cam_intrinsic'][0].data[0][0].cpu().numpy(),
                    result,
                    score_thr=show_score_thr,
                    cov_scale=cov_scale,
                    show=show,
                    out_file=out_file)

        # encode mask results
        if isinstance(result, tuple):
            bbox_results, mask_results = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        results.append(result[0])   
        ### result is a list with a single element which is a dict of 'bbox_results' and 'bbox_3d_results'
        ### each item is a list of 3 arrays corresponding to 3 categories 'Car', 'Pedestrian', 'Cyclist', each array is a n*5 or n*8
        ### for "bbox_results": x_min, y_min, x_max, y_max, conf
        ### for "bbox_3d_results": l,h,w,x,y,z,yaw,conf

        # print("result:", result[0])
        # if i > 5:
        #     break

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results
