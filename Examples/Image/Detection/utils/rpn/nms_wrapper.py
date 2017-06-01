# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
from utils.cython_modules.cpu_nms import cpu_nms
try:
    from utils.cython_modules.gpu_nms import gpu_nms
    gpu_nms_available = True
except ImportError:
    gpu_nms_available = False

try:
    from config import cfg
except ImportError:
    from utils.default_config import cfg

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if gpu_nms_available and cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)

def applyNonMaximaSuppression(coords, labels, scores, nms_threshold=0.5, conf_threshold=0.0):
    # generate input for nms
    allIndices = []
    nmsRects = [[[]] for _ in range(max(labels) + 1)]
    coordsWithScores = np.hstack((coords, np.array([scores]).T))
    for i in range(max(labels) + 1):
        indices = np.where(np.array(labels) == i)[0]
        nmsRects[i][0] = coordsWithScores[indices,:]
        allIndices.append(indices)

    # call nms
    _, nmsKeepIndicesList = apply_nms(nmsRects, nms_threshold, conf_threshold)

    # map back to original roi indices
    nmsKeepIndices = []
    for i in range(max(labels) + 1):
        for keepIndex in nmsKeepIndicesList[i][0]:
            nmsKeepIndices.append(allIndices[i][keepIndex]) # for keepIndex in nmsKeepIndicesList[i][0]]
    assert (len(nmsKeepIndices) == len(set(nmsKeepIndices)))  # check if no roi indices was added >1 times
    return nmsKeepIndices

def apply_nms(all_boxes, nms_threshold, conf_threshold):
    """Apply non-maximum suppression to all predicted boxes output by the test_net method."""
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    nms_keepIndices = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets.astype(np.float32), nms_threshold)

            # also filter out low confidences
            if conf_threshold > 0:
                keep_conf_idx = np.where(dets[:, -1] > conf_threshold)
                keep = list(set(keep_conf_idx[0]).intersection(keep))

            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
            nms_keepIndices[cls_ind][im_ind] = keep
    return nms_boxes, nms_keepIndices

