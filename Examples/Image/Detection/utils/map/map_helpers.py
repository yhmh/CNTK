# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np

#from builtins import str

# def readFile(inputFile):
#     #reading as binary, to avoid problems with end-of-text characters
#     #note that readlines() does not remove the line ending characters
#     with open(inputFile,'rb') as f:
#         lines = f.readlines()
#     return [removeLineEndCharacters(s) for s in lines]
#
# def readTable(inputFile, delimiter='\t', columnsToKeep=None):
#     lines = readFile(inputFile);
#     if columnsToKeep != None:
#         header = lines[0].split(delimiter)
#         columnsToKeepIndices = listFindItems(header, columnsToKeep)
#     else:
#         columnsToKeepIndices = None;
#     return splitStrings(lines, delimiter, columnsToKeepIndices)
#
# def splitString(string, delimiter='\t', columnsToKeepIndices=None):
#     if string == None:
#         return None
#     items = string.decode('utf-8').split(delimiter)
#     if columnsToKeepIndices != None:
#         items = getColumns([items], columnsToKeepIndices)
#         items = items[0]
#     return items;
#
# def splitStrings(strings, delimiter, columnsToKeepIndices=None):
#     table = [splitString(string, delimiter, columnsToKeepIndices) for string in strings]
#     return table;
#
# def readGtAnnotation(imgPath):
#     bboxesPath = imgPath[:-4] + ".bboxes.tsv"
#     labelsPath = imgPath[:-4] + ".bboxes.labels.tsv"
#     bboxes = np.array(readTable(bboxesPath), np.int32)
#     labels = readFile(labelsPath)
#     assert (len(bboxes) == len(labels))
#     return bboxes, labels

# main call to compute per-calass average precision
#   shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
#  (see also test_net() in fastRCNN\test.py)
def evaluate_detections(all_boxes, all_gt_infos, classes, use_07_metric=False):
    aps = []
    for classIndex, className in enumerate(classes):
        if className != '__background__':
            rec, prec, ap = _evaluate_detections(classIndex, all_boxes, all_gt_infos[className], use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {:>15} = {:.4f}'.format(className, ap))
    print('Mean AP = {:.4f}'.format(np.nanmean(aps)))


def _evaluate_detections(classIndex, all_boxes, gtInfos, overlapThreshold=0.5, use_07_metric=False):
    """
    Top level function that does the PASCAL VOC evaluation.

    [overlapThreshold]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation (default False)
    """

    # parse detections for this class
    # shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
    num_images = len(all_boxes[0])
    detBboxes = []
    detImgIndices = []
    detConfidences = []
    for imgIndex in range(num_images):
        dets = all_boxes[classIndex][imgIndex]
        if dets != []:
            for k in range(dets.shape[0]):
                detImgIndices.append(imgIndex)
                detConfidences.append(dets[k, -1])
                # the VOCdevkit expects 1-based indices
                detBboxes.append([dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1])
    detBboxes = np.array(detBboxes)
    detConfidences = np.array(detConfidences)

    # compute precision / recall / ap
    rec, prec, ap = _voc_computePrecisionRecallAp(
        class_recs=gtInfos,
        confidence=detConfidences,
        image_ids=detImgIndices,
        BB=detBboxes,
        ovthresh=overlapThreshold,
        use_07_metric=use_07_metric)
    return rec, prec, ap


#########################################################################
# Python evaluation functions (copied/refactored from faster-RCNN)
##########################################################################

def computeAveragePrecision(recalls, precisions, use_07_metric=False):
    """ ap = voc_ap(recalls, precisions, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrecalls = np.concatenate(([0.], recalls, [1.]))
        mprecisions = np.concatenate(([0.], precisions, [0.]))

        # compute the precision envelope
        for i in range(mprecisions.size - 1, 0, -1):
            mprecisions[i - 1] = np.maximum(mprecisions[i - 1], mprecisions[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrecalls[1:] != mrecalls[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrecalls[i + 1] - mrecalls[i]) * mprecisions[i + 1])
    return ap

def _voc_computePrecisionRecallAp(class_recs, confidence, image_ids, BB, ovthresh=0.5, use_07_metric=False):
    # sort by confidence
    sorted_ind = np.argsort(-confidence)

    if len(BB) == 0:
        #import pdb; pdb.set_trace()
        return 0.0, 0.0, 0.0

    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    npos = sum([len(cr['bbox']) for cr in class_recs])
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = computeAveragePrecision(rec, prec, use_07_metric)
    return rec, prec, ap
