from __future__ import print_function
from utils.rpn.bbox_transform import bbox_transform_inv

def regress_rois(roi_proposals, roi_regression_factors, labels):
    for i in range(len(labels)):
        label = labels[i]
        if label > 0:
            deltas = roi_regression_factors[i:i+1,label*4:(label+1)*4]
            roi_coords = roi_proposals[i:i+1,:]
            regressed_rois = bbox_transform_inv(roi_coords, deltas)
            roi_proposals[i,:] = regressed_rois

    return roi_proposals

