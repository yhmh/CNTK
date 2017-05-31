# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os, sys
import argparse
import cntk
from cntk import Trainer, UnitType, load_model, user_function, Axis, input_variable, parameter, times, combine, relu, \
    softmax, roipooling, reduce_sum, slice, splice, reshape, plus, CloneMethod, minus, element_times, alias, Communicator
from cntk.io import MinibatchSource, ImageDeserializer, CTFDeserializer, StreamDefs, StreamDef, TraceLevel
from cntk.io.transforms import scale
from cntk.initializer import glorot_uniform, normal
from cntk.layers import placeholder, Constant, Sequential
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.graph import find_by_name, plot
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error

try:
    import yaml
except ImportError:
    import pip
    pip.main(['install', '--user', 'pyyaml'])
    import yaml

try:
    import easydict
except ImportError:
    import pip
    pip.main(['install', '--user', 'easydict'])
    import yaml

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from utils.rpn.rpn_helpers import create_rpn, create_proposal_target_layer
from utils.rpn.cntk_smoothL1_loss import SmoothL1Loss
from config import cfg

###############################################################
###############################################################
mb_size = 1
image_width = cfg["CNTK"].IMAGE_WIDTH
image_height = cfg["CNTK"].IMAGE_HEIGHT
num_channels = 3
im_info = cntk.constant([image_width, image_height, 1], (3,))

globalvars = {}
globalvars['output_path'] = os.path.join(abs_path, "Output")

# dataset specific parameters
classes = cfg["CNTK"].CLASSES
num_classes = len(classes)
map_file_path = os.path.join(abs_path, cfg["CNTK"].MAP_FILE_PATH)
globalvars['train_map_file'] = cfg["CNTK"].TRAIN_MAP_FILE
globalvars['test_map_file'] = cfg["CNTK"].TEST_MAP_FILE
globalvars['train_roi_file'] = cfg["CNTK"].TRAIN_ROI_FILE
globalvars['test_roi_file'] = cfg["CNTK"].TEST_ROI_FILE
epoch_size = cfg["CNTK"].NUM_TRAIN_IMAGES
num_test_images = cfg["CNTK"].NUM_TEST_IMAGES

# model specific parameters
model_folder = os.path.join(abs_path, "..", "..", "PretrainedModels")
base_model_file = os.path.join(model_folder, cfg["CNTK"].BASE_MODEL_FILE)
feature_node_name = cfg["CNTK"].FEATURE_NODE_NAME
last_conv_node_name = cfg["CNTK"].LAST_CONV_NODE_NAME
start_train_conv_node_name = cfg["CNTK"].START_TRAIN_CONV_NODE_NAME
pool_node_name = cfg["CNTK"].POOL_NODE_NAME
last_hidden_node_name = cfg["CNTK"].LAST_HIDDEN_NODE_NAME
roi_dim = cfg["CNTK"].ROI_DIM
###############################################################
###############################################################

def parse_arguments():
    parser = argparse.ArgumentParser()

    data_path = map_file_path
    parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located',
                        required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models',
                        required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file',
                        required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int,
                        required=False, default=cfg["CNTK"].MAX_EPOCHS_E2E)
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int,
                        required=False, default=mb_size)
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int,
                        required=False, default=epoch_size)
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int,
                        required=False, default='32')
    parser.add_argument('-r', '--restart',
                        help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)',
                        action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device",
                        required=False, default=None)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        globalvars['output_path'] = args['outputdir']
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['device'] is not None:
        # Setting one worker on GPU and one worker on CPU. Otherwise memory consumption is too high for a single GPU.
        if Communicator.rank() == 0:
            cntk.device.try_set_default_device(cntk.device.gpu(args['device']))
        else:
            cntk.device.try_set_default_device(cntk.device.cpu())

    if args['datadir'] is not None:
        data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    globalvars['train_map_file'] = os.path.join(data_path, globalvars['train_map_file'])
    globalvars['test_map_file'] = os.path.join(data_path, globalvars['test_map_file'])
    globalvars['train_roi_file'] = os.path.join(data_path, globalvars['train_roi_file'])
    globalvars['test_roi_file'] = os.path.join(data_path, globalvars['test_roi_file'])

###############################################################
###############################################################

# Instantiates a composite minibatch source for reading images, roi coordinates and roi labels for training Faster R-CNN
def create_mb_source(img_map_file, roi_map_file, img_height, img_width, img_channels, n_rois, randomize=True):
    if not os.path.exists(img_map_file) or not os.path.exists(roi_map_file):
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_fastrcnn.py from "
                           "Examples/Image/Detection/FastRCNN to fetch them" % (img_map_file, roi_map_file))

    # read images, rois and labels
    transforms = [scale(width=img_width, height=img_height, channels=img_channels, scale_mode="pad", pad_value=114, interpolations='linear')]
    image_source = ImageDeserializer(img_map_file, StreamDefs(features = StreamDef(field='image', transforms=transforms)))
    rois_dim = 5 * n_rois
    roi_source = CTFDeserializer(roi_map_file, StreamDefs(roiAndLabel = StreamDef(field=cfg["CNTK"].ROI_STREAM_NAME, shape=rois_dim, is_sparse=False)))

    return MinibatchSource([image_source, roi_source], randomize=randomize, trace_level=TraceLevel.Error)

def clone_model(base_model, from_node_names, to_node_names, clone_method):
    from_nodes = [find_by_name(base_model, node_name) for node_name in from_node_names]
    if None in from_nodes:
        print("Error: could not find all specified 'from_nodes' in clone. Looking for {}, found {}"
              .format(from_node_names, from_nodes))
    to_nodes = [find_by_name(base_model, node_name) for node_name in to_node_names]
    if None in to_nodes:
        print("Error: could not find all specified 'to_nodes' in clone. Looking for {}, found {}"
              .format(to_node_names, to_nodes))
    input_placeholders = dict(zip(from_nodes, [placeholder() for x in from_nodes]))
    cloned_net = combine(to_nodes).clone(clone_method, input_placeholders)
    return cloned_net

def create_fast_rcnn_predictor(conv_out, rois, fc_layers):
    # for the roipooling layer we convert and scale roi coords back to x, y, w, h relative from x1, y1, x2, y2 absolute
    roi_xy1 = slice(rois, 1, 0, 2)
    roi_xy2 = slice(rois, 1, 2, 4)
    roi_wh = minus(roi_xy2, roi_xy1)
    roi_xywh = splice(roi_xy1, roi_wh, axis=1)
    scaled_rois = element_times(roi_xywh, (1.0 / image_width))

    # RCNN
    roi_out = roipooling(conv_out, scaled_rois, (roi_dim, roi_dim))
    fc_out = fc_layers(roi_out)

    # prediction head
    W_pred = parameter(shape=(4096, num_classes), init=normal(scale=0.01))
    b_pred = parameter(shape=num_classes, init=0)
    cls_score = plus(times(fc_out, W_pred), b_pred, name='cls_score')

    # regression head
    W_regr = parameter(shape=(4096, num_classes*4), init=normal(scale=0.01))
    b_regr = parameter(shape=num_classes*4, init=0)
    bbox_pred = plus(times(fc_out, W_regr), b_regr, name='bbox_regr')

    return cls_score, bbox_pred

# Defines the Faster R-CNN network model for detecting objects in images
def create_faster_rcnn_predictor(features, scaled_gt_boxes):
    # Load the pre-trained classification net and clone layers
    base_model = load_model(base_model_file)
    conv_layers = clone_model(base_model, [feature_node_name], [last_conv_node_name], clone_method=CloneMethod.freeze)
    fc_layers = clone_model(base_model, [pool_node_name], [last_hidden_node_name], clone_method=CloneMethod.clone)

    # Normalization and conv layers
    feat_norm = features - Constant(114)
    conv_out = conv_layers(feat_norm)

    # RPN
    rpn_rois, rpn_losses = create_rpn(conv_out, scaled_gt_boxes, im_info,
                                      proposal_layer_param_string=cfg["CNTK"].PROPOSAL_LAYER_PARAMS)
    rois, label_targets, bbox_targets, bbox_inside_weights = \
        create_proposal_target_layer(rpn_rois, scaled_gt_boxes, num_classes=num_classes)

    # Fast RCNN
    cls_score, bbox_pred = create_fast_rcnn_predictor(conv_out, rois, fc_layers)

    # loss functions
    loss_cls = cross_entropy_with_softmax(cls_score, label_targets, axis=1)
    loss_box = user_function(SmoothL1Loss(bbox_pred, bbox_targets, bbox_inside_weights))
    detection_losses = reduce_sum(loss_cls) + reduce_sum(loss_box)

    loss = rpn_losses + detection_losses
    pred_error = classification_error(cls_score, label_targets, axis=1)

    return cls_score, loss, pred_error

def create_eval_model(model, image_input):
    # modify Faster RCNN model by excluding target layers and losses
    feature_node = find_by_name(model, feature_node_name)
    conv_node = find_by_name(model, last_conv_node_name)
    rpn_roi_node = find_by_name(model, "rpn_rois")
    rpn_target_roi_node = find_by_name(model, "rpn_target_rois")
    cls_score_node = find_by_name(model, "cls_score")
    bbox_pred_node = find_by_name(model, "bbox_regr")

    conv_rpn_layers = combine([conv_node.owner, rpn_roi_node.owner])\
        .clone(CloneMethod.freeze, {feature_node: placeholder()})
    roi_fc_layers = combine([cls_score_node.owner, bbox_pred_node.owner])\
        .clone(CloneMethod.clone, {conv_node: placeholder(), rpn_target_roi_node: placeholder()})

    conv_rpn_net = conv_rpn_layers(image_input)
    conv_out = conv_rpn_net.outputs[0]
    rpn_rois = conv_rpn_net.outputs[1]

    pred_net = roi_fc_layers(conv_out, rpn_rois)
    cls_score = pred_net.outputs[0]
    bbox_regr = pred_net.outputs[1]

    cls_pred = softmax(cls_score, axis=1, name='cls_pred')
    return combine([cls_pred, rpn_rois, bbox_regr])

def convert_gt_boxes(gt_boxes, im_width, name="converted_gt_boxes"):
    # For CNTK: convert and scale gt_box coords from x, y, w, h relative to x1, y1, x2, y2 absolute
    roi_xy1 = slice(gt_boxes, 1, 0, 2)
    roi_wh = slice(gt_boxes, 1, 2, 4)
    roi_label = slice(gt_boxes, 1, 4, 5)
    roi_xy2 = plus(roi_xy1, roi_wh)
    roi_xyxy = splice(roi_xy1, roi_xy2, axis=1)
    scaled_roi_xyxy = element_times(roi_xyxy, (1.0 * im_width))
    scaled_gt_boxes = splice(scaled_roi_xyxy, roi_label, axis=1)

    return alias(scaled_gt_boxes, name=name)

def train_model(image_input, roi_input, loss, pred_error,
                lr_schedule, mm_schedule, l2_reg_weight, epochs_to_train):
    if isinstance(loss, cntk.Variable):
        loss = combine([loss])
    # Instantiate the trainer object
    learner = momentum_sgd(loss.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight,
                           unit_gain=False, use_mean_gradient=cfg["CNTK"].USE_MEAN_GRADIENT)
    trainer = Trainer(None, (loss, pred_error), learner)

    # Create the minibatch source
    minibatch_source = create_mb_source(globalvars['train_map_file'], globalvars['train_roi_file'],
        image_height, image_width, num_channels, cfg["CNTK"].INPUT_ROIS_PER_IMAGE)

    # define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source[cfg["CNTK"].FEATURE_STREAM_NAME],
        roi_input: minibatch_source[cfg["CNTK"].ROI_STREAM_NAME]
    }

    # Get minibatches of images and perform model training
    print("Training model for %s epochs." % epochs_to_train)
    log_number_of_parameters(loss)
    progress_printer = ProgressPrinter(tag='Training', num_epochs=epochs_to_train, gen_heartbeat=True)
    for epoch in range(epochs_to_train):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = minibatch_source.next_minibatch(min(mb_size, epoch_size-sample_count), input_map=input_map)
            trainer.train_minibatch(data)                                    # update model with it
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress
            if sample_count % 100 == 0:
                print("Processed {} samples".format(sample_count))

        progress_printer.epoch_summary(with_metric=True)

# Trains a Faster R-CNN model end-to-end
def train_faster_rcnn_e2e(debug_output=False):
    # Input variables denoting features and labeled ground truth rois (as 5-tuples per roi)
    image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    roi_input = input_variable((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])

    # For CNTK: convert and scale gt_box coords from x, y, w, h relative to x1, y1, x2, y2 absolute
    scaled_gt_boxes = convert_gt_boxes(roi_input, image_width, name='roi_input')

    # Instantiate the Faster R-CNN prediction model and loss function
    predictions, loss, pred_error = create_faster_rcnn_predictor(image_input, scaled_gt_boxes)

    if debug_output:
        print("Storing graphs and models to %s." % globalvars['output_path'])
        plot(loss, os.path.join(globalvars['output_path'], "graph_frcn_train_e2e." + cfg["CNTK"].GRAPH_TYPE))

    # Set learning parameters
    lr_schedule = learning_rate_schedule(cfg["CNTK"].E2E_LR_PER_SAMPLE, unit=UnitType.sample)
    mm_schedule = momentum_schedule(cfg["CNTK"].MOMENTUM_PER_MB)

    train_model(image_input, roi_input, loss, pred_error,
                lr_schedule, mm_schedule, cfg["CNTK"].L2_REG_WEIGHT, cfg["CNTK"].MAX_EPOCHS_E2E)
    return loss

# Trains a Faster R-CNN model using 4-stage alternating training
def train_faster_rcnn_alternating(debug_output=False):
    '''
        4-Step Alternating Training scheme from the Faster R-CNN paper:
        
        # Create initial network, only rpn, without detection network
            # --> train only the rpn (and conv3_1 and up for VGG16)
            # lr = [0.001] * 12 + [0.0001] * 4, momentum = 0.9, weight decay = 0.0005 (cf. stage1_rpn_solver60k80k.pt)
        
        # Create full network, initialize conv layers with imagenet, fix rpn weights
            # --> train only detection network (and conv3_1 and up for VGG16)
            # lr = [0.001] * 6 + [0.0001] * 2, momentum = 0.9, weight decay = 0.0005 (cf. stage1_fast_rcnn_solver30k40k.pt)
        
        # Keep conv weights from detection network and fix them
            # --> train only rpn
            # lr = [0.001] * 12 + [0.0001] * 4, momentum = 0.9, weight decay = 0.0005 (cf. stage2_rpn_solver60k80k.pt)
        
        # Keep conv and rpn weights from stpe 3 and fix them
            # --> train only detection netwrok
            # lr = [0.001] * 6 + [0.0001] * 2, momentum = 0.9, weight decay = 0.0005 (cf. stage2_fast_rcnn_solver30k40k.pt)
    '''

    # Learning parameters
    l2_reg_weight = cfg["CNTK"].L2_REG_WEIGHT
    mm_schedule = momentum_schedule(cfg["CNTK"].MOMENTUM_PER_MB)
    rpn_epochs = 1 if cfg["CNTK"].FAST_MODE else cfg["CNTK"].RPN_EPOCHS
    rpn_lr_schedule = learning_rate_schedule(cfg["CNTK"].RPN_LR_PER_SAMPLE, unit=UnitType.sample)
    frcn_epochs = 1 if cfg["CNTK"].FAST_MODE else cfg["CNTK"].FRCN_EPOCHS
    frcn_lr_schedule = learning_rate_schedule(cfg["CNTK"].FRCN_LR_PER_SAMPLE, unit=UnitType.sample)

    # Input variables denoting features and labeled ground truth rois (as 5-tuples per roi)
    image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()],
                                 name=feature_node_name)
    roi_input = input_variable((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])
    feat_norm = image_input - Constant(114)

    # For CNTK: convert and scale gt_box coords from x, y, w, h relative to x1, y1, x2, y2 absolute
    scaled_gt_boxes = convert_gt_boxes(roi_input, image_width, name='roi_input')

    # base image classification model (e.g. VGG16 or AlexNet)
    base_model = load_model(base_model_file)

    print("Using base model: {}".format(cfg["CNTK"].BASE_MODEL))
    print("rpn_lr_per_sample: {}".format(cfg["CNTK"].RPN_LR_PER_SAMPLE))
    print("rpn_epochs: {}".format(rpn_epochs))
    print("frcn_lr_per_sample: {}".format(cfg["CNTK"].FRCN_LR_PER_SAMPLE))
    print("frcn_epochs: {}".format(frcn_epochs))
    if debug_output:
        print("Storing graphs and models to %s." % globalvars['output_path'])

    print("stage 1a - rpn")
    if True:
        # Create initial network, only rpn, without detection network
            #       initial weights     train?
            # conv: base_model          only conv3_1 and up
            # rpn:  init new            yes
            # frcn: -                   -

        # conv layers
        if start_train_conv_node_name == None:
            conv_layers = clone_model(base_model, [feature_node_name], [last_conv_node_name], clone_method=CloneMethod.freeze)
            conv_out = conv_layers(feat_norm)
        else:
            fixed_conv_layers = clone_model(base_model, [feature_node_name], [start_train_conv_node_name], clone_method=CloneMethod.freeze)
            train_conv_layers = clone_model(base_model, [start_train_conv_node_name], [last_conv_node_name], clone_method=CloneMethod.clone)
            # TODO: it would be nicer to use Sequential(), but then the node name cannot be found in subsequent cloning currently
            # conv_layers = Sequential(fixed_conv_layers, train_conv_layers)
            conv_out_f = fixed_conv_layers(feat_norm)
            conv_out = train_conv_layers(conv_out_f)
        #conv_out = conv_layers(feat_norm)

        # RPN and losses
        rpn_rois, rpn_losses = create_rpn(conv_out, scaled_gt_boxes, im_info,
                                          proposal_layer_param_string=cfg["CNTK"].PROPOSAL_LAYER_PARAMS)
        stage1_rpn_network = combine([rpn_rois, rpn_losses])

        # train
        if debug_output: plot(stage1_rpn_network, os.path.join(globalvars['output_path'], "graph_frcn_train_stage1a_rpn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, rpn_losses, rpn_losses,
                    rpn_lr_schedule, mm_schedule, l2_reg_weight, epochs_to_train=rpn_epochs)

    print("stage 1b - frcn")
    if True:
        # Create full network, initialize conv layers with imagenet, fix rpn weights
            #       initial weights     train?
            # conv: base_model          only conv3_1 and up
            # rpn:  stage1 rpn model    no
            # frcn: base_model + new    yes

        # conv_layers
        if start_train_conv_node_name == None:
            conv_layers = clone_model(base_model, [feature_node_name], [last_conv_node_name], CloneMethod.freeze)
            conv_out = conv_layers(feat_norm)
        else:
            fixed_conv_layers = clone_model(base_model, [feature_node_name], [start_train_conv_node_name], CloneMethod.freeze)
            train_conv_layers = clone_model(base_model, [start_train_conv_node_name], [last_conv_node_name], CloneMethod.clone)
            # TODO: it would be nicer to use Sequential(), but then the node name cannot be found in subsequent cloning
            # conv_layers = Sequential(fixed_conv_layers, train_conv_layers)
            conv_out_f = fixed_conv_layers(feat_norm)
            conv_out = train_conv_layers(conv_out_f)
        # conv_out = conv_layers(feat_norm)

        # RPN
        rpn = clone_model(stage1_rpn_network, [last_conv_node_name, "roi_input"], ["rpn_rois", "rpn_losses"], CloneMethod.freeze)
        rpn_net = rpn(conv_out, scaled_gt_boxes)
        rpn_rois = rpn_net.outputs[0]
        rpn_losses = rpn_net.outputs[1] # required for training rpn in stage 2

        rois, label_targets, bbox_targets, bbox_inside_weights = \
            create_proposal_target_layer(rpn_rois, scaled_gt_boxes, num_classes=num_classes)

        # Fast RCNN
        fc_layers = clone_model(base_model, [pool_node_name], [last_hidden_node_name], CloneMethod.clone)
        cls_score, bbox_pred = create_fast_rcnn_predictor(conv_out, rois, fc_layers)

        # loss functions
        loss_cls = cross_entropy_with_softmax(cls_score, label_targets, axis=1)
        loss_box = user_function(SmoothL1Loss(bbox_pred, bbox_targets, bbox_inside_weights))
        detection_losses = plus(reduce_sum(loss_cls), reduce_sum(loss_box), name="detection_losses")
        stage1_frcn_network = combine([rois, cls_score, bbox_pred, rpn_losses, detection_losses])

        # train
        if debug_output: plot(stage1_frcn_network, os.path.join(globalvars['output_path'], "graph_frcn_train_stage1b_frcn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, detection_losses, detection_losses,
                    frcn_lr_schedule, mm_schedule, l2_reg_weight, epochs_to_train=frcn_epochs)

    print("stage 2a - rpn")
    if True:
        # Keep conv weights from detection network and fix them
            #       initial weights     train?
            # conv: stage1 frcn model   no
            # rpn:  stage1 rpn model    yes
            # frcn: -                   -

        # conv_layers
        conv_layers = clone_model(stage1_frcn_network, [feature_node_name], [last_conv_node_name], CloneMethod.freeze)
        conv_out = conv_layers(image_input)

        # RPN and losses
        rpn = clone_model(stage1_rpn_network, [last_conv_node_name, "roi_input"], ["rpn_rois", "rpn_losses"], CloneMethod.clone)
        rpn_net = rpn(conv_out, scaled_gt_boxes)
        rpn_rois = rpn_net.outputs[0]
        rpn_losses = rpn_net.outputs[1]
        stage2_rpn_network = combine([rpn_rois, rpn_losses])

        # train
        if debug_output: plot(stage2_rpn_network, os.path.join(globalvars['output_path'], "graph_frcn_train_stage2a_rpn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, rpn_losses, rpn_losses,
                    rpn_lr_schedule, mm_schedule, l2_reg_weight, epochs_to_train=rpn_epochs)

    print("stage 2b - frcn")
    if True:
        # Keep conv and rpn weights from step 3 and fix them
            #       initial weights     train?
            # conv: stage2 rpn model    no
            # rpn:  stage2 rpn model    no
            # frcn: stage1 frcn modle   yes                   -

        # conv_layers
        conv_layers = clone_model(stage2_rpn_network, [feature_node_name], [last_conv_node_name], CloneMethod.freeze)
        conv_out = conv_layers(image_input)

        # RPN
        rpn = clone_model(stage2_rpn_network, [last_conv_node_name], ["rpn_rois"], CloneMethod.freeze)
        rpn_rois = rpn(conv_out)

        # Fast RCNN and losses
        frcn = clone_model(stage1_frcn_network, [last_conv_node_name, "rpn_rois", "roi_input"],
                           ["cls_score", "bbox_regr", "rpn_target_rois", "detection_losses"], CloneMethod.clone)
        stage2_frcn_network = frcn(conv_out, rpn_rois, scaled_gt_boxes)
        detection_losses = stage2_frcn_network.outputs[3]

        # train
        if debug_output: plot(stage2_frcn_network, os.path.join(globalvars['output_path'], "graph_frcn_train_stage2b_frcn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, detection_losses, detection_losses,
                    frcn_lr_schedule, mm_schedule, l2_reg_weight, epochs_to_train=frcn_epochs)

    # return stage 2 model
    return stage2_frcn_network

def eval_faster_rcnn_mAP(eval_model, img_map_file, roi_map_file):
    image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    roi_input   = input_variable((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])
    frcn_eval = eval_model(image_input)

    # Create the minibatch source
    minibatch_source = create_mb_source(img_map_file, roi_map_file,
        image_height, image_width, num_channels, cfg["CNTK"].INPUT_ROIS_PER_IMAGE, randomize=False)

    # define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source[cfg["CNTK"].FEATURE_STREAM_NAME],
        roi_input: minibatch_source[cfg["CNTK"].ROI_STREAM_NAME]
    }

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_test_images)] for _ in range(num_classes)]

    # evaluate test images and write netwrok output to file
    print("Evaluating Faster R-CNN model for %s images." % num_test_images)
    for i in range(0, num_test_images):
        mb_data = minibatch_source.next_minibatch(1, input_map=input_map)
        keys = [k for k in mb_data if "features" not in str(k)]
        gt_row = mb_data[keys[0]].asarray()[0,0,:]
        gt_boxes = gt_row.reshape((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5))
        gt_boxes = gt_boxes[np.where(gt_boxes[:,-1] > 0)]

        output = frcn_eval.eval(mb_data)
        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]                      # (300, 17)
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        labels = out_cls_pred.argmax(axis=1)
        scores = out_cls_pred.max(axis=1).tolist()
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels)  # (300, 4)

        import pdb; pdb.set_trace()
        for j in range(1, num_classes):
            all_boxes[j][i] = \
                np.hstack((regressed_rois, out_cls_pred[:, j])) \
                    .astype(np.float32, copy=False)
            # TODO: calculate mAP


# The main method trains and evaluates a Fast R-CNN model.
# If a trained model is already available it is loaded an no training will be performed.
if __name__ == '__main__':
    if os.path.exists(map_file_path):
        os.chdir(map_file_path)
        if not os.path.exists(os.path.join(abs_path, "Output")):
            os.makedirs(os.path.join(abs_path, "Output"))
        if not os.path.exists(os.path.join(abs_path, "Output", "Grocery")):
            os.makedirs(os.path.join(abs_path, "Output", "Grocery"))
        if not os.path.exists(os.path.join(abs_path, "Output", "Pascal")):
            os.makedirs(os.path.join(abs_path, "Output", "Pascal"))

    parse_arguments()
    model_path = os.path.join(globalvars['output_path'], "faster_rcnn_eval_{}_{}.model"
                              .format(cfg["CNTK"].BASE_MODEL, "e2e" if cfg["CNTK"].TRAIN_E2E else "4stage"))

    # Train only if no model exists yet
    if os.path.exists(model_path) and cfg["CNTK"].MAKE_MODE:
        print("Loading existing model from %s" % model_path)
        eval_model = load_model(model_path)
    else:
        if cfg["CNTK"].TRAIN_E2E:
            trained_model = train_faster_rcnn_e2e(debug_output=cfg["CNTK"].DEBUG_OUTPUT)
        else:
            trained_model = train_faster_rcnn_alternating(debug_output=cfg["CNTK"].DEBUG_OUTPUT)

        # create and store eval model
        image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
        eval_model = create_eval_model(trained_model, image_input)
        eval_model.save(model_path)
        if cfg["CNTK"].DEBUG_OUTPUT:
            plot(eval_model, os.path.join(globalvars['output_path'], "graph_frcn_eval_{}_{}.{}"
                                          .format(cfg["CNTK"].BASE_MODEL, "e2e" if cfg["CNTK"].TRAIN_E2E else "4stage", cfg["CNTK"].GRAPH_TYPE)))

        print("Stored eval model at %s" % model_path)

    # Plot results on test set
    if cfg["CNTK"].DEBUG_OUTPUT:
        from cntk_helpers import eval_and_plot_faster_rcnn
        num_eval = min(num_test_images, 100)
        img_shape = (num_channels, image_height, image_width)
        results_map_file_path = os.path.join(globalvars['output_path'], cfg["CNTK"].DATASET)
        eval_and_plot_faster_rcnn(eval_model, num_eval, globalvars['test_map_file'], img_shape,
                                  results_map_file_path, feature_node_name, classes, debug_output=True,
                                  drawNegativeRois=cfg["CNTK"].DRAW_NEGATIVE_ROIS, decisionThreshold=cfg["CNTK"].ROI_PLOT_THRESHOLD)

    #eval_faster_rcnn_mAP(eval_model, globalvars['test_map_file'], globalvars['test_roi_file'])
