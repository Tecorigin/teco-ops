import os
import sys

from tecoalNode import TecoalNode
from nodes.abs_grad_node import TecoalAbsGradNode
from nodes.activation_backward_node import TecoalActivationBackwardNode
from nodes.activation_forward_node import TecoalActivationForwardNode
from nodes.adaptive_pooling_backward_node import TecoalAdaptivePoolingBackwardNode
from nodes.adaptive_pooling_forward_node import TecoalAdaptivePoolingForwardNode
from nodes.add_batched_gemmex_node import TecoalAddBatchedGemmexNode
from nodes.add_batched_gemm_node import TecoalAddBatchedGemmNode
from nodes.addcdiv_node import TecoalAddcdivNode
from nodes.addcmul_node import TecoalAddcmulNode
from nodes.add_n_node import TecoalAddNNode
from nodes.add_tensor_node import TecoalAddTensorNode
from nodes.amax_node import TecoalAmaxNode
from nodes.amin_node import TecoalAminNode
from nodes.arange_node import TecoalArangeNode
from nodes.arg_max_node import TecoalArgMaxNode
from nodes.arg_min_node import TecoalArgMinNode
from nodes.arg_sort_node import TecoalArgSortNode
from nodes.asum_node import TecoalAsumNode
from nodes.atan_tensor_node import TecoalAtanTensorNode
from nodes.axpy_node import TecoalAxpyNode
from nodes.batchnorm_backward_node import TecoalBatchnormBackwardNode
from nodes.batchnorm_forward_inference_node import TecoalBatchnormForwardInferenceNode
from nodes.batchnorm_forward_train_node import TecoalBatchnormForwardTrainNode
from nodes.bce_loss_backward_node import TecoalBceLossBackwardNode
from nodes.bce_loss_forward_node import TecoalBceLossForwardNode
from nodes.bitwise_and_tensor_node import TecoalBitwiseAndTensorNode
from nodes.bitwise_not_tensor_node import TecoalBitwiseNotTensorNode
from nodes.bitwise_or_tensor_node import TecoalBitwiseOrTensorNode
from nodes.bitwise_xor_tensor_node import TecoalBitwiseXorTensorNode
from nodes.ceil_node import TecoalCeilNode
from nodes.celoss_backward_node import TecoalCelossBackwardNode
from nodes.celoss_forward_node import TecoalCelossForwardNode
from nodes.clamp_tensor_node import TecoalClampTensorNode
from nodes.clip_backward_node import TecoalClipBackwardNode
from nodes.clip_tensor_node import TecoalClipTensorNode
from nodes.concat_node import TecoalConcatNode
from nodes.concat_v2_node import TecoalConcatV2Node
from nodes.constant_node import TecoalConstantNode
from nodes.conv_backward_bias_node import TecoalConvBackwardBiasNode
from nodes.conv_backward_data_node import TecoalConvBackwardDataNode
from nodes.conv_backward_filter_node import TecoalConvBackwardFilterNode
from nodes.conv_forward_bias_silu_node import TecoalConvForwardBiasSiluNode
from nodes.conv_forward_node import TecoalConvForwardNode
from nodes.copy_node import TecoalCopyNode
from nodes.copy_stride_node import TecoalCopyStrideNode
from nodes.cornerpool_backward_node import TecoalCornerpoolBackwardNode
from nodes.cornerpool_forward_node import TecoalCornerpoolForwardNode
from nodes.cosine_similarity_backward_node import TecoalCosineSimilarityBackwardNode
from nodes.cosine_similarity_forward_node import TecoalCosineSimilarityForwardNode
from nodes.cosine_similarity_node import TecoalCosineSimilarityNode
from nodes.ctcloss_backward_node import TecoalCtclossBackwardNode
from nodes.ctcloss_forward_node import TecoalCtclossForwardNode
from nodes.ctcloss_node import TecoalCtclossNode
from nodes.cumsum_node import TecoalCumsumNode
from nodes.custom_add_axpy_node import TecoalCustomAddAxpyNode
from nodes.custom_atan_grad_node import TecoalCustomAtanGradNode
from nodes.custom_atan_tensor_node import TecoalCustomAtanTensorNode
from nodes.custom_clip_tensor_node import TecoalCustomClipTensorNode
from nodes.custom_fused_yolo_postprocess_node import TecoalCustomFusedYoloPostprocessNode
from nodes.custom_gather_node import TecoalCustomGatherNode
from nodes.custom_gemm_bias_gelu_node import TecoalCustomGemmBiasGeluNode
from nodes.custom_nan_to_num_node import TecoalCustomNanToNumNode
from nodes.custom_prelu_backward_node import TecoalCustomPreluBackwardNode
from nodes.custom_prelu_forward_node import TecoalCustomPreluForwardNode
from nodes.custom_reciprocal_tensor_node import TecoalCustomReciprocalTensorNode
from nodes.custom_reflection_pad1d_backward_node import TecoalCustomReflectionPad1dBackwardNode
from nodes.custom_reflection_pad1d_node import TecoalCustomReflectionPad1dNode
from nodes.custom_reflection_pad2d_node import TecoalCustomReflectionPad2dNode
from nodes.custom_scale_tensor_node import TecoalCustomScaleTensorNode
from nodes.custom_scale_with_bias_node import TecoalCustomScaleWithBiasNode
from nodes.custom_topk_grad_node import TecoalCustomTopkGradNode
from nodes.custom_topk_node import TecoalCustomTopkNode
from nodes.custom_upsample_bicubic2d_backward_node import TecoalCustomUpsampleBicubic2dBackwardNode
from nodes.custom_upsample_bicubic2d_forward_node import TecoalCustomUpsampleBicubic2dForwardNode
from nodes.custom_upsample_bilinear2d_forward_node import TecoalCustomUpsampleBilinear2dForwardNode
from nodes.custom_warpctc_backward_node import TecoalCustomWarpctcBackwardNode
from nodes.deformable_conv_forward_node import TecoalDeformableConvForwardNode
from nodes.dot_node import TecoalDotNode
from nodes.dropout_backward_node import TecoalDropoutBackwardNode
from nodes.dropout_forward_node import TecoalDropoutForwardNode
from nodes.elementwise_pow_node import TecoalElementwisePowNode
from nodes.embedding_backward_node import TecoalEmbeddingBackwardNode
from nodes.embedding_forward_node import TecoalEmbeddingForwardNode
from nodes.expand_node import TecoalExpandNode
from nodes.flip_node import TecoalFlipNode
from nodes.floor_divide_node import TecoalFloorDivideNode
from nodes.fused_conv_batchnorm_activation_add_forward_node import TecoalFusedConvBatchnormActivationAddForwardNode
from nodes.fused_conv_bias_activation_add_forward_node import TecoalFusedConvBiasActivationAddForwardNode
from nodes.fused_conv_bias_activation_forward_node import TecoalFusedConvBiasActivationForwardNode
from nodes.fused_conv_bias_add_activation_forward_node import TecoalFusedConvBiasAddActivationForwardNode
from nodes.fused_conv_bn_scale_bias_activation_add_forward_node import TecoalFusedConvBnScaleBiasActivationAddForwardNode
from nodes.gather_nd_backward_node import TecoalGatherNdBackwardNode
from nodes.gather_nd_forward_node import TecoalGatherNdForwardNode
from nodes.gather_node import TecoalGatherNode
from nodes.gemm_batched_node import TecoalGemmBatchedNode
from nodes.gemm_batched_v2_node import TecoalGemmBatchedV2Node
from nodes.gemmex_batched_node import TecoalGemmexBatchedNode
from nodes.gemmex_node import TecoalGemmexNode
from nodes.gemmex_stride_batched_node import TecoalGemmexStrideBatchedNode
from nodes.gemm_node import TecoalGemmNode
from nodes.gemm_stride_batched_node import TecoalGemmStrideBatchedNode
from nodes.gemv_node import TecoalGemvNode
from nodes.generate_proposals_node import TecoalGenerateProposalsNode
from nodes.glu_backward_node import TecoalGluBackwardNode
from nodes.glu_forward_node import TecoalGluForwardNode
from nodes.grid_sample_backward_node import TecoalGridSampleBackwardNode
from nodes.grid_sample_forward_node import TecoalGridSampleForwardNode
from nodes.group_norm_backward_node import TecoalGroupNormBackwardNode
from nodes.groupnorm_backward_node import TecoalGroupnormBackwardNode
from nodes.group_norm_forward_node import TecoalGroupNormForwardNode
from nodes.groupnorm_forward_node import TecoalGroupnormForwardNode
from nodes.gumbelsoftmax_backward_node import TecoalGumbelsoftmaxBackwardNode
from nodes.gumbelsoftmax_forward_node import TecoalGumbelsoftmaxForwardNode
from nodes.hard_sigmoid_backwardex_node import TecoalHardSigmoidBackwardexNode
from nodes.hard_sigmoid_backward_node import TecoalHardSigmoidBackwardNode
from nodes.hard_sigmoid_forward_node import TecoalHardSigmoidForwardNode
from nodes.hard_swish_backward_node import TecoalHardSwishBackwardNode
from nodes.hard_swish_forward_node import TecoalHardSwishForwardNode
from nodes.huberloss_backward_node import TecoalHuberlossBackwardNode
from nodes.huberloss_forward_node import TecoalHuberlossForwardNode
from nodes.index_put_node import TecoalIndexPutNode
from nodes.index_sample_node import TecoalIndexSampleNode
from nodes.index_select_node import TecoalIndexSelectNode
from nodes.index_tensor_node import TecoalIndexTensorNode
from nodes.inplace_ops_node import TecoalInplaceOpsNode
from nodes.instance_norm_backward_node import TecoalInstanceNormBackwardNode
from nodes.instance_norm_forward_inference_node import TecoalInstanceNormForwardInferenceNode
from nodes.instance_norm_forward_train_node import TecoalInstanceNormForwardTrainNode
from nodes.isnan_node import TecoalIsnanNode
from nodes.kldivloss_forward_node import TecoalKldivlossForwardNode
from nodes.layernorm_backward_node import TecoalLayernormBackwardNode
from nodes.layernorm_forward_node import TecoalLayernormForwardNode
from nodes.linspace_node import TecoalLinspaceNode
from nodes.log2_node import TecoalLog2Node
from nodes.logical_and_tensor_node import TecoalLogicalAndTensorNode
from nodes.logical_not_tensor_node import TecoalLogicalNotTensorNode
from nodes.logical_or_tensor_node import TecoalLogicalOrTensorNode
from nodes.logical_xor_tensor_node import TecoalLogicalXorTensorNode
from nodes.lrn_backward_node import TecoalLrnBackwardNode
from nodes.lrn_forward_node import TecoalLrnForwardNode
from nodes.masked_fill_node import TecoalMaskedFillNode
from nodes.masked_scatter_node import TecoalMaskedScatterNode
from nodes.masked_select_backward_node import TecoalMaskedSelectBackwardNode
from nodes.masked_select_node import TecoalMaskedSelectNode
from nodes.maximum_grad_node import TecoalMaximumGradNode
from nodes.maximum_node import TecoalMaximumNode
from nodes.memset_node import TecoalMemsetNode
from nodes.meshgrid_node import TecoalMeshgridNode
from nodes.minimum_grad_node import TecoalMinimumGradNode
from nodes.minimum_node import TecoalMinimumNode
from nodes.mish_backward_node import TecoalMishBackwardNode
from nodes.mish_forward_node import TecoalMishForwardNode
from nodes.momentum_node import TecoalMomentumNode
from nodes.mseloss_backward_node import TecoalMselossBackwardNode
from nodes.mseloss_forward_node import TecoalMselossForwardNode
from nodes.multiclass_nms_node import TecoalMulticlassNmsNode
from nodes.nearest_interp_backward_node import TecoalNearestInterpBackwardNode
from nodes.neg_tensor_node import TecoalNegTensorNode
from nodes.nllloss2d_backward_node import TecoalNllloss2dBackwardNode
from nodes.nllloss2d_forward_node import TecoalNllloss2dForwardNode
from nodes.nllloss_backward_node import TecoalNlllossBackwardNode
from nodes.nllloss_forward_node import TecoalNlllossForwardNode
from nodes.nms_node import TecoalNmsNode
from nodes.nonzero_node import TecoalNonzeroNode
from nodes.nrm2_node import TecoalNrm2Node
from nodes.onehot_node import TecoalOnehotNode
from nodes.op_tensor_node import TecoalOpTensorNode
from nodes.pnorm_backward_node import TecoalPnormBackwardNode
from nodes.pnorm_forward_node import TecoalPnormForwardNode
from nodes.pooling_backward_node import TecoalPoolingBackwardNode
from nodes.pooling_forward_node import TecoalPoolingForwardNode
from nodes.prior_box_node import TecoalPriorBoxNode
from nodes.random_exponential_node import TecoalRandomExponentialNode
from nodes.reciprocal_tensor_node import TecoalReciprocalTensorNode
from nodes.reduce_tensor_node import TecoalReduceTensorNode
from nodes.remainder_node import TecoalRemainderNode
from nodes.repeat_interleave_node import TecoalRepeatInterleaveNode
from nodes.rmslayernorm_backward_node import TecoalRmslayernormBackwardNode
from nodes.rmslayernorm_forward_node import TecoalRmslayernormForwardNode
from nodes.rnn_backward_data_node import TecoalRnnBackwardDataNode
from nodes.rnn_backward_weight_node import TecoalRnnBackwardWeightNode
from nodes.rnn_forward_node import TecoalRnnForwardNode
from nodes.rnn_forward_training_node import TecoalRnnForwardTrainingNode
from nodes.roialign_backward_node import TecoalRoialignBackwardNode
from nodes.roialign_forward_node import TecoalRoialignForwardNode
from nodes.roipool_forward_node import TecoalRoipoolForwardNode
from nodes.rotary_position_embedding_backward_node import TecoalRotaryPositionEmbeddingBackwardNode
from nodes.rotary_position_embedding_forward_node import TecoalRotaryPositionEmbeddingForwardNode
from nodes.scale_tensor_node import TecoalScaleTensorNode
from nodes.scal_node import TecoalScalNode
from nodes.scatter_nd_add_node import TecoalScatterNdAddNode
from nodes.scatter_node import TecoalScatterNode
from nodes.scatter_out_node import TecoalScatterOutNode
from nodes.set_tensor_node import TecoalSetTensorNode
from nodes.sigmoid_bceloss_backward_node import TecoalSigmoidBcelossBackwardNode
from nodes.sigmoid_bceloss_forward_node import TecoalSigmoidBcelossForwardNode
from nodes.slice_node import TecoalSliceNode
from nodes.smoothl1loss_backward_node import TecoalSmoothl1lossBackwardNode
from nodes.smoothl1loss_forward_node import TecoalSmoothl1lossForwardNode
from nodes.softmax_backward_node import TecoalSoftmaxBackwardNode
from nodes.softmax_celoss_backward_node import TecoalSoftmaxCelossBackwardNode
from nodes.softmax_celoss_forward_node import TecoalSoftmaxCelossForwardNode
from nodes.softmax_forward_node import TecoalSoftmaxForwardNode
from nodes.softplus_backward_node import TecoalSoftplusBackwardNode
from nodes.softplus_forward_node import TecoalSoftplusForwardNode
from nodes.softsign_backward_node import TecoalSoftsignBackwardNode
from nodes.softsign_forward_node import TecoalSoftsignForwardNode
from nodes.sort_node import TecoalSortNode
from nodes.split_node import TecoalSplitNode
from nodes.split_v2_node import TecoalSplitV2Node
from nodes.squared_difference_node import TecoalSquaredDifferenceNode
from nodes.squared_l2_norm_node import TecoalSquaredL2NormNode
from nodes.swap_node import TecoalSwapNode
from nodes.swish_backward_node import TecoalSwishBackwardNode
from nodes.swish_forward_node import TecoalSwishForwardNode
from nodes.tensor_add_node import TecoalTensorAddNode
from nodes.tensor_addwithalpha_node import TecoalTensorAddwithalphaNode
from nodes.tensor_all_node import TecoalTensorAllNode
from nodes.tensor_any_node import TecoalTensorAnyNode
from nodes.tensor_div_node import TecoalTensorDivNode
from nodes.tensor_equal_greater_node import TecoalTensorEqualGreaterNode
from nodes.tensor_equal_less_node import TecoalTensorEqualLessNode
from nodes.tensor_equal_node import TecoalTensorEqualNode
from nodes.tensor_greater_node import TecoalTensorGreaterNode
from nodes.tensor_less_node import TecoalTensorLessNode
from nodes.tensor_mul_node import TecoalTensorMulNode
from nodes.tensor_not_equal_node import TecoalTensorNotEqualNode
from nodes.tensor_sub_node import TecoalTensorSubNode
from nodes.tile_node import TecoalTileNode
from nodes.topk_node import TecoalTopkNode
from nodes.transform_tensorex_node import TecoalTransformTensorexNode
from nodes.transform_tensor_node import TecoalTransformTensorNode
from nodes.transpose_node import TecoalTransposeNode
from nodes.tril_node import TecoalTrilNode
from nodes.triu_node import TecoalTriuNode
from nodes.unary_ops_node import TecoalUnaryOpsNode
from nodes.unique_node import TecoalUniqueNode
from nodes.upsample_bilinear2d_forward_node import TecoalUpsampleBilinear2dForwardNode
from nodes.upsample_node import TecoalUpsampleNode
from nodes.warpctc_backward_node import TecoalWarpctcBackwardNode
from nodes.warpctc_forward_node import TecoalWarpctcForwardNode
from nodes.where_tensor_node import TecoalWhereTensorNode
from nodes.xlogy_node import TecoalXlogyNode
from nodes.z_fused_celoss_backward_node import TecoalZFusedCelossBackwardNode
from nodes.z_fused_celoss_forward_node import TecoalZFusedCelossForwardNode

api_list = [
    'abs_grad', 'activation_backward', 'activation_forward',
    'adaptive_pooling_backward', 'adaptive_pooling_forward',
    'add_batched_gemmex', 'add_batched_gemm', 'addcdiv', 'addcmul', 'add_n',
    'add_tensor', 'amax', 'amin', 'arange', 'arg_max', 'arg_min', 'arg_sort',
    'asum', 'atan_tensor', 'axpy', 'batchnorm_backward',
    'batchnorm_forward_inference', 'batchnorm_forward_train',
    'bce_loss_backward', 'bce_loss_forward', 'bitwise_and_tensor',
    'bitwise_not_tensor', 'bitwise_or_tensor', 'bitwise_xor_tensor', 'ceil',
    'celoss_backward', 'celoss_forward', 'clamp_tensor', 'clip_backward',
    'clip_tensor', 'concat', 'concat_v2', 'constant', 'conv_backward_bias',
    'conv_backward_data', 'conv_backward_filter', 'conv_forward_bias_silu',
    'conv_forward', 'copy', 'copy_stride', 'cornerpool_backward',
    'cornerpool_forward', 'cosine_similarity_backward',
    'cosine_similarity_forward', 'cosine_similarity', 'ctcloss_backward',
    'ctcloss_forward', 'ctcloss', 'cumsum', 'custom_add_axpy',
    'custom_atan_grad', 'custom_atan_tensor', 'custom_clip_tensor',
    'custom_fused_yolo_postprocess', 'custom_gather', 'custom_gemm_bias_gelu',
    'custom_nan_to_num', 'custom_prelu_backward', 'custom_prelu_forward',
    'custom_reciprocal_tensor', 'custom_reflection_pad1d_backward',
    'custom_reflection_pad1d', 'custom_reflection_pad2d',
    'custom_scale_tensor', 'custom_scale_with_bias', 'custom_topk_grad',
    'custom_topk', 'custom_upsample_bicubic2d_backward',
    'custom_upsample_bicubic2d_forward', 'custom_upsample_bilinear2d_forward',
    'custom_warpctc_backward', 'deformable_conv_forward', 'dot',
    'dropout_backward', 'dropout_forward', 'elementwise_pow',
    'embedding_backward', 'embedding_forward', 'expand', 'flip',
    'floor_divide', 'fused_conv_batchnorm_activation_add_forward',
    'fused_conv_bias_activation_add_forward',
    'fused_conv_bias_activation_forward',
    'fused_conv_bias_add_activation_forward',
    'fused_conv_bn_scale_bias_activation_add_forward', 'gather_nd_backward',
    'gather_nd_forward', 'gather', 'gemm_batched', 'gemm_batched_v2',
    'gemmex_batched', 'gemmex', 'gemmex_stride_batched', 'gemm',
    'gemm_stride_batched', 'gemv', 'generate_proposals', 'glu_backward',
    'glu_forward', 'grid_sample_backward', 'grid_sample_forward',
    'group_norm_backward', 'groupnorm_backward', 'group_norm_forward',
    'groupnorm_forward', 'gumbelsoftmax_backward', 'gumbelsoftmax_forward',
    'hard_sigmoid_backwardex', 'hard_sigmoid_backward', 'hard_sigmoid_forward',
    'hard_swish_backward', 'hard_swish_forward', 'huberloss_backward',
    'huberloss_forward', 'index_put', 'index_sample', 'index_select',
    'index_tensor', 'inplace_ops', 'instance_norm_backward',
    'instance_norm_forward_inference', 'instance_norm_forward_train', 'isnan',
    'kldivloss_forward', 'layernorm_backward', 'layernorm_forward', 'linspace',
    'log2', 'logical_and_tensor', 'logical_not_tensor', 'logical_or_tensor',
    'logical_xor_tensor', 'lrn_backward', 'lrn_forward', 'masked_fill',
    'masked_scatter', 'masked_select_backward', 'masked_select',
    'maximum_grad', 'maximum', 'memset', 'meshgrid', 'minimum_grad', 'minimum',
    'mish_backward', 'mish_forward', 'momentum', 'mseloss_backward',
    'mseloss_forward', 'multiclass_nms', 'nearest_interp_backward',
    'neg_tensor', 'nllloss2d_backward', 'nllloss2d_forward',
    'nllloss_backward', 'nllloss_forward', 'nms', 'nonzero', 'nrm2', 'onehot',
    'op_tensor', 'pnorm_backward', 'pnorm_forward', 'pooling_backward',
    'pooling_forward', 'prior_box', 'random_exponential', 'reciprocal_tensor',
    'reduce_tensor', 'remainder', 'repeat_interleave', 'rmslayernorm_backward',
    'rmslayernorm_forward', 'rnn_backward_data', 'rnn_backward_weight',
    'rnn_forward', 'rnn_forward_training', 'roialign_backward',
    'roialign_forward', 'roipool_forward',
    'rotary_position_embedding_backward', 'rotary_position_embedding_forward',
    'scale_tensor', 'scal', 'scatter_nd_add', 'scatter', 'scatter_out',
    'set_tensor', 'sigmoid_bceloss_backward', 'sigmoid_bceloss_forward',
    'slice', 'smoothl1loss_backward', 'smoothl1loss_forward',
    'softmax_backward', 'softmax_celoss_backward', 'softmax_celoss_forward',
    'softmax_forward', 'softplus_backward', 'softplus_forward',
    'softsign_backward', 'softsign_forward', 'sort', 'split', 'split_v2',
    'squared_difference', 'squared_l2_norm', 'swap', 'swish_backward',
    'swish_forward', 'tensor_add', 'tensor_addwithalpha', 'tensor_all',
    'tensor_any', 'tensor_div', 'tensor_equal_greater', 'tensor_equal_less',
    'tensor_equal', 'tensor_greater', 'tensor_less', 'tensor_mul',
    'tensor_not_equal', 'tensor_sub', 'tile', 'topk', 'transform_tensorex',
    'transform_tensor', 'transpose', 'tril', 'triu', 'unary_ops', 'unique',
    'upsample_bilinear2d_forward', 'upsample', 'warpctc_backward',
    'warpctc_forward', 'where_tensor', 'xlogy', 'z_fused_celoss_backward',
    'z_fused_celoss_forward'
]


def getTecoalNode(pt_path, case_path):
    if case_path is None:
        return TecoalNode()

    for op_name in pt_path.split('/'):
        if (op_name in api_list):
            break

    if op_name is None:
        return TecoalNode()

    if op_name == 'abs_grad':
        return TecoalAbsGradNode(case_path)

    if op_name == 'activation_backward':
        return TecoalActivationBackwardNode(case_path)

    if op_name == 'activation_forward':
        return TecoalActivationForwardNode(case_path)

    if op_name == 'adaptive_pooling_backward':
        return TecoalAdaptivePoolingBackwardNode(case_path)

    if op_name == 'adaptive_pooling_forward':
        return TecoalAdaptivePoolingForwardNode(case_path)

    if op_name == 'add_batched_gemmex':
        return TecoalAddBatchedGemmexNode(case_path)

    if op_name == 'add_batched_gemm':
        return TecoalAddBatchedGemmNode(case_path)

    if op_name == 'addcdiv':
        return TecoalAddcdivNode(case_path)

    if op_name == 'addcmul':
        return TecoalAddcmulNode(case_path)

    if op_name == 'add_n':
        return TecoalAddNNode(case_path)

    if op_name == 'add_tensor':
        return TecoalAddTensorNode(case_path)

    if op_name == 'amax':
        return TecoalAmaxNode(case_path)

    if op_name == 'amin':
        return TecoalAminNode(case_path)

    if op_name == 'arange':
        return TecoalArangeNode(case_path)

    if op_name == 'arg_max':
        return TecoalArgMaxNode(case_path)

    if op_name == 'arg_min':
        return TecoalArgMinNode(case_path)

    if op_name == 'arg_sort':
        return TecoalArgSortNode(case_path)

    if op_name == 'asum':
        return TecoalAsumNode(case_path)

    if op_name == 'atan_tensor':
        return TecoalAtanTensorNode(case_path)

    if op_name == 'axpy':
        return TecoalAxpyNode(case_path)

    if op_name == 'batchnorm_backward':
        return TecoalBatchnormBackwardNode(case_path)

    if op_name == 'batchnorm_forward_inference':
        return TecoalBatchnormForwardInferenceNode(case_path)

    if op_name == 'batchnorm_forward_train':
        return TecoalBatchnormForwardTrainNode(case_path)

    if op_name == 'bce_loss_backward':
        return TecoalBceLossBackwardNode(case_path)

    if op_name == 'bce_loss_forward':
        return TecoalBceLossForwardNode(case_path)

    if op_name == 'bitwise_and_tensor':
        return TecoalBitwiseAndTensorNode(case_path)

    if op_name == 'bitwise_not_tensor':
        return TecoalBitwiseNotTensorNode(case_path)

    if op_name == 'bitwise_or_tensor':
        return TecoalBitwiseOrTensorNode(case_path)

    if op_name == 'bitwise_xor_tensor':
        return TecoalBitwiseXorTensorNode(case_path)

    if op_name == 'ceil':
        return TecoalCeilNode(case_path)

    if op_name == 'celoss_backward':
        return TecoalCelossBackwardNode(case_path)

    if op_name == 'celoss_forward':
        return TecoalCelossForwardNode(case_path)

    if op_name == 'clamp_tensor':
        return TecoalClampTensorNode(case_path)

    if op_name == 'clip_backward':
        return TecoalClipBackwardNode(case_path)

    if op_name == 'clip_tensor':
        return TecoalClipTensorNode(case_path)

    if op_name == 'concat':
        return TecoalConcatNode(case_path)

    if op_name == 'concat_v2':
        return TecoalConcatV2Node(case_path)

    if op_name == 'constant':
        return TecoalConstantNode(case_path)

    if op_name == 'conv_backward_bias':
        return TecoalConvBackwardBiasNode(case_path)

    if op_name == 'conv_backward_data':
        return TecoalConvBackwardDataNode(case_path)

    if op_name == 'conv_backward_filter':
        return TecoalConvBackwardFilterNode(case_path)

    if op_name == 'conv_forward_bias_silu':
        return TecoalConvForwardBiasSiluNode(case_path)

    if op_name == 'conv_forward':
        return TecoalConvForwardNode(case_path)

    if op_name == 'copy':
        return TecoalCopyNode(case_path)

    if op_name == 'copy_stride':
        return TecoalCopyStrideNode(case_path)

    if op_name == 'cornerpool_backward':
        return TecoalCornerpoolBackwardNode(case_path)

    if op_name == 'cornerpool_forward':
        return TecoalCornerpoolForwardNode(case_path)

    if op_name == 'cosine_similarity_backward':
        return TecoalCosineSimilarityBackwardNode(case_path)

    if op_name == 'cosine_similarity_forward':
        return TecoalCosineSimilarityForwardNode(case_path)

    if op_name == 'cosine_similarity':
        return TecoalCosineSimilarityNode(case_path)

    if op_name == 'ctcloss_backward':
        return TecoalCtclossBackwardNode(case_path)

    if op_name == 'ctcloss_forward':
        return TecoalCtclossForwardNode(case_path)

    if op_name == 'ctcloss':
        return TecoalCtclossNode(case_path)

    if op_name == 'cumsum':
        return TecoalCumsumNode(case_path)

    if op_name == 'custom_add_axpy':
        return TecoalCustomAddAxpyNode(case_path)

    if op_name == 'custom_atan_grad':
        return TecoalCustomAtanGradNode(case_path)

    if op_name == 'custom_atan_tensor':
        return TecoalCustomAtanTensorNode(case_path)

    if op_name == 'custom_clip_tensor':
        return TecoalCustomClipTensorNode(case_path)

    if op_name == 'custom_fused_yolo_postprocess':
        return TecoalCustomFusedYoloPostprocessNode(case_path)

    if op_name == 'custom_gather':
        return TecoalCustomGatherNode(case_path)

    if op_name == 'custom_gemm_bias_gelu':
        return TecoalCustomGemmBiasGeluNode(case_path)

    if op_name == 'custom_nan_to_num':
        return TecoalCustomNanToNumNode(case_path)

    if op_name == 'custom_prelu_backward':
        return TecoalCustomPreluBackwardNode(case_path)

    if op_name == 'custom_prelu_forward':
        return TecoalCustomPreluForwardNode(case_path)

    if op_name == 'custom_reciprocal_tensor':
        return TecoalCustomReciprocalTensorNode(case_path)

    if op_name == 'custom_reflection_pad1d_backward':
        return TecoalCustomReflectionPad1dBackwardNode(case_path)

    if op_name == 'custom_reflection_pad1d':
        return TecoalCustomReflectionPad1dNode(case_path)

    if op_name == 'custom_reflection_pad2d':
        return TecoalCustomReflectionPad2dNode(case_path)

    if op_name == 'custom_scale_tensor':
        return TecoalCustomScaleTensorNode(case_path)

    if op_name == 'custom_scale_with_bias':
        return TecoalCustomScaleWithBiasNode(case_path)

    if op_name == 'custom_topk_grad':
        return TecoalCustomTopkGradNode(case_path)

    if op_name == 'custom_topk':
        return TecoalCustomTopkNode(case_path)

    if op_name == 'custom_upsample_bicubic2d_backward':
        return TecoalCustomUpsampleBicubic2dBackwardNode(case_path)

    if op_name == 'custom_upsample_bicubic2d_forward':
        return TecoalCustomUpsampleBicubic2dForwardNode(case_path)

    if op_name == 'custom_upsample_bilinear2d_forward':
        return TecoalCustomUpsampleBilinear2dForwardNode(case_path)

    if op_name == 'custom_warpctc_backward':
        return TecoalCustomWarpctcBackwardNode(case_path)

    if op_name == 'deformable_conv_forward':
        return TecoalDeformableConvForwardNode(case_path)

    if op_name == 'dot':
        return TecoalDotNode(case_path)

    if op_name == 'dropout_backward':
        return TecoalDropoutBackwardNode(case_path)

    if op_name == 'dropout_forward':
        return TecoalDropoutForwardNode(case_path)

    if op_name == 'elementwise_pow':
        return TecoalElementwisePowNode(case_path)

    if op_name == 'embedding_backward':
        return TecoalEmbeddingBackwardNode(case_path)

    if op_name == 'embedding_forward':
        return TecoalEmbeddingForwardNode(case_path)

    if op_name == 'expand':
        return TecoalExpandNode(case_path)

    if op_name == 'flip':
        return TecoalFlipNode(case_path)

    if op_name == 'floor_divide':
        return TecoalFloorDivideNode(case_path)

    if op_name == 'fused_conv_batchnorm_activation_add_forward':
        return TecoalFusedConvBatchnormActivationAddForwardNode(case_path)

    if op_name == 'fused_conv_bias_activation_add_forward':
        return TecoalFusedConvBiasActivationAddForwardNode(case_path)

    if op_name == 'fused_conv_bias_activation_forward':
        return TecoalFusedConvBiasActivationForwardNode(case_path)

    if op_name == 'fused_conv_bias_add_activation_forward':
        return TecoalFusedConvBiasAddActivationForwardNode(case_path)

    if op_name == 'fused_conv_bn_scale_bias_activation_add_forward':
        return TecoalFusedConvBnScaleBiasActivationAddForwardNode(case_path)

    if op_name == 'gather_nd_backward':
        return TecoalGatherNdBackwardNode(case_path)

    if op_name == 'gather_nd_forward':
        return TecoalGatherNdForwardNode(case_path)

    if op_name == 'gather':
        return TecoalGatherNode(case_path)

    if op_name == 'gemm_batched':
        return TecoalGemmBatchedNode(case_path)

    if op_name == 'gemm_batched_v2':
        return TecoalGemmBatchedV2Node(case_path)

    if op_name == 'gemmex_batched':
        return TecoalGemmexBatchedNode(case_path)

    if op_name == 'gemmex':
        return TecoalGemmexNode(case_path)

    if op_name == 'gemmex_stride_batched':
        return TecoalGemmexStrideBatchedNode(case_path)

    if op_name == 'gemm':
        return TecoalGemmNode(case_path)

    if op_name == 'gemm_stride_batched':
        return TecoalGemmStrideBatchedNode(case_path)

    if op_name == 'gemv':
        return TecoalGemvNode(case_path)

    if op_name == 'generate_proposals':
        return TecoalGenerateProposalsNode(case_path)

    if op_name == 'glu_backward':
        return TecoalGluBackwardNode(case_path)

    if op_name == 'glu_forward':
        return TecoalGluForwardNode(case_path)

    if op_name == 'grid_sample_backward':
        return TecoalGridSampleBackwardNode(case_path)

    if op_name == 'grid_sample_forward':
        return TecoalGridSampleForwardNode(case_path)

    if op_name == 'group_norm_backward':
        return TecoalGroupNormBackwardNode(case_path)

    if op_name == 'groupnorm_backward':
        return TecoalGroupnormBackwardNode(case_path)

    if op_name == 'group_norm_forward':
        return TecoalGroupNormForwardNode(case_path)

    if op_name == 'groupnorm_forward':
        return TecoalGroupnormForwardNode(case_path)

    if op_name == 'gumbelsoftmax_backward':
        return TecoalGumbelsoftmaxBackwardNode(case_path)

    if op_name == 'gumbelsoftmax_forward':
        return TecoalGumbelsoftmaxForwardNode(case_path)

    if op_name == 'hard_sigmoid_backwardex':
        return TecoalHardSigmoidBackwardexNode(case_path)

    if op_name == 'hard_sigmoid_backward':
        return TecoalHardSigmoidBackwardNode(case_path)

    if op_name == 'hard_sigmoid_forward':
        return TecoalHardSigmoidForwardNode(case_path)

    if op_name == 'hard_swish_backward':
        return TecoalHardSwishBackwardNode(case_path)

    if op_name == 'hard_swish_forward':
        return TecoalHardSwishForwardNode(case_path)

    if op_name == 'huberloss_backward':
        return TecoalHuberlossBackwardNode(case_path)

    if op_name == 'huberloss_forward':
        return TecoalHuberlossForwardNode(case_path)

    if op_name == 'index_put':
        return TecoalIndexPutNode(case_path)

    if op_name == 'index_sample':
        return TecoalIndexSampleNode(case_path)

    if op_name == 'index_select':
        return TecoalIndexSelectNode(case_path)

    if op_name == 'index_tensor':
        return TecoalIndexTensorNode(case_path)

    if op_name == 'inplace_ops':
        return TecoalInplaceOpsNode(case_path)

    if op_name == 'instance_norm_backward':
        return TecoalInstanceNormBackwardNode(case_path)

    if op_name == 'instance_norm_forward_inference':
        return TecoalInstanceNormForwardInferenceNode(case_path)

    if op_name == 'instance_norm_forward_train':
        return TecoalInstanceNormForwardTrainNode(case_path)

    if op_name == 'isnan':
        return TecoalIsnanNode(case_path)

    if op_name == 'kldivloss_forward':
        return TecoalKldivlossForwardNode(case_path)

    if op_name == 'layernorm_backward':
        return TecoalLayernormBackwardNode(case_path)

    if op_name == 'layernorm_forward':
        return TecoalLayernormForwardNode(case_path)

    if op_name == 'linspace':
        return TecoalLinspaceNode(case_path)

    if op_name == 'log2':
        return TecoalLog2Node(case_path)

    if op_name == 'logical_and_tensor':
        return TecoalLogicalAndTensorNode(case_path)

    if op_name == 'logical_not_tensor':
        return TecoalLogicalNotTensorNode(case_path)

    if op_name == 'logical_or_tensor':
        return TecoalLogicalOrTensorNode(case_path)

    if op_name == 'logical_xor_tensor':
        return TecoalLogicalXorTensorNode(case_path)

    if op_name == 'lrn_backward':
        return TecoalLrnBackwardNode(case_path)

    if op_name == 'lrn_forward':
        return TecoalLrnForwardNode(case_path)

    if op_name == 'masked_fill':
        return TecoalMaskedFillNode(case_path)

    if op_name == 'masked_scatter':
        return TecoalMaskedScatterNode(case_path)

    if op_name == 'masked_select_backward':
        return TecoalMaskedSelectBackwardNode(case_path)

    if op_name == 'masked_select':
        return TecoalMaskedSelectNode(case_path)

    if op_name == 'maximum_grad':
        return TecoalMaximumGradNode(case_path)

    if op_name == 'maximum':
        return TecoalMaximumNode(case_path)

    if op_name == 'memset':
        return TecoalMemsetNode(case_path)

    if op_name == 'meshgrid':
        return TecoalMeshgridNode(case_path)

    if op_name == 'minimum_grad':
        return TecoalMinimumGradNode(case_path)

    if op_name == 'minimum':
        return TecoalMinimumNode(case_path)

    if op_name == 'mish_backward':
        return TecoalMishBackwardNode(case_path)

    if op_name == 'mish_forward':
        return TecoalMishForwardNode(case_path)

    if op_name == 'momentum':
        return TecoalMomentumNode(case_path)

    if op_name == 'mseloss_backward':
        return TecoalMselossBackwardNode(case_path)

    if op_name == 'mseloss_forward':
        return TecoalMselossForwardNode(case_path)

    if op_name == 'multiclass_nms':
        return TecoalMulticlassNmsNode(case_path)

    if op_name == 'nearest_interp_backward':
        return TecoalNearestInterpBackwardNode(case_path)

    if op_name == 'neg_tensor':
        return TecoalNegTensorNode(case_path)

    if op_name == 'nllloss2d_backward':
        return TecoalNllloss2dBackwardNode(case_path)

    if op_name == 'nllloss2d_forward':
        return TecoalNllloss2dForwardNode(case_path)

    if op_name == 'nllloss_backward':
        return TecoalNlllossBackwardNode(case_path)

    if op_name == 'nllloss_forward':
        return TecoalNlllossForwardNode(case_path)

    if op_name == 'nms':
        return TecoalNmsNode(case_path)

    if op_name == 'nonzero':
        return TecoalNonzeroNode(case_path)

    if op_name == 'nrm2':
        return TecoalNrm2Node(case_path)

    if op_name == 'onehot':
        return TecoalOnehotNode(case_path)

    if op_name == 'op_tensor':
        return TecoalOpTensorNode(case_path)

    if op_name == 'pnorm_backward':
        return TecoalPnormBackwardNode(case_path)

    if op_name == 'pnorm_forward':
        return TecoalPnormForwardNode(case_path)

    if op_name == 'pooling_backward':
        return TecoalPoolingBackwardNode(case_path)

    if op_name == 'pooling_forward':
        return TecoalPoolingForwardNode(case_path)

    if op_name == 'prior_box':
        return TecoalPriorBoxNode(case_path)

    if op_name == 'random_exponential':
        return TecoalRandomExponentialNode(case_path)

    if op_name == 'reciprocal_tensor':
        return TecoalReciprocalTensorNode(case_path)

    if op_name == 'reduce_tensor':
        return TecoalReduceTensorNode(case_path)

    if op_name == 'remainder':
        return TecoalRemainderNode(case_path)

    if op_name == 'repeat_interleave':
        return TecoalRepeatInterleaveNode(case_path)

    if op_name == 'rmslayernorm_backward':
        return TecoalRmslayernormBackwardNode(case_path)

    if op_name == 'rmslayernorm_forward':
        return TecoalRmslayernormForwardNode(case_path)

    if op_name == 'rnn_backward_data':
        return TecoalRnnBackwardDataNode(case_path)

    if op_name == 'rnn_backward_weight':
        return TecoalRnnBackwardWeightNode(case_path)

    if op_name == 'rnn_forward':
        return TecoalRnnForwardNode(case_path)

    if op_name == 'rnn_forward_training':
        return TecoalRnnForwardTrainingNode(case_path)

    if op_name == 'roialign_backward':
        return TecoalRoialignBackwardNode(case_path)

    if op_name == 'roialign_forward':
        return TecoalRoialignForwardNode(case_path)

    if op_name == 'roipool_forward':
        return TecoalRoipoolForwardNode(case_path)

    if op_name == 'rotary_position_embedding_backward':
        return TecoalRotaryPositionEmbeddingBackwardNode(case_path)

    if op_name == 'rotary_position_embedding_forward':
        return TecoalRotaryPositionEmbeddingForwardNode(case_path)

    if op_name == 'scale_tensor':
        return TecoalScaleTensorNode(case_path)

    if op_name == 'scal':
        return TecoalScalNode(case_path)

    if op_name == 'scatter_nd_add':
        return TecoalScatterNdAddNode(case_path)

    if op_name == 'scatter':
        return TecoalScatterNode(case_path)

    if op_name == 'scatter_out':
        return TecoalScatterOutNode(case_path)

    if op_name == 'set_tensor':
        return TecoalSetTensorNode(case_path)

    if op_name == 'sigmoid_bceloss_backward':
        return TecoalSigmoidBcelossBackwardNode(case_path)

    if op_name == 'sigmoid_bceloss_forward':
        return TecoalSigmoidBcelossForwardNode(case_path)

    if op_name == 'slice':
        return TecoalSliceNode(case_path)

    if op_name == 'smoothl1loss_backward':
        return TecoalSmoothl1lossBackwardNode(case_path)

    if op_name == 'smoothl1loss_forward':
        return TecoalSmoothl1lossForwardNode(case_path)

    if op_name == 'softmax_backward':
        return TecoalSoftmaxBackwardNode(case_path)

    if op_name == 'softmax_celoss_backward':
        return TecoalSoftmaxCelossBackwardNode(case_path)

    if op_name == 'softmax_celoss_forward':
        return TecoalSoftmaxCelossForwardNode(case_path)

    if op_name == 'softmax_forward':
        return TecoalSoftmaxForwardNode(case_path)

    if op_name == 'softplus_backward':
        return TecoalSoftplusBackwardNode(case_path)

    if op_name == 'softplus_forward':
        return TecoalSoftplusForwardNode(case_path)

    if op_name == 'softsign_backward':
        return TecoalSoftsignBackwardNode(case_path)

    if op_name == 'softsign_forward':
        return TecoalSoftsignForwardNode(case_path)

    if op_name == 'sort':
        return TecoalSortNode(case_path)

    if op_name == 'split':
        return TecoalSplitNode(case_path)

    if op_name == 'split_v2':
        return TecoalSplitV2Node(case_path)

    if op_name == 'squared_difference':
        return TecoalSquaredDifferenceNode(case_path)

    if op_name == 'squared_l2_norm':
        return TecoalSquaredL2NormNode(case_path)

    if op_name == 'swap':
        return TecoalSwapNode(case_path)

    if op_name == 'swish_backward':
        return TecoalSwishBackwardNode(case_path)

    if op_name == 'swish_forward':
        return TecoalSwishForwardNode(case_path)

    if op_name == 'tensor_add':
        return TecoalTensorAddNode(case_path)

    if op_name == 'tensor_addwithalpha':
        return TecoalTensorAddwithalphaNode(case_path)

    if op_name == 'tensor_all':
        return TecoalTensorAllNode(case_path)

    if op_name == 'tensor_any':
        return TecoalTensorAnyNode(case_path)

    if op_name == 'tensor_div':
        return TecoalTensorDivNode(case_path)

    if op_name == 'tensor_equal_greater':
        return TecoalTensorEqualGreaterNode(case_path)

    if op_name == 'tensor_equal_less':
        return TecoalTensorEqualLessNode(case_path)

    if op_name == 'tensor_equal':
        return TecoalTensorEqualNode(case_path)

    if op_name == 'tensor_greater':
        return TecoalTensorGreaterNode(case_path)

    if op_name == 'tensor_less':
        return TecoalTensorLessNode(case_path)

    if op_name == 'tensor_mul':
        return TecoalTensorMulNode(case_path)

    if op_name == 'tensor_not_equal':
        return TecoalTensorNotEqualNode(case_path)

    if op_name == 'tensor_sub':
        return TecoalTensorSubNode(case_path)

    if op_name == 'tile':
        return TecoalTileNode(case_path)

    if op_name == 'topk':
        return TecoalTopkNode(case_path)

    if op_name == 'transform_tensorex':
        return TecoalTransformTensorexNode(case_path)

    if op_name == 'transform_tensor':
        return TecoalTransformTensorNode(case_path)

    if op_name == 'transpose':
        return TecoalTransposeNode(case_path)

    if op_name == 'tril':
        return TecoalTrilNode(case_path)

    if op_name == 'triu':
        return TecoalTriuNode(case_path)

    if op_name == 'unary_ops':
        return TecoalUnaryOpsNode(case_path)

    if op_name == 'unique':
        return TecoalUniqueNode(case_path)

    if op_name == 'upsample_bilinear2d_forward':
        return TecoalUpsampleBilinear2dForwardNode(case_path)

    if op_name == 'upsample':
        return TecoalUpsampleNode(case_path)

    if op_name == 'warpctc_backward':
        return TecoalWarpctcBackwardNode(case_path)

    if op_name == 'warpctc_forward':
        return TecoalWarpctcForwardNode(case_path)

    if op_name == 'where_tensor':
        return TecoalWhereTensorNode(case_path)

    if op_name == 'xlogy':
        return TecoalXlogyNode(case_path)

    if op_name == 'z_fused_celoss_backward':
        return TecoalZFusedCelossBackwardNode(case_path)

    if op_name == 'z_fused_celoss_forward':
        return TecoalZFusedCelossForwardNode(case_path)

    return TecoalNode()
