#include <set>

#include <compiler/fusion.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace chainer_compiler {

void FuseDldtOperations(Graph* graph) {
    // The list was created by
    // $ grep 'op =' dldt/model-optimizer/extensions/front/onnx/*.py
    // and `onnx_op_extractors` in mo/front/onnx/extractor.py.
    const std::set<Node::OpType> fusable_ops = {
            Node::kAdd,
            // Node::kAffine,
            Node::kArgMax,
            Node::kAveragePool,
            Node::kBatchNormalization,
            Node::kCast,
            Node::kClip,
            Node::kConcat,
            Node::kConstant,
            Node::kConstantFill,
            Node::kConv,
            Node::kConvTranspose,
            // Node::kCrop,
            // Node::kDetectionOutput,
            Node::kDropout,
            Node::kElu,
            Node::kExp,
            // Node::kExperimentalDetectronDetectionOutput,
            // Node::kExperimentalDetectronGenerateProposalsSingleImage,
            // Node::kExperimentalDetectronPriorGridGenerator,
            // Node::kExperimentalDetectronROIFeatureExtractor,
            // Node::kExperimentalDetectronTopKROIs,
            Node::kFlatten,
            Node::kGRU,
            Node::kGather,
            Node::kGemm,
            Node::kGlobalAveragePool,
            Node::kGlobalMaxPool,
            Node::kIdentity,
            Node::kImageScaler,
            // Node::kInstanceNormalization,
            Node::kLRN,
            Node::kLSTM,
            Node::kLeakyRelu,
            Node::kMatMul,
            Node::kMaxPool,
            Node::kMul,
            Node::kNeg,
            Node::kPad,
            Node::kPow,
            // Node::kPriorBox,
            // Node::kQuantize,
            Node::kRNN,
            Node::kReduceMean,
            Node::kReduceSum,
            Node::kRelu,
            Node::kReshape,
            // Node::kScale,
            Node::kSigmoid,
            //Node::kSlice,
            Node::kSoftmax,
            Node::kSplit,
            //Node::kSqueeze,
            Node::kSum,
            Node::kTanh,
            Node::kTranspose,
            Node::kUnsqueeze,
            Node::kResize,
            Node::kUpsample,
    };

    auto is_fusable = [&fusable_ops](const Node& node) {
        if (!fusable_ops.count(node.op_type())) {
            return false;
        }
        for (Value* value : node.inputs()) {
            if (!value->type().HasKnownShape()) return false;
        }
        for (Value* value : node.outputs()) {
            if (!value->type().HasKnownShape()) return false;
        }

        if (node.op_type() == Node::kResize || node.op_type() == Node::kUpsample) {
            if (node.inputs() == 2 && !node.input(1)->producer() || node.input(1)->producer()->op_type() != Node::kConstant) {
                return false;
            }
        }

        return true;
    };

    FuseAllConnectedNodes("dldt", graph, 1, true, is_fusable);
}

}  // namespace chainer_compiler
