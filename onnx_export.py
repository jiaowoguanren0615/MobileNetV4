"""
ONNX export script
Export PyTorch models as ONNX graphs.
This export script originally started as an adaptation of code snippets found at
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

The default parameters work with PyTorch 2.0.1 and ONNX 1.13 and produce an optimal ONNX graph
for hosting in the ONNX runtime (see onnx_validate.py). To export an ONNX model compatible
"""

import argparse
import torch
import numpy as np
import onnx
import models
from copy import deepcopy
from timm.models import create_model
from typing import Optional, Tuple, List



## python onnx_export.py --model mobilenetv4_small ./mobilenetv4_small.onnx

parser = argparse.ArgumentParser(description='PyTorch ONNX Deployment')
parser.add_argument('--output', metavar='ONNX_FILE', default=None, type=str,
                    help='output model filename')

# Model & datasets params
parser.add_argument('--model', default='mobilenetv4_conv_large', type=str, metavar='MODEL',
                        choices=['mobilenetv4_hybrid_large', 'mobilenetv4_hybrid_medium', 'mobilenetv4_hybrid_large_075',
                                'mobilenetv4_conv_large', 'mobilenetv4_conv_aa_large', 'mobilenetv4_conv_medium',
                                 'mobilenetv4_conv_aa_medium', 'mobilenetv4_conv_small', 'mobilenetv4_hybrid_medium_075',
                                 'mobilenetv4_conv_small_035', 'mobilenetv4_conv_small_050', 'mobilenetv4_conv_blur_medium'],
                        help='Name of model to train')
parser.add_argument('--extra_attention_block', default=False, type=bool, help='Add an extra attention block')
parser.add_argument('--checkpoint', default='./output/mobilenetv4_conv_large_best_checkpoint.pth', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--img-size', default=384, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--nb-classes', type=int, default=5,
                    help='Number classes in datasets')

parser.add_argument('--opset', type=int, default=10,
                    help='ONNX opset to use (default: 10)')
parser.add_argument('--keep-init', action='store_true', default=False,
                    help='Keep initializers as input. Needed for Caffe2 compatible export in newer PyTorch/ONNX.')
parser.add_argument('--aten-fallback', action='store_true', default=False,
                    help='Fallback to ATEN ops. Helps fix AdaptiveAvgPool issue with Caffe2 in newer PyTorch/ONNX.')
parser.add_argument('--dynamic-size', action='store_true', default=False,
                    help='Export model width dynamic width/height. Not recommended for "tf" models with SAME padding.')
parser.add_argument('--check-forward', action='store_true', default=False,
                    help='Do a full check of torch vs onnx forward after export.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of datasets')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of datasets')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')
parser.add_argument('--training', default=False, action='store_true',
                    help='Export in training mode (default is eval)')
parser.add_argument('--verbose', default=False, action='store_true',
                    help='Extra stdout output')
parser.add_argument('--dynamo', default=False, action='store_true',
                    help='Use torch dynamo export.')



def reparameterize_model(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    if not inplace:
        model = deepcopy(model)

    def _fuse(m):
        for child_name, child in m.named_children():
            if hasattr(child, 'fuse'):
                setattr(m, child_name, child.fuse())
            elif hasattr(child, "reparameterize"):
                child.reparameterize()
            elif hasattr(child, "switch_to_deploy"):
                child.switch_to_deploy()
            _fuse(child)

    _fuse(model)
    return model


def onnx_forward(onnx_file, example_input):
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnx_file, sess_options)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: example_input.numpy()})
    output = output[0]
    return output


def onnx_export(
        model: torch.nn.Module,
        output_file: str,
        example_input: Optional[torch.Tensor] = None,
        training: bool = False,
        verbose: bool = False,
        check: bool = True,
        check_forward: bool = False,
        batch_size: int = 64,
        input_size: Tuple[int, int, int] = None,
        opset: Optional[int] = None,
        dynamic_size: bool = False,
        aten_fallback: bool = False,
        keep_initializers: Optional[bool] = None,
        use_dynamo: bool = False,
        input_names: List[str] = None,
        output_names: List[str] = None,
):
    import onnx

    if training:
        training_mode = torch.onnx.TrainingMode.TRAINING
        model.train()
    else:
        training_mode = torch.onnx.TrainingMode.EVAL
        model.eval()

    if example_input is None:
        if not input_size:
            assert hasattr(model, 'default_cfg')
            input_size = model.default_cfg.get('input_size')
        example_input = torch.randn((batch_size,) + input_size, requires_grad=training)

    # Run model once before export trace, sets padding for models with Conv2dSameExport. This means
    # that the padding for models with Conv2dSameExport (most models with tf_ prefix) is fixed for
    # the input img_size specified in this script.

    # Opset >= 11 should allow for dynamic padding, however I cannot get it to work due to
    # issues in the tracing of the dynamic padding or errors attempting to export the model after jit
    # scripting it (an approach that should work). Perhaps in a future PyTorch or ONNX versions...
    with torch.no_grad():
        original_out = model(example_input)

    print("==> Exporting model to ONNX format at '{}'".format(output_file))

    input_names = input_names or ["input0"]
    output_names = output_names or ["output0"]

    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}
    if dynamic_size:
        dynamic_axes['input0'][2] = 'height'
        dynamic_axes['input0'][3] = 'width'

    if aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX

    if use_dynamo:
        export_options = torch.onnx.ExportOptions(dynamic_shapes=dynamic_size)
        export_output = torch.onnx.dynamo_export(
            model,
            example_input,
            export_options=export_options,
        )
        export_output.save(output_file)
        torch_out = None
    else:
        #TODO, for torch version >= 2.5, use torch.onnx.export()
        torch_out = torch.onnx._export(
            model,
            example_input,
            output_file,
            training=training_mode,
            export_params=True,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            keep_initializers_as_inputs=keep_initializers,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            operator_export_type=export_type
        )

    if check:
        print("==> Loading and checking exported model from '{}'".format(output_file))
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model, full_check=True)  # assuming throw on error
        if check_forward and not training:
            import numpy as np
            onnx_out = onnx_forward(output_file, example_input)
            if torch_out is not None:
                np.testing.assert_almost_equal(torch_out.numpy(), onnx_out, decimal=3)
                np.testing.assert_almost_equal(original_out.numpy(), torch_out.numpy(), decimal=5)
            else:
                np.testing.assert_almost_equal(original_out.numpy(), onnx_out, decimal=3)


def main():
    args = parser.parse_args()

    if args.output == None:
        args.output = f'./{args.model}.onnx'

    print("==> Creating PyTorch {} model".format(args.model))


    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        extra_attention_block=args.extra_attention_block,
        exportable=True
    )

    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()

    if args.reparam:
        model = reparameterize_model(model)

    onnx_export(
        model=model,
        output_file=args.output,
        opset=args.opset,
        dynamic_size=args.dynamic_size,
        aten_fallback=args.aten_fallback,
        keep_initializers=args.keep_init,
        check_forward=args.check_forward,
        training=args.training,
        verbose=args.verbose,
        use_dynamo=args.dynamo,
        input_size=(3, args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    print("==> Passed")


if __name__ == '__main__':
    main()