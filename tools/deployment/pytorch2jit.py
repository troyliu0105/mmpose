# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch

from mmpose.apis import init_pose_model

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=1.0.4')


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2jit(model,
                input_shape,
                device,
                output_file='tmp.jit',
                verify=False,
                use_half=False):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    model.eval().to(device=device)

    one_img = torch.randn(input_shape).to(device=device)
    if use_half:
        model.half()
        one_img = one_img.half()

    # register_extra_symbolics(opset_version)
    with torch.jit.optimized_execution(should_optimize=True):
        traced = torch.jit.trace(model,
                                 example_inputs=one_img,
                                 check_trace=True)
    torch.jit.save(traced, output_file)
    print(f'Successfully exported JIT model: {output_file}')
    if verify:
        # check by onnx
        jit_model = torch.jit.load(output_file, map_location=device)
        jit_model.eval()

        # check the numerical value
        # get pytorch output
        pytorch_results = model(one_img)
        if not isinstance(pytorch_results, (list, tuple)):
            assert isinstance(pytorch_results, torch.Tensor)
            pytorch_results = [pytorch_results]

        # get onnx output
        jit_results = jit_model(one_img)

        # compare results
        assert len(pytorch_results) == len(jit_results)
        for pt_result, jit_result in zip(pytorch_results, jit_results):
            assert np.allclose(
                pt_result.detach().cpu(), jit_result.detach().cpu(), atol=1.e-5
            ), 'The outputs are different between Pytorch and TorchScript'
        print('The numerical values are same between Pytorch and TorchScript')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMPose models to TorchScript')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--output-file', type=str, default='tmp.jit')
    parser.add_argument('--half', action='store_true', default='export float16 TorchScript model')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 256, 192],
        help='input size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model = init_pose_model(args.config, args.checkpoint, device='cpu')
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    # convert model to onnx file
    pytorch2jit(
        model,
        args.shape,
        device=args.device,
        output_file=args.output_file,
        verify=args.verify,
        use_half=args.half)
