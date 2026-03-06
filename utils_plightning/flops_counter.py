# !pip install thop
import torch
import torch.nn as nn
from copy import deepcopy
import warnings

def get_flops(model, input_shape=(256, 1, 8000), device='cpu', method='thop'):

    model = deepcopy(model).to(device)
    model.eval()

    # Create input tensor
    x = torch.randn(*input_shape).to(device)
    if x.dim() == 3:
        x = x.squeeze(1)

    results = {
        'input_shape': input_shape,
        'device': device,
        'method': method,
        'flops_gflops': 0.0,
        'macs_gmacs': 0.0,
        'params_millions': 0.0
    }

    # Calculate parameters
    params = sum(p.numel() for p in model.parameters())
    results['params_millions'] = params / 1e6

    if method == 'thop':
        try:
            import thop
        except ImportError:
            print("thop not installed. Install with: pip install thop")
            return results

        try:
            # thop returns (MACs, params)
            macs, params = thop.profile(model, inputs=(x,), verbose=False)
            results['macs'] = macs
            results['macs_gmacs'] = macs / 1e9
            results['flops'] = macs * 2  # FLOPs typically 2x MACs
            results['flops_gflops'] = (macs * 2) / 1e9

        except Exception as e:
            print(f"thop calculation failed: {e}")

    elif method == 'fvcore':
        try:
            from fvcore.nn import FlopCountAnalysis, parameter_count_table
        except ImportError:
            print("fvcore not installed. Install with: pip install fvcore")
            return results

        try:
            # fvcore returns FLOPs directly
            flops = FlopCountAnalysis(model, x)
            results['flops'] = flops.total()
            results['flops_gflops'] = flops.total() / 1e9
            results['macs_gmacs'] = (flops.total() / 2) / 1e9  # Approximate MACs

            # Get parameter count
            results['params_millions'] = sum(p.numel() for p in model.parameters()) / 1e6

        except Exception as e:
            print(f"fvcore calculation failed: {e}")

    elif method == 'ptflops':
        try:
            from ptflops import get_model_complexity_info
        except ImportError:
            print("ptflops not installed. Install with: pip install ptflops")
            return results

        try:
            # ptflops expects input shape without batch dimension
            input_res = (input_shape[1], input_shape[2])  # (channels, seq_len)
            macs, params = get_model_complexity_info(
                model,
                input_res,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            results['macs'] = macs
            results['macs_gmacs'] = macs / 1e9
            results['flops'] = macs * 2
            results['flops_gflops'] = (macs * 2) / 1e9
            results['params_millions'] = params / 1e6

        except Exception as e:
            print(f"ptflops calculation failed: {e}")

    return results

def print_flops(results):
    """Pretty print the FLOPs results for audio models."""
    print("=" * 60)
    print("AUDIO MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)
    print(f"Input shape: {results['input_shape']}")
    print(f"Device: {results['device']}")
    print(f"Method: {results['method']}")
    print("-" * 60)
    print(f"Parameters: {results['params_millions']:.3f} M")
    print(f"MACs: {results['macs_gmacs']:.3f} G ({results['macs_gmacs']*1e3:.2f} M MACs)")
    print(f"FLOPs: {results['flops_gflops']:.3f} G ({results['flops_gflops']*1e3:.2f} M FLOPs)")
    print("=" * 60)

    # Audio-specific insights
    samples = results['input_shape'][2]
    batch_size = results['input_shape'][0]
    flops_per_sample = results['flops'] / batch_size
    print(f"\nPer-sample analysis:")
    print(f"  FLOPs per sample: {flops_per_sample/1e6:.2f} M FLOPs")
    print(f"  FLOPs per time-step: {flops_per_sample/samples:.2f} FLOPs/sample")