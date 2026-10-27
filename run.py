"""Windows launcher for the original TinyTL trainer (fixed resolution)."""
from pathlib import Path
import argparse
import os
import sys
import runpy
import json
import datetime

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['smoke', 'train'], default='smoke')
    p.add_argument('--method', choices=['tinytl-lite_residual+bias', 'tinytl-lite_residual', 'tinytl-bias', 'bn+last', 'last', 'full'], default='tinytl-lite_residual+bias')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--resume-from', dest='resume_from', default=None,
        help='Existing runs/<dir> to continue from its checkpoint instead of starting a new run.')
    args = p.parse_args()
    os.environ['TINYTL_FLOWERS_PATH'] = str(ROOT / 'data' / 'flowers102')
    os.environ['TINYTL_CACHE'] = str(ROOT / 'cache')
    os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
    os.environ['OMP_NUM_THREADS'] = '4'
    sys.path[:0] = [str(ROOT / 'upstream' / 'once-for-all'), str(ROOT / 'upstream' / 'tinytl')]
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use the project .venv and check the NVIDIA driver.')
    from tinytl.data_providers.fgvc_data_providers import Flowers102DataProvider
    if args.mode == 'smoke':
        # Only checks execution, not accuracy. Full training is never subsampled.
        for method in ['train_dataset', 'test_dataset']:
            original = getattr(Flowers102DataProvider, method)
            def subset(self, transform, original=original):
                dataset = original(self, transform)
                dataset.samples = dataset.samples[::max(1, len(dataset.samples)//16)][:16]
                dataset.imgs = dataset.samples
                dataset.targets = [target for _, target in dataset.samples]
                return dataset
            setattr(Flowers102DataProvider, method, subset)
    if args.resume_from:
        output = Path(args.resume_from)
        if not output.is_absolute():
            output = ROOT / output
        ckpt = output / 'checkpoint' / 'checkpoint.pth.tar'
        if not ckpt.is_file():
            raise SystemExit(f'No checkpoint to resume: {ckpt}')
        # latest.txt stores an absolute path from the original machine/run; rewrite it.
        (output / 'checkpoint' / 'latest.txt').write_text(str(ckpt) + chr(10), encoding='utf-8')
        print('Resuming:', output, flush=True)
    else:
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output = ROOT / 'runs' / f'{args.mode}-{args.method}-seed{args.seed}-{stamp}'
        output.mkdir(parents=True, exist_ok=False)
        print('Output:', output, flush=True)
    sys.argv = ['tinytl_fgvc_train.py', '--path', str(output), '--dataset', 'flowers102',
        '--transfer_learning_method', args.method, '--train_batch_size', str(args.batch_size),
        '--test_batch_size', '16' if args.mode == 'smoke' else '64',
        '--n_epochs', '1' if args.mode == 'smoke' else str(args.epochs),
        '--init_lr', '2e-4', '--opt_type', 'adam', '--label_smoothing', '0.7',
        '--distort_color', 'None', '--frozen_param_bits', '8', '--n_worker', str(args.workers),
        '--manual_seed', str(args.seed), '--gpu', '0']
    if args.resume_from:
        # Upstream --resume reloads weights/optimizer/epoch and skips re-quantization.
        # It also reseeds from the clock, so a resumed run is not seed-identical to an unbroken one.
        sys.argv.append('--resume')
    torch.cuda.reset_peak_memory_stats()
    runpy.run_path(str(ROOT / 'upstream' / 'tinytl' / 'tinytl_fgvc_train.py'), run_name='__main__')
    report = {'status': 'passed', 'mode': args.mode, 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(), 'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
        'note': 'Smoke accuracy is not a reproduction result. CUDA memory is not the paper memory estimate.'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
