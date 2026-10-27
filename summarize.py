"""Collect final-epoch / best / memory numbers from runs/ and compare with the paper."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parent

# arXiv 2007.11622 Table 2, ProxylessNAS-Mobile, Flowers102.
PAPER = {
    'last':                        ('FT-Last',       90.1,  31),
    'tinytl-bias':                 ('TinyTL-B',      93.5,  32),
    'bn+last':                     ('FT-Norm+Last',  94.3, 192),
    'tinytl-lite_residual':        ('TinyTL-L',      95.3,  37),
    'tinytl-lite_residual+bias':   ('TinyTL-L+B',    95.5,  37),
    'full':                        ('FT-Full',       96.8, 391),
}

VALID = re.compile(r'Valid \[(\d+)/(\d+)\].*?top1 ([\d.]+) \(([\d.]+)\)')
# Upstream appends the closing validate_all_resolution line to the same file.
FINAL_TEST = re.compile(r'test_acc1: ([\d.]+)')


def summarize(run: Path):
    args_file = run / 'args.txt'
    valid_log = run / 'logs' / 'valid_console.txt'
    if not args_file.is_file() or not valid_log.is_file():
        return None
    args = json.loads(args_file.read_text(encoding='utf-8'))
    lines = [l for l in valid_log.read_text(encoding='utf-8').splitlines() if l.strip()]
    # The last line is the closing test line once a run completes, so scan backwards.
    m = next((m for m in (VALID.search(l) for l in reversed(lines)) if m), None)
    if not m:
        return None
    epoch, total, final_top1, best_top1 = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
    t = next((t for t in (FINAL_TEST.search(l) for l in reversed(lines)) if t), None)
    final_test_top1 = float(t.group(1)) if t else None

    memory = None
    net_info = run / 'net_info.txt'
    if net_info.is_file():
        blobs = re.findall(r'\{\s*"memory_cost".*?\}', net_info.read_text(encoding='utf-8'), re.S)
        if blobs:
            memory = json.loads(blobs[-1])['memory_cost']

    return {
        'run': run.name,
        'method': args['transfer_learning_method'],
        'seed': args['manual_seed'],
        'epochs_done': epoch,
        'epochs_planned': total,
        'complete': (run / 'verification.json').is_file(),
        # Upstream selects "best" on the test set; report both and say so.
        'final_top1': final_top1,
        'best_top1': best_top1,
        'final_test_top1': final_test_top1,
        'memory_cost_MB': memory,
    }


def main():
    rows = []
    for run in sorted((ROOT / 'runs').iterdir()):
        if not run.is_dir() or run.name.startswith('smoke'):
            continue
        row = summarize(run)
        if row:
            rows.append(row)

    if not rows:
        print('No runs with results yet.')
        return

    hdr = f"{'run':<52} {'method':<28} {'ep':>7} {'final':>7} {'best':>7} {'mem MB':>7}  {'paper':>13}"
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        name, acc, mem = PAPER.get(r['method'], ('?', float('nan'), float('nan')))
        flag = '' if r['complete'] else '  (INCOMPLETE)'
        print(f"{r['run']:<52} {r['method']:<28} "
              f"{r['epochs_done']:>3}/{r['epochs_planned']:<3} "
              f"{r['final_top1']:>7.2f} {r['best_top1']:>7.2f} "
              f"{(r['memory_cost_MB'] or float('nan')):>7.2f}  "
              f"{name} {acc}/{mem}MB{flag}")

    print()
    print('final = last epoch. best = upstream best-on-test selection; not a clean model-selection protocol.')
    print('mem MB = paper-style profile_memory_cost estimate, not PyTorch CUDA usage.')
    (ROOT / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f"\nWrote {ROOT / 'summary.json'}")


if __name__ == '__main__':
    sys.exit(main())
