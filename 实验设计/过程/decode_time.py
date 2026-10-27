"""CPU cost per training image (open+decode+train transform), Flowers vs Aircraft-sized JPEGs. No GPU, no training."""
import time, glob, os, sys, json
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
torch.set_num_threads(4)
RAW = Path(r"C:\Users\zhita\Documents\Codex\2026-09-10\this-class-will-feature-a-quarter\outputs\TinyTL\data\flowers102_raw\jpg")
TMP = Path(__file__).with_name('decode_tmp')
files = sorted(RAW.glob('*.jpg'))[:40]
sizes = [Image.open(f).size for f in files]
print('flowers mean size', tuple(sum(s[i] for s in sizes) / len(sizes) for i in (0, 1)))
def make(tag, w, h):
    out = []
    for i, f in enumerate(files[:20]):
        p = TMP / f'{tag}_{i}.jpg'
        if not p.exists():
            Image.open(f).convert('RGB').resize((w, h), Image.BICUBIC).save(p, quality=95)
        out.append(p)
    return out
sets = {'flowers': files[:20], 'air_1.3MP(1400x950)': make('a13', 1400, 950),
        'air_2.7MP(2000x1350)': make('a27', 2000, 1350), 'air_prep(754x512)': make('ap', 754, 512)}
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
base = [transforms.RandomResizedCrop(224, scale=(0.22, 1)), transforms.RandomHorizontalFlip()]
tf_none = transforms.Compose(base + [transforms.ToTensor(), norm])
tf_torch = transforms.Compose(base + [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.ToTensor(), norm])
res = {}
for name, fs in sets.items():
    for tn, tf in (('None', tf_none), ('torch', tf_torch)):
        t = time.perf_counter(); n = 0
        for rep in range(3):
            for f in fs:
                tf(Image.open(f).convert('RGB')); n += 1
        ms = (time.perf_counter() - t) / n * 1000
        res[f'{name}/{tn}'] = round(ms, 2)
        print(f'{name:<24} distort={tn:<6} {ms:6.2f} ms/image', flush=True)
Path(__file__).with_name('decode_time_result.json').write_text(json.dumps(res, indent=1))
