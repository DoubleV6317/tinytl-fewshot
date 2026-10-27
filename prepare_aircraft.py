"""FGVC-Aircraft 2013b for the depth experiment (实验设计/最终实验设计.md §1.1). CPU only.

  python prepare_aircraft.py      extract data/aircraft_raw/fgvc-aircraft-2013b.tar.gz, then build data/aircraft_prep/

Declared deviations from upstream TinyTL (which uses hanlab's make_aircraft.py train/val lists):
  * the official variant-level 3-way split, images_variant_{train,val,test}.txt (100 variants);
  * the 20-pixel copyright banner at the bottom of every image is cropped;
  * images are pre-resized so the shorter side is 512 px (JPEG quality 95). Every RandomResizedCrop(224, scale >= 0.22)
    is still a downsample, and the test transform Resize(256) / CenterCrop(224) is unchanged.
Class folders are named by the zero-padded index of the variant in variants.txt, because names contain spaces and '/'.

The k-shot roots mirror fewshot_resolution.build_root: k images per variant from the train split with Random(seed);
val is a fixed 10-per-variant subset of the val split (Random(12345)); the test root holds the whole test split.
"""
from pathlib import Path
import json
import os
import random
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'data' / 'aircraft_raw' / 'fgvc-aircraft-2013b.tar.gz'
RAW = ROOT / 'data' / 'aircraft_raw' / 'fgvc-aircraft-2013b' / 'data'
PREP = ROOT / 'data' / 'aircraft_prep'
WORK = ROOT / 'fewshot_aircraft'
BANNER = 20
SHORT_SIDE = 512
VAL_PER_CLASS, VAL_SEED = 10, 12345
SPLIT_SIZES = {'train': 3334, 'val': 3333, 'test': 3333}


def extract():
    if (RAW / 'variants.txt').exists():
        return
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        tar.extractall(ARCHIVE.parent, filter='data')  # refuses absolute paths, '..' and links outside the folder


def read_splits():
    variants = [l.strip() for l in (RAW / 'variants.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(variants) == 100 and len(set(variants)) == 100
    index = {v: i for i, v in enumerate(variants)}
    splits = {}
    for split, size in SPLIT_SIZES.items():
        rows = []
        for line in (RAW / ('images_variant_%s.txt' % split)).read_text(encoding='utf-8').splitlines():
            if line.strip():
                img, variant = line.rstrip('\n').split(' ', 1)  # variant names contain spaces
                rows.append([img, index[variant]])
        assert len(rows) == size, '%s has %d images' % (split, len(rows))
        assert {c for _, c in rows} == set(range(100)), '%s does not cover all 100 variants' % split
        splits[split] = rows
    ids = [i for rows in splits.values() for i, _ in rows]
    assert len(ids) == len(set(ids)) == 10000
    return variants, splits


def prepare_images(splits):
    from PIL import Image
    out = PREP / 'images'
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for s in splits.values() for r in s]
    for n, (img, _) in enumerate(rows):
        dst = out / ('%s.jpg' % img)
        if dst.exists():
            continue
        with Image.open(RAW / 'images' / ('%s.jpg' % img)) as im:
            im = im.convert('RGB')
            w, h = im.size
            im = im.crop((0, 0, w, h - BANNER))
            w, h = im.size
            scale = SHORT_SIDE / min(w, h)
            if scale < 1:
                im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            tmp = dst.with_suffix('.tmp')
            im.save(tmp, 'JPEG', quality=95)
            os.replace(tmp, dst)
        if (n + 1) % 1000 == 0:
            print('prepared %d / %d' % (n + 1, len(rows)), flush=True)


def load_prepared():
    return json.loads((PREP / 'splits.json').read_text(encoding='utf-8'))


def link(src, dst):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def by_class(rows):
    out = {}
    for img, c in rows:
        out.setdefault(c, []).append(img)
    return out


def place(root, split_dir, imgs_by_class):
    for c, imgs in imgs_by_class.items():
        for img in imgs:
            link(PREP / 'images' / ('%s.jpg' % img), root / split_dir / ('%03d' % c) / ('%s.jpg' % img))


def val_subset(splits):
    rng = random.Random(VAL_SEED)
    return {c: sorted(rng.sample(sorted(imgs), VAL_PER_CLASS)) for c, imgs in sorted(by_class(splits['val']).items())}


def build_root_aircraft(k, seed):
    """Dataset root whose train is k images per variant and whose val is the fixed val subset."""
    root = WORK / ('k%d_seed%d' % (k, seed))
    if (root / '.done').exists():
        return root
    splits = load_prepared()
    rng = random.Random(seed)
    train = {c: sorted(rng.sample(sorted(imgs), k)) for c, imgs in sorted(by_class(splits['train']).items())}
    place(root, 'train', train)
    place(root, 'val', val_subset(splits))
    (root / '.done').write_text('ok', encoding='utf-8')
    return root


def test_root_aircraft():
    """Root whose val/ is the whole test split (the provider also needs a train/ folder: one image per variant)."""
    root = WORK / 'testset'
    if (root / '.done').exists():
        return root
    splits = load_prepared()
    place(root, 'val', by_class(splits['test']))
    place(root, 'train', {c: sorted(imgs)[:1] for c, imgs in by_class(splits['train']).items()})
    (root / '.done').write_text('ok', encoding='utf-8')
    return root


def main():
    extract()
    variants, splits = read_splits()
    prepare_images(splits)
    (PREP / 'splits.json').write_text(json.dumps(dict(variants=variants, **splits), indent=0), encoding='utf-8')
    n = len(list((PREP / 'images').glob('*.jpg')))
    assert n == 10000, '%d prepared images' % n
    val = val_subset(splits)
    assert sum(len(v) for v in val.values()) == 1000
    print('ok: 100 variants, train/val/test %s, val subset 1000, prepared images in %s'
          % ('/'.join(str(len(splits[s])) for s in SPLIT_SIZES), PREP / 'images'))


if __name__ == '__main__':
    main()
