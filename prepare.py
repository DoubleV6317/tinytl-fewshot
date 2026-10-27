"""Download original assets and prepare the official Flowers102 train+val/test split."""
from pathlib import Path
import urllib.request
import tarfile
import shutil
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print('Downloading', url, flush=True)
        temp = path.with_name(path.name + '.part')
        with urllib.request.urlopen(url, timeout=45) as response, temp.open('wb') as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        temp.replace(path)
    return path

def main():
    from scipy.io import loadmat
    base = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
    raw = ROOT / 'data' / 'flowers102_raw'
    for name in ['102flowers.tgz', 'imagelabels.mat', 'setid.mat']:
        download(base + name, raw / name)
    from torchvision.datasets import Flowers102
    for name, expected_md5 in Flowers102._file_dict.values():
        actual = hashlib.md5((raw / name).read_bytes()).hexdigest()
        if actual != expected_md5:
            raise RuntimeError(f'Checksum mismatch: {name}; remove the incomplete download and retry.')
    if not (raw / 'jpg' / 'image_08189.jpg').exists():
        with tarfile.open(raw / '102flowers.tgz') as archive:
            archive.extractall(raw, filter='data')
    split = loadmat(raw / 'setid.mat')
    labels = loadmat(raw / 'imagelabels.mat')['labels'][0]
    counts = {}
    for folder, ids in [('train', list(split['trnid'][0]) + list(split['valid'][0])), ('val', split['tstid'][0])]:
        for i in ids:
            name = f'image_{i:05d}.jpg'
            target = ROOT / 'data' / 'flowers102' / folder / str(labels[i - 1]) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(raw / 'jpg' / name, target)
        counts[folder] = len(ids)
    assert counts == {'train': 2040, 'val': 6149}, counts
    urls = [
        'https://hanlab18.mit.edu/files/proxylessNAS/proxyless_mobile.config',
        'https://hanlab18.mit.edu/projects/tinyml/tinyTL/files/proxylessnas_mobile+lite_residual@imagenet@ws+gn',
    ]
    records = []
    for url in urls:
        path = download(url, ROOT / 'cache' / url.rsplit('/', 1)[1])
        records.append({'url': url, 'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT / 'assets.json').write_text(json.dumps({'split': counts, 'assets': records}, indent=2), encoding='utf-8')
    print('Ready:', counts, flush=True)

if __name__ == '__main__':
    main()
