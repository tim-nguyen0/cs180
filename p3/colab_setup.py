"""Executed by notebook setup after staging the support bundle locally."""
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

_restart_required = []


def install_missing(wanted):
    """Install first, then request a restart only for loaded packages that changed."""
    global _restart_required
    if _restart_required:
        raise RuntimeError('Packages were updated in this process. Restart the Colab session '
                           'before rerunning setup: '+', '.join(_restart_required))
    from packaging.specifiers import SpecifierSet
    missing=[]
    for package,spec in wanted.items():
        try: version=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: version=None
        if version is None or version not in SpecifierSet(spec): missing.append(package+spec)
    if not missing:
        return
    # Include transitive dependencies (e.g. tokenizers), not unrelated imported modules.
    loaded_roots={name.split('.')[0] for name in sys.modules}
    before={}
    for module,packages in importlib.metadata.packages_distributions().items():
        if module not in loaded_roots: continue
        for package in packages:
            try: before[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: pass
    print('Installing required packages:', ', '.join(missing))
    subprocess.check_call([sys.executable,'-m','pip','install','--quiet',*missing])
    importlib.invalidate_caches()
    for package,old in before.items():
        try: new=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: new=None
        if new != old: _restart_required.append(package)
    if _restart_required:
        raise RuntimeError('Installation completed, but these already-loaded packages changed: '
                           +', '.join(_restart_required)+'. Restart the Colab session once, '
                           'then run setup again. The installed packages will be reused.')


def configure(part, install_dependencies=False):
    root=Path.cwd()
    cache=Path(os.environ.setdefault('CS180_CACHE_DIR',str(root/'.cache')))
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('HF_HOME',str(cache/'huggingface'))
    os.environ.setdefault('CS180_PROMPT_CACHE',str(cache/'prompt_embeddings'))
    os.environ.setdefault('CS180_TRAIN_CACHE',str(cache/'training'))
    os.environ.setdefault('PIP_CACHE_DIR',str(cache/'pip'))
    if install_dependencies:
        # Keep Colab's CUDA-compatible torch/torchvision installation intact.
        wanted={'matplotlib':'>=3.7,<4','Pillow':'>=9.2'}
        if part=='parta':
            wanted.update({'transformers':'==4.51.3','huggingface-hub':'==0.36.2','einops':'>=0.8,<1'})
        install_missing(wanted)
    import torch
    import torchvision
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU accelerator in Colab or Kaggle, then rerun setup.')
    props=torch.cuda.get_device_properties(0)
    print('GPU:',props.name,'VRAM GiB:',round(props.total_memory/2**30,1))
    print('PyTorch:',torch.__version__,'torchvision:',torchvision.__version__)
    print('Local cache:',cache)
    if part=='parta':
        print('Part A precision: float32 (text encoder and image model).')
    return cache
