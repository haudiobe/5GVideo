import numpy as np
import scipy
import os
import subprocess
from statistics import mean
from anchor import VariantCfg
from utils import VideoSequence

def  _ffavf_metric(ref:VideoSequence, dist:VideoSequence, measurement="psnr"):
    assert measurement in ["psnr", "ssim"]
    graph = [
        'gpac',
        '-i', f'{dist.path}:size={dist.width}x{dist.height}:spfmt={dist.pixfmt}:fps={dist.fps}:#ffid=dist:FID=V1',
        '-i', f'{ref.path}:size={ref.width}x{ref.height}:spfmt={ref.pixfmt}:fps={ref.fps}:#ffid=ref:FID=V2',
    ]
    # avfilter config
    avfg = "ffavf:SID=V1,V2::f="
    # up-sample reconstructed frame to match the original
    avfg += f'[dist]scale={ref.width}:{ref.height},'
    # if framerates do not match, use closest frame in the past from reconstructed sequence
    avfg += f'fps=round=up:fps={ref.fps}[main];'
    # run measurement filter
    fout = dist.path.parent / f'{dist.path.stem}.{measurement}.log'
    # note the order: [recon_seq][orig_seq], with [recon_seq] being the main graph input 
    avfg += f'[main][ref]{measurement}=f={fout}'
    graph.extend([ avfg, '@', '-o', 'null'])
    return graph, fout

def compute_psnr_stats(v:VariantCfg):
    ref = v.anchor.reference
    dist = VideoSequence(v.reconstructed, **ref.metadata_dict)
    cmd, statfile = _ffavf_metric(ref, dist, 'psnr')
    with subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) as proc:
        proc.wait()
        if proc.returncode == 0:
            return statfile
        raise Exception(f'command failed {proc.returncode}')

def read_psnr_stats(v:VariantCfg):
    f = v.anchor.working_dir / f'{v.basename}.psnr.log'
    with open(f, 'r', encoding='utf-8') as r:
        for l in r.readlines():
            row = { x[0]: float(x[1]) for x in [ m.split(":") for m in l.rstrip().split(" ")] }
            row.pop('n', None)
            yield row

def psnr_stats(v:VariantCfg):
    compute_psnr_stats(v)
    return read_psnr_stats(v)

def compute_ssim_stats(v:VariantCfg):
    ref = v.anchor.reference
    dist = VideoSequence(v.reconstructed, **ref.metadata_dict)
    cmd, statfile = _ffavf_metric(ref, dist, 'ssim')
    with subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) as proc:
        proc.wait()
        if proc.returncode == 0:
            return statfile
        raise Exception(f'command failed {proc.returncode}')

def read_ssim_stats(v:VariantCfg):
    f = v.anchor.working_dir / f'{v.basename}.ssim.log'
    with open(f, 'r', encoding='utf-8') as r:
        for l in r.readlines():
            row = {  f'ssim_{str(x[0]).lower()}': float(x[1]) for x in [ m.split(":") for m in l.rstrip().split(" ")][:-1] }
            row.pop('ssim_n', None)
            yield row

def ssim_stats(v:VariantCfg):
    compute_ssim_stats(v)
    return read_ssim_stats(v)

def avg(seq:list, *keys) -> dict:
    avg = { k: 0 for k in keys }
    c = len(seq)
    if c == 0:
        print("empty dataset")
        return {}
    for f in seq :
        for k in keys:
            avg[k] += f[k]
    for k in keys:
        avg[k] = avg[k] / c
    return avg

def bd_q(RA, QA, RT, QT, piecewise=0):
    lRA = np.log(RA)
    lRT = np.log(RT)
    QA = np.array(QA)
    QT = np.array(QT)
    p1 = np.polyfit(lRA, QA, 3)
    p2 = np.polyfit(lRT, QT, 3)

    # integration interval
    min_int = max(min(lRA), min(lRT))
    max_int = min(max(lRA), max(lRT))
    
    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lRA), QA[np.argsort(lRA)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lRT), QT[np.argsort(lRT)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)
    return avg_diff

class VariantData:

    def __init__(self, variant_id:str, avg_bitrate:float, **metrics):
        self.variant_id = variant_id
        self.avg_bitrate = avg_bitrate
        self.metrics = metrics

    def to_string(self, *metrics):
        if len(metrics) == 0:
            metrics = self.metrics.keys()
        s = f'{self.variant_id} :\n - avg_bitrate: {self.avg_bitrate:.2f}'
        for m in metrics:
            v = self.metrics.get(m, None)
            s += f'\n - {m}: {v:.2f}'
        return s

    @property
    def data(self):
        return { 'variant_id': self.variant_id, 'avg_bitrate': self.avg_bitrate, **self.metrics }


