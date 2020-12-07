#!/usr/bin/python3
from pathlib import Path
import struct
import io
import os

from utils import VideoSequence

from anchor import AnchorCfg, VariantCfg

from utils import run_process

def iter_section(f:io.FileIO, esc="-", eof_raises=True):
    l = f.readline()
    while len(l) > 0:
        if l[0] == esc:
            return
        yield l
        l = f.readline()
    assert not eof_raises, 'parser error: unexpected end of file'

def read_meta(f):
    l = f.readline()
    # prints out command line options
    meta = [*iter_section(f, esc="=")]
    # prints out input sequences' descriptions
    meta += [*iter_section(f, esc="=")]
    l = f.readline()
    assert len(l) > 0 and l[0] == '-', 'failed to parse header'
    return meta

def read_log(f, stats={}):
    stats['meta'] = read_meta(f)
    stats['metrics'] = [l.split() for l in iter_section(f)][0]
    stats['frames'] = [l.split() for l in iter_section(f)]
    stats['avg'] = [l.split() for l in iter_section(f)][0]
    amplitude = [l.split() for l in iter_section(f)]
    stats['min'] = amplitude[0]
    stats['max'] = amplitude[1]
    stats['perfs'] = "".join([*iter_section(f)])
    return stats

def reduce_log(raw):
    parse_hex = lambda indices, values: [struct.unpack('!d', bytes.fromhex(values[i]))[0] for i in indices]
    parse_frames = lambda indices, frames: [[f[0], *parse_hex(indices, f)] for f in frames]
    indices, metrics = zip(*[(i, h[3:]) for i, h in enumerate(raw['metrics']) if str(h).startswith('hex')])
    return {
        'metrics': metrics,
        'avg': parse_hex(indices, raw['avg']),
        'min': parse_hex(indices, raw['min']),
        'max': parse_hex(indices, raw['max']),
        'frames': [['frame', *metrics], *parse_frames(indices, raw['frames'])]
    }

def parse_metrics(log):
    metrics = {}
    with open(Path(log), 'r') as f:
        metrics = read_log(f, metrics)
    return reduce_log(metrics)

def hdrtools_input(v:VideoSequence, ref=True, start_frame=0):
    i = 0 if ref else 1
    opts = [
        '-p', f'Input{i}File={v.path}',
        '-p', f'Input{i}Width={v.width}',
        '-p', f'Input{i}Height={v.height}',
        '-p', f'Input{i}BitDepthCmp0={v.bitdepth}',
        '-p', f'Input{i}BitDepthCmp1={v.bitdepth_chroma}',
        '-p', f'Input{i}BitDepthCmp2={v.bitdepth_chroma}',
        '-p', f'Input{i}StartFrame={start_frame}',
        '-p', f'Input{i}FileHeader={0}',
        '-p', f'Input{i}StartFrame={0}',
        '-p', f'Input{i}Rate={v.fps}',
        '-p', f'Input{i}Interleaved={0}', # Planar YUV
        '-p', f'Input{i}Interlaced={0}',
        '-p', f'Input{i}ColorSpace=0', # 0:CM_YCbCr, 1:CM_RGB, 2:CM_XYZ
        # '-p', f'Input{i}SampleRange={0}'
        # '-p', f'Input{i}FourCCCode={0}'
    ]

    cf = { '400': 0, '420': 1, '422': 2, '444': 3 }
    opts += ['-p', f'Input{i}ChromaFormat={cf.get(v.chroma_subsampling, 1)}']

    cs = { 'bt.709':0, 'bt.2020':1, 'bt.p3d60':2, 'bt.p3d65': 3 }
    opts += ['-p', f'Input{i}ColorPrimaries={cs.get(v.color_space, 4)}']
    return opts


def hdrtools_metrics(v:VariantCfg):
    ref = v.anchor.reference
    dist = VideoSequence(v.reconstructed, **ref.metadata_dict)
    input0 = hdrtools_input(ref, ref=True, start_frame=v.anchor.start_frame)
    input1 = hdrtools_input(dist, ref=False, start_frame=0)
    # this may be specified as an external .cfg file 
    run = [
        '-p', f'EnablehexMetric=1',
        '-p', f'EnablePSNR=1',
        '-p', f'EnableShowMSE=1',
        '-p', f'EnableJVETPSNR=1',
        '-p', f'EnableSSIM=1',
        '-p', f'EnableMSSSIM=1',
        '-p', f'SilentMode=0',
        '-p', f'NumberOfFrames={v.anchor.frame_count}'
    ]
    log = dist.path.parent / 'distortion.txt'
    run += ['-p', f'LogFile={log}']

    tool = os.getenv('HDRMETRICS_TOOL')

    cmd = [tool, *input0, *input1, *run]
    log = dist.path.parent / f'{dist.path.stem}.metrics.log'
    run_process(log, *cmd, dry_run=v.anchor.dry_run)
    if v.anchor.dry_run:
        return
    return parse_metrics(log)


def avg(seq:list, *keys) -> dict:
    avg = { k: 0 for k in keys }
    c = len(seq)
    if c == 0:
        print("empty dataset. the sequence may be too short.")
        return {}
    for f in seq :
        for k in keys:
            avg[k] += f[k]
    for k in keys:
        avg[k] = avg[k] / c
    return avg

def bd_q(RA, QA, RT, QT, piecewise=0):
    import numpy as np
    import scipy
    
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


