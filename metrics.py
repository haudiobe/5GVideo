#!/usr/bin/python3
from pathlib import Path
import struct
import io
import os
import json

from utils import VideoSequence, ColourPrimaries, ChromaFormat, ChromaSubsampling, TransferFunction

from anchor import VariantCfg
from utils import run_process

class VariantData:

    required = ["PSNR", "PSNR-Y", "MSSSIM", "VMAF"]

    def __init__(self, variant_id:str, avg_bitrate:float, metrics:dict):
        self.variant_id = variant_id
        self.avg_bitrate = avg_bitrate
        for m in self.required:
            assert m in metrics, f'missing required metric {m}'
        self.metrics = metrics

    def get_keys(self):
        return [
            "Key",
            "Bitrate",
            *self.required
        ]

    def to_dict(self):
        filtered = { m: self.metrics[m] for m in self.required }
        return {
            "Key": self.variant_id,
            "Bitrate": self.avg_bitrate,
            **filtered
        }

################################################################################

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

def parse_metrics(log):
    raw_data = None
    p = Path(log).resolve()
    with open(p, 'r') as f:
        raw_data = read_log(f)
    assert raw_data != None, f'failed to parse {p}'
    parse_hex = lambda indices, values: [struct.unpack('!d', bytes.fromhex(values[i]))[0] for i in indices]
    parse_frames = lambda indices, frames: [[f[0], *parse_hex(indices, f)] for f in frames]
    indices, metrics = zip(*[(i, h[3:]) for i, h in enumerate(raw_data['metrics']) if str(h).startswith('hex')])
    return {
        'metrics': metrics,
        'avg': parse_hex(indices, raw_data['avg']),
        'min': parse_hex(indices, raw_data['min']),
        'max': parse_hex(indices, raw_data['max']),
        'frames': [['frame', *metrics], *parse_frames(indices, raw_data['frames'])]
    }

def hdrtools_input(v:VideoSequence, ref=True, start_frame=0, file_header=0):
    i = 0 if ref else 1
    opts = [
        '-p', f'Input{i}File={v.path}',
        '-p', f'Input{i}Width={v.width}',
        '-p', f'Input{i}Height={v.height}',
        '-p', f'Input{i}BitDepthCmp0={v.bit_depth}',
        '-p', f'Input{i}BitDepthCmp1={v.bit_depth}',
        '-p', f'Input{i}BitDepthCmp2={v.bit_depth}',
        '-p', f'Input{i}StartFrame={start_frame}',
        '-p', f'Input{i}FileHeader={file_header}',
        '-p', f'Input{i}Rate={v.frame_rate}',
        '-p', f'Input{i}SampleRange=0', # SR_STANDARD is HDRMetrics' default, (16-235)*k
        # '-p', f'Input{i}FourCCCode={0}' # PF_UYVY is HDRMetrics' default, specifies custom pixel formats, mostly for interleaved and custom component ordering (eg. BGR instead of RGB)
    ]
    
    if v.interleaved:
        opts += ['-p', f'Input{i}Interleaved=0']
    else:
        opts += ['-p', f'Input{i}Interleaved=1']

    if v.interlaced:
        opts += ['-p', f'Input{i}Interlaced=0']
    else:
        opts += ['-p', f'Input{i}Interlaced=1']

    if v.chroma_format == ChromaFormat.YUV:
        opts += ['-p', f'Input{i}ColorSpace=0'] # 0:CM_YCbCr
    elif v.chroma_format == ChromaFormat.RGB:
        opts += ['-p', f'Input{i}ColorSpace=1'] # 1:CM_RGB
    else:
        # out of scope
        raise ValueError('Unexpected color space')
    
    if v.chroma_subsampling == ChromaSubsampling.CS_400:
        opts += ['-p', f'Input{i}ChromaFormat=0']
    elif v.chroma_subsampling == ChromaSubsampling.CS_420:
        opts += ['-p', f'Input{i}ChromaFormat=1']
    elif v.chroma_subsampling == ChromaSubsampling.CS_422:
        opts += ['-p', f'Input{i}ChromaFormat=2']
    elif v.chroma_subsampling == ChromaSubsampling.CS_444:
        opts += ['-p', f'Input{i}ChromaFormat=3']

    if v.colour_primaries == ColourPrimaries.BT_709:
        opts += ['-p', f'Input{i}ColorPrimaries=0']
    elif v.colour_primaries == ColourPrimaries.BT_2020:
        opts += ['-p', f'Input{i}ColorPrimaries=1']

    return opts


def hdrtools_metrics(v:VariantCfg) -> dict:

    ref = v.anchor.reference
    # PQ content metrics to be parsed from VTM encoder log
    assert ref.transfer_characteristics != TransferFunction.BT2020_PQ, 'unsupported transfer function'

    dist = VideoSequence(v.reconstructed, **ref.properties)
    input0 = hdrtools_input(ref, ref=True, start_frame=v.anchor.start_frame)
    input1 = hdrtools_input(dist, ref=False, start_frame=0)
    run = [
        '-p', f'EnablehexMetric=1', # the parser collects hex values only
        '-p', f'EnableJVETPSNR=1',
        '-p', f'EnableShowMSE=1',
        '-p', f'EnablePSNR=1',
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
        return {}
    
    data = parse_metrics(log)
    metrics = { k:v for k,v in zip(data['metrics'], data['avg']) }
    
    if v.anchor.reference.chroma_format == ChromaFormat.YUV:
        return {
            'PSNR': ((6 * metrics["PSNR-Y"]) + metrics["PSNR-U"] + metrics["PSNR-V"]) / 8,
            'SSIM': ((6 * metrics["SSIM-Y"]) + metrics["SSIM-U"] + metrics["SSIM-V"]) / 8,
            'MSSSIM': ((6 * metrics["MSSSIM-Y"]) + metrics["MSSSIM-U"] + metrics["MSSSIM-V"]) / 8,
            **metrics
        }
    elif v.anchor.reference.chroma_format == ChromaFormat.RGB:
        return {
            'PSNR': ( metrics["PSNR-R"] + metrics["PSNR-G"] + metrics["PSNR-B"] ) / 3,
            'SSIM': ( metrics["SSIM-R"] + metrics["SSIM-G"] + metrics["SSIM-B"] ) / 3,
            'MSSSIM': ( metrics["MSSSIM-R"] + metrics["MSSSIM-G"] + metrics["MSSSIM-B"] ) / 3,
            **metrics
        }


################################################################################

def vmaf_metrics(v:VariantCfg, model="version=vmaf_v0.6.1"):
    output =  v.anchor.working_dir / f'{v.basename}.vmaf.json'
    log = v.anchor.working_dir / f'{v.basename}.vmaf.log'
    cmd = [
        "vmaf",
        "-r", f'{v.anchor.reference.path}', 
        "-d", f'{v.reconstructed}',
        "-w", f'{v.anchor.reference.width}',
        "-h", f'{v.anchor.reference.height}',
        "-p", f'{v.anchor.reference.chroma_subsampling.value}',
        "-b", f'{v.anchor.reference.bit_depth}',
        "--json", "-o", str(output),
        "-m", model
    ]
    run_process(log, *cmd, dry_run=v.anchor.dry_run)
    with open(output, "rb") as fp:
        data = json.load(fp)
        return data["pooled_metrics"]["vmaf"]["mean"]


################################################################################


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

