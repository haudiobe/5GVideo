from pathlib import Path
import struct
import io
import os
import json
import csv
import math

from utils import VideoSequence, ColorPrimaries, ChromaFormat, ChromaSubsampling, TransferFunction

from anchor import AnchorTuple, VariantData, Metric, VariantMetricSet, iter_variants
from utils import run_process
from encoders import get_encoder, parse_encoding_bitdepth, parse_variant_qp
from conversion import Conversion, as_10bit_sequence, as_exr2020_sequence, get_anchor_conversion_type, hdr_convert_cmd_8to10bits

class VideoFormatException(BaseException):
    pass


BASE_METRICS = (
    Metric.BITRATELOG,
    Metric.BITRATE,
    Metric.ENCODETIME,
    Metric.DECODETIME
)

SDR_METRICS = (
    Metric.PSNR_Y,
    Metric.PSNR_U,
    Metric.PSNR_V,
    Metric.PSNR,
    Metric.MSSSIM, 
    Metric.VMAF,
)

HDR_METRICS = (
    Metric.WTPSNR,
    # Metric.GSSIM,
    Metric.DELTAE100,
    Metric.PSNR,
    Metric.MSSSIM, 
    Metric.VMAF,
)



def iter_section(f: io.FileIO, esc="-", eof_raises=False):
    line = f.readline()
    while len(line) > 0:
        if line[0] == esc:
            return
        yield line
        line = f.readline()
    assert not eof_raises, 'parser error: unexpected end of file'


def read_meta(f):
    line = f.readline()
    # prints out command line options
    meta = [*iter_section(f, esc="=")]
    # prints out input sequences' descriptions
    meta += [*iter_section(f, esc="=")]
    line = f.readline()
    assert len(line) > 0 and line[0] == '-', 'failed to parse header'
    return meta


def read_log(f, stats={}):
    stats['meta'] = read_meta(f)
    stats['metrics'] = [line.split() for line in iter_section(f)][0]
    stats['frames'] = [line.split() for line in iter_section(f)]
    stats['avg'] = [line.split() for line in iter_section(f)][0]
    amplitude = [line.split() for line in iter_section(f)]
    stats['min'] = amplitude[0]
    stats['max'] = amplitude[1]
    stats['perfs'] = "".join([*iter_section(f)])
    return stats


def parse_metrics(log):
    raw_data = None
    p = Path(log).resolve()
    with open(p, 'r') as f:
        raw_data = read_log(f)
    
    assert raw_data is not None, f'failed to parse {p}'
    
    def parse_hex(indices, values): 
        return [struct.unpack('!d', bytes.fromhex(values[i]))[0] for i in indices]
    
    def parse_frames(indices, frames):
        return [[f[0], *parse_hex(indices, f)] for f in frames]
    
    indices, metrics = zip(*[(i, h[3:]) for i, h in enumerate(raw_data['metrics']) if str(h).startswith('hex')])
    return {
        'metrics': metrics,
        'avg': parse_hex(indices, raw_data['avg']),
        # 'min': parse_hex(indices, raw_data['min']),
        # 'max': parse_hex(indices, raw_data['max']),
        'frames': [['frame', *metrics], *parse_frames(indices, raw_data['frames'])]
    }


def hdrtools_input(v: VideoSequence, ref=True, file_header=0):
    i = 0 if ref else 1
    opts = [
        '-p', f'Input{i}File={v.path}',
        '-p', f'Input{i}Width={v.width}',
        '-p', f'Input{i}Height={v.height}',
        '-p', f'Input{i}BitDepthCmp0={v.bit_depth}',
        '-p', f'Input{i}BitDepthCmp1={v.bit_depth}',
        '-p', f'Input{i}BitDepthCmp2={v.bit_depth}',
        '-p', f'Input{i}StartFrame={v.start_frame-1}',
        '-p', f'Input{i}FileHeader={file_header}',
        '-p', f'Input{i}Rate={v.frame_rate}',
        '-p', f'Input{i}SampleRange=0',  # SR_STANDARD is HDRMetrics' default, (16-235)*k
        # '-p', f'Input{i}FourCCCode={0}'  # PF_UYVY is HDRMetrics' default, specifies custom pixel formats, mostly for interleaved and custom component ordering (eg. BGR instead of RGB)
    ]
    
    if v.interleaved:
        opts += ['-p', f'Input{i}Interleaved=1']
    else:
        opts += ['-p', f'Input{i}Interleaved=0']

    if v.interlaced:
        opts += ['-p', f'Input{i}Interlaced=1']
    else:
        opts += ['-p', f'Input{i}Interlaced=0']

    if v.chroma_format == ChromaFormat.YUV:
        opts += ['-p', f'Input{i}ColorSpace=0']  # 0:CM_YCbCr
    elif v.chroma_format == ChromaFormat.RGB:
        opts += ['-p', f'Input{i}ColorSpace=1']  # 1:CM_RGB
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

    if v.colour_primaries == ColorPrimaries.BT_709:
        opts += ['-p', f'Input{i}ColorPrimaries=0']
    elif v.colour_primaries == ColorPrimaries.BT_2020:
        opts += ['-p', f'Input{i}ColorPrimaries=1']

    return opts


def hdrtools_metrics(ref: VideoSequence, dist: VideoSequence, dry_run=False, cfg: Path = None) -> dict:
    if os.getenv('DISABLE_HDRMETRICS'):
        return
    
    input0 = hdrtools_input(ref, ref=True)
    input1 = hdrtools_input(dist, ref=False)
    run = ['-f', str(cfg)] if cfg else []
    run += [
        '-p', 'EnablehexMetric=1',  # the parser collects hex values only
        '-p', 'SilentMode=0',
        '-p', f'NumberOfFrames={dist.frame_count}'
    ]

    tool = os.getenv('HDRMETRICS_TOOL', None)
    assert tool, 'missing environment variable, path to exe: HDRMETRICS_TOOL'
    cmd = [tool, *input0, *input1, *run]

    log = dist.path.with_suffix('.metrics.log')
    run_process(log, *cmd, dry_run=dry_run)
    if log.exists():
        data = parse_metrics(log)
        metrics = {k: v for k, v in zip(data['metrics'], data['avg'])}
        return metrics
    else:
        assert not dry_run, f'HDRMetrics log file not found: {log}'
        return {}


def bitstream_size(bitstream: Path, drop_sei=False) -> int:
    tmp = None
    if drop_sei:
        # expecting SEIRemovalAppStatic built from HM16.23
        tool = os.getenv('SEI_REMOVAL_APP')
        assert tool, 'missing env variable: SEI_REMOVAL_APP'
        tmp = bitstream.with_suffix('.tmp')
        cmd = [tool, '-b', str(bitstream), '-o', str(tmp), '--DiscardPrefixSEI=1', '--DiscardSuffixSEI=1']
        log = bitstream.with_suffix('.seiremoval.log')
        run_process(log, *cmd, dry_run=False)
    s = int(os.path.getsize(tmp if tmp else bitstream))
    if tmp:
        os.remove(tmp)
        os.remove(log)
    return s


def vmaf_metrics(ref: VideoSequence, dist: VideoSequence, model="version=vmaf_v0.6.1", dry_run=False):
    # VMAF may not support skipping frames at the beginning of input sequences
    output = dist.path.with_suffix('.vmaf.json')
    log = dist.path.with_suffix('.vmaf.log')
    vmaf_exec = os.getenv('VMAF_EXEC', 'vmafossexec')
    """
    cmd = [
        vmaf_exec,
        "-r", f'{ref.path}', 
        "-d", f'{dist.path}',
        "-w", f'{ref.width}',
        "-h", f'{ref.height}',
        "-p", f'{ref.chroma_subsampling.value}',
        "-b", f'{ref.bit_depth}',
        "--json", "-o", str(output),
        "-m", model
   ]
    """
    print(f'[{ref.bit_depth}\tbits]:: ', ref.path.name)
    print(f'[{dist.bit_depth}\tbits]:: ', dist.path.name)
    try:
        assert not dist.interlaced and not ref.interlaced
        assert not dist.interleaved and not ref.interleaved
        # dist.chroma_format
        # dist.chroma_subsampling
    except AssertionError:
        print('Invalid format for VMAF. Must be one of yuv420p, yuv420p10le, yuv420p12le, yuv420p16le')

    pixfmt = None 
    if ref.bit_depth == 8:
        pixfmt = 'yuv420p'
    elif ref.bit_depth == 10:
        pixfmt = 'yuv420p10le'
    elif ref.bit_depth == 12:
        pixfmt = 'yuv420p12le'
    elif ref.bit_depth == 16:
        pixfmt = 'yuv420p16le'

    cmd = [
        vmaf_exec,
        str(pixfmt),
        str(ref.width),
        str(ref.height),
        str(ref.path),
        str(dist.path),
        str(model),
        '--log', str(output),
        '--log-fmt', 'json'
    ]
    
    run_process(log, *cmd, dry_run=dry_run)
    if output.exists():
        with open(output, "rb") as fp:
            data = json.load(fp)
            return data["pooled_metrics"]["vmaf"]["mean"]
    else:
        print(f"VMAF output not found : {output}")
        return None


def compute_sdr_metrics(a: AnchorTuple, vd: VariantData):
    hdr_metrics_cfg = os.getenv('HDRMETRICS_CFG', '/home/cfg/HDRMetrics_PSNR_MSSSIM.cfg')

    conv = get_anchor_conversion_type(a)
    if conv == Conversion.NONE:
        ref = a.reference
        dist = VideoSequence.from_sidecar_metadata( a.working_dir / f'{vd.variant_id}.yuv.json')
        assert ref.bit_depth == dist.bit_depth

    elif conv == Conversion.HDRCONVERT_8TO10BIT:
        ref = as_10bit_sequence(a.reference)
        assert ref.path.exists(), f'reference sequence needs pre-processing - Not found: {ref.path}'
        dist = VideoSequence.from_sidecar_metadata( a.working_dir / f'{vd.variant_id}.yuv.json')
        assert ref.bit_depth == dist.bit_depth

    else:
        raise ValueError("Invalid conversion type for SDR metrics")
    
    d = hdrtools_metrics(ref, dist, dry_run=a.dry_run, cfg= hdr_metrics_cfg if hdr_metrics_cfg is None else Path(hdr_metrics_cfg)) 
    result = {
        Metric.PSNR_Y.key: d["PSNR-Y"],
        Metric.PSNR_U.key: d["PSNR-U"],
        Metric.PSNR_V.key: d["PSNR-V"],
        Metric.MSSSIM.key: d["JMSSSIM-Y"]
    }
    return VariantMetricSet(result)
    

def compute_hdr_metrics_yuv(a: AnchorTuple, vd: VariantData):
    hdr_metrics_cfg = os.getenv('HDRMETRICS_CFG', '/home/cfg/HDRMetricsYUV_PQ10.cfg')
    dist = VideoSequence.from_sidecar_metadata(a.working_dir / f'{vd.variant_id}.yuv.json')
    
    cfg = hdr_metrics_cfg if hdr_metrics_cfg is None else Path(hdr_metrics_cfg)
    
    d = hdrtools_metrics(a.reference, dist, dry_run=a.dry_run, cfg= cfg)
    result = {
        Metric.WTPSNR_Y.key: d["wtPSNR-Y"],
        Metric.WTPSNR_U.key: d["wtPSNR-U"],
        Metric.WTPSNR_V.key: d["wtPSNR-V"]
    }
    return VariantMetricSet(result)


def compute_hdr_metrics_exr(a: AnchorTuple, vd: VariantData):
    hdr_metrics_cfg = os.getenv('HDRMETRICS_CFG', '/home/cfg/HDRMetrics_DeltaE100.cfg')
 
    ref = as_exr2020_sequence(a.reference)
    fp = ref.path.with_suffix('.json')
    assert fp.exists(), f'reference sequence needs pre-processing - Not found: {fp}'
 
    yuv_dist = VideoSequence.from_sidecar_metadata( a.working_dir / f'{vd.variant_id}.yuv.json')
    dist = as_exr2020_sequence(yuv_dist)
    fp = dist.path.with_suffix('.json')
    assert fp.exists(), f'reference sequence needs pre-processing - Not found: {fp}'

    d = hdrtools_metrics(ref, dist, dry_run=a.dry_run, cfg= hdr_metrics_cfg if hdr_metrics_cfg is None else Path(hdr_metrics_cfg)) 
    result = {
        Metric.PSNR_DE0100.key: d["PSNR_DE0100"],
        Metric.PSNR_L0100.key: d["PSNR_L0100"],
        Metric.PSNR_MD0100.key: d["PSNR_L0100"]
    }
    return VariantMetricSet(result)


def compute_vmaf_metrics(a: AnchorTuple, vd: VariantData):
    conv = get_anchor_conversion_type(a)
    if conv == Conversion.NONE:
        ref = a.reference
        dist = VideoSequence.from_sidecar_metadata( a.working_dir / f'{vd.variant_id}.yuv.json')
        assert ref.bit_depth == dist.bit_depth

    elif conv == Conversion.HDRCONVERT_8TO10BIT:
        ref = as_10bit_sequence(a.reference)
        assert ref.path.exists(), f'reference sequence needs pre-processing - Not found: {ref.path}'
        dist = VideoSequence.from_sidecar_metadata( a.working_dir / f'{vd.variant_id}.yuv.json')
        assert ref.bit_depth == dist.bit_depth

    else:
        raise ValueError("Invalid conversion type for vmaf")

    vmaf_model = os.getenv('VMAF_MODEL', "version=vmaf_v0.6.1")
    d = vmaf_metrics(ref, dist, model=vmaf_model, dry_run=a.dry_run)
    filtered = {m : d[m] for m in [ Metric.VMAF ]}
    return VariantMetricSet(filtered)


def compute_metrics(a: AnchorTuple, vd: VariantData) -> VariantMetricSet:

    conv = get_anchor_conversion_type(a)
    metrics = VariantMetricSet()
    metrics[Metric.BITRATELOG.key] = 0
    metrics[Metric.BITRATE.key] = 0
    metrics[Metric.ENCODETIME.key] = 0
    metrics[Metric.DECODETIME.key] = 0

    if not os.getenv('DISABLE_HDRMETRICS'):
        if conv in (Conversion.NONE, Conversion.HDRCONVERT_8TO10BIT):
            metrics_sdr = compute_sdr_metrics(a, vd)
            metrics.update(metrics_sdr)

        elif conv == Conversion.HDRCONVERT_YCBR420TOEXR2020:
            # os.setenv("DISABLE_VMAF", "1")
            metrics_hdr = compute_hdr_metrics_yuv(a, vd)
            metrics.update(metrics_hdr)
            metrics_exr = compute_hdr_metrics_exr(a, vd)
            metrics.update(metrics_exr)
        
    if not os.getenv('DISABLE_VMAF'):
        if conv in (Conversion.NONE, Conversion.HDRCONVERT_8TO10BIT):
            metrics_vmaf = compute_vmaf_metrics(a, vd)    
            metrics.update(metrics_vmaf)

    if a.dry_run:
        return vd.variant_id, metrics

    bitstream = a.working_dir / vd.bitstream['URI']
    s = bitstream_size(bitstream, drop_sei=False)
    metrics[Metric.BITRATE.key] = int(s * 8 / a.duration) * 1e-3

    enc = get_encoder(vd.generation['encoder'])
    # parse additional metrics from ENCODER log 
    if vd.generation and vd.generation.get('log-file', None):
        enc_log = a.working_dir / vd.generation['log-file']
        if enc_log.exists():
            encoder_metrics = enc.encoder_log_metrics(enc_log)
            if Metric.BITRATELOG.key in encoder_metrics:
                metrics[Metric.BITRATELOG.key] = float(encoder_metrics[Metric.BITRATELOG.key])
            if Metric.ENCODETIME.key in encoder_metrics:
                metrics[Metric.ENCODETIME.key] = float(encoder_metrics[Metric.ENCODETIME.key])
        else:
            print(f'#\tencoder log not found: {enc_log}')
    # parse additional metrics from DECODER log 
    if vd.reconstruction and vd.reconstruction.get('log-file', None):
        dec_log = a.working_dir / vd.reconstruction['log-file']
        if dec_log.exists():
            decoder_metrics = enc.decoder_log_metrics(dec_log)
            if Metric.DECODETIME.key in decoder_metrics:
                metrics[Metric.DECODETIME.key] = float(decoder_metrics[Metric.DECODETIME.key])
        else:
            print(f'#\tdecoder log not found: {dec_log}')

    return metrics


def anchor_metrics_to_csv(a: AnchorTuple, dst: Path = None):
    fieldnames = None
    for variant_path, variant_data in iter_variants(a):
        assert variant_path.exists(), f'{variant_path} not found'
        assert variant_data.metrics, f'metrics not defined in: {variant_path}'
        if not fieldnames:
            fieldnames = ["parameter", *variant_data.metrics.keys()]

    if dst is None:
        dst = a.working_dir.parent / 'Metrics' / f'{a.working_dir.stem}.csv'
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)
    with open(dst, 'w') as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        for variant_path, variant_data in iter_variants(a):
            row = {}
            for k in fieldnames:
                if k == "parameter":
                    row[k] = parse_variant_qp(variant_data.variant_cli)
                else:
                    row[k] = variant_data.metrics[k]
            writer.writerow(row)

