from hashlib import md5
import logging
from re import template
import click
from datetime import datetime
from pathlib import Path
import struct
import io
import os
import json
import csv
from typing import Any, Dict, List, Tuple
from copy import deepcopy

from anchor import AnchorTuple, VariantData, Metric, VariantMetricSet, load_variants
from encoders import get_encoder
from conversion import Conversion, get_anchor_conversion_type
from constants import BITSTREAMS_DIR, SEQUENCES_DIR, ENCODING

from sequences import VideoSequence, as_8bit_sequence, as_10bit_sequence, as_exr2020_sequence, ColorPrimaries, ChromaFormat, ChromaSubsampling, TransferFunction
from utils import run_process, from_enum

class VideoFormatException(BaseException):
    pass

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
    try:
        stats['meta'] = read_meta(f)
        stats['metrics'] = [line.split() for line in iter_section(f)][0]
        stats['frames'] = [line.split() for line in iter_section(f)]
        stats['avg'] = [line.split() for line in iter_section(f)][0]
        amplitude = [line.split() for line in iter_section(f)]
        stats['min'] = amplitude[0]
        stats['max'] = amplitude[1]
        stats['perfs'] = "".join([*iter_section(f)])
        return stats
    except BaseException:
        logging.error(f"Failed to parse metrics on {f}")
        

def parse_metrics(log):
    raw_data = None
    p = Path(log).resolve()
    with open(p, 'r') as f:
        raw_data = read_log(f)
    
    assert raw_data is not None, f'failed to parse {p}'
    
    def parse_hex(indices, values):
        r = []
        for i in indices:
            try:
                b = bytes.fromhex(values[i][:16])
                r.append(struct.unpack('!d', b)[0])
            except BaseException as e:
                raise Exception(f'{values[i]}  => {e}')
        return r
    
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
        '-p', f'Input{i}SampleRange={v.video_full_range}',  # 0: Standard/Video range, 1: Full range
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


def hdrtools_metrics(ref: VideoSequence, dist: VideoSequence, log: Path, dry_run=False, cfg: Path = None) -> dict:
    if os.getenv('DISABLE_HDRMETRICS'):
        return {}
    else:
        assert cfg is not None and Path(cfg).exists(), 'HDRTools config file not found'
    
    input0 = hdrtools_input(ref, ref=True)
    input1 = hdrtools_input(dist, ref=False)
    run = ['-f', str(cfg)] if cfg else []
    run += [
        '-p', 'EnablehexMetric=1',
        '-p', 'SilentMode=0',
        '-p', f'NumberOfFrames={dist.frame_count}'
    ]

    tool = os.getenv('HDRMETRICS_TOOL', 'HDRMetrics')
    cmd = [tool, *input0, *input1, *run]

    run_process(log, *cmd, dry_run=dry_run)
    if log.exists():
        data = parse_metrics(log)
        metrics = {k: v for k, v in zip(data['metrics'], data['avg'])}
        return metrics
    else:
        assert dry_run, f'HDRMetrics log file not found: {log}'
        return {}


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


def compute_sdr_metrics(a: AnchorTuple, vd: VariantData, dry_run=False):

    hdr_metrics_cfg = Path(os.getenv('HDRMETRICS_CFG_DIR', '/home/cfg')) / 'HDRMetrics_PSNR_MSSSIM.cfg'

    ref = a.reference
    dist = VideoSequence.from_sidecar_metadata( a.working_dir / f'{vd.variant_id}.yuv.json')
    conv = get_anchor_conversion_type(a)
    if conv[0] == Conversion.HDRCONVERT_8TO10BIT:
        ref = as_10bit_sequence(ref)
    elif conv[0] == Conversion.HDRCONVERT_10TO8BIT:
        # ref is 10 bit, encoded on 8bit 
        pass
    assert ref.path.exists(), f'reference sequence needs pre-processing - Not found: {ref.path}'

    if conv[1] == Conversion.HDRCONVERT_8TO10BIT:
        dist = as_10bit_sequence(dist)
        assert dist.path.exists(), f'bitstream needs pre-processing - Not found: {dist.path}'
    assert ref.bit_depth == dist.bit_depth
    
    log = dist.path.with_suffix('.metrics.log')
    d = hdrtools_metrics(ref, dist, log = log, dry_run = dry_run, cfg = hdr_metrics_cfg ) 
    res = VariantMetricSet({
        Metric.PSNR_Y: d["PSNR-Y"],
        Metric.PSNR_U: d["PSNR-U"],
        Metric.PSNR_V: d["PSNR-V"],
        Metric.MSSSIM: d["JMSSSIM-Y"]
    })
    res.compute_avg_psnr()
    return res
    

def compute_hdr_metrics_yuv(a: AnchorTuple, vd: VariantData, dry_run=False):
    hdr_metrics_cfg = Path(os.getenv('HDRMETRICS_CFG_DIR', '/home/cfg')) / 'HDRMetricsYUV_PQ10.cfg'
    dist = VideoSequence.from_sidecar_metadata(a.working_dir / f'{vd.variant_id}.yuv.json')
    log = dist.path.with_suffix('.metrics.log')
    d = hdrtools_metrics(a.reference, dist, log = log, dry_run = dry_run, cfg = hdr_metrics_cfg)
    res = VariantMetricSet({
        Metric.WTPSNR_Y: d["wtPSNR-Y"],
        Metric.WTPSNR_U: d["wtPSNR-U"],
        Metric.WTPSNR_V: d["wtPSNR-V"]
    })
    res.compute_avg_wpsnr()
    return res


def compute_hdr_metrics_exr(a: AnchorTuple, vd: VariantData, dry_run=False):
    hdr_metrics_cfg = Path(os.getenv('HDRMETRICS_CFG_DIR', '/home/cfg')) / 'HDRMetrics_DeltaE100.cfg'
 
    ref = as_exr2020_sequence(a.reference)
    fp = ref.path.with_suffix('.json')
    assert fp.exists(), f'reference sequence needs pre-processing - Not found: {fp}'
 
    yuv_dist = VideoSequence.from_sidecar_metadata( a.working_dir / f'{vd.variant_id}.yuv.json')
    dist = as_exr2020_sequence(yuv_dist)
    fp = dist.path.with_suffix('.json')
    assert fp.exists(), f'reference sequence needs pre-processing - Not found: {fp}'

    log = dist.path.with_suffix('.metrics.log')
    d = hdrtools_metrics(ref, dist, log = log, dry_run = dry_run, cfg = hdr_metrics_cfg) 
    result = {
        Metric.PSNR_L0100: d["PSNR_L0100"],
        Metric.DELTAE100: d["PSNR_DE0100"]
    }
    return VariantMetricSet(result)


def compute_vmaf_metrics(a: AnchorTuple, vd: VariantData, dry_run=False):

    ref = a.reference
    dist = VideoSequence.from_sidecar_metadata( a.working_dir / f'{vd.variant_id}.yuv.json')
    
    conv = get_anchor_conversion_type(a)
    if conv[0] == Conversion.HDRCONVERT_8TO10BIT:
        ref = as_10bit_sequence(ref)
    elif conv[0] == Conversion.HDRCONVERT_10TO8BIT:
        ref = as_8bit_sequence(ref)
    assert ref.path.exists(), f'reference sequence needs pre-processing - Not found: {ref.path}'
    
    if conv[1] == Conversion.HDRCONVERT_8TO10BIT:
        dist = as_10bit_sequence(dist)
        assert dist.path.exists(), f'bitstream needs pre-processing - Not found: {dist.path}'
    
    assert ref.bit_depth == dist.bit_depth

    vmaf_model = os.getenv('VMAF_MODEL', "version=vmaf_v0.6.1")
    d = vmaf_metrics(ref, dist, model=vmaf_model, dry_run=dry_run)
    if d is None:
        return VariantMetricSet({Metric.VMAF: 0})
    return VariantMetricSet({Metric.VMAF: d})


def compute_metrics(a: AnchorTuple, vd: VariantData, digits=3, dry_run=False) -> VariantMetricSet:

    metrics = None
    if vd.metrics is None:
        metrics = { m: None for m in a.get_metrics_set() }
    else:
        metrics = vd.metrics.copy()

    if not os.getenv('VCC_DISABLE_HDRMETRICS'):

        if a.reference.transfer_characteristics == TransferFunction.BT2020_PQ:
            metrics_hdr = compute_hdr_metrics_yuv(a, vd, dry_run=dry_run)
            metrics.update(metrics_hdr)
            metrics_exr = compute_hdr_metrics_exr(a, vd, dry_run=dry_run)
            metrics.update(metrics_exr)
        else:
            metrics_sdr = compute_sdr_metrics(a, vd, dry_run=dry_run)
            metrics.update(metrics_sdr)

    if not bool(os.getenv('VCC_DISABLE_VMAF')):

        if a.reference.transfer_characteristics != TransferFunction.BT2020_PQ:
            metrics_vmaf = compute_vmaf_metrics(a, vd, dry_run=dry_run)
            metrics.update(metrics_vmaf)


    enc = get_encoder(vd.generation['encoder'])
    bitstream = Path(vd.bitstream['URI']) if Path(vd.bitstream['URI']).exists() else  a.working_dir / Path(vd.bitstream['URI']).name

    if not dry_run:
        s = enc.bitstream_size(bitstream)
        metrics[Metric.BITRATE] = int(s * 8 / a.duration) * 1e-3
        
    # parse additional metrics from ENCODER log 
    metrics[Metric.BITRATELOG] = 0
    metrics[Metric.ENCODETIME] = 0
    if vd.generation and vd.generation.get('log-file', None):
        enc_log = a.working_dir / vd.generation['log-file']
        if enc_log.exists():
            encoder_metrics = enc.encoder_log_metrics(enc_log)
            if Metric.BITRATELOG in encoder_metrics:
                metrics[Metric.BITRATELOG] = float(encoder_metrics[Metric.BITRATELOG])
            if Metric.ENCODETIME in encoder_metrics:
                metrics[Metric.ENCODETIME] = float(encoder_metrics[Metric.ENCODETIME])
        else:
            print(f'#\tencoder log not found: {enc_log}')

    # parse additional metrics from DECODER log
    metrics[Metric.DECODETIME] = 0
    if vd.reconstruction and vd.reconstruction.get('log-file', None):
        dec_log = a.working_dir / vd.reconstruction['log-file']
        if dec_log.exists():
            decoder_metrics = enc.decoder_log_metrics(dec_log)
            if Metric.DECODETIME in decoder_metrics:
                metrics[Metric.DECODETIME] = float(decoder_metrics[Metric.DECODETIME])
        else:
            print(f'#\tdecoder log not found: {dec_log}')
    
    for k, v in metrics.items():
        metrics[k] = round(v, digits) if v is not None else 0

    return metrics



def anchor_metrics_csv_rows(a: AnchorTuple) -> List[Dict[str, Any]]:
    rows = []
    for qp, vd in load_variants(a):
        r = {} 
        r['parameter'] = qp
        if vd.metrics is None:
            logging.error(f'metrics not set on {vd.variant_id}')
        else:
            for k, v in vd.metrics.items():
                r[k.csv_key] = v
        rows.append(r)
    return rows


def anchor_metrics_to_csv(a: AnchorTuple, save = False, dst: Path = None):

    rows = anchor_metrics_csv_rows(a)
    if not save:
        return rows

    fieldnames = ['parameter'] + [m.csv_key for m in a.get_metrics_set()]

    if dst is None:
        dst = a.working_dir.parent / 'Metrics' / f'{a.working_dir.stem}.csv'
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)
    
    with open(dst, 'w', newline='') as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def anchor_metrics_from_csv(csv_path: Path) -> Dict[Metric, Any]:
    r = {}
    with open(csv_path, 'r', encoding = ENCODING ) as fo:
        csv_reader = csv.DictReader(fo)
        for row in csv_reader:
            qp = None
            metrics = {}
            for k, v in row.items():
                if k == 'parameter':
                    qp = v
                else:
                    m = Metric.from_csv_key(k)
                    assert m, f'Unknown metric key "{k}" used in csv metadata: {k}'
                    metrics[m] = float(v)
                    
            r[qp] = metrics
    return r


def load_csv_metrics(a:AnchorTuple, streams:List[VariantData]):
    """update variant metrics with csv data 
    """
    a_csv = a.working_dir.parent / 'Metrics' / (a.working_dir.with_suffix('.csv')).name
    metrics = anchor_metrics_from_csv(a_csv)
    for vd in streams:
        vd.metrics = metrics[vd.variant_qp]


@click.group()
@click.pass_context
@click.option('--working-dir', envvar='VCC_WORKING_DIR', required=True, type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True, readable=True), help="Directory containing bitstreams and pre-computed metrics, alternatively this can be set with VCC_WORKING_DIR environment variable." )
@click.option('-s/-c', required=True, default=True, show_default=True, help="Signals whether KEY is a sequence IDs, or an encoder config ID")
@click.argument('key', required=True)
def main(ctx, working_dir:str, s:bool, key:str):
    """
    export metrics to csv, if the scope is an encoder config ID, 
        all anchors for that config will be listed in the csv output. 
    """

    ctx.ensure_object(dict)
    ctx.obj['anchor_key'] = key

    working_dir = Path(working_dir)

    parts = key.split('-')
    if len(parts) != 3:
        ctx.get_help()
    
    bitstreams_dir = working_dir / BITSTREAMS_DIR
    sequences_dir = working_dir / SEQUENCES_DIR
    if s:
        ctx.obj['anchors'] = [AnchorTuple.load(key, bitstreams_dir, sequences_dir)]
    else:
        ctx.obj['anchors'] = AnchorTuple.iter_cfg_anchors(key, bitstreams_dir, sequences_dir)



@main.command()
@click.pass_context
def csv_metrics(ctx):
    """
    export metrics to csv, if the scope is an encoder config ID, 
        all anchors for that config will be listed in the csv output. 
    """

    if len(ctx.obj['anchors']) == 1:
        a = ctx.obj['anchors'][0]
        anchor_metrics_to_csv(a, save = True)
        return

    rows = []
    metrics = None
    for a in ctx.obj['anchors']:
        metrics = a.get_metrics_set()
        # compare.py currently needs individual CSV files 
        a_rows = anchor_metrics_to_csv(a, save = True)
        a_rows[0]['sequence'] = a.reference.sequence['Key']
        rows.append(a_rows)
    
    key = ctx.obj['anchor_key']
    dst = ctx.obj['anchors'][0].working_dir.parent / 'Metrics' / f'{key}.csv'
    fieldnames = ['sequence', 'parameter'] + [m.csv_key for m in metrics]

    with open(dst, 'w', newline='') as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerows(r)



def __verify_metrics(a:AnchorTuple, orig_dir:Path, metric_keys:Tuple[str], row_template={}):
    
    rows = []
    v = load_variants(a)
    v_orig = load_variants(a, orig_dir / a.anchor_key)
    
    for ((_, vd), (__, vd_orig)) in zip(v, v_orig):

        info_reconstruction = [("md5", vd.reconstruction["md5"])]
        orig_info_reconstruction = [("md5", vd_orig.reconstruction["md5"])]
        info_metrics = []
        orig_info_metrics = []

        reconstruction_ok = vd.reconstruction["md5"] == vd_orig.reconstruction["md5"]
        metrics_ok = True

        for key, val in vd.metrics.items():
            k = key.csv_key
            if len(metric_keys):
                if not k in metric_keys:
                    continue
                elif k in ('bitrate_log', 'encode_time', 'decode_time'):
                    continue
            val_orig = vd_orig.metrics[key]
            info_metrics.append((k, val))
            orig_info_metrics.append((k, val_orig))
            metrics_ok = metrics_ok and abs(val - val_orig) < 0.01

        row = {
            **row_template,
            "key": vd.variant_id,
            "file": "/".join(a.working_dir.parts[-4:]) + f'/{vd.variant_id}.json',
            "orig-date": vd.bitstream.get("date", None),
            "info-reconstruction": str(info_reconstruction),
            "orig-info-reconstruction": str(orig_info_reconstruction),
            "status-reconstruction": "successful" if reconstruction_ok else "failed",
            "info-metrics": str(info_metrics),
            "orig-info-metrics": str(orig_info_metrics),
            "status-metrics": "successful" if metrics_ok else "failed",
            # "type": "metrics",
        }

        rows.append(row)
    
    return rows

@main.command()
@click.pass_context
@click.option('--company')
@click.option('--email')
@click.option('--doc')
@click.option('--info')
@click.option('-o', '--orig-dir', required=True, type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True))
@click.option('-r', '--report', required=True, type=click.Path(exists=False, dir_okay=False, file_okay=True, writable=True))
@click.argument('metric_keys', nargs=-1, required=False)
def verify_metrics(ctx, company:str, email:str, doc:str, info:str, orig_dir:str, report:str, metric_keys:Tuple[str]):
    """
    verify that metrics in json and metrics in csv match.
    reference metrics are expected to be available as csv files in the {anchor}/Metrics folder
    verification metrics are expected to be available in the local bitstream .json metadata
    """
    rows = []
    row = {
        "company": company,
        "e-mail": email,
        "vdate": datetime.now().ctime(),
        "document": doc
    }
    fieldnames = [
        "key", "file", "orig-date", 
        "status-reconstruction", "info-reconstruction", "orig-info-reconstruction", 
        "status-metrics", "info-metrics", "orig-info-metrics",
        *row.keys()
    ]
    for a in ctx.obj['anchors']:
        a_report = __verify_metrics(a, Path(orig_dir), metric_keys, row)
        rows += a_report
    with open(report, 'w', newline='') as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@main.command()
@click.pass_context
def import_csv_metrics(ctx):
    for a in ctx.obj['anchors']:
        a_csv = a.working_dir.parent / 'Metrics' / (a.working_dir.with_suffix('.csv')).name
        csv_metrics = anchor_metrics_from_csv(a_csv)
        for variant_id, _ in a.iter_variants_params():
            vfp = a.working_dir / f'{variant_id}.json'
            vd = VariantData.load(vfp, variant_id)
            m = csv_metrics[variant_id]
            vd.metrics = m
            vd.save_as(a.working_dir / f'{variant_id}.json')



if __name__ == "__main__":
    main()