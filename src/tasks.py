#!/usr/bin/python3

from pathlib import Path
import os
from anchor import AnchorTuple, VariantData, iter_variants
from celery import Celery
from celery.utils.log import get_task_logger
from encoders import get_encoder
from anchor import AnchorTuple
from metrics import compute_metrics
from conversion import get_anchor_conversion_type, Conversion, convert_sequence
from utils import VideoSequence
import click

logger = get_task_logger(__name__)

BROKER_URL = os.getenv('BROKER_URL', 'redis://127.0.0.1:6379/0')
BACKEND_URL = os.getenv('BACKEND_URL', None)
VCC_WORKING_DIR = Path(os.getenv('VCC_WORKING_DIR', '/data'))

app = Celery('tasks', broker=BROKER_URL, backend=BACKEND_URL)

@app.task
def encode_variant_task(anchor_key:str, variant_id:str, variant_cli:str, dry_run=False):
    a = AnchorTuple.load(anchor_key, root_dir=VCC_WORKING_DIR)
    a.dry_run = dry_run
    vd = get_encoder(a.encoder_id).encode_variant(a, variant_id, variant_cli)
    vd.save_as(a.working_dir / f'{variant_id}.json')

@app.task
def decode_variant_task(variant_id:str, dry_run=False):
    anchor_key = '-'.join(variant_id.split('-')[0:3])
    a = AnchorTuple.load(anchor_key, root_dir=VCC_WORKING_DIR)
    a.dry_run = dry_run
    vp = a.working_dir / f'{variant_id}.json'
    vd = None
    try:
        vd = VariantData.load(vp)
    except FileNotFoundError:
        b = {'key': variant_id, 'URI': f'{variant_id}.bin'}
        vd = VariantData(bitstream=b)
        vd.save_as(vp)
        logger.warn(f'creating missing bitstream metadata: {vp}')
    finally:
        get_encoder(a.encoder_id).decode_variant(a, vd)


# @app.task
def encode_task(anchor_key, variant_id=None, dry_run=False):
    a = AnchorTuple.load(anchor_key, root_dir=VCC_WORKING_DIR)
    for vid, vargs in a.iter_variants_args():
        if (variant_id is not None) and (variant_id != vid):
                continue
        # enqueue each variant as async task
        encode_variant_task.delay(anchor_key, vid, vargs)

# @app.task
def decode_task(anchor_key, variant_id=None, dry_run=False):

    if variant_id:
        decode_variant_task.delay(variant_id)
        return 

    a = AnchorTuple.load(anchor_key, root_dir=VCC_WORKING_DIR)
    for vid, vargs in a.iter_variants_args():
        if (variant_id is not None) and (variant_id != vid):
                continue
        # enqueue each variant as async task
        decode_variant_task.delay(vid)


@app.task
def convert_sequence_task(conv:str, vs:str):
    conv = int(conv)
    if conv == Conversion.NONE.value:
        conv = Conversion.NONE
    elif conv == Conversion.HDRCONVERT_8TO10BIT.value:
        conv = Conversion.HDRCONVERT_8TO10BIT
    elif conv == Conversion.HDRCONVERT_YCBR420TOEXR2020.value:
        conv = Conversion.HDRCONVERT_YCBR420TOEXR2020
    vs = VideoSequence.from_sidecar_metadata(vs)
    convert_sequence(conv, vs)

# @app.task
def convert_task(anchor_key, variant_id=None, dry_run=False):
    a = AnchorTuple.load(anchor_key, root_dir=VCC_WORKING_DIR)
    conv = get_anchor_conversion_type(a)

    if conv == Conversion.NONE:
        return
    
    elif conv == Conversion.HDRCONVERT_8TO10BIT:
        convert_sequence_task.delay(conv.value, str(a.reference.path.with_suffix('.json')))

    elif conv == Conversion.HDRCONVERT_YCBR420TOEXR2020:
        convert_sequence_task.delay(conv.value, str(a.reference.path.with_suffix('.json')))
        for _, vd in iter_variants(a):
            if (vd is None) or ((variant_id is not None) and (variant_id != vd.variant_id)):
                continue
            vs = a.working_dir / f'{vd.variant_id}.yuv.json'
            convert_sequence_task.delay(conv.value, str(vs))

@app.task
def compute_variant_metrics_task(anchor_key, vfp, dry_run=False):
    a = AnchorTuple.load(anchor_key, root_dir=VCC_WORKING_DIR)
    a.dry_run = dry_run
    vd = VariantData.load(Path(vfp))
    vd.metrics = compute_metrics(a, vd)
    vd.dumps(vfp)

# @app.task
def metrics_task(anchor_key, variant_id=None, dry_run=False):
    a = AnchorTuple.load(anchor_key, root_dir=VCC_WORKING_DIR)
    for vfp, vd in iter_variants(a):
        if (vd is not None) and ((variant_id is not None) and (variant_id != vd.variant_id)):
                continue
        compute_variant_metrics_task.delay(anchor_key, str(vfp))

def parse_tasks(cmd):
    if cmd == 'encode':
        return encode_task
    elif cmd == 'decode':
        return decode_task
    elif cmd == 'convert':
        return convert_task
    elif cmd == 'metrics':
        return metrics_task
    raise ValueError(f'invalid command: {cmd}')


@click.command()
@click.option('--dry-run', required=False, default=False)
@click.option('-s/-c', required=True, default=True, show_default=True, help="signals whether KEY is a sequence IDs, or an encoder config ID")
@click.argument('key', nargs=1, required=True)
@click.argument('cmd', nargs=1, required=True)
def main(dry_run, s, key, cmd):
    """
    process anchors identiifed through KEY task CMD
    """
    fn = parse_tasks(cmd)
    
    if s:
        parts = key.split('-')
        variant_id=None
        if len(parts) == 4:
            variant_id = '-'.join(parts)
            key = '-'.join(parts[0:3])
        elif len(parts) != 3:
            raise ValueError(f'invalid anchor key {key}')
        fn(key, variant_id=variant_id, dry_run=dry_run)
    
    else:
        anchors = AnchorTuple.iter_anchors(cfg_keys=[key])
        for a in anchors:
            fn(a.anchor_key, dry_run=dry_run)


if __name__ == "__main__":
    main()