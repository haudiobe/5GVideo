import logging
import click
from celery import Celery
from celery.utils.log import get_task_logger
import os
from pathlib import Path

from anchor import AnchorTuple, VariantData, load_variants
from constants import BITSTREAMS_DIR, SEQUENCES_DIR
from conversion import Conversion, get_anchor_conversion_type, convert_sequence
from encoders import get_encoder
from metrics import compute_metrics
from sequences import VideoSequence


logger = get_task_logger(__name__)

BROKER_URL = os.getenv('BROKER_URL', 'redis://0.0.0.0:6379/0')
BACKEND_URL = os.getenv('BACKEND_URL', 'redis://0.0.0.0:6379/1')
VCC_WORKING_DIR = Path(os.getenv('VCC_WORKING_DIR', '/data'))

app = Celery('vcc', broker=BROKER_URL, backend=BACKEND_URL)
# visibility_timeout should be more than the expected maximum tasks time to avoid automatic re-submission of tasks 
app.conf.broker_transport_options.update({ 'visibility_timeout': 360000 })
app.conf.worker_send_task_events = True
app.conf.task_track_started = True
app.conf.task_send_sent_event = True
app.conf.task_publish_retry = False


@app.task
def encode_variant_task(anchor_key:str, variant_id:str, variant_qp:str, dry_run=False):
    bitstreams_dir = VCC_WORKING_DIR / BITSTREAMS_DIR
    sequences_dir = VCC_WORKING_DIR / SEQUENCES_DIR
    a = AnchorTuple.load(anchor_key, bitstreams_dir, sequences_dir)
    vd = get_encoder(a.encoder_id).encode_variant(a, variant_id, variant_qp, dry_run = dry_run, dst_dir = a.working_dir)
    vd.save_as(a.working_dir / f'{variant_id}.json')


@app.task
def decode_variant_task(anchor_key:str, variant_id:str, dry_run=False):
    bitstreams_dir = VCC_WORKING_DIR / BITSTREAMS_DIR
    sequences_dir = VCC_WORKING_DIR / SEQUENCES_DIR
    a = AnchorTuple.load(anchor_key, bitstreams_dir, sequences_dir)
    vp = a.working_dir / f'{variant_id}.json'
    vd = None
    try:
        vd = VariantData.load(vp, variant_id)
    except FileNotFoundError:
        b = {'key': variant_id, 'URI': f'{variant_id}.bin'}
        vd = VariantData(bitstream=b)
        vd.save_as(vp)
        logger.warn(f'creating missing bitstream metadata: {vp}')
    finally:
        rec = get_encoder(a.encoder_id).decode_variant(a, vd, dry_run=dry_run)
        rec.update(vp, vd)


@app.task
def convert_sequence_task(conv:str, vs:str, dry_run=False):
    conv = int(conv)
    if conv == Conversion.NONE.value:
        conv = Conversion.NONE
    elif conv == Conversion.HDRCONVERT_8TO10BIT.value:
        conv = Conversion.HDRCONVERT_8TO10BIT
    elif conv == Conversion.HDRCONVERT_10TO8BIT.value:
        conv = Conversion.HDRCONVERT_10TO8BIT
    elif conv == Conversion.HDRCONVERT_YCBR420TOEXR2020.value:
        conv = Conversion.HDRCONVERT_YCBR420TOEXR2020
    vs = VideoSequence.from_sidecar_metadata(vs)
    convert_sequence(conv, vs, dry_run=dry_run)


@app.task
def compute_variant_metrics_task(anchor_key:str, variant_id:str, dry_run=False):
    bitstreams_dir = VCC_WORKING_DIR / BITSTREAMS_DIR
    sequences_dir = VCC_WORKING_DIR / SEQUENCES_DIR
    a = AnchorTuple.load(anchor_key, bitstreams_dir, sequences_dir)
    sequence_uri = a.reference.sequence.get('URI', None)
    sequence_key = a.reference.sequence.get('Key', None)
    assert sequence_uri is not None and Path(sequence_uri).exists, f'Reference sequence {sequence_key} not found in {sequences_dir}'
    vfp = a.working_dir / f'{variant_id}.json'
    vd = VariantData.load(vfp, variant_id)
    vd.metrics = compute_metrics(a, vd, dry_run=dry_run)
    vd.save_as(vfp)


@click.group()
@click.pass_context
@click.option('--queue/--no-queue', is_flag=True, required=False, default=False, help='Process sequentialy (--no-queue, default), or use distributed processing with a Celery backend (--queue).')
@click.option('--dry-run', is_flag=True, required=False, default=False, help='Print subprocess commands (eg. encoder, metrics computation, sequence conversion) to stdout instead of actually running them. All other processing steps are be performed as ususal (eg. log parsing, .json or .csv file generation/update) or fail when files they depend upon are missing.')
@click.option('-s/-c', default=True, help="Signals whether KEY is a sequence IDs (-s, default), or an encoder config ID (-c)")
@click.argument('key', nargs=1, required=True)
def main(ctx, queue:bool, dry_run:bool, s:bool, key:str):
    """
    \b
    to get detailed usage of a specific command, use:
    \b
        vcc.py COMMAND help
    """
    ctx.ensure_object(dict)

    parts = key.split('-')
    ctx.obj['variant_id'] = None
    if len(parts) == 4:
        ctx.obj['variant_id'] = '-'.join(parts)
        key = '-'.join(parts[0:3])
    elif len(parts) != 3:
        ctx.obj['command_key'] = key
        ctx.obj['anchors'] = []
        ctx.obj['dry_run'] = True
        ctx.obj['queue'] = False
        return

    ctx.obj['dry_run'] = dry_run
    ctx.obj['queue'] = queue

    bitstreams_dir = VCC_WORKING_DIR / BITSTREAMS_DIR
    sequences_dir = VCC_WORKING_DIR / SEQUENCES_DIR
    if s:
        ctx.obj['anchors'] = [AnchorTuple.load(key, bitstreams_dir, sequences_dir)]
    else:
        ctx.obj['anchors'] = AnchorTuple.iter_cfg_anchors(key, bitstreams_dir, sequences_dir)



@main.command(add_help_option=False)
@click.pass_context
def help(ctx):
    """
    Get detailed help on a command.
    \b
    Use --help for global options
    """
    key = None
    try:
        key = ctx.obj['command_key']
        cmd = main.commands[key]
        h = cmd.get_help(click.Context(cmd, parent=ctx.parent, info_name=key))
        click.echo(h)
        return
    
    except KeyError:
        pass
    
    h = main.get_help(ctx)
    click.echo(h)
    if key != 'help':
        msg = f"\nhelp error: no such command '{key}'\n"
        click.echo(msg)
    


@main.command(add_help_option=False)
@click.pass_context
def decode(ctx):
    """
    \b
    Reconstruction and json metadatas.
    \b
    """
    dry_run = ctx.obj['dry_run']
    variant_id = ctx.obj['variant_id']
    queue = ctx.obj['queue']

    for a in ctx.obj['anchors']:
        for vp, vd in load_variants(a):
            if (vd is None) or ((variant_id is not None) and (variant_id != vd.variant_id)):
                    continue
            if not queue:
                rec = get_encoder(a.encoder_id).decode_variant(a, vd, dry_run = dry_run)
                rec.update(vp, vd)
            else:
                decode_variant_task.delay(a.anchor_key, vd.variant_id, dry_run = dry_run)


@main.command(add_help_option=False)
@click.pass_context
def encode(ctx):
    """
    \b
    Anchor/test bitstreams, with reconstruction and json metadatas. md5 is computed for all bitstreams and reconstructions.
    \b
    \b
    The default encoder executable name is used, by default.
    \b
    It can be specified throuh environment variables: JM, HM, VTM, ETM, ...
    \b
    Specific encoder versions are supported through per version environment variables.
    for instance: JM19_0_ENCODER, HM16_24_ENCODER, VTM_13_0_ENCODER, ETM_7_5_ENCODER, ...
    """
    dry_run = ctx.obj['dry_run']
    variant_id = ctx.obj['variant_id']
    queue = ctx.obj['queue']
    
    for a in ctx.obj['anchors']:
        for vid, vqp in a.iter_variants_params():
            if (variant_id is not None) and (variant_id != vid):
                    continue
            if not queue:
                vd = get_encoder(a.encoder_id).encode_variant(a, vid, vqp, dry_run = dry_run, dst_dir = a.working_dir)
                vd.save_as(a.working_dir / f'{variant_id}.json')

            else:
                encode_variant_task.delay(a.anchor_key, vid, vqp, dry_run = dry_run)


@main.command(add_help_option=False)
@click.pass_context
@click.option('--reconstructions/--reference',  is_flag=True, required=False, default=True, help="process reference sequences (default), or reconstructed")
def convert(ctx, reconstructions):
    """
    \b
    Format conversion of reference and reconstructed video sequences.
    \b
    the HDRCONVERT_TOOL environment variable should provide the executable path.
    """
    dry_run = ctx.obj['dry_run']
    variant_id = ctx.obj['variant_id']
    queue = ctx.obj['queue']

    for a in ctx.obj['anchors']:
        assert a.reference != None
        conv = get_anchor_conversion_type(a)

        if reconstructions:
            if conv[1] == Conversion.NONE:
                return
            for _, vd in load_variants(a):
                if (vd is None) or ((variant_id is not None) and (variant_id != vd.variant_id)):
                    continue
                vs = a.working_dir / f'{vd.variant_id}.yuv.json'
                if not queue:
                    convert_sequence(conv[1], VideoSequence.from_sidecar_metadata(str(vs)), dry_run=dry_run)
                else:
                    convert_sequence_task.delay(conv[1].value, str(vs), dry_run=dry_run)
        else:
            if conv[0] == Conversion.NONE:
                return
            if not queue:
                convert_sequence(conv[0], a.reference, dry_run=dry_run)
            else:
                convert_sequence_task.delay(conv[0].value, str(a.reference.path.with_suffix('.json')), dry_run=dry_run)



@main.command(add_help_option=False)
@click.pass_context
def metrics(ctx):

    dry_run = ctx.obj['dry_run']
    variant_id = ctx.obj['variant_id']
    queue = ctx.obj['queue']

    for a in ctx.obj['anchors']:
        match_found = variant_id is None
        for _, vd in load_variants(a, a.working_dir):
            if vd is None:
                continue
            if ((variant_id is not None) and (variant_id != vd.variant_id)):
                continue
            vfp = a.working_dir / f'{vd.variant_id}.json'
            match_found = True
            if not queue:
                try:
                    vd.metrics = compute_metrics(a, vd, dry_run=dry_run)
                    vd.save_as(vfp)
                except BaseException:
                    logging.error('='*32)
                    logging.error(vd.variant_id)
                    logging.error('='*32)
            else:
                compute_variant_metrics_task.delay(a.anchor_key, vd.variant_id, dry_run=dry_run)
        assert match_found, f'{variant_id} not found'



if __name__ == "__main__":
    main()