#!.venv/bin/python3

import argparse, re, sys, json, threading, time, asyncio, functools
from pathlib import Path
from encoders import parse_encoding_bitdepth

from anchor import AnchorTuple, VariantData, reference_sequences_dict, iter_anchors, iter_variants, iter_ref_locations
from typing import Tuple, List, Iterable, Callable, Dict
from utils import VideoSequence
import requests

from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def __fix_sequence_name(name) -> str:
    # @FIXME: the file names in the metadata do not match the actual filenames
    if name == "Mission-Control":
        return "mission-control"
    elif name == "Baolei-Man":
        return "baolei-man"
    elif name == "MineCraft":
        return "MINECRAFT"
    elif name == "CS-GO":
        return "cs-go"
    elif name == "StarCraft":
        return "starcraft"
    elif name == "world-of-warCraft":
        return "World-of-WarCraft"
    return name

dl_executor = ThreadPoolExecutor(max_workers=10)
dl_errors = []
dl_count = 0
dl_complete = 0
dl_skipped = 0
dl_overwrite = True
dl_file_changed = []

tasks = []

async def dl_print_status():
    global tasks
    while True:
        pending = [t for t in tasks if not t.done()]
        if len(pending) == 1:
            break
        err_count = len(dl_errors)
        dl_done = dl_count - dl_complete - dl_skipped - err_count
        msg = f'downloading: {dl_done}/{dl_count} - errors: {err_count} - skipped: {dl_skipped} - file changed: {len(dl_file_changed)}\r'
        sys.stdout.write(msg)
        sys.stdout.flush()
        await asyncio.sleep(0)


class AnchorTupleCtx:

    @classmethod
    def parse_args(cls, parser:argparse.ArgumentParser=None) -> 'AnchorTool':
        if parser == None:
            parser = argparse.ArgumentParser()
        parser.add_argument('--scenario_dir', required=True, type=str, help='scenario directory, eg. Scenario-3-Screen/265')
        parser.add_argument('-k', '--key', required=False, type=str, default=None, help='an optional anchor key')
        parser.add_argument('-a', '--anchors-list', required=False, type=str, default='./streams.csv', help='streams.csv file containing the list of anchors for a scenario')
        parser.add_argument('-s','--sequences-list', required=False, type=str, default='../reference-sequence.csv', help='sequences.csv file containing the list of reference raw sequences')
        parser.add_argument('--sequences_dir', required=False, type=str, help='the directory containing the reference sequences')
        parser.add_argument('--dry-run', action='store_true', default=False)
        parser.add_argument('--dl-streams', action='store_true', default=False, help='download streams')
        parser.add_argument('--dl-ref-sequences', action='store_true', default=False, help='download reference sequences')
        cli_args = parser.parse_args()
        return cls(cli_args=cli_args)
    
    def __init__(self, cli_args=None, **kwargs):
        
        if cli_args:
            scenario_dir = Path(cli_args.scenario_dir)
            if not scenario_dir.exists():
                scenario_dir.mkdir(parents=True)
            
            references_csv = scenario_dir / Path(cli_args.sequences_list)
            if not references_csv.exists():
                logging.info(f'reference-sequence.csv not found {references_csv}')

            anchors_csv = scenario_dir / cli_args.anchors_list
            if not anchors_csv.exists():
                logging.info(f'streams.csv list not found {anchors_csv}')
                        
            sequences_dir = scenario_dir / '../../../ReferenceSequences'
            if cli_args.sequences_dir != None:
                sequences_dir = Path(cli_args.sequences_dir).resolve()
                assert sequences_dir.is_dir(), f'invalid sequence directory {sequences_dir}'

            self.scenario_dir = scenario_dir
            self.anchors_csv = anchors_csv
            self.references_csv = references_csv
            self.sequences_dir = sequences_dir
            self.dry_run = cli_args.dry_run

            self.dl_streams = cli_args.dl_streams
            self.dl_ref_sequences = not self.dl_streams
            self.key = cli_args.key

        else:
            self.scenario_dir = Path(kwargs['scenario_dir'])
            if 'anchors_csv' in kwargs:
                self.anchors_csv = kwargs['anchors_csv']
            else:
                self.anchors_csv = self.scenario_dir / 'streams.csv'

            if 'references_csv' in kwargs:
                self.references_csv = kwargs['references_csv']
            else:
                self.references_csv = self.scenario_dir.parent / 'reference-sequence.csv'

            if 'sequences_dir' in kwargs:
                self.sequences_dir = kwargs['sequences_dir']
            else:
                self.sequences_dir = self.base_dir() / 'ReferenceSequences'

            if 'dry_run' in kwargs:
                self.dry_run = bool(kwargs['dry_run'])
            else:
                self.dry_run = False

    def base_dir(self) -> Path:
        # scenario_dir is $base_dir/Bitstreams/$scenario/$codec/
        return self.scenario_dir.parent.parent.parent

    def iter_ref_sequences(self) -> Tuple[str, VideoSequence]:
        for key, video_sequence in reference_sequences_dict(self.references_csv, self.sequences_dir).items():
            yield key, video_sequence
    
    def iter_anchors(self, keys:List[str]=None):
        if (keys == None) and hasattr(self, "key"):
            keys = [self.key]
        refs = reference_sequences_dict(self.references_csv, self.sequences_dir)
        return iter_anchors(self.anchors_csv, refs, self.scenario_dir, keys=keys, raise_if_not_exists=False)

def bytesize_match(local:Path, remote:str):
    r = requests.head(remote, allow_redirects=True)
    size = r.headers.get('content-length', -1)
    return int(size) == local.stat().st_size

def noop(*args, **kwargs):
    global dl_count, dl_skipped
    dl_count += 1
    dl_skipped += 1

def download_file(url:str, local_filename:Path, chunk_size=8192) -> BaseException:
    try:
        # logging.info(f'[downloading] {url}')
        global dl_count, dl_complete
        dl_count += 1
        local_filename.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size): 
                    f.write(chunk)
            dl_complete += 1
        # logging.info(f'[complete] {url}')
    except BaseException as e:
        dl_errors.append(e)
        return e


async def dl_if_not_exists(target:str, base_dir:Path, base_uri:str, overwrite=False, dry_run=False):
    global dl_executor
    loop = asyncio.get_running_loop()
    local = base_dir / target
    remote = base_uri + target
    exists = local.exists()
    task = download_file
    if exists and bytesize_match(local, remote):
        logging.info(f'[ok  ] {target}')
        task = noop
    else:
        dl_file_changed.append(target)
        assert overwrite, "file exists but size doesn't match : overwriting is disabled"
    fn = functools.partial(task, remote, local)
    err = await loop.run_in_executor(dl_executor, fn)
    return err


def bitstream_uri(p:Path) -> str:
    if p.exists():
        return p.parent / VariantData.load( p ).bitstream['URI']


async def dl_variant(variant:str, base_dir:Path, base_uri:str, overwrite=False, dry_run=False) -> BaseException:
    err = await dl_if_not_exists(
            str(variant.relative_to(base_dir)), 
            base_dir, 
            base_uri, 
            overwrite=overwrite, 
            dry_run=dry_run
        )
    if not err:
        err = await dl_if_not_exists(
                str(bitstream_uri(variant).relative_to(base_dir)), 
                base_dir, 
                base_uri, 
                overwrite=overwrite, 
                dry_run=dry_run
            )
    return err


async def download_variant_data(ctx:AnchorTupleCtx, base_uri:str):

    base_dir = ctx.base_dir()
    dl_all_data = []

    for a in ctx.iter_anchors():

        assert a.reference.path.exists(), 'reference sequence not found'

        # anchor-key directory
        if not a.working_dir.exists():
            a.working_dir.mkdir(parents=True, exist_ok=True)

        # variant encoder base config
        enc_cfg = a.encoder_cfg.relative_to(base_dir)
        pending = dl_if_not_exists(
            str(enc_cfg), 
            base_dir, 
            base_uri, 
            overwrite=dl_overwrite, 
            dry_run=ctx.dry_run )
        dl_all_data.append(pending)

        # variant bitstreams
        for variant, _ in iter_variants(a):
            if ctx.dry_run and not variant.exists():
                # meta will not be downloaded in dry_run
                continue
            pending = dl_variant(variant, base_dir, base_uri, overwrite=dl_overwrite, dry_run=ctx.dry_run)
            dl_all_data.append(pending)

    await asyncio.gather( *dl_all_data )


async def download_scenario_streams(scenario:str, codecs:List[str], local:Path, remote:str, dry_run=False):

    async def dl_codec(codec):
        scenario_codec_dir = f'Bitstreams/{scenario}/{codec}/' 
        streams = f'{scenario_codec_dir}streams.csv'
        err = await dl_if_not_exists(
            streams, 
            local, 
            remote, 
            overwrite=dl_overwrite, 
            dry_run=dry_run
        )
        if not err:
            if dry_run and not (local / streams).exists():
                logging.info(f'\t\'streams.csv\' not downloaded - processing individual streams is disabled')
            else:
                ctx = AnchorTupleCtx(scenario_dir=local/scenario_codec_dir, dry_run=dry_run)
                await download_variant_data(ctx, remote)

    await asyncio.gather(*[dl_codec(c) for c in codecs])


async def download_reference_sequence(meta_location:str, base_dir:Path, base_uri:str, overwrite=False, dry_run=True):
    err = await dl_if_not_exists(meta_location, base_dir, base_uri, overwrite=overwrite, dry_run=dry_run)
    if not err:
        vs = VideoSequence.from_sidecar_metadata( base_dir / meta_location )
        stem = __fix_sequence_name(vs.path.stem)
        sequence_location = f'ReferenceSequences/{vs.path.parent.stem}/{stem}{vs.path.suffix}'
        await dl_if_not_exists(sequence_location, base_dir, base_uri, overwrite=overwrite, dry_run=dry_run)

async def download_reference_sequences(scenario:str, base_dir:Path, base_uri:str, dry_run=False):

    reference_sequences_csv = f'Bitstreams/{scenario}/reference-sequence.csv'
    
    await dl_if_not_exists(
        reference_sequences_csv, 
        base_dir, 
        base_uri, 
        overwrite=dl_overwrite, 
        dry_run=dry_run
    )

    if dry_run and not (base_dir / reference_sequences_csv).exists():
        logging.info(f'\t\'reference-sequence.csv\' not downloaded - processing individual sequences is disabled')
    
    else:
        dl_all_sequences = []
        
        for meta in iter_ref_locations(base_dir/reference_sequences_csv):
            meta_location =  f'ReferenceSequences/{meta}'
            pending = download_reference_sequence(meta_location, base_dir, base_uri, overwrite=dl_overwrite, dry_run=dry_run)
            dl_all_sequences.append(pending)
        
        await asyncio.gather( *dl_all_sequences )

async def main():

    global tasks
    base_uri = 'https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/'

    ctx = AnchorTupleCtx.parse_args()
    assert len(ctx.scenario_dir.parts) >= 2, f'invalid scenario directory: {ctx.scenario_dir}'
    scenario = ctx.scenario_dir.parts[-2]
    codec = ctx.scenario_dir.parts[-1]
    base_dir = ctx.scenario_dir.parent.parent.parent
    
    status = dl_print_status()
    tasks = [asyncio.create_task(status)]

    if ctx.dl_ref_sequences:
        dl_ref_sequences = download_reference_sequences(scenario, base_dir, base_uri, dry_run=ctx.dry_run)
        tasks.append(asyncio.create_task(dl_ref_sequences))

    elif ctx.dl_streams:
        dl_streams = download_scenario_streams(scenario, [codec], base_dir, base_uri, dry_run=ctx.dry_run)
        tasks.append(asyncio.create_task(dl_streams))
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
