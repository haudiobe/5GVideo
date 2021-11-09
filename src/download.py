import sys
import asyncio
import functools

from pathlib import Path

from anchor import AnchorTupleCtx, VariantData, iter_variants, iter_ref_locations
from typing import Any, List
from utils import VideoSequence
import requests

from concurrent.futures import ThreadPoolExecutor

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

dl_executor = ThreadPoolExecutor(max_workers=10)
dl_errors: List[BaseException] = []
dl_count = 0
dl_complete = 0
dl_skipped = 0
dl_overwrite = True
dl_file_changed: List[str] = []

tasks: List[asyncio.Task] = []


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


def bytesize_match(local: Path, remote: str):
    r = requests.head(remote, allow_redirects=True)
    size = r.headers.get('content-length', -1)
    return int(size) == local.stat().st_size


def noop(*args, **kwargs):
    global dl_count, dl_skipped
    dl_count += 1
    dl_skipped += 1


def download_file(url: str, local_filename: Path, chunk_size=8192, try_harder=True) -> Any:
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
    except BaseException as e:
        err = None
        if (e.response.status_code == 404) and try_harder:
            url_parts = url.split('/')
            fix = str(url_parts[-1])
            url_parts[-1] = fix[0].lower() + fix[1:]
            url_fixed = '/'.join(url_parts)
            try:
                download_file(url_fixed, local_filename, try_harder=False)
                return
            except BaseException as e:
                err = e
        else:
            err = e
        
        if err is not None:
            dl_errors.append(e)
            return err

async def dl_if_not_exists(target: str, base_dir: Path, base_uri: str, overwrite=False, dry_run=False):
    global dl_executor
    loop = asyncio.get_running_loop()
    local = base_dir / target
    remote = base_uri + target
    exists = local.exists()
    task = download_file
    if exists and bytesize_match(local, remote):
        logging.info(f'[ok ] {target}')
        task = noop
    else:
        dl_file_changed.append(target)
        assert overwrite, "file exists but size doesn't match : overwriting is disabled"
    fn = functools.partial(task, remote, local)
    err = await loop.run_in_executor(dl_executor, fn)
    return err


def bitstream_uri(p: Path) -> Path:
    if p.exists():
        return p.parent / VariantData.load(p).bitstream['URI']
    else:
        raise FileNotFoundError(p)


async def dl_variant(variant: Path, base_dir: Path, base_uri: str, overwrite=False, dry_run=False) -> Any:
    err = await dl_if_not_exists(
        str(variant.relative_to(base_dir)), 
        base_dir, 
        base_uri, 
        overwrite=overwrite, 
        dry_run=dry_run)
    if not err:
        # err = await dl_if_not_exists(
        #     str(bitstream_uri(variant).relative_to(base_dir)), 
        #     base_dir, 
        #     base_uri, 
        #     overwrite=overwrite, 
        #     dry_run=dry_run)
        return None
    return err


async def download_variant_data(ctx: AnchorTupleCtx, base_uri: str):

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
            dry_run=ctx.dry_run)
        dl_all_data.append(pending)

        # variant bitstreams
        for variant_path, _ in iter_variants(a):
            if ctx.dry_run and not variant_path.exists():
                continue
            pending = dl_variant(variant_path, base_dir, base_uri, overwrite=dl_overwrite, dry_run=ctx.dry_run)
            dl_all_data.append(pending)

    await asyncio.gather(*dl_all_data)


async def download_scenario_streams(scenario: str, codecs: List[str], local: Path, remote: str, dry_run=False):

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
                logging.info('\t\'streams.csv\' not downloaded - processing individual streams is disabled')
            else:
                ctx = AnchorTupleCtx(scenario_dir=local / scenario_codec_dir, dry_run=dry_run)
                await download_variant_data(ctx, remote)

    await asyncio.gather(*[dl_codec(c) for c in codecs])


async def download_reference_sequence(meta_location: str, base_dir: Path, base_uri: str, overwrite=False, dry_run=True):
    err = await dl_if_not_exists(meta_location, base_dir, base_uri, overwrite=overwrite, dry_run=dry_run)
    if not err:
        vs = VideoSequence.from_sidecar_metadata(base_dir / meta_location)
        stem = vs.path.stem
        sequence_location = f'ReferenceSequences/{vs.path.parent.stem}/{stem}{vs.path.suffix}'
        await dl_if_not_exists(sequence_location, base_dir, base_uri, overwrite=overwrite, dry_run=dry_run)


async def download_reference_sequences(scenario: str, base_dir: Path, base_uri: str, dry_run=False):

    reference_sequences_csv = f'Bitstreams/{scenario}/reference-sequence.csv'
    
    await dl_if_not_exists(
        reference_sequences_csv, 
        base_dir, 
        base_uri, 
        overwrite=dl_overwrite, 
        dry_run=dry_run
    )

    if dry_run and not (base_dir / reference_sequences_csv).exists():
        logging.info('\t\'reference-sequence.csv\' not downloaded - processing individual sequences is disabled')
    
    else:
        dl_all_sequences = []
        
        for meta in iter_ref_locations(base_dir / reference_sequences_csv):
            meta_location = f'ReferenceSequences/{meta}'
            pending = download_reference_sequence(meta_location, base_dir, base_uri, overwrite=dl_overwrite, dry_run=dry_run)
            dl_all_sequences.append(pending)
        
        await asyncio.gather(*dl_all_sequences)


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

    for err in dl_errors:
        print(err)


if __name__ == "__main__":
    asyncio.run(main())
