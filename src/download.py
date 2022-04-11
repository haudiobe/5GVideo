import click
import logging
import requests
import json
from pathlib import Path
from urllib.parse import urlparse, urljoin
from anchor import iter_anchors, iter_ref_sequence_locations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

DL_JOBS = []
CANCELED = False

def __download_file(url: str, local_filename: Path, chunk_size=8192) -> Any:
    global CANCELED
    if CANCELED:
        return
    
    local_filename.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            logging.error(e)
            return
        
        logging.info(f'downloading: {url}')
        with open(local_filename, 'wb') as f:
            try:
                for chunk in r.iter_content(chunk_size=chunk_size): 
                    f.write(chunk)

            except KeyboardInterrupt as e:
                logging.info(f'canceled: {url}')
                CANCELED = True


def __download_all(dl_pool_size=5) -> Any:
    global DL_JOBS
    with ThreadPoolExecutor(dl_pool_size) as executor:
        futures = [executor.submit(__download_file, *args) for args in DL_JOBS]
        try:
            for future in as_completed(futures):
                _ = future.result()

        except KeyboardInterrupt as e:
            logging.error(e)
            raise
    

def download_file(url: str, local_filename: Path, overwrite=False, dry_run=False, queue=True) -> Any:
    global DL_JOBS, CANCELED
    try:
        if dry_run:
            with requests.head(url) as r:
                try:
                    r.raise_for_status()
                except requests.HTTPError as e:
                    logging.error(e)
            return
        if local_filename.exists():
            if not overwrite:
                logging.info(f'skipped existing file: {local_filename}')
                return
            else:
                logging.warning(f'overwriting: {local_filename}')
        if queue:
            DL_JOBS.append((url, local_filename))
        else:
            __download_file(url, local_filename)
    except KeyboardInterrupt:
        CANCELED = True
        raise


def download_reference_sequence(ref_meta_location:str, remote_sequences_dir:str, local_sequences_dir:Path, overwrite=False, dry_run=False, preflight=False) -> Any:

    local_sequences_dir.mkdir(exist_ok=True, parents=True)
    local_sequence_metadata = Path(local_sequences_dir / ref_meta_location)
    url = urljoin(remote_sequences_dir, ref_meta_location)
    download_file(url, local_sequence_metadata, overwrite=overwrite, dry_run=dry_run, queue=False)
    if dry_run:
        return
    elif preflight and not local_sequence_metadata.exists():
        # logging.warning(f'Failed to download {local_sequence_metadata}')
        return
    
    with open(local_sequence_metadata, 'r') as reader:
        data = json.load(reader)
        url = urlparse(data['Sequence']['URI'])
        if url.scheme == '':
            logging.warning(f'{local_sequence_metadata}\n\tcontains an invalid URI: "{url.geturl()}"')
            return
        stem = '/'.join(url.path.split('/')[-2:])
        raw_video_local = local_sequences_dir / stem
        o = (False if (dry_run or preflight) else overwrite)
        download_file(url.geturl(), raw_video_local, overwrite=o, dry_run=preflight, queue=True)


@click.group()
@click.pass_context
@click.option('--dry-run', is_flag=True, required=False, default=False)
@click.option('--overwrite', is_flag=True, required=False, default=False)
@click.option('--pool-size', required=False, default=5, type=click.IntRange(min=1))
@click.option('--verbose/--quiet', is_flag=True, required=False, default=True)
def cli(ctx, dry_run:bool, overwrite:bool, pool_size:int, verbose:bool):
    """
    \b
    download.py sequences ./reference-sequences.csv ../../ReferenceSequences
    
    \b
    download.py streams http://hosted/streams.csv .
    
    """
    lvl = logging.INFO if verbose else logging.WARN
    logging.basicConfig(format='%(message)s', level=lvl)
    ctx.ensure_object(dict)
    ctx.obj['dry_run'] = dry_run
    ctx.obj['overwrite'] = overwrite
    ctx.obj['pool_size'] = pool_size


@cli.command()
@click.pass_context
@click.option('--json/--no-json', is_flag=True, default=True, help='./*/*.json bitstream metadata')
@click.option('--bitstream/--no-bitstream', is_flag=True, default=False, help='./*/*.bin bitstream')
@click.option('--metrics/--no-metrics', is_flag=True, default=False, help='./Metrics/*.csv for each bistream')
@click.option('--configs/--no-configs', is_flag=True, default=True, help='./CFG/encoder.cfg')
@click.option('--sequences/--no-sequences', is_flag=True, default=True, help='../references.csv')
@click.argument('streams_list_url', required=True)
@click.argument('download_dir', required=True, type=click.Path(file_okay=False, dir_okay=True))
def streams(ctx, json:bool, bitstream:bool, metrics:bool, configs:bool, sequences:bool , streams_list_url:str, download_dir:Path):
    """
    \b
    Downloads streams.csv, and the content related to the streams in that list.  
    \b
    STREAMS_LIST_URL - URL of an hosted streams.csv file, all listed content will be downloaded. 
    \b
    DOWNLOAD_DIR - where content will be downloaded
    \b
    to serve streams.csv located in the current directory :
    python -m http.server
    \b
    download.py streams https://localhost:8000/streams.csv ./Bitstreams/Scenario-1-FHD/264
    

    """
    overwrite = ctx.obj['overwrite']
    dry_run = ctx.obj['dry_run']

    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True, parents=True)
    
    p = Path(urlparse(streams_list_url).path).parts
    streams_csv_local = download_dir / p[-1]
    download_file(streams_list_url, streams_csv_local, overwrite=overwrite, dry_run=dry_run, queue=False)

    encoder_cfg_keys = set()
    
    for a in iter_anchors(streams_csv_local, streams_dir=download_dir, sequences=None):
        for variant_id, _ in a.iter_variants_params():
            key = a.working_dir.name
            encoder_cfg_keys.add(a.encoder_cfg_key)
            if json:
                variant_json = f'{key}/{variant_id}.json'
                url = urljoin(streams_list_url, variant_json)
                download_file(url, download_dir/variant_json, overwrite=overwrite, dry_run=dry_run, queue=True)
            if bitstream:
                variant_bitstream = f'{key}/{variant_id}.bin'
                url = urljoin(streams_list_url, variant_bitstream)
                download_file(url, download_dir/variant_bitstream, overwrite=overwrite, dry_run=dry_run, queue=True)
            if metrics:
                variant_metrics = f'Metrics/{key}.csv'
                url = urljoin(streams_list_url, variant_metrics)
                download_file(url, download_dir/variant_metrics, overwrite=overwrite, dry_run=dry_run, queue=True)
    
    if configs:
        for cfg in encoder_cfg_keys:
            encoder_cfg = f'CFG/{str(cfg).lower()}.cfg'
            url = urljoin(streams_list_url, encoder_cfg)
            download_file(url, download_dir/encoder_cfg, overwrite=overwrite, dry_run=dry_run, queue=True)

    if sequences:
        url = urljoin(streams_list_url, '../reference-sequence.csv')
        download_file(url, download_dir.parent/'reference-sequence.csv', overwrite=overwrite, dry_run=dry_run, queue=True)

    __download_all(ctx.obj['pool_size'])


@cli.command()
@click.pass_context
@click.option('--preflight', is_flag=True, required=False, default=False, show_default=True, help='download json metadata, and check files at remote URI exists without actually downloading them')
@click.argument('sequences_csv', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('download_dir', required=True, type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument('remote_dir', required=False, default="https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/ReferenceSequences/")
def sequences(ctx, preflight, sequences_csv:Path, download_dir:Path=None, remote_dir:str=None):
    """
    \b
    Downloads content listed in reference-sequences.csv into specified directory.
    \b
    SEQUENCES_CSV - a local reference-sequences.csv file listing reference sequences 
    \b
    DOWNLOAD_DIR - where content will be downloaded
    \b
    REMOTE_DIR - base URL to download the content - default: https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/ReferenceSequences/
    \b
    """
    overwrite = ctx.obj['overwrite']
    dry_run = ctx.obj['dry_run']

    sequences_csv = Path(sequences_csv)
    download_dir = Path(download_dir)

    with requests.head(remote_dir) as r:
            r.raise_for_status()

    processed = set()
    for ref_key, ref_location in iter_ref_sequence_locations(sequences_csv):
        if ref_location in processed:
            continue
        download_reference_sequence(ref_location, remote_dir, download_dir, overwrite=overwrite, dry_run=dry_run, preflight=preflight)
        processed.add(ref_key)

    __download_all(ctx.obj['pool_size'])

if __name__ == "__main__":
    cli()