import argparse
from pathlib import Path

from encoders import get_encoder
from encoders import encode_anchor_bitstreams
from metrics import bitstream_size, compute_metrics, anchor_metrics_to_csv
from anchor import AnchorTuple, ReconstructionMeta, Metric, reference_sequences_dict, iter_anchors, iter_variants


def compute_anchor_metrics(*anchors: AnchorTuple, decode=True, overwrite=False, dry_run=False, vmaf=True):
    for a in anchors:
        a.dry_run = dry_run
        enc = get_encoder(a.encoder_id)
        assert enc is not None, f'unknown encoder {a.encoder_id}'
        for vf, vd in iter_variants(a):
            assert vd is not None, f'bitstream metadata not found: {vf}'
            if (vd.metrics is not None) and (not overwrite):
                print('[skipping, use -y to overwrite] found existing metrics in ', vf)
                continue
            if decode:
                rec = enc.decode_variant(a, vd, md5=(not a.dry_run))
                vd.reconstruction = rec.to_dict()
            else:
                log_file = getattr(vd.reconstruction, 'log-file', None)
                rec = ReconstructionMeta(a.encoder_id, a.working_dir / f'{vd.variant_id}.yuv', Path(log_file) if log_file else None, md5=False)
            vd.metrics = compute_metrics(a, vd, vmaf=vmaf)
            vd.save_as(vf)
        anchor_metrics_to_csv(a)


def parse_args():
    parser = argparse.ArgumentParser()
    cmds = ['encoder', 'decoder', 'metrics', 'csv_metrics', 'bitrate_log']
    parser.add_argument('cmd', help='|'.join(cmds))
    parser.add_argument('--scenario_dir', required=True, type=str, help='scenario directory')
    parser.add_argument('-k', '--key', required=False, type=str, default=None, help='an optional anchor key')
    parser.add_argument('-a', '--anchors-list', required=False, type=str, default='./streams.csv', help='streams.csv file containing the list of anchors for a scenario')
    parser.add_argument('-s', '--sequences-list', required=False, type=str, default='../reference-sequence.csv', help='sequences.csv file containing the list of reference raw sequences')
    parser.add_argument('-y', '--overwrite', required=False, action='store_true', default=False, help='overwrite if data already exists')
    parser.add_argument('--dry-run', required=False, action='store_true', default=False)
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir)
    assert scenario_dir.is_dir(), f'invalid scenario directory {scenario_dir}'
    
    anchors_csv = scenario_dir / args.anchors_list
    assert anchors_csv.exists(), f'anchor list not found {anchors_csv.resolve()}'
    
    references_csv = scenario_dir.parent / 'reference-sequence.csv'
    assert references_csv.exists(), f'reference list not found {references_csv.resolve()}'

    sequences_dir = scenario_dir.parent.parent.parent / 'ReferenceSequences'
    assert sequences_dir.is_dir(), f'sequence directory not found {sequences_dir.resolve()}'
    
    cfg_dir = scenario_dir / 'CFG'
    if args.cmd == "encoder":
        assert cfg_dir.exists(), f'config dir not found {cfg_dir.resolve()}'
    elif args.cmd not in cmds:
        parser.error(f'invalid command "{args.cmd}" must be one of {cmds}')
    
    anchor_keys = None
    if args.key is not None:
        anchor_keys = [args.key]

    return args.cmd, scenario_dir, anchors_csv, anchor_keys, references_csv, sequences_dir, cfg_dir, args.dry_run, args.overwrite


def main():

    cmd, scenario_dir, anchors_csv, anchor_keys, references_csv, sequences_dir, cfg_dir, dry_run, y = parse_args()
    print(references_csv, sequences_dir)
    assert Path(references_csv).exists() and Path(sequences_dir).exists()
    refs = reference_sequences_dict(references_csv, sequences_dir)
    anchors = iter_anchors(anchors_csv, refs, scenario_dir, keys=anchor_keys)

    if cmd == "decoder":
        compute_anchor_metrics(*anchors, decode=True, overwrite=y, dry_run=dry_run, vmaf=False)
    elif cmd == "metrics":
        compute_anchor_metrics(*anchors, decode=False, overwrite=y, dry_run=dry_run, vmaf=True)
    elif cmd == "csv_metrics":
        for a in anchors:
            anchor_metrics_to_csv(a)
    elif cmd == "bitrate_log":
        for a in anchors:
            for vf, vd in iter_variants(a):
                bs = a.locate_bitstream(vd, md5_check=False)
                s = bitstream_size(bs, drop_sei=True)
                if dry_run:
                    print(f'[bitstream_size] {bs}: {s}')
                else:
                    vd.metrics[Metric.BITRATELOG.key] = s
                    vd.save_as(vf)

    elif cmd == "encoder":
        encode_anchor_bitstreams(*anchors, overwrite=y, dry_run=dry_run)


if __name__ == "__main__":
    main()
