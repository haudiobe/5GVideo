
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

from numpy.core.fromnumeric import var

from anchor import VariantData, iter_variants, AnchorTuple, VariantMetricSet2
from download import AnchorTupleCtx
from metrics import SDR_METRICS, BD_RATE, compute_avg_psnr, Metric
import sys, csv

# plot rd for 1 anchor
def rd_plot(*anchors):
    # for (rate, dist) in data:
    #     print(rate, dist)
    for a in anchors:
        # [(r,d), ... ] = a
        print(*a)


def rd_metrics(variants:List[VariantData], rate="BitrateLog", dist="PSNR") -> Iterable[Any]:
        return zip(*[(v.metrics[rate], v.metrics[dist]) for v in variants])

def ensure_avg_psnr(data:List[VariantData]):
    for vd in data:
        compute_avg_psnr(vd)

def compare_anchors_metrics( anchor:List[VariantData], test:List[VariantData], rate="BitrateLog", dist="PSNR", piecewise=1) -> float:
    if dist.lower() == str(Metric.PSNR.key):
        ensure_avg_psnr(anchor)
        ensure_avg_psnr(test)
    anchor_metrics = [*rd_metrics(anchor, rate=rate, dist=dist)]
    test_metrics = [*rd_metrics(test, rate=rate, dist=dist)]
    return BD_RATE(*anchor_metrics, *test_metrics, piecewise=piecewise)


def variant_metrics_to_csv(variants, path='./variants.csv', csv_append=False, csv_headers=True):
    # eg. 5GVideo/Bitstreams/Scenario-5-Gaming/265/Metrics/hm-01.csv
    # sequence,QPISlice,kbps,Y psnr,U psnr,V psnr,Enc T [s],Dec T [s],VMAF,MS SSIM,Enc T [h]

    # eg. 
    # parameter,bitrate,y_psnr,u_psnr,v_psnr,ms_ssim,vmaf,bitrate_log,encode_time,decode_time
    with open(path, 'a' if csv_append else 'w', newline='') as csvfile:
        fieldnames = [
            "ID",
            "BitrateLog",
            "YPSNR",
            "UPSNR",
            "VPSNR",
            "MS_SSIM",
            "VMAF",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csv_headers:
            writer.writeheader()
        for v in variants:
            row = { k: v for [k, v] in v.metrics.items() if k in fieldnames }
            row['ID'] = v.variant_id
            writer.writerow( row )


def compare_anchors_directories(refdir:Path, testdir:Path, metrics:Iterable[str], strict=False) -> Iterable[Tuple[str, VariantMetricSet2]]:

    def load_anchor_directory(ad:Path) -> Tuple[AnchorTuple, List[VariantData]]:
        assert ad.is_dir(), f'not a directory {ad}'
        ctx, anchor_key = AnchorTupleCtx.from_anchor_directory(ad)
        a = ctx.iter_anchors(keys=[anchor_key])[0]
        v = [ vdata for (_, vdata) in iter_variants(a) ]
        return a, v

    aref, vref = load_anchor_directory(refdir)
    _, vtest = load_anchor_directory(testdir)

    vms = VariantMetricSet2()
    for key in metrics:
        try:
            d = compare_anchors_metrics(vref, vtest, rate="BitrateLog", dist=key)
            vms[key] = f'{round(d, 2):.2f}'
        except BaseException as e:
            if strict:
                raise
            vms[key] = f'Err.'
            # vms[key] = f'[err]: {e}'
    return aref.reference.sequence['Key'], vms

#######################################################

def _parse_filter(fp:Path):
    arr = fp.name.split("@")
    if len(arr) > 1:
        return fp.parent / "@".join(arr[:-1]), arr[-1]
    else:
        return fp, None

def compare_encoder_configs(ref:Path, test:Path, metrics:List[str], strict=False):

    refdir, refkey = _parse_filter(ref)
    refctx = AnchorTupleCtx(scenario_dir=refdir)
    ref_anchors = refctx.iter_anchors(cfg_keys=[refkey])
    
    testdir, testkey = _parse_filter(test)
    testctx = AnchorTupleCtx(scenario_dir=testdir)
    test_anchors = testctx.iter_anchors(cfg_keys=[testkey])

    assert len(ref_anchors) == len(test_anchors), f'reference has {len(ref_anchors)} anchors, test has {len(test_anchors)} anchors'

    bd_rates = []

    for aref, atest in zip(ref_anchors, test_anchors):
        vref = [ v for (_, v) in iter_variants(aref) ]
        vtest = [ v for (_, v) in iter_variants(atest) ]
        arates = {}
        for key in metrics:
            try:
                d = compare_anchors_metrics(vref, vtest, rate="BitrateLog", dist=key)
                arates[key] = f'{round(d, 2):.2f}'
            except BaseException as e:
                if strict:
                    raise
                # arates[key] = f'[err]: {e}'
                arates[key] = '[err]'

        bd_rates.append((aref, atest, arates))

    return [(r.reference.sequence['Key'], bdr) for (r, _, bdr) in bd_rates]

def csv_dump(data, fp):
    # ignore = ['bitrate', 'bitratelog', 'encodetime', 'decodetime']
    # fieldnames = [ k for k in data[0].keys() if k not in ignore ]
    fieldnames = [ 
        'reference',
        Metric.PSNR_Y.key,
        Metric.PSNR_U.key,
        Metric.PSNR_V.key,
        Metric.PSNR.key,
        Metric.MSSSIM.key,
        Metric.VMAF.key
    ]
    if not fp.parent.exists():
        fp.parent.mkdir(exist_ok=True, parents=True)
    with open(fp, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in data:
            writer.writerow({ k: r[k] for k in fieldnames })


def main():

    assert len(sys.argv) == 3

    ref = Path(sys.argv[1])
    test = Path(sys.argv[2])

    print("Anchor:", ref)
    print("Test:", test)
    
    metrics = [str(m.key) for m in SDR_METRICS]

    if ref.is_dir() and (not (ref.parent / 'streams.csv').exists()) \
        and test.is_dir() and (not (test.parent / 'streams.csv').exists()):
        # eg. compare.py ./scenario/codec1/a_key1 ./scenario/codec2/a_key2
        seqid, r = compare_anchors_directories(ref, test, metrics)
        print(seqid, r)
    else:
        # eg. compare.py ./scenario/codec1@cfg_id1 ./scenario/codec2@cfg_id2
        _, refkey = _parse_filter(ref)
        testdir, testkey = _parse_filter(test)
        outp = testdir / 'Metrics' / f'{refkey}.{testkey}.csv'.lower()
        data = []
        for (seqid, r) in compare_encoder_configs(ref, test, metrics):
            r['reference'] = seqid
            data.append(r)
        csv_dump(data, outp)
        
if __name__ == "__main__":
    main()