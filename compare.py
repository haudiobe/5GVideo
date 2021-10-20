
from pathlib import Path
from typing import Iterable, List, Tuple, Any
import matplotlib.pyplot as plt

# plt.ioff()

import numpy as np
import scipy.interpolate

from anchor import VariantData, VariantMetricSet, Metric, iter_variants, AnchorTuple
from download import AnchorTupleCtx
from metrics import Metric
import sys, csv



def sort_rates_on(rates, metric):
    """
    sort [(rate, psnr), ...] samples based on psnr 
    """
    return rates[np.argsort(metric)], np.sort(metric)

def sort_on_rates(rates, metric):
    """
    sort [(rate, psnr), ...] samples based on rate 
    """
    return np.sort(rates), metric[np.argsort(rates)]


def sanitize_rd_data(rates, dist, step=0.001):
    """workaround for saturated dist values.
        returns sanitized rates & dist, sorted on dist. 
        consecutive samples that have the same values are modified with the given step 
        so that the sequence is increasing instead of stagnating.
        if a sequence is decreasing rather than stagnating, it is not modified.
        eg. [50., 50., 50., 50., 50.] becomes [50., 50.001, 50.002, 50.003, 50.004]
            [98.999, 99.999, 99.999, 99.999, 100.] becomes [98.999, 99.999, 100., 100.001, 100.002]
            [98.999, 99.999, 99.999, 97., 100.] becomes [97, 98., 99.999, 100., 100.001]
        """
    rate = np.array(rates)
    dist = np.array(dist)
    sorted = np.lexsort((rate, dist))
    rate = rate[sorted]
    dist = dist[sorted]
    dist_fix = []
    sanitized = False
    for i, _ in enumerate(rate):
        d = dist[i]
        if i and (d == dist[i-1]):
            d = dist_fix[-1] + step
            sanitized = True
        elif i and (d > dist[i-1]) and (d <= dist_fix[-1]):
            d = dist_fix[-1] + step
            sanitized = True
        dist_fix.append(d)
    if sanitized:
        print("/!\ data has been sanitized:")
        print(f" -  replaced: {dist}")
        print(f" -  with    : {dist_fix}")
    return rate, np.array(dist_fix, dtype=np.float64)

def rd_metrics(variants:List[VariantData], rate="BitrateLog", dist="PSNR") -> Iterable[Any]:
        return zip(*[(v.metrics[rate], v.metrics[dist]) for v in variants])

def compare_anchors_metrics( anchor:List[VariantData], test:List[VariantData], rate="BitrateLog", dist="psnr", title="", strict=False, sanitize=True ) -> float:
    anchor_metrics = [*rd_metrics(anchor, rate=rate, dist=dist)]
    test_metrics = [*rd_metrics(test, rate=rate, dist=dist)]
    try:
        print("#", dist, "#"*(32-len(str(dist))))
        return bd_rate_plot(*anchor_metrics, *test_metrics, sanitize=sanitize, title=title, dist_label=dist)
    except BaseException as e:
        if strict:
            raise
        raise ValueError(f'A:{anchor_metrics} | T:{test_metrics} | Err: {e}')

def variant_metrics_to_csv(variants, path='./variants.csv', csv_append=False, csv_headers=True):
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


def compare_anchors_directories(refdir:Path, testdir:Path, metrics:Iterable[str], strict=False, sanitize=True, plots=None) -> Iterable[Tuple[str, VariantMetricSet]]:

    def load_anchor_directory(ad:Path) -> Tuple[AnchorTuple, List[VariantData]]:
        assert ad.is_dir(), f'not a directory {ad}'
        ctx, anchor_key = AnchorTupleCtx.from_anchor_directory(ad)
        a = ctx.iter_anchors(keys=[anchor_key])[0]
        v = [ vdata for (_, vdata) in iter_variants(a) ]
        return a, v

    aref, vref = load_anchor_directory(refdir)
    atest, vtest = load_anchor_directory(testdir)

    vms = VariantMetricSet()
    fig = None
    for key in metrics:
        try:
            # plot_title = f'RD-curve & BD-rate for {key}\n'\
            #     f'[sequence] {aref.reference.sequence["Key"]} | {aref.reference.sequence["Name"]}\n'\
            #     f'[anchor] {aref.encoder_cfg.name} @ {aref._variants}\n'\
            #     f'[test] {atest.encoder_cfg.name} @ {atest._variants}'
            fig, bd, *_ = compare_anchors_metrics(vref, vtest, rate="BitrateLog", dist=key, strict=strict, sanitize=sanitize, title=None)
            vms[key] = f'{round(bd, 3):.3f}'
            if plots and (key in plots):
                # t = f'bd-rate @{key} | {vms[key]}'
                # if sanitize:
                #     t += ' [sanitized]'
                # fig = rd_plot(r0, d0, r1, d1, key, title=t, anchor_label=aref.anchor_key, test_label=atest.anchor_key, show=False)
                fname = testdir.parent / 'Metrics' / f'{refdir.name}.{testdir.name}.{key}.png'.lower()
                fig.savefig(fname)
                plt.close()

        except BaseException as e:
            if fig:
                fig.close()
            if strict:
                raise
            vms[key] = str(e)
    
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

    assert len(ref_anchors) == len(test_anchors), f'reference has {len(ref_anchors)} anchors, test has {len(test_anchors)} anchors.'

    bd_rates = []

    for aref, atest in zip(ref_anchors, test_anchors):
        vref = [ v for (_, v) in iter_variants(aref) ]
        vtest = [ v for (_, v) in iter_variants(atest) ]
        arates = {}
        print("\n"+"="*35)
        print(f'{aref.anchor_key} vs {atest.anchor_key}\n'+'='*35)
        for key in metrics:
            try:
                _, bd, *_  = compare_anchors_metrics(vref, vtest, rate="BitrateLog", dist=key)
                arates[key] = bd
            except BaseException as e:
                if strict:
                    raise
                arates[key] = str(e)

        bd_rates.append((aref, atest, arates))

    return [(r.reference.sequence['Key'], bdr) for (r, _, bdr) in bd_rates]

def csv_dump(data, fp):
    fieldnames = [ 
        'reference',
        Metric.PSNR_Y.key,
        Metric.PSNR.key,
        Metric.MSSSIM.key,
        Metric.VMAF.key
    ]

    stats = {
        'min': {},
        'max': {},
        'avg': {},
    }

    if not fp.parent.exists():
        fp.parent.mkdir(exist_ok=True, parents=True)

    with open(fp, 'w', newline='') as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in data:
            
            row = {}

            for k in fieldnames:
                row[k] = r[k]
                if len(data) == 1:
                    continue
                if k == "reference":
                    continue
                
                v = row[k]
                if v == None or type(v) == str:
                    continue
                if k in stats["min"]:
                    stats["min"][k] = min(v, stats["min"][k])
                else:
                    stats["min"][k] = v
                if k in stats["max"]:
                    stats["max"][k] = max(v, stats["max"][k])
                else:
                    stats["max"][k] = v
                if k in stats["avg"]:
                    stats["avg"][k].append(v)
                else:
                    stats["avg"][k] = [v]
            
            writer.writerow(row)

        if len(data) == 1:
            return

        r = { "reference": "Min" }
        for k in fieldnames[1:]:
            r[k] = stats["min"].get(k, None)
        writer.writerow(r)

        r = { "reference": "Max" }
        for k in fieldnames[1:]:
            r[k] = stats["max"].get(k, None)
        writer.writerow(r)

        r = { "reference": "Avg" }
        for k in fieldnames[1:]:
            avg = None
            if k in stats["avg"] and len(stats["avg"][k]):
                avg = 0
                for j in stats["avg"][k]:
                    avg += j
                avg /= len(stats["avg"][k])
            r[k] = avg
        writer.writerow(r)



#####################################################################################################

def strictly_increasing(samples):
    for i, v in enumerate(samples):
        if i and v <= samples[i-1]:
            return False
    return True

def bd_rate_plot(R1, DIST1, R2, DIST2, sanitize=False, title="", dist_label="dist"):
    
    """
    adapted from https://github.com/Anserw/Bjontegaard_metric
    which computes bd-rate according to:
        [1] G. Bjontegaard, Calculation of average PSNR differences between RD-curves (VCEG-M33) 
        [2] S. Pateux, J. Jung, An excel add-in for computing Bjontegaard metric and its evolution
    """

    if sanitize:
        R1, DIST1 = sanitize_rd_data(R1, DIST1)
        R2, DIST2 = sanitize_rd_data(R2, DIST2)
        b = strictly_increasing(DIST1) and strictly_increasing(DIST2)

    else:
        DIST1 = np.array(DIST1)
        DIST2 = np.array(DIST2)

    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # integration interval
    min_int = max(min(DIST1), min(DIST2))
    max_int = min(max(DIST1), max(DIST2))

    samples, interval = np.linspace(min_int, max_int, num=100, retstep=True)
    [r1, d1] = sort_on_rates(lR1, DIST1)
    assert strictly_increasing(d1)

    [r2, d2] = sort_on_rates(lR2, DIST2)
    assert strictly_increasing(d2)
    
    v1, v2, avg_diff, fig = None, None, 0, None
    try:
        v1 = scipy.interpolate.pchip_interpolate(d1, r1, samples)
        v2 = scipy.interpolate.pchip_interpolate(d2, r2, samples)

        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)
        avg_exp_diff = (int2-int1)/(max_int-min_int)
        avg_diff = (np.exp(avg_exp_diff)-1) * -100

        fig, axs = plt.subplots(1, 2, figsize=(20, 12))

        axs[0].plot(R1, DIST1, 'o-', R2, DIST2, 'o-')
        axs[0].set_xlabel('bitrate', fontsize=21)
        axs[0].set_ylabel('quality', fontsize=21)
        axs[0].grid(True)
        axs[0].tick_params(axis='both', which='major', labelsize=21)
        axs[0].set_title('Rate-Quality Curve', fontdict={'fontsize': 21, 'fontweight': 'medium'})
        axs[0].legend(['anchor', 'test'])
        axs[0].axhline(min_int, linestyle='dashed', color='red')
        axs[0].axhline(max_int, linestyle='dashed', color='red')

        axs[1].plot(r1, d1, 'o:', label="anchor (measured)")
        axs[1].plot(r2, d2, 'o:', label="test (measured)")
        axs[1].plot(v1, samples, '-', label="anchor (interpolated)")
        axs[1].plot(v2, samples, '-', label="test (interpolated)")
        
        axs[1].legend()
        axs[1].set_xlabel('bitrate (log)', fontsize=21)
        axs[1].set_ylabel(f'quality', fontsize=21)
        axs[1].grid(True)
        axs[1].tick_params(axis='both', which='major', labelsize=21)
        axs[1].set_title(f'BD rate gain: {avg_diff:.3f}', fontdict={'fontsize': 21, 'fontweight': 'medium'})
        axs[1].axhline(min_int, linestyle='dashed', color='red')
        axs[1].axhline(max_int, linestyle='dashed', color='red')
        axs[1].fill_betweenx(samples, v1, v2, color='red', alpha=0.25)

        if title and title != "":
            fig.suptitle(title, fontsize=21)

    except ValueError as ve:
        print(ve)
        print('d1:', d1)
        print('d2:', d2)
    
    return fig, avg_diff, R1, DIST1, R2, DIST2


#####################################################################################################



def main():

    assert len(sys.argv) == 3

    ref = Path(sys.argv[1])
    test = Path(sys.argv[2])

    print("Anchor:", ref)
    print("Test:", test)
    
    metrics = [ m.key for m in (
            Metric.PSNR_Y, 
            Metric.PSNR, 
            Metric.MSSSIM, 
            Metric.VMAF 
        )]

    if ref.is_dir() and (ref.parent / 'streams.csv').exists() \
        and test.is_dir() and (test.parent / 'streams.csv').exists():
        # eg. compare.py ./scenario/codec1/a_ker1 ./scenario/codec2/a_ker2
        plots = [ m.key for m in (
            Metric.PSNR_Y, 
            Metric.PSNR, 
            Metric.MSSSIM, 
            Metric.VMAF
        )]
        seqid, r = compare_anchors_directories(ref, test, metrics, strict=True, sanitize=True, plots=plots)
        r['reference'] = seqid
        outp = test.parent / 'Metrics' / f'{ref.name}.{test.name}.csv'.lower()
        csv_dump([r], outp)
    
    else:
        # eg. compare.py ./scenario/codec1@cfg_id1 ./scenario/codec2@cfg_id2
        _, refkey = _parse_filter(ref)
        testdir, testkey = _parse_filter(test)
        outp = testdir / 'Metrics' / f'{refkey}.{testkey}.csv'.lower()
        data = []
        for (seqid, r) in compare_encoder_configs(ref, test, metrics, strict=True):
            r['reference'] = seqid
            data.append(r)
        csv_dump(data, outp)
        
if __name__ == "__main__":
    main()