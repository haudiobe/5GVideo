from pathlib import Path
from typing import Iterable, List, Tuple, Any
import matplotlib.pyplot as plt
plt.ioff()

from matplotlib.figure import Figure

import numpy as np
# from numpy.typing import ArrayLike, DTypeLike

import scipy.interpolate

from anchor import VariantData, VariantMetricSet, Metric, iter_variants, AnchorTuple, iter_anchors
import csv

import click


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
        if i and (d == dist[i - 1]):
            d = dist_fix[-1] + step
            sanitized = True
        elif i and (d > dist[i - 1]) and (d <= dist_fix[-1]):
            d = dist_fix[-1] + step
            sanitized = True
        dist_fix.append(d)
    if sanitized:
        print("/!\\ data has been sanitized:")
        print(f" -  replaced: {dist}")
        print(f" -  with    : {dist_fix}")
    return rate, np.array(dist_fix, dtype=np.float64)


def rd_metrics(variants: List[VariantData], rate="BitrateLog", dist="PSNR") -> Iterable[Any]:
    # return zip(*[(v.metrics[rate], v.metrics[dist]) for v in variants])
    rd = []
    for v in variants:
        r = v.metrics[rate]
        if int(r) == 0:
            r = v.metrics["Bitrate"]
        if dist in v.metrics:
            rd.append((r, v.metrics[dist]))
        else:
            rd.append((r, 0.000))
    return zip(*rd)

def compare_anchors_metrics(anchor: List[VariantData], test: List[VariantData], rate="BitrateLog", dist=None, title="", strict=False, sanitize=True) -> Tuple[Figure, int, Any, Any, Any, Any]:
    anchor_metrics = [*rd_metrics(anchor, rate=rate, dist=dist)]
    test_metrics = [*rd_metrics(test, rate=rate, dist=dist)]
    try:
        print("#", dist, "#" * (32 - len(str(dist))))
        return bd_rate_plot(*anchor_metrics, *test_metrics, sanitize=sanitize, title=title, dist_label=None)
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
            row = {k: v for [k, v] in v.metrics.items() if k in fieldnames}
            row['ID'] = v.variant_id
            writer.writerow(row)


def compare_sequences(anchor: AnchorTuple, test: AnchorTuple, metrics: Iterable[str], strict=False, sanitize=True, save_plots=None) -> Iterable[Tuple[str, VariantMetricSet]]:

    anchor_variants = [v[1] for v in iter_variants(anchor)]
    test_variants = [v[1] for v in iter_variants(test)]

    vms = VariantMetricSet()
    fig = None
    for key in metrics:
        try:
            # plot_title = f'RD-curve & BD-rate for {key}\n'\
            #     f'[sequence] {anchor.reference.sequence["Key"]} | {anchor.reference.sequence["Name"]}\n'\
            #     f'[anchor] {anchor.encoder_cfg.name} @ {anchor._variants}\n'\
            #     f'[test] {test.encoder_cfg.name} @ {test._variants}'
            fig, bd, *_ = compare_anchors_metrics(anchor_variants, test_variants, rate="BitrateLog", dist=key, strict=strict, sanitize=sanitize, title=None)
            vms[key] = f'{round(bd, 3):.3f}'
            if save_plots and (key in save_plots):
                # t = f'bd-rate @{key} | {vms[key]}'
                # if sanitize:
                #     t += ' [sanitized]'
                # fig = rd_plot(r0, d0, r1, d1, key, title=t, anchor_label=anchor.anchor_key, test_label=test.anchor_key, show=False)
                fname = test.working_dir / 'Metrics' / f'{anchor.working_dir.name}.{test.working_dir.name}.{key}.png'.lower()
                if not fname.parent.exists():
                    fname.parent.mkdir()
                fig.savefig(fname)
                plt.close()

        except KeyError as e:
            print(e)
        except BaseException as e:
            if fig:
                fig.close()
            if strict:
                raise
            vms[key] = str(e)

    return anchor.reference.sequence['Key'], vms


def _parse_filter(fp: Path):
    arr = fp.name.split("@")
    if len(arr) > 1:
        return fp.parent / "@".join(arr[:-1]), arr[-1]
    else:
        return fp, None


def compare_anchors(anchors: List[AnchorTuple], tests: List[AnchorTuple], metrics: List[str], strict=False, save_plots=True):

    assert len(anchors) == len(tests), f'reference has {len(anchors)} anchors, test has {len(tests)} anchors.'

    bd_rates = []

    for aref, atest in zip(anchors, tests):
        vref = [v for (_, v) in iter_variants(aref)]
        vtest = [v for (_, v) in iter_variants(atest)]
        arates = {}
        print("\n" + "=" * 35)
        print(f"{aref.anchor_key} vs {atest.anchor_key}\n" + "=" * 35)
        for key in metrics:
            try:
                fig, bd, *_ = compare_anchors_metrics(vref, vtest, rate="BitrateLog", dist=key)
                arates[key] = f'{round(bd, 3):.3f}'
                if save_plots:
                    fname = atest.working_dir.parent / 'Metrics' / (f'{aref.working_dir.name}.{atest.working_dir.name}'.upper() + f'{key}.png'.lower())
                    if not fname.parent.exists():
                        fname.parent.mkdir()
                    fig.savefig(fname)
                    try:
                        plt.close()
                    except BaseException as e:
                        print(e)
                    
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
                if v is None or type(v) == str:
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

        r = {"reference": "Min"}
        for k in fieldnames[1:]:
            r[k] = stats["min"].get(k, None)
        writer.writerow(r)

        r = {"reference": "Max"}
        for k in fieldnames[1:]:
            r[k] = stats["max"].get(k, None)
        writer.writerow(r)

        r = {"reference": "Avg"}
        for k in fieldnames[1:]:
            avg = None
            if k in stats["avg"] and len(stats["avg"][k]):
                avg = 0
                for j in stats["avg"][k]:
                    avg += j
                avg /= len(stats["avg"][k])
            r[k] = avg
        writer.writerow(r)


def strictly_increasing(samples):
    for i, v in enumerate(samples):
        if i and v <= samples[i - 1]:
            return False
    return True


def bd_rate_plot(R1, DIST1, R2, DIST2, sanitize=False, title="", dist_label=None) -> Tuple[Figure, int, Any, Any, Any, Any]:
    """adapted from https://github.com/Anserw/Bjontegaard_metric
    which computes bd-rate according to:
        [1] G. Bjontegaard, Calculation of average PSNR differences between RD-curves (VCEG-M33)
        [2] S. Pateux, J. Jung, An excel add-in for computing Bjontegaard metric and its evolution
    """
    if not dist_label:
        dist_label = "quality"

    if sanitize:
        R1, DIST1 = sanitize_rd_data(R1, DIST1)
        R2, DIST2 = sanitize_rd_data(R2, DIST2)
        _ = strictly_increasing(DIST1) and strictly_increasing(DIST2)

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
        avg_exp_diff = (int2 - int1) / (max_int - min_int)
        avg_diff = (np.exp(avg_exp_diff) - 1) * -100

        fig, axs = plt.subplots(1, 3, figsize=(30, 12))

        axs[0].plot(R1, DIST1, 'o-', R2, DIST2, 'o-')
        axs[0].set_xlabel('bitrate', fontsize=21)
        axs[0].set_ylabel(dist_label, fontsize=21)
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
        axs[1].set_ylabel(dist_label, fontsize=21)
        axs[1].grid(True)
        axs[1].tick_params(axis='both', which='major', labelsize=21)
        axs[1].set_title(f'BD rate gain: {avg_diff:.3f}', fontdict={'fontsize': 21, 'fontweight': 'medium'})
        axs[1].axhline(min_int, linestyle='dashed', color='red')
        axs[1].axhline(max_int, linestyle='dashed', color='red')
        axs[1].fill_betweenx(samples, v1, v2, color='red', alpha=0.25)

        if title and title != "":
            fig.suptitle(title, fontsize=21)

        x = [i * interval for i in range(100)]
        axs[2].set_xlabel('composite trapezoidal integration', fontsize=21)
        axs[2].plot(x, v1, '-', label="anchor (interpolated)")
        axs[2].plot(x, v2, '-', label="test (interpolated)")
        axs[2].fill_between(x, v1, v2, color='red', alpha=0.75)

    except ValueError as ve:
        print(ve)
        print('d1: ', d1)
        print('d2: ', d2)

    return fig, avg_diff, R1, DIST1, R2, DIST2


@click.command()
@click.option('--root-dir', envvar='VCC_WORKING_DIR', type=click.Path(exists=True, file_okay=False, writable=True, readable=True))
@click.option('-s/-c', required=True, default=True, help="whether anchor/test keys are:\n 1. sequence IDs, eg. '-s S1-A01-264 S1-A01-265'\n 2. encoder configs IDs, eg. '-c S1-JM-01 S1-HM-01'")
@click.argument('anchor', nargs=1)
@click.argument('test', nargs=1)
def main(root_dir, s, anchor, test):
    root_dir = Path(root_dir)
    # TODO: check support HDR keys
    metric_keys = (Metric.PSNR_Y, Metric.PSNR, Metric.MSSSIM, Metric.VMAF)
    if s:  # eg. '-s S1-A01-264 S1-A01-265'
        anchor = AnchorTuple.load(anchor, root_dir)
        test = AnchorTuple.load(test, root_dir)
        metrics = [m.key for m in metric_keys]
        compare_sequences(anchor, test, metrics, strict=True, save_plots=metrics)

    else:
        data = []
        anchors = AnchorTuple.iter_cfg_anchors(anchor, root_dir=root_dir)
        tests = AnchorTuple.iter_cfg_anchors(test, root_dir=root_dir)
        if tests[-1] is None:
            raise ValueError(f'{test} data not found')
        outp = tests[-1].working_dir.parent  / 'Metrics' / f'{anchor}.{test}.csv'.lower()
        metrics = [m.key for m in metric_keys]
        for (seqid, r) in compare_anchors(anchors, tests, metrics, strict=True, save_plots=True):
            r['reference'] = seqid
            data.append(r)

        csv_dump(data, outp)

if __name__ == "__main__":
    main()
