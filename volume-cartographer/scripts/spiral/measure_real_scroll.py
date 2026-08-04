"""Measure real-scroll CT statistics to calibrate synth_phantom realism knobs.

Streams two small crops (native-res texture + downsampled overview) from an
OME-Zarr scroll volume over plain HTTP -- the Vesuvius open-data bucket needs
no auth -- and reports the quantities the phantom imitates: gap/sheet
intensity levels, layer-spacing autocorrelation, and the high-frequency noise
floor. The pherc0009b preset in synth_phantom.PRESETS records the values this
script measured at the PHerc0009B volume centre; rerun it (or point it at
another scroll/position) to recalibrate or add presets.

Total transfer is ~100 MB; nothing is written except the comparison PNG.
"""

import click
import numpy as np
import scipy.signal
import zarr
from scipy.ndimage import gaussian_filter

DEFAULT_URL = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
               'PHerc0009B/volumes/20250820154339-2.401um-0.3m-77keV-masked.zarr')


@click.command()
@click.option('--url', default=DEFAULT_URL, show_default=False,
              help='OME-Zarr volume URL (default: PHerc0009B 2.4um masked).')
@click.option('--position', nargs=3, type=int, default=(14556, 14129, 14129),
              help='z y x centre of the native-res crop.')
@click.option('--crop-half', default=384, type=int,
              help='Half-size of the native crop (crop is 2*half squared).')
@click.option('--voxel-um', default=2.401, type=float,
              help='Voxel size in microns, for reporting spacing in um.')
@click.option('--out-png', default=None, type=click.Path(dir_okay=False),
              help='Optionally write an overview/crop comparison figure.')
def main(url, position, crop_half, voxel_um, out_png):
    cz, cy, cx = position
    group = zarr.open_group(url, mode='r')
    levels = sorted(int(k) for k in group.array_keys())
    coarse = levels[min(3, len(levels) - 1)]
    ds = 2 ** coarse

    crop = group['0'][cz, cy - crop_half:cy + crop_half, cx - crop_half:cx + crop_half]
    overview = group[str(coarse)][cz // ds, :, :]
    click.echo(f'native crop {crop.shape} {crop.dtype}; ds{ds} overview {overview.shape}')

    fg = crop[crop > 0]
    if len(fg) == 0:
        raise click.UsageError('crop is entirely masked; pick a position inside the scroll')
    click.echo(f'nonzero fraction {(crop > 0).mean():.3f}')
    for q in (1, 10, 25, 50, 75, 90, 99):
        click.echo(f'  p{q:02d}: {np.percentile(fg, q):.0f}/255')
    gap_level = np.percentile(fg, 25) / 255
    sheet_level = np.percentile(fg, 85) / 255
    click.echo(f'gap level ~p25 = {gap_level:.2f}; sheet level ~p85 = {sheet_level:.2f}')

    # Layer spacing: averaged autocorrelation of mean-subtracted line profiles;
    # the first prominent off-centre peak is the dominant layer period. At a
    # bunched location this measures pack-to-pack spacing rather than
    # sheet-to-sheet -- inspect the PNG before trusting a single number.
    rows = crop[range(0, crop.shape[0], 24)].astype(np.float32)
    rows = rows - rows.mean(axis=1, keepdims=True)
    ac = np.mean([np.correlate(r, r, mode='full') for r in rows], axis=0)
    ac = ac[len(ac) // 2:] / ac[len(ac) // 2]
    peaks, _ = scipy.signal.find_peaks(ac, prominence=0.02)
    if len(peaks):
        click.echo(f'layer-spacing autocorrelation peaks (voxels): {peaks[:6].tolist()}'
                   f' => ~{peaks[0] * voxel_um:.0f} um')

    residual = crop.astype(np.float32) - gaussian_filter(crop.astype(np.float32), 2.)
    click.echo(f'high-frequency noise std inside papyrus: '
               f'{residual[crop > 0].std() / 255:.3f}')

    if out_png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(overview, cmap='gray')
        axes[0].set_title(f'ds{ds} overview, z={cz}')
        axes[1].imshow(crop, cmap='gray')
        axes[1].set_title(f'native crop at ({cy},{cx})')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_png, dpi=110)
        click.echo(f'wrote {out_png}')


if __name__ == '__main__':
    main()
