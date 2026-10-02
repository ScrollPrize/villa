"""Paired spatial views of measured features, attention, history and decisions."""
import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .interpret import VARIANTS

BG, INK, MUTED = '#f1f5f9', '#172b43', '#53677c'
COLORS = ['#008876', '#246da8', '#8853b1', '#cb4d61', '#ca7c17']


class Atlas:
    def __init__(self, directory):
        self.dest = Path(directory)
        with np.load(self.dest / 'sample_activations.npz') as data:
            self.a = dict(data)
        with np.load(self.dest / 'analysis_arrays.npz') as data:
            self.b = dict(data)
        self.m = json.loads((self.dest / 'sample_provenance.json').read_text())
        self.r = json.loads((self.dest / 'analysis_summary.json').read_text())
        self.c = self.m['model_cfg']
        self.crop = self.c['fine']
        self.xyz = self.a['token_xyz']
        self.shape = self.xyz.shape[:3]
        self.nimage = int(np.prod(self.shape))
        self.ref = np.unravel_index(np.linalg.norm(self.xyz, axis=-1).argmin(), self.shape)
        self.pages, self.pairs = [], []
        s, w, d, behind = (self.crop[k] for k in ('spacing', 'width', 'depth', 'behind'))
        self.ext = (-w*s/2, w*s/2, (-behind-.5)*s, (d-behind-.5)*s)
        self.tex = (self.xyz[0, 0, 0, 0]-2*s, self.xyz[0, 0, -1, 0]+2*s,
                    self.xyz[0, 0, 0, 2]-2*s, self.xyz[-1, 0, 0, 2]+2*s)
        self.ct_limits = np.percentile(self.a['input'][0], [2, 98])

    def page(self, title, subtitle, size=(20, 15)):
        fig = plt.figure(figsize=size, facecolor=BG)
        fig.text(.04, .97, title, fontsize=24, weight='bold', va='top')
        fig.text(.04, .927, subtitle, fontsize=12, color=MUTED, va='top')
        return fig

    def axis(self, fig, rect, title=''):
        ax = fig.add_axes(rect)
        ax.set_title(title, fontsize=11, pad=9)
        ax.tick_params(labelsize=9)
        return ax

    def pair(self, fig, rect, title, kind='central slices'):
        x, y, w, h = rect
        gap = .035
        axes = [self.axis(fig, [x+i*(w+gap)/2, y, (w-gap)/2, h],
                          title+'\n'+('u–forward' if i == 0 else 'v–forward')) for i in range(2)]
        self.pairs.append(dict(page=len(self.pages)+1, title=title, kind=kind, views=['u-forward', 'v-forward']))
        return axes

    def spatial(self, ax, volume, comp, token=False, cmap='gray', limits=None, marginal=False):
        if marginal:
            values = volume.sum(1 if comp == 0 else 2)
        else:
            middle = (int(np.abs(self.xyz[0, :, 0, 1]).argmin()) if comp == 0 else
                      int(np.abs(self.xyz[0, 0, :, 0]).argmin())) if token else self.crop['width']//2
            values = volume[:, middle, :] if comp == 0 else volume[:, :, middle]
        im = ax.imshow(values, origin='lower', extent=self.tex if token else self.ext,
                       interpolation='nearest', cmap=cmap,
                       vmin=limits[0] if limits is not None else None,
                       vmax=limits[1] if limits is not None else None)
        ax.set(xlim=self.ext[:2], ylim=self.ext[2:], xlabel=('u' if comp == 0 else 'v')+' (trace voxels)', ylabel='forward')
        return im

    def curves(self, ax, comp, variants=('baseline',), history=False):
        points = self.a['points']
        gt = self.a['plane_ab'].copy()
        gt[self.a['plane_mask'] == 0] = np.nan
        ax.plot(gt[:, comp], points[:, 2], '--', color='#d765ed', label='Annotation reference')
        if history:
            hist = self.a['hist'][self.a['hmask'] > 0]
            ax.plot(hist[:, comp], hist[:, 2], color='#ff6078', label='Observed history')
        for name in variants:
            p = self.b[name+'_points']
            ax.plot(p[:, comp], p[:, 2], color=COLORS[list(VARIANTS).index(name)],
                    lw=2, label=VARIANTS[name], linestyle='-' if name == 'baseline' else '--')
        ax.scatter([0], [0], marker='D', s=25, c='#55d9f3', label='Current head', zorder=5)

    def note(self, fig, text):
        fig.text(.04, .045, text, fontsize=11, color=MUTED, va='top', linespacing=1.5)

    def finish(self, fig, name, title, summary, pdf):
        for suffix, dpi in [('', 160), ('_preview', 75)]:
            fig.savefig(self.dest / (name+suffix+'.png'), dpi=dpi, facecolor=BG)
        fig.savefig(self.dest / (name+'.pdf'), facecolor=BG)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)
        self.pages.append(dict(name=name, title=title, summary=summary))

    def overview(self, pdf):
        c, m, r = self.c, self.m, self.r
        fig = self.page('01  What the current model sees and decides',
                        f"Fiber {m['fiber_index']} · {m['fiber_name']} · step {m['checkpoint_step']:,} EMA · {m['precision']}")
        generation = (f"{c['flow_steps']} midpoint integration steps" if m['capabilities']['solver']
                      else f"Up to {c['recurrent_refinement_steps']} retries")
        boxes = [
            ('INPUT', f"{self.a['input'].shape}\nZ-scored CT\nObserved references and seed"),
            ('PATCH ENCODER', f"6³ overlapping patches, stride 4\n{' × '.join(map(str, self.shape))} tokens, width {c['hidden']}\n{c['layers']} axial blocks + normalization"),
            ('HISTORY', f"{m['observations']} valid CT/path slabs\n{m['history_tokens_per_slab']} tokens per slab\nSeparate residual decoder reads"),
            ('PATH AND SCORE', f"{c['n_future']} future points; {c['decoder_layers']} generator layers\nCausal segment survival scorer\n{generation}")]
        for i, (title, body) in enumerate(boxes):
            fig.text(.045+i*.24, .85, title+'\n\n'+body, fontsize=12, va='top', linespacing=1.6,
                     bbox=dict(boxstyle='round,pad=.7', facecolor='white', edgecolor='#cad7e5'))
        axes = self.pair(fig, [.05, .19, .43, .40], 'Selected prediction on CT')
        for comp, ax in enumerate(axes):
            self.spatial(ax, self.a['input'][0], comp, limits=self.ct_limits)
            self.curves(ax, comp, history=True)
        axes[0].legend(fontsize=8, loc='lower left')
        ax = self.axis(fig, [.67, .36, .27, .23], 'Sensitivity to historical evidence')
        names = list(VARIANTS)[1:]
        ax.barh([VARIANTS[n].replace(' history read disabled', ' off') for n in names],
                [r['metrics'][n]['max_path_shift'] for n in names], color=COLORS[1:])
        ax.invert_yaxis()
        ax.set_xlabel('Maximum path shift (trace voxels)')
        base, none = r['metrics']['baseline'], r['metrics']['no_history']
        fig.text(.55, .25, f"Selected attempt {m['selected_refinement']+1} of {m['attempts']}\n"
                 f"Baseline commits {base['commit']} / {m['n_commit']}; final survival {base['last_confidence']:.3f}\n"
                 f"Both history reads off: max shift {none['max_path_shift']:.3f}, commit {none['commit']}", fontsize=14, linespacing=1.7)
        self.note(fig, 'CT panels are central slices; curves are 3D projections and may lie outside the slice.\nFeature summaries and attention are measured representations. Interventions describe one decision, not dataset accuracy.')
        self.finish(fig, '01_overview', 'Architecture and decision', 'Current overlapping-patch model, live historical slabs and measured decision sensitivity.', pdf)

    def inputs(self, pdf):
        fig = self.page('02  Inputs and overlapping patch embeddings', 'Each token sees a 6 × 6 × 6 neighborhood; adjacent token centers are four input voxels apart.')
        for col, channel in enumerate(range(len(self.a['input']))):
            axes = self.pair(fig, [.045+col*.49, .56, .425, .30], 'CT and observed history')
            for comp, ax in enumerate(axes):
                self.spatial(ax, self.a['input'][channel], comp, limits=self.ct_limits if channel == 0 else (0, 1))
                if channel == 0:
                    self.curves(ax, comp, history=True)
        patch = self.a['patch']
        ref = patch[self.ref]
        similarity = patch @ ref / (np.linalg.norm(patch, axis=-1)*np.linalg.norm(ref)).clip(1e-12)
        axes = self.pair(fig, [.045, .12, .425, .28], 'Patch token cosine similarity to the head token')
        for comp, ax in enumerate(axes):
            self.spatial(ax, similarity, comp, token=True, cmap='RdBu_r', limits=(-1, 1))
        ax = self.axis(fig, [.56, .24, .38, .16], 'Actual embedding of the head token')
        ax.plot(ref, lw=1)
        ax.set(xlabel='Feature index', ylabel='Value')
        fig.text(.54, .16, f"Conv3d: {self.a['input'].shape[0]} channels × 6³ → {self.c['hidden']} features\n"
                 f"Stride 4; one-voxel halo; high edge padded to a multiple of 4.\n"
                 'Position and observed-reference conditioning enter before block 1.', fontsize=12, va='top', linespacing=1.5)
        self.note(fig, 'CT is normalized per crop, without clipping or a background sentinel.\nFeature axes have no assigned anatomical meanings. All similarity maps use the same [−1, 1] scale.')
        self.finish(fig, '02_inputs', 'Inputs and patches', 'Actual CT and overlapping convolution embeddings; all channels are available separately.', pdf)

    def encoder(self, pdf):
        count = len(self.r['encoder_blocks'])
        fig = self.page('03  What the axial encoder changes', 'Paired views of full-vector cosine similarity and relative updates, with actual block-bypass measurements.', size=(21, max(15, 5*count)))
        height = .75/count
        vmax = max(np.percentile(self.b[f'block_{i+1}_relative_change'], 99) for i in range(count))
        for i, stats in enumerate(self.r['encoder_blocks']):
            y = .86-(i+1)*height
            values = self.a[f'axial_{i}']
            ref = values[self.ref]
            sim = values @ ref / (np.linalg.norm(values, axis=-1)*np.linalg.norm(ref)).clip(1e-12)
            for col, (data, title, cmap, limits) in enumerate([
                (sim, f'Block {i+1} output cosine similarity', 'RdBu_r', (-1, 1)),
                (self.b[f'block_{i+1}_relative_change'], f'Block {i+1}: ‖after − before‖ / ‖before‖', 'viridis', (0, vmax))]):
                axes = self.pair(fig, [.05+col*.49, y+.045, .39, height*.57], title)
                for comp, ax in enumerate(axes):
                    im = self.spatial(ax, data, comp, token=True, cmap=cmap, limits=limits)
                fig.colorbar(im, ax=axes, fraction=.025, pad=.025)
            effect = self.r['metrics'][f'without_block_{i+1}']
            fig.text(.055, y+.008, f"Median vector update {stats['median_relative_change']*100:.1f}% · median rotation {stats['median_angle_degrees']:.1f}°"
                     f"     Bypass: max path shift {effect['max_path_shift']:.3f}; final survival {effect['last_confidence']:.3f}", fontsize=11)
        self.note(fig, f"Statistics use all {self.nimage:,} tokens and all {self.c['hidden']} features. Position/history conditioning precedes block 1.\nLayerNorm follows the final block. Slab encoding is independent of this grid; block bypasses leave historical inputs fixed.")
        self.finish(fig, '03_encoder', 'Encoder changes', 'Layer updates, similarities and single-block intervention effects.', pdf)

    def memory(self, pdf):
        valid = np.flatnonzero(self.a['input_history_valid'])
        fig = self.page('04  Historical CT and observed-path slabs',
                        f"{len(valid)} valid slots of {len(self.a['input_history_valid'])}; each 2 × 8 × 65 × 65 slab becomes {self.m['history_tokens_per_slab']} tokens.", size=(22, 17))
        row_height = .48 / max(1, (len(valid)+3)//4)
        for j, slot in enumerate(valid):
            col, row = j % 4, j // 4
            bottom = .86-(row+1)*row_height
            ct, heat = self.a['input_history_slabs'][slot]
            ax = self.axis(fig, [.04+col*.245, bottom+row_height*.37, .205, row_height*.48],
                           f"Slot {slot} {'(seed)' if slot == 0 else ''} · age {self.a['input_history_ages'][slot]:.1f}\nCT + path at slab mid-depth")
            ax.imshow(ct[ct.shape[0]//2], origin='lower', cmap='gray', extent=(-16.25, 16.25, -16.25, 16.25), vmin=0, vmax=1)
            overlay = heat[heat.shape[0]//2]
            ax.imshow(np.ma.masked_less(overlay, .05), origin='lower', cmap='autumn', alpha=.6, vmin=0, vmax=1,
                      extent=(-16.25, 16.25, -16.25, 16.25))
            ax.set(xlabel='slab u (trace voxels)', ylabel='slab v')
            axes = self.pair(fig, [.04+col*.245, bottom, .205, row_height*.18],
                             f'Slot {slot} longitudinal slices')
            for comp, ax in enumerate(axes):
                values = ct[:, 32, :] if comp == 0 else ct[:, :, 32]
                path = heat[:, 32, :] if comp == 0 else heat[:, :, 32]
                extent = (-16.25, 16.25, -2.25, 1.75)
                ax.imshow(values, origin='lower', cmap='gray', vmin=self.ct_limits[0], vmax=self.ct_limits[1],
                          extent=extent, aspect='equal', interpolation='nearest')
                ax.imshow(np.ma.masked_less(path, .05), origin='lower', cmap='autumn',
                          alpha=.6, vmin=0, vmax=1, extent=extent, aspect='equal', interpolation='nearest')
                ax.set(xlabel='slab '+('u' if comp == 0 else 'v'), ylabel='forward')
        attempt = self.m['selected_refinement']
        for col, family in enumerate(('generator', 'scorer')):
            layers = self.c['decoder_layers'] if family == 'generator' else self.c['scorer_layers']
            keys = [k for k in self.a if k.startswith(f'{family}_history_attention_')]
            index = max(int(k.rsplit('_', 1)[1]) for k in keys) if self.m.get('capabilities', {}).get('solver') else attempt*layers+layers-1
            weights = self.a[f'{family}_history_attention_{index}']
            mass = weights.reshape(len(weights), -1, self.m['history_tokens_per_slab']).sum(-1)*100
            ax = self.axis(fig, [.07+col*.49, .13, .38, .17], f'{family.title()}: last-layer history attention by slot')
            im = ax.imshow(mass.T, origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=100,
                           extent=(.5, len(weights)+.5, -.5, mass.shape[1]-.5))
            ax.set(xlabel='Future point / segment', ylabel='History slot')
            fig.colorbar(im, ax=ax, label='% of history attention')
        self.note(fig, 'Slab features include relative pose, age, seed role, spatial position and slot identity. Padded slots are masked. Longitudinal views preserve physical aspect ratio.\nHistory attention has its own softmax and residual update in every generator/scorer layer; its mass is not comparable to image-attention mass.')
        self.finish(fig, '04_memory', 'Historical slabs', 'Real CT/path slabs and separate historical attention reads for the selected attempt.', pdf)

    def interventions(self, pdf):
        fig = self.page('05  Controlled history interventions', 'Current CT, observed references and baseline curve stay fixed. Each intervention reruns proposal, retries and scoring.')
        axes = self.pair(fig, [.045, .48, .43, .37], 'Actual path overlays')
        for comp, ax in enumerate(axes):
            self.spatial(ax, self.a['input'][0], comp, limits=self.ct_limits)
            self.curves(ax, comp, variants=tuple(VARIANTS))
            ax.set(xlim=(-6, 6), ylim=(-1, self.c['n_future']*self.c['future_step']+1))
        ax = self.axis(fig, [.58, .53, .36, .29], 'Prefix survival on each newly predicted curve')
        for i, name in enumerate(VARIANTS):
            ax.plot(np.arange(1, self.c['n_future']+1), self.b[name+'_confidence'], color=COLORS[i], label=VARIANTS[name])
        ax.axhline(self.m['confidence_threshold'], ls='--', color=MUTED)
        ax.set(xlabel='Prefix length', ylabel='Survival probability', ylim=(0, 1.02))
        ax.legend(fontsize=8)
        rows = []
        for name, label in VARIANTS.items():
            metric = self.r['metrics'][name]
            rows.append([label, f"{metric['max_path_shift']:.4f}", f"{metric['last_confidence']:.4f}",
                         f"{metric['fixed_curve_last_confidence']:.4f}", metric['commit'], metric['selected_attempt']+1])
        ax = self.axis(fig, [.05, .12, .89, .25])
        ax.axis('off')
        table = ax.table(cellText=rows, colLabels=['Setting', 'Max shift', 'Own-curve S', 'Fixed-curve S', 'Commit', 'Attempt'],
                         colWidths=[.35, .13, .15, .15, .1, .1], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        self.note(fig, 'Fixed-curve scoring isolates the evidence change from proposal geometry. A scorer change can also alter retry/selection behavior.\nThese interventions can be out of distribution; confidence increases are not evidence of better tracing.')
        self.finish(fig, '05_memory_effects', 'History interventions', 'Separate generator/scorer history reads, both reads disabled, and seed-only evidence.', pdf)

    def attention_page(self, pdf, family):
        generator = family == 'generator'
        title = '06  How the decoder reads current evidence' if generator else '07  How the causal scorer evaluates the path'
        fig = self.page(title, 'Selected attempt, final layer, mean over heads. Each panel is a separate query; spatial maps marginalize one axis.', size=(22, 17))
        layer = self.c['decoder_layers']-1 if generator else self.c['scorer_layers']-1
        index = self.m['selected_refinement']
        if generator and self.m.get('capabilities', {}).get('solver'):
            index = max(int(k.rsplit('_', 1)[1]) for k in self.a if k.startswith(f'{family}_attention_{layer}_'))
        weights = self.a[f'{family}_attention_{layer}_{index}']
        queries = [0, self.c['n_future']-1]
        volumes = [weights[q, :self.nimage].reshape(self.shape)*100 for q in queries]
        vmax = max(volume.sum(axis).max() for volume in volumes for axis in (1, 2))
        for col, (q, volume) in enumerate(zip(queries, volumes)):
            axes = self.pair(fig, [.045+col*.49, .54, .425, .31], f"{'Point' if generator else 'Segment'} {q+1}", 'attention marginal')
            for comp, ax in enumerate(axes):
                im = self.spatial(ax, volume, comp, token=True, marginal=True, cmap='magma', limits=(0, vmax))
                p = self.a['points']
                ax.plot(p[:, comp], p[:, 2], color='#2ee1bb', lw=1.5)
                ax.scatter([p[q, comp]], [p[q, 2]], color='#65f5d5', edgecolors=INK, s=40)
            fig.colorbar(im, ax=axes, fraction=.025, pad=.025, label='% of image/reference attention per bin')
            fig.text(.05+col*.49, .48, f"Image mass: {weights[q, :self.nimage].sum()*100:.2f}% · observed-reference mass: {weights[q, self.nimage:].sum()*100:.2f}%", fontsize=11)
        if generator:
            ax = self.axis(fig, [.055, .14, .4, .23], 'Executed generation steps')
            for attempt, points in enumerate(self.a.get('solver_points', self.a['refinement_points'])):
                ax.plot(points[:, 2], points[:, 0], label=f'Step {attempt}')
            ax.set(xlabel='Forward plane', ylabel='u coordinate')
            ax.legend(fontsize=10)
            flow = self.m['capabilities']['solver']
            description = ('Velocity queries combine image features, lateral state, forward plane and time.\n'
                           'The shared decoder reads image, references and historical tokens.\n'
                           'Midpoint integration starts at lateral x=0, y=0.\n'
                           'Forward planes stay fixed; only the final path receives survival scores.\n'
                           'Integration stages are not separate candidate proposals.' if flow else
                           f"{self.c['hidden']} sampled image features + forward coordinate initialize each query.\n"
                     'Self-attention couples future points. Cross-attention reads image/references,\n'
                     'then a separate residual attention reads historical slab tokens.\n'
                     'LayerNorm and a bounded lateral readout produce u and v.\n'
                     'Forward planes stay fixed. Refinement refreshes path evidence\n'
                     'and uses detached failure/survival feedback with the shared decoder.')
            fig.text(.54, .35, description, fontsize=13, va='top', linespacing=1.7)
        else:
            ax = self.axis(fig, [.055, .15, .4, .22], 'Conditional failure and prefix survival')
            logits = self.a['hazard_logits']
            hazard = np.exp(-np.logaddexp(0, -logits))
            ax.plot(np.arange(1, len(logits)+1), hazard, label='Conditional failure', color=COLORS[3])
            ax.plot(np.arange(1, len(logits)+1), self.a['confidence'], label='Prefix survival', color=COLORS[0])
            ax.axhline(self.m['confidence_threshold'], ls='--', color=MUTED)
            ax.set(xlabel='Segment / prefix', ylim=(0, 1.02))
            ax.legend()
            metric = self.r['metrics']['baseline']
            fig.text(.54, .35, 'Four ordered feature samples describe each incoming segment.\n'
                     'Causal path self-attention excludes the proposed suffix.\n'
                     'Image/reference and history cross-attention read all observed context.\n'
                     'Survival(j) = product over i ≤ j of (1 − failure(i)).\n'
                     f"Selected attempt: {self.m['selected_refinement']+1} / {self.m['attempts']}\n"
                     f"Committed prefix: {metric['commit']} / {self.m['n_commit']} at threshold {self.m['confidence_threshold']:g}", fontsize=13, va='top', linespacing=1.7)
        self.note(fig, 'Attention indicates read allocation, not feature importance. Spatial marginals preserve probability mass; they are not central CT slices.\nHistorical-slab attention is a separate distribution shown on page 04.')
        self.finish(fig, '06_decoder' if generator else '07_confidence', 'Trajectory decoder' if generator else 'Causal confidence',
                    'Measured spatial attention and executed proposals.' if generator else 'Measured scorer attention, conditional failures and monotone survival.', pdf)

    def raw_inputs(self):
        names = ['CT']
        rows = (len(names)+1)//2
        fig = self.page('Exact current-crop inputs', 'Z-scored CT in paired orthogonal central slices.', size=(22, 5*rows+2))
        for k, name in enumerate(names):
            row, col = divmod(k, 2)
            axes = self.pair(fig, [.045+col*.49, .84-(row+1)*(.74/rows), .425, .74/rows*.72], name)
            limits = self.ct_limits if k == 0 else ((-.5, .5) if k >= 5 else (0, 1))
            for comp, ax in enumerate(axes):
                im = self.spatial(ax, self.a['input'][k], comp, cmap='RdBu_r' if k >= 5 else 'gray', limits=limits)
            fig.colorbar(im, ax=axes, fraction=.025, pad=.025)
        for suffix, dpi in [('', 160), ('_preview', 75)]:
            fig.savefig(self.dest / ('input_channels'+suffix+'.png'), dpi=dpi, facecolor=BG)
        plt.close(fig)

    def browser(self):
        m = self.m
        header = f"Fiber {m['fiber_index']} · replay row {m['replay_row']} · step {m['checkpoint_step']:,} EMA · {m['model_type']}"
        sections = ''.join(f'<section id="{p["name"]}"><h2>{escape(p["title"])}</h2><p>{escape(p["summary"])}</p>'
                           f'<a href="{p["name"]}.png"><img loading="lazy" src="{p["name"]}_preview.png" alt="{escape(p["title"])}"></a></section>' for p in self.pages)
        links = ''.join(f'<a href="#{p["name"]}">{escape(p["title"])}</a>' for p in self.pages)
        html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        html += '<title>Fiber model interpretation</title><style>body{background:#f1f5f9;color:#172b43;font:17px/1.5 system-ui;margin:auto;max-width:1600px;padding:24px}img{width:100%;height:auto}nav{display:flex;gap:20px;flex-wrap:wrap}section{margin:50px 0}a{color:#246da8}pre{white-space:pre-wrap}</style>'
        html += f'<h1>What the model sees, reads and decides</h1><p>{escape(header)}</p><p>{escape(m["history_source"])}</p>'
        html += '<p><a href="model_interpretation.pdf">Complete PDF</a> · <a href="README.txt">Methods and command</a> · <a href="analysis_summary.json">Metrics</a> · <a href="validation.json">Validation</a></p>'
        html += f'<nav>{links}</nav>{sections}<h2>Exact input channels</h2><a href="input_channels.png"><img src="input_channels_preview.png" alt="Exact input channels"></a></html>'
        (self.dest / 'index.html').write_text(html)
        (self.dest / 'README.txt').write_text(header+'\n\n'+self.r['protocol']+'\n\nHistory: '+m['history_source']+
            '\n\nCurrent model: overlapping 6-cube stride-4 convolution, axial token encoder, separate live CT/path slab encoder, '
            'per-layer history residual reads, coordinate refinement or flow integration, and causal segment survival.\n'
            '\nInstrumentation is removed before an exactly equal baseline rerun. All interventions are restored and the baseline is checked again.\n'
            'Spatial overlays are projections over central CT slices. Attention marginals sum the omitted dimension and average heads; '
            'historical attention has an independent normalization. Cosine and update maps summarize all hidden features.\n'
            '\nReproduce from fiber_follow with the existing project Python environment:\n'+m['command']+'\n'
            '\nUse a fresh --output-dir on reruns. sample_provenance.json records checkpoint SHA256, input sources, shapes, and old-input comparison.\n'
            '\nOld-atlas comparison:\n'+json.dumps(m['prior_comparison'], indent=2)+'\n')
        (self.dest / 'figure_manifest.json').write_text(json.dumps(dict(pages=self.pages, spatial_pairs=self.pairs), indent=2))


def render_common(directory):
    """A useful core report even without component-specific instrumentation."""
    dest = Path(directory)
    meta = json.loads((dest/'sample_provenance.json').read_text())
    with np.load(dest/'sample_activations.npz') as data:
        a = dict(data)
    fig, axes = plt.subplots(1, 3, figsize=(20, 15))
    crop = meta['model_cfg']['fine']
    ct = a['input'][0]
    half = crop['width']*crop['spacing']/2
    extent = (-half, half, (-crop['behind']-.5)*crop['spacing'],
              (crop['depth']-crop['behind']-.5)*crop['spacing'])
    for axis, component in zip(axes[:2], (0, 1)):
        middle = (ct.shape[1]-1)/2
        indices = [int(np.floor(middle)), int(np.ceil(middle))]
        section = ct[:, indices, :].mean(1) if component == 0 else ct[:, :, indices].mean(2)
        axis.imshow(section, origin='lower', extent=extent, cmap='gray', vmin=-4, vmax=4, aspect='auto')
        axis.plot(a['hist'][a['hmask'] > 0, component], a['hist'][a['hmask'] > 0, 2], label='Observed history')
        for i, points in enumerate(a.get('solver_points', a['refinement_points'])):
            axis.plot(points[:, component], points[:, 2], label=f'Generation step {i}')
        axis.plot(a['points'][:, component], a['points'][:, 2], label='Selected path', linewidth=3)
        axis.set(xlabel=('u', 'v')[component], ylabel='Forward plane'); axis.legend()
    axes[2].plot(a['confidence']); axes[2].set(xlabel='Prefix', ylabel='Survival confidence', ylim=(0, 1))
    fig.suptitle('Model decision — component attention and interventions unavailable')
    for suffix, dpi in [('', 160), ('_preview', 75)]: fig.savefig(dest/('01_decision'+suffix+'.png'), dpi=dpi)
    fig.savefig(dest/'01_decision.pdf'); fig.savefig(dest/'model_interpretation.pdf'); plt.close(fig)
    pages = [dict(name='01_decision', title='Model decision', summary='CT, observed history, generated path and confidence')]
    (dest/'figure_manifest.json').write_text(json.dumps(dict(pages=pages, spatial_pairs=[])))
    (dest/'index.html').write_text('<!doctype html><title>Model decision</title><h1>Model decision</h1>'
        '<p>Component attention and history interventions are unavailable.</p><img src="01_decision_preview.png">')


def render(directory):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    meta = json.loads((Path(directory)/'sample_provenance.json').read_text())
    if not all(meta.get('capabilities', {}).get(key) for key in ('patch_features', 'history', 'attention')):
        return render_common(directory)
    atlas = Atlas(directory)
    with PdfPages(atlas.dest / 'model_interpretation.pdf') as pdf:
        for method in (atlas.overview, atlas.inputs, atlas.encoder, atlas.memory, atlas.interventions):
            method(pdf)
        atlas.attention_page(pdf, 'generator')
        atlas.attention_page(pdf, 'scorer')
    atlas.raw_inputs()
    atlas.browser()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    render(parser.parse_args().directory)
