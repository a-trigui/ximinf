from astropy.cosmology import Planck18
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_HD(
    example,
    params,
    default_params,
    N_total,
    survey_name=None,
    images_dir=None,
    n_bins=15,
):
    # ============================================
    # Simulation parameters
    # ============================================

    mabs = params['mabs']
    beta = params['beta']
    alpha = params['alpha']

    # ============================================
    # Data from simulation
    # ============================================

    z = np.asarray(example['z'])
    magobs = np.asarray(example['magobs'])
    c = np.asarray(example['c'])

    # ============================================
    # Apply selection
    # ============================================

    mask = magobs > 0

    z = z[mask]
    magobs = magobs[mask]
    c = c[mask]

    # ============================================
    # Number of SNe
    # ============================================

    N_sne_selected = len(z)

    # If N_total is the number before selection
    N_sne = N_total

    # ============================================
    # Fiducial cosmology distance modulus
    # ============================================

    mu_th = Planck18.distmod(z).value

    # ============================================
    # Raw Hubble residuals
    # ============================================

    residuals = magobs - mu_th - mabs

    # ============================================
    # Equal-count (quantile) binning
    # ============================================

    mask_finite = (
        np.isfinite(z)
        & np.isfinite(residuals)
    )

    z_clean = z[mask_finite]
    res_clean = residuals[mask_finite]

    # Quantile bins
    bins = np.quantile(
        z_clean,
        np.linspace(0, 1, n_bins + 1)
    )

    bins = np.unique(bins)

    bin_idx = np.digitize(z_clean, bins) - 1

    z_bin = []
    res_bin = []
    res_err = []

    for i in range(len(bins) - 1):

        mask_bin = bin_idx == i

        if np.sum(mask_bin) == 0:
            continue

        z_i = z_clean[mask_bin]
        r_i = res_clean[mask_bin]

        z_bin.append(np.mean(z_i))
        res_bin.append(np.mean(r_i))
        res_err.append(np.std(r_i) / np.sqrt(len(r_i)))

    z_bin = np.array(z_bin)
    res_bin = np.array(res_bin)
    res_err = np.array(res_err)

    # ============================================
    # Colormap
    # ============================================

    cmap = LinearSegmentedColormap.from_list(
        'custom_red_beige_blue',
        ['#1F487E', 'beige', '#A31621']
    )

    norm = TwoSlopeNorm(
        vmin=-0.2,
        vcenter=0.0,
        vmax=0.3
    )

    # ============================================
    # Figure with two panels
    # ============================================

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # ============================================
    # TOP PANEL: observed magnitudes
    # ============================================

    scatter = ax1.scatter(
        z,
        magobs - mabs,
        c=c,
        cmap=cmap,
        norm=norm,
        edgecolor='none',
        alpha=0.8,
    )

    # ============================================
    # Fiducial cosmology curve
    # ============================================

    z_plot = np.linspace(
        z.min(),
        z.max(),
        1000
    )

    mu_plot = Planck18.distmod(z_plot).value

    ax1.plot(
        z_plot,
        mu_plot,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Planck18'
    )

    # ============================================
    # Low-z selection cut
    # ============================================

    if survey_name is not None:

        if f'cut_loc_{survey_name}' not in params:
            cut_loc = default_params[f'cut_loc_{survey_name}']
        else:
            cut_loc = params[f'cut_loc_{survey_name}']

        # Convert observed-magnitude cut to the quantity
        # plotted on the y-axis: magobs - mabs
        ax1.hlines(
            cut_loc - mabs,
            xmin=z.min(),
            xmax=z.max(),
            colors='grey',
            linestyles='dashed',
            linewidth=1.5,
            label='low-$z$ cut'
        )

    ax1.legend()

    # ============================================
    # Histogram above top panel
    # ============================================

    divider = make_axes_locatable(ax1)

    ax_hist = divider.append_axes(
        "top",
        1.2,
        pad=0.0,
        sharex=ax1
    )

    ax_hist.hist(
        z,
        bins=20,
        color='gray',
        edgecolor='black'
    )

    ax_hist.axis('off')

    # ============================================
    # Inset histogram of colour parameter c
    # ============================================

    color_clean_hist = c[np.isfinite(c)]

    ax_inset = ax1.inset_axes(
        [0.55, 0.2, 0.35, 0.25]
    )

    counts, bins_hist = np.histogram(
        color_clean_hist,
        bins=20
    )

    bin_centers = 0.5 * (
        bins_hist[:-1] + bins_hist[1:]
    )

    bin_colors = cmap(norm(bin_centers))

    ax_inset.bar(
        bins_hist[:-1],
        counts,
        width=bins_hist[1:] - bins_hist[:-1],
        color=bin_colors,
        edgecolor='black'
    )

    ax_inset.set_xlabel(r'$c$')
    ax_inset.set_yticks([])
    ax_inset.set_frame_on(False)

    # ============================================
    # BOTTOM PANEL: residuals
    # ============================================

    ax2.scatter(
        z,
        residuals,
        c=c,
        cmap=cmap,
        norm=norm,
        edgecolor='none',
        alpha=0.4,
        s=10
    )

    # ============================================
    # Binned residuals
    # ============================================

    ax2.errorbar(
        z_bin,
        res_bin,
        yerr=res_err,
        fmt='o',
        color='black',
        markersize=5,
        capsize=3
    )

    # ============================================
    # Reference line
    # ============================================

    ax2.axhline(
        0.0,
        color='black',
        linestyle='--',
        linewidth=1
    )

    # ============================================
    # Labels
    # ============================================

    ax1.set_ylabel(r'$\mu$')
    ax2.set_xlabel(r'Redshift $z$')
    ax2.set_ylabel(r'$\Delta\mu$')

    # ============================================
    # Figure title
    # ============================================

    fig.suptitle(
        rf'$M_{{\rm abs}} = {mabs:.2f}$, '
        rf'$\beta = {beta:.2f}$, '
        rf'$\alpha = {alpha:.2f}$'
        '\n'
        rf'$N_{{\rm SNe}} = {N_sne}$, '
        rf'$N_{{\rm selected}} = {N_sne_selected}$',
        fontsize=12
    )

    # ============================================
    # Layout
    # ============================================

    plt.tight_layout()

    plt.subplots_adjust(
        hspace=0.05,
        top=0.90
    )

    # ============================================
    # Save figure
    # ============================================

    if images_dir is not None:
        plt.savefig(
            images_dir / f"sim_example.png",
            bbox_inches='tight'
        )

    plt.show()

    # ============================================
    # Diagnostics
    # ============================================

    print('Mean residual:', np.mean(residuals))
    print('Residual scatter:', np.std(residuals))

    return fig, (ax1, ax2)