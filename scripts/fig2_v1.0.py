"""

**IN DEVELOPMENT**

Chaplin Figure 2 Recreation

Goal:
    Construct approximate absolute carbon spectra comparable to Figure 2
    of Chaplin.

The script:
    1. Computes metastable-resolved carbon equilibrium populations.
    2. Computes bound-bound PECs for C I through C VI.
    3. Weights each PEC by its corresponding metastable population.
    4. Converts PECs into line-of-sight photon brightness.
    5. Broadens each line in wavelength space.
    6. Converts the spectrum from photons per nm into:

        erg s^-1 cm^-2 sr^-1 eV^-1

Author: Ryan Rauenzahn
Date: July 10, 2026
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from colradpy import colradpy
from colradpy.ionization_balance_class import ionization_balance

# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ATOMIC_DATA_DIR = PROJECT_ROOT / "atomic_data"

FILES = [
    str(ATOMIC_DATA_DIR / "mom97_ls#c0.dat"),  # C I
    str(ATOMIC_DATA_DIR / "mom97_ls#c1.dat"),  # C II
    str(ATOMIC_DATA_DIR / "mom97_ls#c2.dat"),  # C III
    str(ATOMIC_DATA_DIR / "mom97_ls#c3.dat"),  # C IV
    str(ATOMIC_DATA_DIR / "mom97_ls#c4.dat"),  # C V
    str(ATOMIC_DATA_DIR / "mom97_ls#c5.dat"),  # C VI
]

ION_LABELS = [
    "C I",
    "C II",
    "C III",
    "C IV",
    "C V",
    "C VI",
]

METAS = [
    np.array([0]),      # C I
    np.array([0, 1]),   # C II
    np.array([0, 1]),   # C III
    np.array([0]),      # C IV
    np.array([0, 1]),   # C V
    np.array([0]),      # C VI
]

# ============================================================
# State locations in the equilibrium eigenvector
# ============================================================

STATE_SLICES = {
    "C I":  slice(0, 1),
    "C II": slice(1, 3),
    "C III": slice(3, 5),
    "C IV": slice(5, 6),
    "C V":  slice(6, 8),
    "C VI": slice(8, 9),
}

# State 9 is C VII, the fully stripped carbon population-it participates in ionization balance but emits no bound-bound lines.

# ============================================================
# Plasma conditions
# ============================================================

TEMPERATURES_EV = np.array([
    20.0,
    40.0,
])

ELECTRON_DENSITY_CM3 = 5.0e14

# Approximate carbon abundance.
CARBON_FRACTION = 0.01

CARBON_DENSITY_CM3 = (
    CARBON_FRACTION * ELECTRON_DENSITY_CM3
)

PLASMA_LENGTH_CM = 40.0

NE = np.array([
    ELECTRON_DENSITY_CM3,
])

# ============================================================
# Physical constants
# ============================================================

EV_TO_ERG = 1.602176634e-12
HC_EV_NM = 1239.841984

# ============================================================
# Spectral grid
# ============================================================

WAVELENGTH_MIN_NM = 50.0
WAVELENGTH_MAX_NM = 300.0

N_WAVELENGTH_POINTS = 50000

WAVELENGTH_GRID_NM = np.linspace(
    WAVELENGTH_MIN_NM,
    WAVELENGTH_MAX_NM,
    N_WAVELENGTH_POINTS,
)

# Broadened line width in wavelength space.

GAUSSIAN_SIGMA_NM = 0.02
GAUSSIAN_WINDOW_SIGMA = 6.0

# ============================================================
# Plot limits
# ============================================================

INTENSITY_MIN = 1.0e0
INTENSITY_MAX = 1.0e11

# ============================================================
# Optional line threshold
# ============================================================

# Integrated line brightness threshold before broadening.
# Set to 0.0 to retain all positive lines.
MIN_LINE_PHOTON_BRIGHTNESS = 0.0

# ============================================================
# File checks
# ============================================================

def check_input_files(files):
    """Verify that every required ADF04 file exists."""

    for file in files:
        if not Path(file).exists():
            raise FileNotFoundError(
                f"Missing ADF04 file: {file}"
            )

# ============================================================
# Ionization balance
# ============================================================

def build_ionization_balance(te, ne):
    """
    Build and solve the metastable-resolved carbon ionization balance.
    """

    ib = ionization_balance(
        FILES,
        METAS,
        te,
        ne,
        use_ionization=True,
        suppliment_with_ecip=True,
        use_recombination_three_body=True,
        use_recombination=True,
        use_cx=False,
        keep_charge_state_data=False,
    )

    ib.populate_ion_matrix()
    ib.solve_no_source()

    return ib

def get_equilibrium_populations(ib):
    """
    Extract the normalized metastable-resolved equilibrium population
    vector from the eigenmode whose eigenvalue is closest to zero.

    Returns
    -------
    equilibrium_populations : ndarray
        Shape:

            temperature x population state
    """

    eigvals = np.asarray(
        ib.data["processed"]["eigen_val"]
    )

    eigvecs = np.asarray(
        ib.data["processed"]["eigen_vec"]
    )

    n_temperatures = eigvals.shape[0]
    n_states = eigvecs.shape[2]

    equilibrium_populations = np.zeros(
        (n_temperatures, n_states)
    )

    for i_temperature in range(n_temperatures):

        zero_mode_index = np.argmin(
            np.abs(
                eigvals[
                    i_temperature,
                    0,
                    :,
                ]
            )
        )

        population_vector = np.real(
            eigvecs[
                i_temperature,
                0,
                :,
                zero_mode_index,
            ]
        )

        # Eigenvector sign is arbitrary.
        if np.sum(population_vector) < 0.0:
            population_vector *= -1.0

        population_sum = np.sum(population_vector)

        if population_sum == 0.0:
            raise ValueError(
                "Equilibrium eigenvector has zero total population."
            )

        population_vector /= population_sum

        equilibrium_populations[
            i_temperature,
            :
        ] = population_vector

    return equilibrium_populations

def print_equilibrium_populations(equilibrium_populations):
    """Print metastable-resolved and charge-state populations."""

    for i_temperature, temperature in enumerate(
        TEMPERATURES_EV
    ):
        populations = equilibrium_populations[
            i_temperature,
            :
        ]

        print("\n" + "=" * 70)
        print(
            f"Equilibrium populations at "
            f"Te = {temperature:.1f} eV"
        )
        print("=" * 70)

        for ion_label in ION_LABELS:
            state_population = populations[
                STATE_SLICES[ion_label]
            ]

            print(
                f"{ion_label:5s}: "
                f"total = {np.sum(state_population):.6e}, "
                f"metastables = {state_population}"
            )

        print(
            f"C VII: total = {populations[9]:.6e}"
        )

        print(
            f"Sum:   {np.sum(populations):.6e}"
        )


# ============================================================
# Single-ion collisional-radiative model
# ============================================================

def solve_single_ion(
    adf04_file,
    metastables,
    te,
    ne,
):
    """
    Solve one ion's collisional-radiative model.

    Ionization and recombination are disabled because charge-state and
    metastable populations are supplied by the separate equilibrium model.
    """

    cr = colradpy(
        adf04_file,
        metastables,
        te,
        ne,
        use_ionization=False,
        suppliment_with_ecip=False,
        use_recombination=False,
        use_recombination_three_body=False,
        use_cx=False,
    )

    cr.solve_cr()

    return cr


def extract_line_data(cr):
    """
    Extract transition wavelengths, PECs, and level indices.
    """

    wavelengths_nm = np.asarray(
        cr.data["processed"]["wave_vac"]
    )

    pecs = np.asarray(
        cr.data["processed"]["pecs"]
    )

    pec_levels = np.asarray(
        cr.data["processed"]["pec_levels"]
    )

    if pecs.ndim != 4:
        raise ValueError(
            "Unexpected PEC array shape. "
            f"Expected four dimensions but received {pecs.shape}."
        )

    return wavelengths_nm, pecs, pec_levels


# ============================================================
# Physical brightness calculations
# ============================================================

def calculate_integrated_photon_brightness(
    effective_pec,
):
    """
    Convert a population-weighted PEC into integrated photon brightness.

    Since effective_pec already includes the fractional carbon population
    in the relevant metastable states:

        B_ph =
            ne * n_carbon * effective_PEC * L / (4 pi)

    Units:

        photons cm^-2 s^-1 sr^-1
    """

    return (
        ELECTRON_DENSITY_CM3
        * CARBON_DENSITY_CM3
        * effective_pec
        * PLASMA_LENGTH_CM
        / (4.0 * np.pi)
    )

# ============================================================
# Gaussian broadening
# ============================================================

def add_area_normalized_gaussian(
    spectrum,
    wavelength_grid_nm,
    center_nm,
    integrated_photon_brightness,
    sigma_nm,
):
    """
    Add an area-normalized Gaussian line in wavelength space.

    Input line brightness:

        photons cm^-2 s^-1 sr^-1

    Output spectrum units:

        photons cm^-2 s^-1 sr^-1 nm^-1
    """

    lower_wavelength = (
        center_nm
        - GAUSSIAN_WINDOW_SIGMA * sigma_nm
    )

    upper_wavelength = (
        center_nm
        + GAUSSIAN_WINDOW_SIGMA * sigma_nm
    )

    lower_index = np.searchsorted(
        wavelength_grid_nm,
        lower_wavelength,
        side="left",
    )

    upper_index = np.searchsorted(
        wavelength_grid_nm,
        upper_wavelength,
        side="right",
    )

    lower_index = max(
        lower_index,
        0,
    )

    upper_index = min(
        upper_index,
        len(wavelength_grid_nm),
    )

    if lower_index >= upper_index:
        return

    local_wavelengths = wavelength_grid_nm[
        lower_index:upper_index
    ]

    gaussian_normalization = (
        1.0
        / (
            sigma_nm
            * np.sqrt(2.0 * np.pi)
        )
    )

    profile_per_nm = (
        gaussian_normalization
        * np.exp(
            -0.5
            * (
                (
                    local_wavelengths
                    - center_nm
                )
                / sigma_nm
            ) ** 2
        )
    )

    spectrum[
        lower_index:upper_index
    ] += (
        integrated_photon_brightness
        * profile_per_nm
    )

# ============================================================
# Unit conversion
# ============================================================

def convert_photon_spectrum_to_energy_spectrum(
    wavelength_grid_nm,
    photon_spectra_per_nm,
):
    """
    Convert photon spectral brightness per nm into energy intensity per eV.

    Input units:

        photons cm^-2 s^-1 sr^-1 nm^-1

    Output units:

        erg s^-1 cm^-2 sr^-1 eV^-1

    Conversion:

        I_E = I_lambda * E_photon(erg) * |d lambda / dE|

    with:

        E_photon(eV) = hc / lambda

        |d lambda / dE| = lambda^2 / hc
    """

    photon_energy_ev = (
        HC_EV_NM / wavelength_grid_nm
    )

    photon_energy_erg = (
        photon_energy_ev * EV_TO_ERG
    )

    jacobian_nm_per_ev = (
        wavelength_grid_nm**2
        / HC_EV_NM
    )

    energy_spectra_per_ev = (
        photon_spectra_per_nm
        * photon_energy_erg[np.newaxis, :]
        * jacobian_nm_per_ev[np.newaxis, :]
    )

    return energy_spectra_per_ev

# ============================================================
# Spectrum construction
# ============================================================

def build_photon_spectra(equilibrium_populations):
    """
    Build carbon bound-bound spectra in photon units per nm.
    """

    photon_spectra = np.zeros(
        (
            len(TEMPERATURES_EV),
            len(WAVELENGTH_GRID_NM),
        )
    )

    line_records = {
        temperature: []
        for temperature in TEMPERATURES_EV
    }

    for ion_label, adf04_file, metastables in zip(
        ION_LABELS,
        FILES,
        METAS,
    ):
        print("\n" + "-" * 70)
        print(f"Processing {ion_label}")
        print("-" * 70)

        cr = solve_single_ion(
            adf04_file,
            metastables,
            TEMPERATURES_EV,
            NE,
        )

        wavelengths_nm, pecs, pec_levels = extract_line_data(
            cr
        )

        print(
            f"Transitions returned: {len(wavelengths_nm)}"
        )

        for i_temperature, temperature in enumerate(
            TEMPERATURES_EV
        ):
            metastable_populations = (
                equilibrium_populations[
                    i_temperature,
                    STATE_SLICES[ion_label],
                ]
            )

            pec_by_metastable = pecs[
                :,
                :,
                i_temperature,
                0,
            ]

            if (
                pec_by_metastable.shape[1]
                != len(metastable_populations)
            ):
                raise ValueError(
                    f"{ion_label}: PEC metastable dimension "
                    f"{pec_by_metastable.shape[1]} does not match "
                    f"population count {len(metastable_populations)}."
                )

            # Each metastable PEC is multiplied by the absolute fractional
            # population of carbon in that metastable.
            effective_pec = np.sum(
                pec_by_metastable
                * metastable_populations[np.newaxis, :],
                axis=1,
            )

            valid = (
                np.isfinite(wavelengths_nm)
                & np.isfinite(effective_pec)
                & (effective_pec > 0.0)
                & (
                    wavelengths_nm
                    >= WAVELENGTH_MIN_NM
                )
                & (
                    wavelengths_nm
                    <= WAVELENGTH_MAX_NM
                )
            )

            valid_indices = np.where(valid)[0]

            print(
                f"Te = {temperature:4.1f} eV: "
                f"{len(valid_indices)} positive transitions "
                f"in wavelength range"
            )

            for transition_index in valid_indices:

                wavelength_nm = wavelengths_nm[
                    transition_index
                ]

                transition_effective_pec = effective_pec[
                    transition_index
                ]

                integrated_photon_brightness = (
                    calculate_integrated_photon_brightness(
                        transition_effective_pec
                    )
                )

                if (
                    not np.isfinite(
                        integrated_photon_brightness
                    )
                ):
                    continue

                if (
                    integrated_photon_brightness
                    <= MIN_LINE_PHOTON_BRIGHTNESS
                ):
                    continue

                add_area_normalized_gaussian(
                    spectrum=photon_spectra[
                        i_temperature,
                        :
                    ],
                    wavelength_grid_nm=WAVELENGTH_GRID_NM,
                    center_nm=wavelength_nm,
                    integrated_photon_brightness=(
                        integrated_photon_brightness
                    ),
                    sigma_nm=GAUSSIAN_SIGMA_NM,
                )

                photon_energy_erg = (
                    HC_EV_NM
                    / wavelength_nm
                    * EV_TO_ERG
                )

                integrated_energy_brightness = (
                    integrated_photon_brightness
                    * photon_energy_erg
                )

                line_records[temperature].append(
                    {
                        "ion": ion_label,
                        "wavelength_nm": wavelength_nm,
                        "effective_pec": (
                            transition_effective_pec
                        ),
                        "photon_brightness": (
                            integrated_photon_brightness
                        ),
                        "energy_brightness": (
                            integrated_energy_brightness
                        ),
                        "levels": pec_levels[
                            transition_index
                        ],
                    }
                )

    return photon_spectra, line_records

# ============================================================
# Diagnostics
# ============================================================

def print_strongest_lines(
    line_records,
    number_to_print=30,
):
    """
    Print strongest integrated-energy lines at each temperature.
    """

    for temperature in TEMPERATURES_EV:

        sorted_lines = sorted(
            line_records[temperature],
            key=lambda line: line["energy_brightness"],
            reverse=True,
        )

        print("\n" + "=" * 90)
        print(
            f"Strongest integrated lines at "
            f"Te = {temperature:.1f} eV"
        )
        print("=" * 90)

        for line in sorted_lines[:number_to_print]:

            print(
                f"{line['ion']:5s}  "
                f"{line['wavelength_nm']:10.4f} nm  "
                f"energy brightness = "
                f"{line['energy_brightness']:.4e} "
                f"erg cm^-2 s^-1 sr^-1  "
                f"effective PEC = "
                f"{line['effective_pec']:.4e}  "
                f"levels = {line['levels']}"
            )

# ============================================================
# Plotting
# ============================================================

def plot_energy_spectra(
    energy_spectra_per_ev,
):
    """
    Plot the spectra in Chaplin-style energy-intensity units.
    """

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 8),
        sharex=True,
    )

    for i_temperature, (
        temperature,
        axis,
    ) in enumerate(
        zip(
            TEMPERATURES_EV,
            axes,
        )
    ):
        spectrum = energy_spectra_per_ev[
            i_temperature,
            :
        ]

        # Logarithmic axes cannot display zero.
        plotted_spectrum = np.where(
            spectrum > 0.0,
            spectrum,
            np.nan,
        )

        axis.semilogy(
            WAVELENGTH_GRID_NM,
            plotted_spectrum,
            linewidth=1.0,
        )

        axis.set_ylabel(
            r"Intensity"
            "\n"
            r"(erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ eV$^{-1}$)"
        )

        axis.set_title(
            rf"$T_e={temperature:g}$ eV, "
            rf"$n_e={ELECTRON_DENSITY_CM3:.1e}\ "
            rf"\mathrm{{cm^{{-3}}}}$"
        )

        axis.set_xlim(
            WAVELENGTH_MIN_NM,
            WAVELENGTH_MAX_NM,
        )

        axis.set_ylim(
            INTENSITY_MIN,
            INTENSITY_MAX,
        )

        axis.grid(
            True,
            which="both",
            alpha=0.3,
        )

    axes[-1].set_xlabel(
        "Wavelength (nm)"
    )

    plt.tight_layout()

    output_file = (
        PROJECT_ROOT
        / "fig2_v3_0_energy_spectra.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
    )

    plt.show()

    print(f"\nSaved: {output_file}")

# ============================================================
# Main
# ============================================================

def main():
    check_input_files(FILES)

    print("\n--- Model configuration ---")

    print(
        f"Electron density: "
        f"{ELECTRON_DENSITY_CM3:.4e} cm^-3"
    )

    print(
        f"Carbon fraction:  "
        f"{CARBON_FRACTION:.4e}"
    )

    print(
        f"Carbon density:   "
        f"{CARBON_DENSITY_CM3:.4e} cm^-3"
    )

    print(
        f"Plasma length:    "
        f"{PLASMA_LENGTH_CM:.4f} cm"
    )

    print(
        f"Gaussian sigma:   "
        f"{GAUSSIAN_SIGMA_NM:.4f} nm"
    )

    print(
        f"Gaussian FWHM:    "
        f"{2.35482 * GAUSSIAN_SIGMA_NM:.4f} nm"
    )

    print("\n--- Building ionization balance ---")

    ib = build_ionization_balance(
        TEMPERATURES_EV,
        NE,
    )

    equilibrium_populations = (
        get_equilibrium_populations(ib)
    )

    print_equilibrium_populations(
        equilibrium_populations
    )

    print("\n--- Building photon spectra ---")

    photon_spectra_per_nm, line_records = (
        build_photon_spectra(
            equilibrium_populations
        )
    )

    print("\n--- Converting to energy intensity per eV ---")

    energy_spectra_per_ev = (
        convert_photon_spectrum_to_energy_spectrum(
            WAVELENGTH_GRID_NM,
            photon_spectra_per_nm,
        )
    )

    print_strongest_lines(
        line_records,
        number_to_print=30,
    )

    for i_temperature, temperature in enumerate(
        TEMPERATURES_EV
    ):
        spectrum = energy_spectra_per_ev[
            i_temperature,
            :
        ]

        finite_positive = spectrum[
            np.isfinite(spectrum)
            & (spectrum > 0.0)
        ]

        print(
            f"\nTe = {temperature:.1f} eV spectrum diagnostics:"
        )

        if len(finite_positive) == 0:
            print("  No positive spectral values.")
        else:
            print(
                f"  Minimum positive intensity: "
                f"{np.min(finite_positive):.4e}"
            )

            print(
                f"  Maximum intensity: "
                f"{np.max(finite_positive):.4e}"
            )

    plot_energy_spectra(
        energy_spectra_per_ev
    )

    print("\nDone.")

if __name__ == "__main__":
    main()