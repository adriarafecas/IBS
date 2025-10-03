#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilayer slab — Complex conductivity σ*(ω) & resistivity ρ*(ω) using the Gabriel model

Features
--------
- Console UI: choose any number of layers; each with thickness (mm/µm/nm/m).
- If a tissue name is not in the built-in DB, you can enter its Gabriel parameters on the fly.
- Electromagnetic combination: ALWAYS SERIES along the normal field (layered stack).
- Frequency range: 1 kHz – 100 MHz.
- Plots: user can choose one combined plot (σ* on left axis, ρ* on right axis) or two separate plots.

Math (kept exactly as requested)
--------------------------------
1) Per tissue (Gabriel / multi-term Cole–Cole):
   εr*(ω) = ε∞ + Σ_n Δε_n / [1 + (jωτ_n)^(1-α_n)] + σ_s/(jωε0)
   ε*(ω)  = ε0 εr*(ω)
   σ*(ω)  = jω ε*(ω) = jω ε0 εr*(ω)
   ρ*(ω)  = 1 / σ*(ω)

2) Series stacking (normal field), thickness-weighted:
   εr,eq*(ω) = d_tot / Σ_i [ d_i / εr,i*(ω) ]
   (Equivalently in conductivity: σeq*(ω) = d_tot / Σ_i [ d_i / σi*(ω) ],
    but we implement the εr* formula above and then convert to σ* and ρ*.)
"""

import sys
import cmath
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Physical constant
# -----------------------------
EPS0 = 8.854187817e-12  # F/m
PI = np.pi

# -----------------------------
# Gabriel DB (excerpt from 1996/97 tables)
# Units:
#   - τ in seconds
#   - σ in S/m
#   - ε∞ and Δε are dimensionless (relative permittivity)
# -----------------------------
DB = {
    "blood": {
        "eps_inf": 4.0,
        "sigma_s": 0.700,
        "terms": [
            (56.0,   8.38e-12, 0.10),
            (5200.0, 132.63e-9, 0.10),
            (2.0e4,  159.15e-6, 0.20),
            (2.0e7,  15.915e-3, 0.00),
        ],
    },
    "bone_cancellous": {
        "eps_inf": 2.5,
        "sigma_s": 0.020,
        "terms": [
            (18.0,   13.26e-12, 0.22),
            (300.0,  79.58e-9,  0.25),
            (2.0e4,  159.15e-6, 0.20),
            (4.5e7,  5.305e-3,  0.02),
        ],
    },
    "bone_cortical": {
        "eps_inf": 2.5,
        "sigma_s": 0.020,
        "terms": [
            (10.0,   13.26e-12, 0.20),
            (180.0,  79.58e-9,  0.20),
            (5.0e3,  159.15e-6, 0.20),
            (1.0e5,  15.915e-3, 0.00),
        ],
    },
    "fat_infiltrated": {
        "eps_inf": 2.5,
        "sigma_s": 0.035,
        "terms": [
            (9.0,   7.96e-12, 0.20),
            (20.0,  15.92e-9, 0.10),
            (1.0e4, 159.15e-6, 0.20),
            (1.0e5, 15.915e-3, 0.01),
        ],
    },
    "fat_not_infiltrated": {
        "eps_inf": 2.5,
        "sigma_s": 0.010,
        "terms": [
            (3.0,   7.96e-12, 0.20),
            (15.0,  15.92e-9, 0.10),
            (1.0e4, 159.15e-6, 0.20),
            (1.0e5, 7.958e-3, 0.01),
        ],
    },
    "heart": {  # muscle proxy
        "eps_inf": 4.0,
        "sigma_s": 0.700,
        "terms": [
            (50.0,   7.96e-12, 0.10),
            (1200.0, 159.15e-9, 0.05),
            (4.5e5,  72.34e-6, 0.02),
            (2.5e7,  4.547e-3, 0.05),
        ],
    },
}

# -----------------------------
# Gabriel / Cole–Cole (multi-term)
# Returns εr*(ω) — complex relative permittivity
# -----------------------------
def gabriel_epsr(freq_hz, eps_inf, sigma_s, terms):
    w = 2 * PI * freq_hz
    eps_r = np.full_like(freq_hz, complex(eps_inf, 0.0), dtype=complex)
    for (d_eps, tau, alpha) in terms:
        # (jωτ)^(1-α) = exp( (1-α) * log(jωτ) ) — numerically robust
        one_minus_alpha = 1.0 - alpha
        jwta = 1j * w * tau
        term = np.array([cmath.exp(one_minus_alpha * cmath.log(z)) for z in jwta], dtype=complex)
        eps_r += d_eps / (1.0 + term)
    # Conduction term inside εr*: + σs / (j ω ε0)
    eps_r += sigma_s / (1j * w * EPS0)
    return eps_r

# -----------------------------
# Series combination for layered stack (normal field)
# εr,eq*(ω) = d_tot / Σ_i [ d_i / εr,i*(ω) ]
# -----------------------------
def combine_series_epsr(epsr_list, d_list):
    d_tot = np.sum(d_list)
    acc = np.zeros_like(epsr_list[0], dtype=complex)
    for epsr_i, d_i in zip(epsr_list, d_list):
        acc += d_i / epsr_i
    return d_tot / acc

# -----------------------------
# Console helpers
# -----------------------------
def parse_thickness(s: str) -> float:
    """
    Parse thickness with units (mm, µm/um, nm, m). Empty input -> 1.0 (relative).
    Returns thickness in meters (or 1.0 as a relative weight if empty).
    """
    s = s.strip().lower()
    if s == "":
        return 1.0  # identical thickness by default (equal weighting)
    if s.endswith("mm"):
        return float(s[:-2]) * 1e-3
    if s.endswith("um") or s.endswith("µm"):
        return float(s[:-2]) * 1e-6
    if s.endswith("nm"):
        return float(s[:-2]) * 1e-9
    if s.endswith("m"):
        return float(s[:-1])
    # bare number -> assume meters
    return float(s)

def ask_layers():
    """
    Ask the user for a comma-separated list of tissue names and per-layer thicknesses.
    If a tissue is unknown, prompt for its Gabriel parameters and add it to the DB.
    """
    print("\nAvailable tissues in DB:")
    for k in DB.keys():
        print(f"  - {k}")
    raw = input("\nEnter layer names separated by commas (≥1), e.g.\n"
                "  fat_not_infiltrated, heart, bone_cortical\n> ").strip()
    names = [c.strip().lower() for c in raw.split(",") if c.strip()]
    if len(names) < 1:
        print("You must enter at least one layer.")
        sys.exit(1)

    layers = []
    for name in names:
        if name not in DB:
            print(f"\n⚠️ '{name}' not found in DB. Please enter Gabriel parameters to create it:")
            eps_inf = float(input("  eps_inf (dimensionless): "))
            sigma_s = float(input("  sigma_s [S/m]: "))
            n = int(input("  Number of Cole–Cole terms (0–4 recommended): "))
            terms = []
            for i in range(n):
                print(f"   - Term {i+1}:")
                d_eps = float(input("       Δε: "))
                tau   = float(input("       τ [s]: "))
                alpha = float(input("       α [0–1): "))
                terms.append((d_eps, tau, alpha))
            DB[name] = {"eps_inf": eps_inf, "sigma_s": sigma_s, "terms": terms}

        th = input(f"Thickness of '{name}' (mm, µm, nm, m; Enter = identical to others): ").strip()
        d_m = parse_thickness(th)
        layers.append((name, d_m))

    print("\nSelected layers and thicknesses:")
    for n, d in layers:
        print(f"  * {n:>20s} — d = {d:.6e} m")
    return layers

def ask_plot_mode():
    """
    Ask whether to draw a single combined plot (σ* left, ρ* right) or two separate plots.
    """
    ans = input("\nPlot mode: [1] single (σ* & ρ* together)  |  [2] separate figures  → choose 1/2 [1]: ").strip()
    if ans == "" or ans == "1":
        return "single"
    if ans == "2":
        return "separate"
    print("Unrecognized choice. Using single combined plot.")
    return "single"

# -----------------------------
# Main
# -----------------------------
def main():
    print("\n===========================================================")
    print("Multilayer slab — Complex σ*(ω) & ρ*(ω)  [1 kHz – 100 MHz]")
    print("Series (layered, normal field) — Gabriel model per tissue")
    print("===========================================================\n")

    layers = ask_layers()
    plot_mode = ask_plot_mode()

    # Frequency grid: 1e3–1e8 Hz (log-spaced)
    f = np.logspace(3, 8, 2000)
    w = 2 * np.pi * f

    # Per-layer εr*(ω)
    epsr_layers, d_layers = [], []
    for name, d in layers:
        pars = DB[name]
        epsr = gabriel_epsr(f, pars["eps_inf"], pars["sigma_s"], pars["terms"])
        epsr_layers.append(epsr)
        d_layers.append(d)

    # Series combination in εr*(ω)
    epsr_eq = combine_series_epsr(epsr_layers, d_layers)

    # Convert to σ*(ω) and ρ*(ω)
    sigma_eq = 1j * w * EPS0 * epsr_eq
    rho_eq   = 1.0 / sigma_eq

    # Console checkpoints at a few frequencies
    print("\nCheckpoints:")
    for ff in [1e3, 1e5, 1e7, 1e8]:
        idx = np.argmin(np.abs(f - ff))
        print(f"  f={ff:>8.0f} Hz → σ* = {sigma_eq[idx].real:.3e}{sigma_eq[idx].imag:+.3e}j  S/m"
              f"  |  ρ* = {rho_eq[idx].real:.3e}{rho_eq[idx].imag:+.3e}j  Ω·m")

    # -------------------------
    # Plotting
    # -------------------------
    title = ", ".join([f"{n} ({(d if d != 1.0 else 1.0)*1e3:.2f} mm)" for n, d in layers])

    if plot_mode == "single":
        # One combined figure: left axis σ*, right axis ρ*, both real/imag
        fig, ax1 = plt.subplots(figsize=(10, 5.5))
        ax1.set_xscale("log")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("σ*(ω) [S/m]")
        l1, = ax1.semilogx(f, sigma_eq.real, linewidth=2.0, label="Re{σ*}")
        l2, = ax1.semilogx(f, sigma_eq.imag, linewidth=2.0, label="Im{σ*}")
        ax1.grid(True, which="both", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("ρ*(ω) [Ω·m]")
        l3, = ax2.semilogx(f, rho_eq.real, linestyle="--", linewidth=2.0, label="Re{ρ*}")
        l4, = ax2.semilogx(f, rho_eq.imag, linestyle="--", linewidth=2.0, label="Im{ρ*}")

        lines = [l1, l2, l3, l4]
        labels = [ln.get_label() for ln in lines]
        ax1.legend(lines, labels, loc="best")

        fig.suptitle(f"Layered slab (series, normal field) — {title}", fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    else:
        # Two separate figures: σ*(ω) and ρ*(ω)
        plt.figure(figsize=(9.5, 5.3))
        plt.semilogx(f, sigma_eq.real, label="Re{σ*} [S/m]")
        plt.semilogx(f, sigma_eq.imag, label="Im{σ*} [S/m]")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Complex Conductivity σ*(ω) [S/m]")
        plt.title(f"Layered slab (series, normal field) — {title}")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(9.5, 5.3))
        plt.semilogx(f, rho_eq.real, label="Re{ρ*} [Ω·m]")
        plt.semilogx(f, rho_eq.imag, label="Im{ρ*} [Ω·m]")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Complex Resistivity ρ*(ω) [Ω·m]")
        plt.title(f"Layered slab (series, normal field) — {title}")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
