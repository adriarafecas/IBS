# IBS
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
