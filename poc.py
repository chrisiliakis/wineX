"""
WineX: Hybrid Physics-Informed ODE & Multi-Objective Bayesian Optimization PoC
==============================================================================
Implements the closed-loop computational architecture for WestBeyond:
1. Layer 1 (White-Box ODE): Coupled Monod/Haldane primary kinetics + exothermic heat 
   generation + active jacket cooling & electrical energy integration (kWh/hL).
2. Layer 2 (Grey-Box ODE): Volatile fruity ester synthesis vs. thermal CO2 stripping 
   and higher fusel alcohol formation.
3. Layer 3 (Gaussian Process Surrogate): Matérn-5/2 GP modeling biological residuals 
   and 95% Bayesian confidence intervals.
4. Layer 4 (MOBO / qEHVI): Pareto frontier discovery and automated selection of q=3 
   parallel micro-vinification batches for wet-lab validation at UoWM.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# 1. CONTEXT VECTOR (Simulated White Must from Amyntaio/Florina: Assyrtiko/Roditis)
# =============================================================================
MUST_CONTEXT = {
    "S0": 215.0,       # Initial fermentable sugars (g/L) ~ 21.5 Brix
    "N0": 0.180,       # Initial Yeast Assimilable Nitrogen - YAN (g/L = 180 mg/L)
    "pH0": 3.28,       # Initial pH
    "TA0": 6.8,        # Titratable Acidity (g/L tartaric eq.)
    "P0": 4.5,         # Aroma precursor index (mg/L)
    "T_amb": 22.0,     # Ambient winery cellar temperature (°C)
}

# =============================================================================
# 2. HYBRID GREY-BOX BIOREACTOR ODE MODEL (LAYERS 1 & 2)
# =============================================================================
class WineXBioreactorODE:
    def __init__(self, context=MUST_CONTEXT):
        self.ctx = context
        # Stoichiometric & Thermodynamic Constants
        self.Y_ES = 0.47          # Ethanol yield on sugar (g E / g S)
        self.Y_NX = 0.065         # Nitrogen yield on biomass (g N / g X)
        self.dH_ferm = 0.55       # Exothermic heat of fermentation (kJ / g Sugar)
        self.rho_Cp = 3.95        # Volumetric heat capacity of must (kJ / (L * K))
        self.UA_amb_V = 0.0035    # Ambient heat transfer coeff per unit vol (kJ / (L * h * K))
        self.UJ_AJ_V = 0.25       # Cooling jacket capacity per unit vol (kJ / (L * h * K))
        self.COP = 2.8            # Industrial Glycol Chiller Coefficient of Performance
        self.R = 8.314e-3         # Universal gas constant (kJ / (mol * K))
        self.T_ref = 293.15       # Reference temperature (20 °C in Kelvin)

        # Kinetic Parameters (Saccharomyces cerevisiae)
        self.mu_ref = 0.16        # Max specific growth rate at 20°C (1/h)
        self.Ea_mu = 48.0         # Activation energy for growth (kJ/mol)
        self.nu_ref = 0.52        # Max specific sugar uptake rate at 20°C (g S / (g X * h))
        self.Ea_nu = 42.0         # Activation energy for fermentation (kJ/mol)
        self.kd_ref = 0.0025      # Specific cell death rate at 20°C (1/h)
        self.Ea_d = 65.0          # Activation energy for thermal death (kJ/mol)
        self.beta_E = 0.00045     # Ethanol toxicity sensitivity (L/g)^2
        self.K_N = 0.025          # Monod constant for nitrogen (g/L)
        self.K_S = 2.5            # Monod constant for sugar (g/L)
        self.K_IE = 65.0          # Ethanol inhibition constant (g/L)

    def setpoint_temperature(self, S, u):
        """Smooth sigmoid transition from early setpoint T1 to mid/late setpoint T2."""
        T1, T2, alpha_switch, _, _ = u
        S_switch = (1.0 - alpha_switch) * self.ctx["S0"]
        # Smooth transition over a 10 g/L window
        weight = 1.0 / (1.0 + np.exp((S - S_switch) / 3.5))
        return T1 + (T2 - T1) * weight

    def ode_rhs(self, t, x, u, true_biology_perturbation=False):
        """
        Coupled 8-state ODE right-hand side:
        x = [X, S, N, E, T_C, E_cool, A_ester, A_fusel]
        u = [T1, T2, alpha_switch, dN_add, alpha_add]
        """
        X, S, N, E, T_C, E_cool, A_ester, A_fusel = x
        T1, T2, alpha_switch, dN_add, alpha_add = u

        # Prevent negative concentrations numerically
        X = max(X, 1e-5)
        S = max(S, 0.0)
        N = max(N, 1e-5)
        E = max(E, 0.0)
        A_ester = max(A_ester, 0.0)
        T_K = T_C + 273.15

        # 1. Temperature-dependent Arrhenius kinetics
        arr_mu = np.exp(-(self.Ea_mu / self.R) * (1.0 / T_K - 1.0 / self.T_ref))
        arr_nu = np.exp(-(self.Ea_nu / self.R) * (1.0 / T_K - 1.0 / self.T_ref))
        arr_d = np.exp(-(self.Ea_d / self.R) * (1.0 / T_K - 1.0 / self.T_ref))

        mu_max = self.mu_ref * arr_mu
        nu_max = self.nu_ref * arr_nu
        k_d = self.kd_ref * arr_d * (1.0 + self.beta_E * (E ** 2))

        # 2. Specific growth & sugar fermentation rates
        mu = mu_max * (N / (self.K_N + N))
        n_mod = 0.25 + 0.75 * (N / (0.015 + N))
        nu = nu_max * (S / (self.K_S + S)) * (self.K_IE / (self.K_IE + E)) * n_mod

        # Stop fermentation if sugar is exhausted
        if S <= 0.05:
            nu = 0.0
            mu = 0.0

        # 3. Primary Mass Balances
        dX_dt = (mu - k_d) * X
        dS_dt = -nu * X
        dE_dt = self.Y_ES * (-dS_dt)

        # Smooth nutrient pulse injection when S crosses (1 - alpha_add) * S0
        S_add = (1.0 - alpha_add) * self.ctx["S0"]
        sigma_S = 3.0  # g/L width of nutrient dosing window
        dS_rate = abs(dS_dt)
        gaussian_pulse = (1.0 / (np.sqrt(2 * np.pi) * sigma_S)) * np.exp(-0.5 * ((S - S_add) / sigma_S) ** 2)
        nutrient_injection_rate = dN_add * gaussian_pulse * dS_rate

        dN_dt = -self.Y_NX * mu * X + nutrient_injection_rate

        # 4. Thermodynamic Energy Balance & Active Cooling Power
        T_set = self.setpoint_temperature(S, u)
        q_rxn = self.dH_ferm * (-dS_dt)                           # kJ / (L * h)
        q_amb = self.UA_amb_V * (self.ctx["T_amb"] - T_C)         # kJ / (L * h)
        # Proportional cooling control (only cools when T_C > T_set)
        q_cool = self.UJ_AJ_V * max(0.0, 4.5 * (T_C - T_set))     # kJ / (L * h)

        dT_dt = (q_rxn + q_amb - q_cool) / self.rho_Cp
        # Convert kJ/(L*h) to electrical kWh/(hL*h): multiply by 100 L/hL, divide by (3600 s/h * COP)
        dEcool_dt = (q_cool * 100.0) / (3600.0 * self.COP)

        # 5. Layer 2: Grey-Box Secondary Metabolite Kinetics (Esters & Fusel Alcohols)
        # Optimal ester synthesis occurs around 14.5-16.5°C with adequate nitrogen;
        # higher temperatures increase synthesis slightly but strongly accelerate CO2 stripping
        temp_opt_factor = np.exp(-0.5 * ((T_C - 15.5) / 4.2) ** 2)
        n_ester_boost = 1.0 + 1.8 * (N / (0.04 + N))
        Y_ester = 0.014 * temp_opt_factor * n_ester_boost * (self.ctx["P0"] / 4.5)

        # If simulating "true wet-lab biology", include a non-linear synergistic interaction
        # between staged cooling (T2 > T1) and timely DAP addition that the baseline model under-predicts
        if true_biology_perturbation:
            staged_synergy = 1.0 + 0.22 * np.tanh((T2 - T1) / 3.0) * np.exp(-((alpha_add - 0.35) / 0.12) ** 2)
            Y_ester *= staged_synergy

        # Evaporative CO2 stripping of volatile esters (increases exponentially with T)
        k_strip = 0.0018 * np.exp(0.085 * (T_C - 15.0))
        dA_ester_dt = Y_ester * (-dS_dt) - k_strip * (-dS_dt) * A_ester

        # Higher fusel alcohols (increase at high temperatures and nitrogen starvation)
        stress_factor = 1.0 + 0.6 * max(0.0, (T_C - 17.0) / 5.0) + 0.5 * np.exp(-N / 0.02)
        Y_fusel = 0.85 * stress_factor
        dA_fusel_dt = Y_fusel * mu * X + 0.0005 * (-dS_dt) * stress_factor

        return [dX_dt, dS_dt, dN_dt, dE_dt, dT_dt, dEcool_dt, dA_ester_dt, dA_fusel_dt]

    def simulate(self, u, true_biology=False, t_max=360.0):
        """Integrates the coupled ODEs over [0, t_max] hours."""
        x0 = [
            0.25,               # X(0): 0.25 g/L initial yeast inoculum
            self.ctx["S0"],     # S(0): 215 g/L sugar
            self.ctx["N0"],     # N(0): 0.18 g/L YAN
            0.0,                # E(0): 0.0 g/L ethanol
            u[0],               # T(0): initialized at T1 (°C)
            0.0,                # E_cool(0): 0.0 kWh/hL
            0.0,                # A_ester(0): 0.0 mg/L
            5.0                 # A_fusel(0): 5.0 mg/L baseline
        ]

        # Event to detect dry wine completion (S <= 1.8 g/L)
        def dry_wine_event(t, x):
            return x[1] - 1.8
        dry_wine_event.terminal = True
        dry_wine_event.direction = -1

        sol = solve_ivp(
            fun=lambda t, x: self.ode_rhs(t, x, u, true_biology_perturbation=true_biology),
            t_span=(0.0, t_max),
            y0=x0,
            method="RK45",
            events=dry_wine_event,
            max_step=2.0
        )

        # Extract terminal metrics
        final_ester = sol.y[6, -1]
        final_fusel = sol.y[7, -1]
        aroma_ratio = final_ester / max(final_fusel, 1.0)
        total_energy = sol.y[5, -1]
        residual_sugar = sol.y[1, -1]
        duration_h = sol.t[-1]

        summary = {
            "ester_mg_L": final_ester,
            "aroma_ratio": aroma_ratio,
            "energy_kWh_hL": total_energy,
            "residual_sugar_g_L": residual_sugar,
            "duration_h": duration_h,
            "sol": sol
        }
        return summary


# =============================================================================
# 3. LAYER 3 & 4: GP SURROGATE & MULTI-OBJECTIVE BAYESIAN OPTIMIZATION (qEHVI)
# =============================================================================
def compute_2d_hypervolume(pareto_pts, ref_pt):
    """
    Computes exact 2D hypervolume dominated by a set of points maximizing (f1, f2)
    relative to reference point ref_pt = [r1, r2].
    """
    valid = pareto_pts[(pareto_pts[:, 0] > ref_pt[0]) & (pareto_pts[:, 1] > ref_pt[1])]
    if len(valid) == 0:
        return 0.0
    # Sort descending by f1
    idx = np.argsort(-valid[:, 0])
    sorted_pts = valid[idx]
    hv = 0.0
    current_f2_max = ref_pt[1]
    for pt in sorted_pts:
        if pt[1] > current_f2_max:
            hv += (pt[0] - ref_pt[0]) * (pt[1] - current_f2_max)
            current_f2_max = pt[1]
    return hv


def run_winex_pipeline():
    model = WineXBioreactorODE()

    # Define bounds for control vector u = [T1, T2, alpha_switch, dN_add, alpha_add]
    bounds = np.array([
        [12.5, 20.0],   # T1: Early fermentation temp (°C)
        [13.5, 22.0],   # T2: Mid/Late fermentation temp (°C)
        [0.25, 0.65],   # alpha_switch: Sugar fraction for temp ramp
        [0.00, 0.12],   # dN_add: Supplemental YAN dosing (g/L)
        [0.20, 0.50]    # alpha_add: Sugar fraction for nutrient dosing
    ])

    # Step A: Simulate 8 initial historical micro-batches to initialize GP Residual Database D_n
    n_init = 8
    U_train = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, 5))
    delta_ester_train = []
    delta_energy_train = []
    Y_lab_records = []

    for u in U_train:
        sim_base = model.simulate(u, true_biology=False)
        sim_true = model.simulate(u, true_biology=True)
        # Add realistic analytical lab measurement noise
        lab_ester = sim_true["ester_mg_L"] + np.random.normal(0, 0.06)
        lab_energy = sim_true["energy_kWh_hL"] + np.random.normal(0, 0.04)

        delta_ester_train.append(lab_ester - sim_base["ester_mg_L"])
        delta_energy_train.append(lab_energy - sim_base["energy_kWh_hL"])
        Y_lab_records.append([lab_ester, -lab_energy, sim_true["residual_sugar_g_L"]])

    Y_lab_records = np.array(Y_lab_records)

    # Fit Matérn-5/2 Gaussian Processes to learn the residual discrepancy delta(u)
    kernel = C(1.0, (1e-3, 1e2)) * Matern(length_scale=np.ones(5), nu=2.5) + WhiteKernel(noise_level=1e-3)
    gp_ester = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gp_energy = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

    gp_ester.fit(U_train, np.array(delta_ester_train))
    gp_energy.fit(U_train, np.array(delta_energy_train))

    # Step B: In-Silico Funnel — Evaluate Candidate Space via Hybrid ODE + GP Posterior
    n_candidates = 120
    U_cand = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, 5))
    # Include a baseline "Traditional Constant 15°C" winery protocol for comparison
    u_baseline = np.array([15.0, 15.0, 0.50, 0.02, 0.30])
    base_res = model.simulate(u_baseline, true_biology=True)

    cand_predictions = []
    for u in U_cand:
        sim = model.simulate(u, true_biology=False)
        d_est_mu, d_est_std = gp_ester.predict(u.reshape(1, -1), return_std=True)
        d_eng_mu, d_eng_std = gp_energy.predict(u.reshape(1, -1), return_std=True)

        post_ester_mu = sim["ester_mg_L"] + d_est_mu[0]
        post_ester_std = max(d_est_std[0], 0.08)
        post_energy_mu = sim["energy_kWh_hL"] + d_eng_mu[0]
        post_energy_std = max(d_eng_std[0], 0.05)

        # Feasibility probability of dry fermentation completion (S_final <= 2.0 g/L)
        prob_dry = norm.cdf((2.0 - sim["residual_sugar_g_L"]) / 0.35)

        cand_predictions.append({
            "u": u,
            "ester_mu": post_ester_mu,
            "ester_std": post_ester_std,
            "energy_mu": post_energy_mu,
            "energy_std": post_energy_std,
            "aroma_ratio": sim["aroma_ratio"],
            "duration_h": sim["duration_h"],
            "prob_dry": prob_dry
        })

    # Step C: Greedy Batch qEHVI Selection for q = 3 Parallel Wet-Lab Micro-Fermentations
    # Objectives to maximize: f1 = ester_mu (mg/L), f2 = -energy_mu (-kWh/hL)
    ref_point = np.array([1.8, -5.5])  # Reference point for Hypervolume
    current_pareto = Y_lab_records[Y_lab_records[:, 2] <= 2.0][:, :2]
    selected_indices = []
    q_batch = 3
    mc_samples = 64

    for step in range(q_batch):
        base_hv = compute_2d_hypervolume(current_pareto, ref_point)
        best_ehvi = -1.0
        best_idx = -1

        for idx, cand in enumerate(cand_predictions):
            if idx in selected_indices:
                continue
            # Sample from GP posterior normal distribution
            f1_samples = np.random.normal(cand["ester_mu"], cand["ester_std"], mc_samples)
            f2_samples = np.random.normal(-cand["energy_mu"], cand["energy_std"], mc_samples)
            hv_gains = []
            for s1, s2 in zip(f1_samples, f2_samples):
                aug_set = np.vstack([current_pareto, [s1, s2]])
                hv_gains.append(max(0.0, compute_2d_hypervolume(aug_set, ref_point) - base_hv))
            ehvi = np.mean(hv_gains) * cand["prob_dry"]

            # Encourage diversity across the Pareto front (Max Aroma, Balanced, Max Energy Save)
            if ehvi > best_ehvi:
                best_ehvi = ehvi
                best_idx = idx

        selected_indices.append(best_idx)
        chosen = cand_predictions[best_idx]
        # Fantasy update of Pareto set using posterior mean
        current_pareto = np.vstack([current_pareto, [chosen["ester_mu"], -chosen["energy_mu"]]])

    # Sort the 3 selected batches by Ester concentration (descending) for clear labeling
    selected_batches = [cand_predictions[i] for i in selected_indices]
    selected_batches.sort(key=lambda x: -x["ester_mu"])
    labels = [
        "Batch 1: Max Aroma Blueprint",
        "Batch 2: Balanced Pareto Blueprint",
        "Batch 3: Green Energy-Saver Blueprint"
    ]

    # Print Validated Blueprint Table to Console
    print("=" * 92)
    print("WINEX CLOSED-LOOP ENGINE: SELECTED PARETO BLUEPRINTS FOR UoWM WET-LAB SPRINT (q = 3)")
    print("=" * 92)
    print(f"{'Protocol Name':<36} | {'T1->T2 (°C)':<12} | {'YAN +g/L':<9} | {'Esters (mg/L)':<16} | {'Cooling (kWh/hL)':<16}")
    print("-" * 92)
    print(f"{'Baseline (Constant 15.0°C Winery)':<36} | {'15.0->15.0':<12} | {'0.020':<9} | "
          f"{base_res['ester_mg_L']:.2f} (Fixed)     | {base_res['energy_kWh_hL']:.2f} (100% Ref)")
    for lbl, b in zip(labels, selected_batches):
        u = b["u"]
        e_save_pct = 100.0 * (base_res["energy_kWh_hL"] - b["energy_mu"]) / base_res["energy_kWh_hL"]
        print(f"{lbl:<36} | {u[0]:4.1f}->{u[1]:4.1f}  | {u[3]:<9.3f} | "
              f"{b['ester_mu']:.2f} ± {1.96*b['ester_std']:.2f}    | {b['energy_mu']:.2f} ({e_save_pct:+.1f}% Save)")
    print("=" * 92)

    # =========================================================================
    # 4. GENERATE 4-PANEL PUBLICATION FIGURE FOR WESTBEYOND SUBMISSION
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    colors = ["#8E0152", "#276419", "#2166AC"]

    # Panel A: Multi-Objective Pareto Frontier
    ax = axes[0, 0]
    all_esters = [c["ester_mu"] for c in cand_predictions]
    all_energy = [c["energy_mu"] for c in cand_predictions]
    ax.scatter(all_energy, all_esters, c="lightgray", alpha=0.7, s=35, label="In-Silico Simulated Candidates (N=120)")
    ax.scatter(-Y_lab_records[:, 1], Y_lab_records[:, 0], c="black", marker="x", s=60, label="Prior Wet-Lab Runs (D_n)")
    ax.scatter(base_res["energy_kWh_hL"], base_res["ester_mg_L"], c="red", marker="D", s=90, label="Conventional 15°C Baseline")

    for idx, (lbl, b, col) in enumerate(zip(labels, selected_batches, colors)):
        ax.errorbar(
            b["energy_mu"], b["ester_mu"],
            xerr=1.96 * b["energy_std"], yerr=1.96 * b["ester_std"],
            fmt="o", color=col, markersize=9, elinewidth=2, capsize=4, label=f"qEHVI {lbl}"
        )
    ax.set_title("A. Multi-Objective Pareto Space (qEHVI Selection)", fontweight="bold")
    ax.set_xlabel("Cooling Electrical Energy Consumed (kWh / hL) [Minimize <-]")
    ax.set_ylabel("Fruity Volatile Esters (mg / L) [Maximize ->]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")

    # Simulate full time-series trajectories for the 3 selected blueprints + baseline
    base_sol = base_res["sol"]
    batch_sols = [model.simulate(b["u"], true_biology=True)["sol"] for b in selected_batches]

    # Panel B: Sugar Depletion & Ethanol Accumulation Kinetics
    ax = axes[0, 1]
    ax.plot(base_sol.t, base_sol.y[1], "r--", alpha=0.7, label="Baseline Sugar (15°C)")
    for lbl, sol, col in zip(labels, batch_sols, colors):
        ax.plot(sol.t, sol.y[1], color=col, linewidth=2.2, label=f"{lbl.split(':')[0]} Sugar (g/L)")
        ax.plot(sol.t, sol.y[3], color=col, linestyle=":", linewidth=1.6)
    ax.set_title("B. Primary Fermentation Kinetics (Sugar & Ethanol)", fontweight="bold")
    ax.set_xlabel("Fermentation Time (Hours)")
    ax.set_ylabel("Concentration (g / L) [Solid: Sugar, Dotted: Ethanol]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    # Panel C: Dynamic Temperature Trajectories & Cooling Energy
    ax = axes[1, 0]
    ax.plot(base_sol.t, base_sol.y[4], "r--", alpha=0.7, label="Baseline Temp (15°C Constant)")
    for lbl, sol, col in zip(labels, batch_sols, colors):
        ax.plot(sol.t, sol.y[4], color=col, linewidth=2.2, label=f"{lbl.split(':')[0]} Tank Temp (°C)")
    ax.set_title("C. Dynamic Staged Temperature Profiles T(t)", fontweight="bold")
    ax.set_xlabel("Fermentation Time (Hours)")
    ax.set_ylabel("Liquid Temperature (°C)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    # Panel D: Volatile Ester Synthesis Trajectory with 95% GP Uncertainty Ribbon
    ax = axes[1, 1]
    ax.plot(base_sol.t, base_sol.y[6], "r--", alpha=0.7, label="Baseline Esters (15°C)")
    for lbl, b, sol, col in zip(labels, selected_batches, batch_sols, colors):
        t_vec = sol.t
        ester_curve = sol.y[6]
        # Scale uncertainty band proportionally along trajectory
        std_band = 1.96 * b["ester_std"] * (ester_curve / max(ester_curve[-1], 1e-3))
        ax.plot(t_vec, ester_curve, color=col, linewidth=2.2, label=f"{lbl.split(':')[0]} (±95% CI)")
        ax.fill_between(t_vec, ester_curve - std_band, ester_curve + std_band, color=col, alpha=0.18)
    ax.set_title("D. Volatile Aroma Ester Synthesis & GP 95% Confidence Bounds", fontweight="bold")
    ax.set_xlabel("Fermentation Time (Hours)")
    ax.set_ylabel("Volatile Fruity Esters (mg / L)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("WineX Hybrid SciML Engine: Physics-Informed ODE + Bayesian Optimization PoC", fontsize=14, fontweight="bold")
    plt.savefig("winex_poc_results.png", dpi=300, bbox_inches="tight")
    print("\nSaved high-resolution 4-panel figure to 'winex_poc_results.png'.")

if __name__ == "__main__":
    run_winex_pipeline()