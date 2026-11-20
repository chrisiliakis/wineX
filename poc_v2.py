"""
WineX v2: Closed-Loop Hybrid SciML & Industrial Scale-Up Proof-of-Concept
=========================================================================
Aligned end-to-end with the WineX LaTeX Architecture & WestBeyond Proposal:
- Layer 1 (White-Box ODEs): Growth-uncoupled kinetics via cumulative assimilated
  nitrogen N_eff(t) (Eq. 6), chain-rule Gaussian nutrient impulse (Eq. 12), and
  unilateral saturation-bounded chiller thermodynamics (Eqs. 14-18).
- Layer 2 (Grey-Box UDE): Intracellular ester synthesis decoupled from Henry's
  Law CO2 bubble stripping k_strip(T, E) * Y_CO2/S * (-dS/dt) (Eqs. 19-20).
- Layer 3 (Probabilistic Calibration): Normalized ARD Matérn-5/2 Gaussian Process
  surrogates across all 4 outputs [A_ester, A_ester/A_fusel, E_cool, S_final] (Eqs. 21-25).
- Layer 4 (Constrained Pareto Selection): Sobol quasi-Monte Carlo candidates +
  feasible max-aroma, minimum-energy, and balanced blueprints.
- Layer 5 (Scale-Up): Inverse thermodynamic transfer function mapping 10 L lab
  trajectories (A_J/V = 22 m^-1) to 10,000 L industrial PLC setpoints (A_J/V = 2.1 m^-1) (Eq. 28).
"""

import argparse
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =============================================================================
# 1. SYSTEM SPACES: MUST CONTEXT (Eq. 1), YEAST LIBRARY (Eq. 2), VESSELS (Eq. 3)
# =============================================================================
MUST_CONTEXT = {
    "S0": 215.0,       # Initial fermentable hexoses (g/L) ~ 21.5 Brix
    "N0": 0.180,       # Initial Yeast Assimilable Nitrogen - YAN (g/L = 180 mg/L)
    "pH0": 3.28,       # Initial must pH (-)
    "TA0": 6.8,        # Titratable Acidity (g/L tartaric acid eq.)
    "P0": 4.5,         # Bound varietal aroma precursor & phytosterol index (mg/L)
    "T_amb": 22.0,     # Ambient cellar temperature (°C)
}

# Commercial S. cerevisiae strain embeddings z_yeast (Eq. 2)
YEAST_LIBRARY = {
    0: {"name": "Strain A (Clean/Crisp)",  "mu_mod": 1.00, "nu_mod": 1.00, "ester_mod": 1.00, "fusel_mod": 0.95},
    1: {"name": "Strain B (Thiol/Ester+)", "mu_mod": 0.96, "nu_mod": 0.98, "ester_mod": 1.12, "fusel_mod": 1.00},
    2: {"name": "Strain C (Cryotolerant)", "mu_mod": 1.04, "nu_mod": 1.05, "ester_mod": 1.05, "fusel_mod": 0.92},
}

# Vessel Geometry & Thermal Hardware Vectors g_v = [V_v, A_J, U_J, UA_amb, T_coolant_min] (Eq. 3)
LAB_VESSEL = {
    "name": "10 L Jacketed Benchtop Bioreactor",
    "V": 10.0,               # Liquid working volume (L)
    "A_J": 0.22,             # Wetted jacket area (m^2) -> A_J / V_m3 = 22.0 m^-1
    "U_J": 180.0,            # Jacket heat-transfer coeff (kJ / (m^2 * h * K))
    "UA_amb": 1.85,          # Ambient conductance (kJ / (h * K)) -> 0.185 kJ/(L*h*K)
    "T_coolant_min": -5.0,   # Lab chiller minimum supply (°C)
}

IND_VESSEL = {
    "name": "10,000 L Stainless Industrial Fermenter",
    "V": 10000.0,            # Liquid working volume (L)
    "A_J": 21.0,             # Wetted jacket area (m^2) -> A_J / V_m3 = 2.1 m^-1
    "U_J": 135.0,            # Industrial jacket U_J (kJ / (m^2 * h * K))
    "UA_amb": 180.0,         # Industrial ambient heat gain (kJ / (h * K))
    "T_coolant_min": -4.0,   # Winery glycol header limit (°C)
    "dT_margin": 1.5,        # Safety margin above glycol limit (°C) -> min allowed -2.5°C
}


# =============================================================================
# 2. LAYERS 1 & 2: MECHANISTIC + GREY-BOX UDE BIOREACTOR SIMULATOR
# =============================================================================
class WineXBioreactorODE:
    def __init__(self, context=MUST_CONTEXT, vessel=LAB_VESSEL):
        self.ctx = dict(context)
        self.vessel = dict(vessel)

        # Table 1 Stoichiometric & Thermodynamic Parameters
        self.dH_S = 0.562         # Exothermic heat of hexose fermentation (kJ/g) [101.2 / 180.16]
        self.Y_ES = 0.47          # Ethanol mass yield on fermented hexose (g/g)
        self.Y_CO2_S = 0.45       # Evolved CO2 mass yield on fermented hexose (g/g)
        self.Y_NX = 0.065         # YAN consumed per unit yeast biomass synthesized (g/g)
        self.rho_Cp = 3.95        # Volumetric specific heat capacity of must (kJ / (L * K))
        self.R = 8.314e-3         # Universal gas constant (kJ / (mol * K))
        self.T_ref = 293.15       # Reference temperature 20°C (K)
        self.COP_nom = 2.8        # Nominal glycol chiller Coefficient of Performance

        # Arrhenius & Monod/Michaelis-Menten Kinetic Constants (Eqs. 5-7)
        self.mu_ref = 0.20        # Allows biomass to reach realistic cellar concentrations (~2.0 g/L)
        self.nu_ref = 0.65        # Standard stationary hexose uptake rate for S. cerevisiae
        self.kd_ref = 0.0018      # Avoids premature cell death before sugar depletion
        self.Ea_mu = 48.0         # Growth activation energy (kJ/mol)
        self.Ea_nu = 42.0         # Hexose transport activation energy (kJ/mol)
        self.Ea_d = 65.0          # Cell inactivation activation energy (kJ/mol)
        self.beta_E = 0.00045     # Quadratic ethanol toxicity coefficient (L/g)^2
        self.K_N = 0.025          # Monod half-saturation for nitrogen (g/L)
        self.K_S = 2.5            # Michaelis-Menten constant for hexose (g/L)
        self.K_IE = 65.0          # Non-competitive ethanol inhibition constant (g/L)
        self.gamma_min = 0.25     # Basal transporter capacity fraction (-)
        self.K_N_nu = 0.08        # Proper saturation for HXT transporters under YAN pulse

        # Control & Nutrient Impulse Constants (Eqs. 12-15)
        self.sigma_S = 2.0        # Gaussian nutrient dosing window width (g/L)
        self.dS_ramp = 3.5        # Logistic temperature transition smoothness (g/L)
        self.K_c = 4.5            # Proportional jacket controller gain (-)

    def setpoint_temperature(self, S, u):
        """Eq. (13): Logistic temperature schedule T_set(S; u)."""
        T1, T2, alpha_switch = u[0], u[1], u[2]
        S_switch = (1.0 - alpha_switch) * self.ctx["S0"]
        return T1 + (T2 - T1) / (1.0 + np.exp((S - S_switch) / self.dS_ramp))

    def chiller_cop(self, T_amb_C, T_coolant_C):
        """Temperature-dependent chiller COP around nominal COP = 2.8."""
        lift = max(8.0, T_amb_C + 8.0 - T_coolant_C)
        return np.clip(self.COP_nom * (25.0 / lift), 1.8, 4.2)

    def ode_rhs(self, t, x, u, true_biology_perturbation=False):
        """
        State vector x = [X_v, S, N, E, T_C, E_cool, A_ester, A_fusel, N_eff]
        Control vector u = [T1, T2, alpha_switch, dN_add, alpha_add, z_yeast]
        """
        X_v, S, N, E, T_C, E_cool, A_ester, A_fusel, N_eff = x
        T1, T2, alpha_switch, dN_add, alpha_add, z_idx = u
        strain = YEAST_LIBRARY[int(round(z_idx))]

        X_v = max(X_v, 1e-6)
        S = max(S, 0.0)
        N = max(N, 0.0)
        E = max(E, 0.0)
        A_ester = max(A_ester, 0.0)
        N_eff = max(N_eff, self.ctx["N0"])
        T_K = T_C + 273.15

        # Eqs. (5)-(7): Growth-uncoupled nitrogen-limited Arrhenius kinetics
        arr_mu = np.exp(-(self.Ea_mu / self.R) * (1.0 / T_K - 1.0 / self.T_ref))
        arr_nu = np.exp(-(self.Ea_nu / self.R) * (1.0 / T_K - 1.0 / self.T_ref))
        arr_d = np.exp(-(self.Ea_d / self.R) * (1.0 / T_K - 1.0 / self.T_ref))

        mu = (self.mu_ref * strain["mu_mod"]) * arr_mu * (N / (self.K_N + N)) if N > 1e-7 else 0.0
        Phi_N = self.gamma_min + (1.0 - self.gamma_min) * (N_eff / (self.K_N_nu + N_eff))
        nu = (
            (self.nu_ref * strain["nu_mod"])
            * arr_nu
            * (S / (self.K_S + S))
            * (self.K_IE / (self.K_IE + E))
            * Phi_N
        )
        k_d = self.kd_ref * arr_d * (1.0 + self.beta_E * (E ** 2))

        if S <= 0.02:
            mu = 0.0
            nu = 0.0

        # Eqs. (8)-(12): Primary mass balances + exact chain-rule Gaussian YAN pulse
        dX_dt = (mu - k_d) * X_v
        dS_dt = -nu * X_v
        dE_dt = self.Y_ES * (-dS_dt)

        S_add = (1.0 - alpha_add) * self.ctx["S0"]
        gauss_S = (1.0 / (np.sqrt(2.0 * np.pi) * self.sigma_S)) * np.exp(
            -0.5 * ((S - S_add) / self.sigma_S) ** 2
        )
        N_add_rate = dN_add * gauss_S * (-dS_dt)
        dN_dt = -self.Y_NX * mu * X_v + N_add_rate
        if N <= 1e-7 and dN_dt < 0.0:
            dN_dt = 0.0
        dNeff_dt = N_add_rate

        # Eqs. (14)-(18): Thermodynamic energy balance & unilateral saturated jacket cooling
        T_set = self.setpoint_temperature(S, u)
        q_rxn = self.dH_S * (-dS_dt)  # kJ / (L * h)
        q_amb = (self.vessel["UA_amb"] / self.vessel["V"]) * (self.ctx["T_amb"] - T_C)

        if self.vessel["U_J"] * self.vessel["A_J"] > 0.0:
            T_cool_desired = T_C - self.K_c * (T_C - T_set)
            T_coolant = max(self.vessel["T_coolant_min"], min(T_C, T_cool_desired))
            q_cool = (self.vessel["U_J"] * self.vessel["A_J"] / self.vessel["V"]) * (T_C - T_coolant)
        else:
            T_coolant = T_C
            q_cool = 0.0

        dT_dt = (q_rxn + q_amb - q_cool) / self.rho_Cp
        cop = self.chiller_cop(self.ctx["T_amb"], T_coolant)
        dEcool_dt = (100.0 * q_cool) / (3600.0 * cop)  # kWh / (hL * h)

        # Eqs. (19)-(20): Layer 2 Grey-Box UDE Secondary Aroma Kinetics
        temp_ester_opt = np.exp(-0.5 * ((T_C - 15.0) / 4.2) ** 2)
        n_ester_factor = 1.0 + 1.55 * (N_eff / (0.12 + N_eff))
        pH_factor = 1.0 + 0.08 * (3.35 - self.ctx["pH0"])
        f_theta_ester = (
            0.0128 * strain["ester_mod"] * temp_ester_opt * n_ester_factor * pH_factor * (self.ctx["P0"] / 4.5)
        )

        if true_biology_perturbation:
            # Unmodeled varietal co-factor synergy learned by the Matérn-5/2 GP residual layer
            staged_synergy = 1.0 + 0.18 * np.tanh((T2 - T1) / 3.0) * np.exp(-((alpha_add - 0.34) / 0.11) ** 2)
            f_theta_ester *= staged_synergy

        # White-box Henry's law CO2 stripping modulated by T and ethanol co-solvency
        k_strip = 0.0038 * np.exp(0.082 * (T_C - 15.0) - 0.0035 * E)
        co2_rate = self.Y_CO2_S * (-dS_dt)
        dA_ester_dt = f_theta_ester * (-dS_dt) - k_strip * co2_rate * A_ester

        # Ehrlich pathway fusel alcohol accumulation
        stress_factor = (1.0 + 0.55 * max(0.0, (T_C - 17.5) / 5.0) + 0.35 * np.exp(-N_eff / 0.10)) * strain["fusel_mod"]
        dA_fusel_dt = 5.2 * stress_factor * mu * X_v + 0.025 * stress_factor * (-dS_dt)

        return [dX_dt, dS_dt, dN_dt, dE_dt, dT_dt, dEcool_dt, dA_ester_dt, dA_fusel_dt, dNeff_dt]

    def simulate(self, u, true_biology=False, t_max=360.0):
        x0 = [
            0.25,               # X_v(0): 0.25 g/L dry cell weight inoculum
            self.ctx["S0"],     # S(0)
            self.ctx["N0"],     # N(0)
            0.0,                # E(0)
            u[0],               # T(0) initialized at T1 (°C)
            0.0,                # E_cool(0) (kWh/hL)
            0.0,                # A_ester(0) (mg/L)
            6.0,                # A_fusel(0) baseline (mg/L)
            self.ctx["N0"],     # N_eff(0)
        ]

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
            rtol=1e-6,
            atol=1e-8,
            max_step=1.0,
        )

        final_ester = sol.y[6, -1]
        final_fusel = sol.y[7, -1]
        return {
            "ester_mg_L": final_ester,
            "fusel_mg_L": final_fusel,
            "aroma_ratio": final_ester / max(final_fusel, 1e-3),
            "energy_kWh_hL": sol.y[5, -1],
            "residual_sugar_g_L": sol.y[1, -1],
            "ethanol_g_L": sol.y[3, -1],
            "N_eff_final": sol.y[8, -1],
            "duration_h": sol.t[-1],
            "sol": sol,
        }

    def compute_industrial_scaleup(self, sol, u, ind_vessel=IND_VESSEL):
        """
        Eq. (28): Inverse Thermodynamic Scale-Up Transfer Function (10 L -> 10,000 L+).
        Computes exact industrial glycol jacket setpoint trajectory T_coolant,ind*(t).
        """
        t_vec = sol.t
        T_lab = sol.y[4]
        rhs_evals = np.array([self.ode_rhs(t, sol.y[:, i], u, False) for i, t in enumerate(t_vec)])
        neg_dS_dt = -rhs_evals[:, 1]
        dT_lab_dt = rhs_evals[:, 4]

        # Lab jacket trajectory for direct visual comparison
        T_set_vec = np.array([self.setpoint_temperature(s, u) for s in sol.y[1]])
        T_cool_lab = np.maximum(
            self.vessel["T_coolant_min"],
            np.minimum(T_lab, T_lab - self.K_c * (T_lab - T_set_vec)),
        )

        # Eq. (28) feedforward industrial jacket coolant requirement
        q_net_ind = (
            self.dH_S * neg_dS_dt
            + (ind_vessel["UA_amb"] / ind_vessel["V"]) * (self.ctx["T_amb"] - T_lab)
            - self.rho_Cp * dT_lab_dt
        )
        thermal_resistance_ind = ind_vessel["V"] / (ind_vessel["U_J"] * ind_vessel["A_J"])
        T_cool_ind_req = T_lab - thermal_resistance_ind * np.maximum(0.0, q_net_ind)

        min_allowed_glycol = ind_vessel["T_coolant_min"] + ind_vessel["dT_margin"]
        is_feasible = bool(np.min(T_cool_ind_req) >= min_allowed_glycol)
        return {
            "t": t_vec,
            "T_lab": T_lab,
            "T_cool_lab": T_cool_lab,
            "T_cool_ind": T_cool_ind_req,
            "min_T_cool_ind": float(np.min(T_cool_ind_req)),
            "is_chiller_feasible": is_feasible,
        }


# =============================================================================
# 3. AUTOMATED PHYSICS & STOICHIOMETRY VERIFICATION SUITE
# =============================================================================
def run_verification_suite(model):
    u_test = np.array([15.0, 18.5, 0.45, 0.080, 0.35, 1])
    res = model.simulate(u_test, true_biology=False)

    # 1. Ethanol mass yield verification
    expected_E = model.Y_ES * (model.ctx["S0"] - res["residual_sugar_g_L"])
    assert abs(res["ethanol_g_L"] - expected_E) < 0.05, "Ethanol stoichiometry mismatch!"

    # 2. Chain-rule Gaussian YAN pulse exact integral verification (Eq. 12)
    added_N = res["N_eff_final"] - model.ctx["N0"]
    assert abs(added_N - 0.080) < 1e-3, f"Gaussian pulse mass error: {added_N:.5f} != 0.080"

    # 3. Unilateral cooling energy monotonicity & state non-negativity
    assert np.all(np.diff(res["sol"].y[5]) >= -1e-12), "Cooling energy decreased (violates unilateral cooling)!"
    assert np.min(res["sol"].y) >= -1e-9, "Negative concentration detected!"

    # 4. Adiabatic exothermic temperature rise check
    adiabatic_vessel = dict(LAB_VESSEL, U_J=0.0, UA_amb=0.0)
    adiabatic_model = WineXBioreactorODE(MUST_CONTEXT, adiabatic_vessel)
    adiabatic_model.kd_ref = 0.0
    res_ad = adiabatic_model.simulate(np.array([15.0, 15.0, 0.50, 0.05, 0.30, 0]))
    expected_dT = (adiabatic_model.dH_S * (MUST_CONTEXT["S0"] - res_ad["residual_sugar_g_L"])) / adiabatic_model.rho_Cp
    actual_dT = res_ad["sol"].y[4, -1] - 15.0
    assert abs(actual_dT - expected_dT) < 0.15, f"Adiabatic dT mismatch: {actual_dT:.2f} vs {expected_dT:.2f}"

    print("[PASS] Verification Suite: Stoichiometry, Gaussian YAN Conservation, Unilateral Chiller, & Adiabatic 1st Law OK.")


# =============================================================================
# 4. MAIN CLOSED-LOOP PIPELINE & 6-PANEL PUBLICATION FIGURE
# =============================================================================
def normalize_u(U, bounds):
    return (U - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def run_winex_v2(n_candidates=512, csv_path=None):
    rng = np.random.default_rng(42)
    model = WineXBioreactorODE(MUST_CONTEXT, LAB_VESSEL)
    run_verification_suite(model)

    # Control space bounds: [T1, T2, alpha_switch, dN_add, alpha_add, z_yeast]
    bounds = np.array([
        [12.5, 20.0],   # T1 (°C)
        [14.0, 22.0],   # T2 (°C)
        [0.25, 0.65],   # alpha_switch (-)
        [0.00, 0.14],   # dN_add (g/L, strictly below EU OIV limits)
        [0.20, 0.45],   # alpha_add (-)
        [0.00, 2.00],   # z_yeast index {0, 1, 2}
    ])

    # Optional external public CSV ingestion for calibration overlay
    external_sugar_curve = None
    if csv_path is not None:
        raw = np.genfromtxt(csv_path, delimiter=",", names=True)
        external_sugar_curve = (raw["time_h"], raw["sugar_g_L"])
        print(f"[INFO] Loaded external experimental time-series from '{csv_path}' ({len(raw)} points).")

    # Step A: Initialize Historical Wet-Lab Database D_n (n = 10 runs)
    n_init = 10
    try:
        sobol_init = qmc.Sobol(d=5, scramble=True, rng=42)
    except TypeError:
        sobol_init = qmc.Sobol(d=5, scramble=True, seed=42)
    u_cont_init = qmc.scale(sobol_init.random(n_init), bounds[:5, 0], bounds[:5, 1])
    z_init = (np.arange(n_init) % 3).reshape(-1, 1)
    U_train = np.hstack([u_cont_init, z_init])

    residuals = {k: [] for k in ["ester", "ratio", "energy", "sugar"]}
    Y_lab = []

    for u in U_train:
        sim_prior = model.simulate(u, true_biology=False)
        sim_true = model.simulate(u, true_biology=True)
        y1 = sim_true["ester_mg_L"] + rng.normal(0.0, 0.05)
        y2 = sim_true["aroma_ratio"] + rng.normal(0.0, 0.003)
        y3 = sim_true["energy_kWh_hL"] + rng.normal(0.0, 0.04)
        y4 = max(0.0, sim_true["residual_sugar_g_L"] + rng.normal(0.0, 0.12))

        residuals["ester"].append(y1 - sim_prior["ester_mg_L"])
        residuals["ratio"].append(y2 - sim_prior["aroma_ratio"])
        residuals["energy"].append(y3 - sim_prior["energy_kWh_hL"])
        residuals["sugar"].append(y4 - sim_prior["residual_sugar_g_L"])
        Y_lab.append([y1, y2, -y3, y4])

    Y_lab = np.array(Y_lab)
    U_train_norm = normalize_u(U_train, bounds)

    # Fit Normalized ARD Matérn-5/2 Gaussian Processes for all 4 outputs (Eqs. 22-25)
    gps = {}
    for key in ["ester", "ratio", "energy", "sugar"]:
        kernel = (
            C(1.0, (1e-3, 1e2))
            * Matern(length_scale=np.ones(6), length_scale_bounds=(0.05, 25.0), nu=2.5)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        )
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42)
        gp.fit(U_train_norm, np.array(residuals[key]))
        gps[key] = gp

    # Conventional 15.0°C Isothermal Winery Baseline
    u_baseline = np.array([15.0, 15.0, 0.50, 0.020, 0.30, 0])
    base_res = model.simulate(u_baseline, true_biology=True)

    # Step B: Sobol Quasi-Monte Carlo In-Silico Candidate Funnel
    try:
        sobol_cand = qmc.Sobol(d=5, scramble=True, rng=101)
    except TypeError:
        sobol_cand = qmc.Sobol(d=5, scramble=True, seed=101)
    u_cont_cand = qmc.scale(sobol_cand.random(n_candidates), bounds[:5, 0], bounds[:5, 1])
    z_cand = rng.integers(0, 3, size=(n_candidates, 1))
    U_cand = np.hstack([u_cont_cand, z_cand])
    U_cand_norm = normalize_u(U_cand, bounds)

    # Vectorized GP posterior inference across all candidates
    gp_preds = {}
    for key, gp in gps.items():
        mu, std = gp.predict(U_cand_norm, return_std=True)
        gp_preds[key] = (mu, std)

    candidates = []
    S_dry = 2.0
    for i, u in enumerate(U_cand):
        sim = model.simulate(u, true_biology=False)
        scaleup = model.compute_industrial_scaleup(sim["sol"], u, IND_VESSEL)

        est_mu = sim["ester_mg_L"] + gp_preds["ester"][0][i]
        est_std = max(gp_preds["ester"][1][i], 0.06)
        rat_mu = sim["aroma_ratio"] + gp_preds["ratio"][0][i]
        rat_std = max(gp_preds["ratio"][1][i], 0.004)
        eng_mu = sim["energy_kWh_hL"] + gp_preds["energy"][0][i]
        eng_std = max(gp_preds["energy"][1][i], 0.05)
        sug_mu = sim["residual_sugar_g_L"] + gp_preds["sugar"][0][i]
        sug_std = max(gp_preds["sugar"][1][i], 0.18)

        # Eq. (26): Posterior probability of dry completion S(t_f) <= 2.0 g/L
        prob_dry = float(norm.cdf((S_dry - sug_mu) / sug_std))

        candidates.append({
            "u": u,
            "ester_mu": est_mu,
            "ester_std": est_std,
            "ratio_mu": rat_mu,
            "ratio_std": rat_std,
            "energy_mu": eng_mu,
            "energy_std": eng_std,
            "prob_dry": prob_dry,
            "chiller_feasible": scaleup["is_chiller_feasible"],
            "min_T_cool_ind": scaleup["min_T_cool_ind"],
        })

    # Select distinct high-performing archetypes from the feasible Pareto pool.
    valid_cands = [c for c in candidates if c["chiller_feasible"] and c["prob_dry"] >= 0.95]
    if not valid_cands:
        raise RuntimeError("No candidates meet the industrial chiller and dry-completion constraints.")

    b_aroma = max(valid_cands, key=lambda c: c["ester_mu"])
    b_energy = min(valid_cands, key=lambda c: c["energy_mu"])

    min_eng = min(c["energy_mu"] for c in valid_cands)
    max_eng = max(c["energy_mu"] for c in valid_cands)
    min_est = min(c["ester_mu"] for c in valid_cands)
    max_est = max(c["ester_mu"] for c in valid_cands)

    def trade_off_score(c):
        norm_est = (c["ester_mu"] - min_est) / max(1e-5, max_est - min_est)
        norm_eng = (max_eng - c["energy_mu"]) / max(1e-5, max_eng - min_eng)
        dist_aroma = abs(c["ester_mu"] - b_aroma["ester_mu"])
        dist_energy = abs(c["energy_mu"] - b_energy["energy_mu"])
        if dist_aroma < 0.2 or dist_energy < 0.2:
            return -1.0
        return norm_est + norm_eng

    b_balanced = max(valid_cands, key=trade_off_score)
    selected_batches = [b_aroma, b_balanced, b_energy]
    labels = [
        "Batch 1: Max-Aroma Blueprint",
        "Batch 2: Balanced Pareto Blueprint",
        "Batch 3: Green Energy Blueprint",
    ]

    # Print Aligned Benchmark Table
    print("=" * 108)
    print(f"WINEX v2 CLOSED-LOOP ENGINE (N={n_candidates} Sobol Candidates -> q=3 Validated Pareto Blueprints)")
    print("=" * 108)
    header = f"{'Protocol Designation':<34} | {'T1->T2 (°C)':<13} | {'+YAN g/L':<8} | {'Yeast':<8} | {'Esters (mg/L)':<19} | {'Chiller (kWh/hL)':<18}"
    print(header)
    print("-" * 108)
    print(
        f"{'Conventional Isothermal Baseline':<34} | {'15.0->15.0':<13} | {'0.020':<8} | {'Strain A':<8} | "
        f"{base_res['ester_mg_L']:.2f} (Reference)    | {base_res['energy_kWh_hL']:.2f} (Reference)"
    )
    for lbl, b in zip(labels, selected_batches):
        u = b["u"]
        est_pct = 100.0 * (b["ester_mu"] - base_res["ester_mg_L"]) / base_res["ester_mg_L"]
        eng_pct = 100.0 * (b["energy_mu"] - base_res["energy_kWh_hL"]) / base_res["energy_kWh_hL"]
        strain_short = f"Strain {'ABC'[int(round(u[5]))]}"
        print(
            f"{lbl:<34} | {u[0]:4.1f}->{u[1]:4.1f}   | {u[3]:<8.3f} | {strain_short:<8} | "
            f"{b['ester_mu']:.2f}±{1.96*b['ester_std']:.2f} ({est_pct:+.1f}%) | "
            f"{b['energy_mu']:.2f}±{1.96*b['energy_std']:.2f} ({eng_pct:+.1f}%)"
        )
    print("=" * 108)

    # =========================================================================
    # 6. GENERATE 6-PANEL PUBLICATION FIGURE (300 DPI)
    # =========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    plt.subplots_adjust(hspace=0.35, wspace=0.24)
    colors = ["#7F0037", "#1B6B36", "#1D4E89"]

    base_sol = base_res["sol"]
    batch_sims = [model.simulate(b["u"], true_biology=True) for b in selected_batches]
    batch_scaleups = [model.compute_industrial_scaleup(sim["sol"], b["u"], IND_VESSEL) for sim, b in zip(batch_sims, selected_batches)]

    # Panel A: Multi-Objective Pareto Space
    ax = axes[0, 0]
    feas_c = [c for c in candidates if c["chiller_feasible"]]
    infeas_c = [c for c in candidates if not c["chiller_feasible"]]
    ax.scatter([c["energy_mu"] for c in feas_c], [c["ester_mu"] for c in feas_c], c="#C7CED6", s=26, alpha=0.75, label=f"Feasible Sobol Pool (N={len(feas_c)})")
    if infeas_c:
        ax.scatter([c["energy_mu"] for c in infeas_c], [c["ester_mu"] for c in infeas_c], c="#E07A5F", marker="x", s=28, alpha=0.6, label="Rejected (Chiller Saturation)")
    ax.scatter(-Y_lab[:, 2], Y_lab[:, 0], c="black", marker="+", s=65, label=f"Prior Wet-Lab Calibration (n={n_init})")
    ax.scatter(base_res["energy_kWh_hL"], base_res["ester_mg_L"], c="#D90429", marker="D", s=85, zorder=5, label="Isothermal 15°C Baseline")
    for lbl, b, col in zip(labels, selected_batches, colors):
        ax.errorbar(b["energy_mu"], b["ester_mu"], xerr=1.96 * b["energy_std"], yerr=1.96 * b["ester_std"], fmt="o", color=col, markersize=8.5, elinewidth=2, capsize=4, zorder=6, label=lbl)
    ax.set_title("A. Feasible Pareto Pool and Selected Blueprints", fontweight="bold")
    ax.set_xlabel("Chiller Electrical Energy E_cool (kWh / hL) [Minimize <-]")
    ax.set_ylabel("Volatile Fruity Esters A_ester (mg / L) [Maximize ->]")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.8, loc="lower right")

    # Panel B: Primary Kinetics (Sugar Depletion + External Dataset Hook)
    ax = axes[0, 1]
    ax.plot(base_sol.t, base_sol.y[1], "r--", linewidth=1.8, alpha=0.8, label="Baseline Sugar (15°C)")
    for lbl, sim, col in zip(labels, batch_sims, colors):
        sol = sim["sol"]
        ax.plot(sol.t, sol.y[1], color=col, linewidth=2.1, label=f"{lbl.split(':')[0]} Sugar")
        ax.plot(sol.t, sol.y[3], color=col, linestyle=":", linewidth=1.5)
    if external_sugar_curve is not None:
        ax.scatter(external_sugar_curve[0], external_sugar_curve[1], c="black", s=30, zorder=6, label="Public Dataset Sugar (g/L)")
    ax.axhline(2.0, color="gray", linestyle="-.", linewidth=1.0, label="Dry Threshold (2.0 g/L)")
    ax.set_title("B. Growth-Uncoupled Primary Kinetics (Solid: Sugar, Dotted: EtOH)", fontweight="bold")
    ax.set_xlabel("Fermentation Time (Hours)")
    ax.set_ylabel("Concentration (g / L)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.8)

    # Panel C: Dynamic Liquid Temperature Trajectories
    ax = axes[1, 0]
    ax.plot(base_sol.t, base_sol.y[4], "r--", linewidth=1.8, label="Isothermal 15°C Baseline")
    for lbl, sim, col in zip(labels, batch_sims, colors):
        sol = sim["sol"]
        ax.plot(sol.t, sol.y[4], color=col, linewidth=2.1, label=f"{lbl.split(':')[0]} Liquid T(t)")
    ax.set_title("C. Sugar-Indexed Staged Liquid Temperature Profiles T(t)", fontweight="bold")
    ax.set_xlabel("Fermentation Time (Hours)")
    ax.set_ylabel("Bulk Must Temperature (°C)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    # Panel D: Grey-Box UDE Volatile Esters + 95% GP Epistemic Bands
    ax = axes[1, 1]
    ax.plot(base_sol.t, base_sol.y[6], "r--", linewidth=1.8, label="Baseline Esters (15°C)")
    for lbl, b, sim, col in zip(labels, selected_batches, batch_sims, colors):
        sol = sim["sol"]
        est_curve = sol.y[6]
        ci_band = 1.96 * b["ester_std"] * (est_curve / max(est_curve[-1], 1e-3))
        ax.plot(sol.t, est_curve, color=col, linewidth=2.1, label=f"{lbl.split(':')[0]} (±95% GP CI)")
        ax.fill_between(sol.t, est_curve - ci_band, est_curve + ci_band, color=col, alpha=0.16)
    ax.set_title("D. Grey-Box UDE Volatile Ester Synthesis vs. CO2 Stripping", fontweight="bold")
    ax.set_xlabel("Fermentation Time (Hours)")
    ax.set_ylabel("Volatile Fruity Esters (mg / L)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="lower right")

    # Panel E: Eq. (28) Inverse Thermodynamic Scale-Up (10 L -> 10,000 L PLC Setpoints)
    ax = axes[2, 0]
    sc_bal = batch_scaleups[1]  # Balanced Pareto Blueprint
    ax.plot(sc_bal["t"], sc_bal["T_lab"], color="#1B6B36", linewidth=2.2, label="Target Must Temp T*(t) (Batch 2)")
    ax.plot(sc_bal["t"], sc_bal["T_cool_lab"], color="#52B788", linestyle="--", linewidth=1.8, label="10 L Lab Jacket Coolant (A_J/V = 22 m⁻¹)")
    ax.plot(sc_bal["t"], sc_bal["T_cool_ind"], color="#081C15", linestyle="-.", linewidth=2.1, label="10,000 L Ind. PLC Glycol Setpoint (A_J/V = 2.1 m⁻¹)")
    ax.axhline(IND_VESSEL["T_coolant_min"] + IND_VESSEL["dT_margin"], color="#D90429", linestyle=":", linewidth=1.6, label="Winery Glycol Saturation Limit (-2.5°C)")
    ax.set_title("E. Inverse Scale-Up Transfer Function: 10 L Lab -> 10,000 L Tank (Eq. 28)", fontweight="bold")
    ax.set_xlabel("Fermentation Time (Hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.8, loc="lower right")

    # Panel F: Cumulative Chiller Electrical Energy Consumption
    ax = axes[2, 1]
    ax.plot(base_sol.t, base_sol.y[5], "r--", linewidth=2.0, label=f"15°C Baseline ({base_res['energy_kWh_hL']:.2f} kWh/hL)")
    for lbl, sim, col in zip(labels, batch_sims, colors):
        sol = sim["sol"]
        ax.plot(sol.t, sol.y[5], color=col, linewidth=2.1, label=f"{lbl.split(':')[0]} ({sim['energy_kWh_hL']:.2f} kWh/hL)")
    ax.set_title("F. Cumulative Chiller Electrical Consumption E_cool(t)", fontweight="bold")
    ax.set_xlabel("Fermentation Time (Hours)")
    ax.set_ylabel("Electrical Cooling Energy (kWh / hL)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("WineX Hybrid SciML Architecture: Physics-Informed UDE + Feasible Pareto Selection + Scale-Up PoC", fontsize=14, fontweight="bold", y=0.995)
    out_file = "winex_poc_v2_results.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"\n[OK] Saved 6-panel 300 DPI publication figure to '{out_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WineX Hybrid SciML PoC v2")
    parser.add_argument("--candidates", type=int, default=512, help="Power-of-2 Sobol candidate count (e.g., 512 or 8192)")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV with columns time_h,sugar_g_L for Panel B overlay")
    args = parser.parse_args()
    run_winex_v2(n_candidates=args.candidates, csv_path=args.csv)