import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from cAMP_model_definition import cAMP_model_ion, set_parameters_cAMP, update_params_from_dict

# =============================================================================
# 1. LOAD INITIAL CONDITIONS FROM EXCEL
# =============================================================================
file_path = "new_equilibrium_results.xlsx"  

def get_ic(sheet):
    return pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")["value"].dropna().tolist()

try:
    core_ic_values = get_ic('IC_literature')
    print(f"Loaded {len(core_ic_values)} core ICs from Excel")
except:
    print("Could not load Excel. Using fallback hardcoded values (may be wrong length).")
    core_ic_values = [1.0, 0.0, 0.488, 0.0, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0,
                      0.173, 0.023, 0.023, 0.0, 0.011493185, 0.0, 0.0, 0.0,
                      0.011493185, 0.0, 0.0, 0.0, 0.011493185, 0.0, 0.0, 0.0,
                      0.011493185, 0.0, 0.0, 0.0, 0.011493185, 0.0, 0.0, 0.0,
                      0.011493185, 0.0, 0.0, 0.0, 0.010836431, 0.0, 0.0, 0.0,
                      0.009358736, 0.0, 0.0, 0.0, 0.009358736, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.085, 0.015, 0.087360595, 0.0,
                      0.214126394, 0.0, 0.0]

# Fixed parts
vhsrw_ics = [-65.0, 0.5, 0.5, 0.5, 0.5]
ca_cam_ics = [1.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0]
extra_gates = [0.0, 1.0, 0.0, 0.0]

initial_conditions = vhsrw_ics + core_ic_values + ca_cam_ics + extra_gates
print(f"Initial conditions length: {len(initial_conditions)} (should be 106)")

if len(initial_conditions) != 106:
    print("WARNING: Length mismatch. Adjust core_ic_values or check Excel.")
    if len(initial_conditions) > 106:
        initial_conditions = initial_conditions[:106]
    else:
        initial_conditions = initial_conditions + [0.0]*(106 - len(initial_conditions))

# =============================================================================
# 2. PARAMETER FUNCTION (with GEBR-7b inhibition)
# =============================================================================
def get_parameters(condition):
    p = set_parameters_cAMP()
    # Baseline parameters
    missing_params = {
        'Tmp': 37.0, 'Cm': 0.9,
        'gNaF': 1000.0, 'gNaP': 1.0, 'gKS': 40.0, 'gL': 11.3,
        'DgHCN': 12.0, 'DgM': 50.0, 'gM': 50.0, 'gHCN': 23.0,
        'ENa': 50.0, 'EK': -84.0, 'ELK': -83.38, 'EHCN': -50.0,
        'Ps': 3.0 ** ((37.0 - 20) / 10),
        'Ph': 2.9 ** ((37.0 - 20) / 10),
        'Pr': 3.0 ** ((37.0 - 35) / 10),
        'gCaL': 20.0, 'gBK': 60.0, 'ECa': 60.0,
        'Kd_BK': 5.0, 'n_BK': 2.0,
        'caL_pka_gain': 2.0, 'Kd_PKA_CaL': 0.5,
        'bk_pka_gain': 3.0, 'Kd_PKA_BK': 0.5,
        'gNMDA_max': 0.29, 'ENMDA': 0.0, 'k': 0.28, 'b': 0.06, 'CaM_gain_NMDA': 1.5,
        'v_rel': 0.5, 'C_er_fixed': 100.0, 'K_a_RyR': 0.2, 'K_b_RyR': 0.3,
        'K_c_RyR': 0.4, 'K_d_w_RyR': 10.0, 'v_leak_RyR': 0.01, 'PKA_switch_RyR': 0.5,
        'k_Ca2_CaM': 100.0, 'k_inv_Ca2_CaM': 0.1,
        'k_Ca4_CaM': 100.0, 'k_inv_Ca4_CaM': 0.04,
        'k_AC1_Ca4_CaM': 0.5, 'k_inv_AC1_Ca4_CaM': 0.01,
        'Km4': 0.5, 'k_AC1_Ca2_CaM': 0.01, 'k_inv_AC1_Ca2_CaM': 0.001,
        'k_PDE1_Ca2_CaM': 0.005, 'k_inv_PDE1_Ca2_CaM': 0.005,
        'Km6': 1.8, 'k_PDE1_Ca4_CaM': 10.0, 'k_inv_PDE1_Ca4_CaM': 0.005,
        'k_c_Ca2_to_Ca4_AC1': 0.5, 'K_m_Ca2_to_Ca4_AC1': 1.0,
        'k_deg_Ca4_to_Ca2_AC1': 0.05, 'K_m_Ca4_to_Ca2_AC1': 1.0,
        'k_PDE1_Ca2_to_Ca4': 0.2, 'Km_PDE1_Ca2_to_Ca4': 1.0,
        'k_inv_PDE1_Ca4_to_Ca2': 0.005, 'Km_PDE1_Ca4_to_Ca2': 1.0,
        'kb6': 0.05,
    }
    update_params_from_dict(p, missing_params)

    # GEBR-7b PDE4 inhibition
    if condition == 'GEBR7b':
        p.kcat_D3 *= 0.33
        p.kcat_D3_ERK *= 0.33
        p.kcat_D3_PKA *= 0.332
        p.kcat_D3_DP *= 0.33
        p.kcat_D2 *= 0.24
        p.kcat_D2_ERK *= 0.24
        p.kcat_D1 *= 0.43
        p.kcat_D1_ERK *= 0.43
        p.kcat_D1_PKA *= 0.43
        p.kcat_D1_DP *= 0.43
        for iso in ['D4', 'D5', 'D6', 'D7', 'D8', 'D9']:
            setattr(p, f'kcat_{iso}', getattr(p, f'kcat_{iso}') * 0.16)
            setattr(p, f'kcat_{iso}_ERK', getattr(p, f'kcat_{iso}_ERK') * 0.16)
            setattr(p, f'kcat_{iso}_PKA', getattr(p, f'kcat_{iso}_PKA') * 0.16)
            setattr(p, f'kcat_{iso}_DP', getattr(p, f'kcat_{iso}_DP') * 0.16)
    return p

# =============================================================================
# 3. SPIKE ANALYSIS FUNCTIONS 
# =============================================================================
def detect_spikes(V, t, threshold=0):
    crossings = np.where((V[:-1] < threshold) & (V[1:] >= threshold))[0]
    return t[crossings], len(crossings)

def first_spike_latency(V, t, stim_start=0, threshold=0):
    spike_times, n = detect_spikes(V, t, threshold)
    if n == 0:
        return np.nan
    return spike_times[0] - stim_start

def compute_fI_curve(results, I_values, stim_start=0):
    fI = {'Control': [], 'GEBR7b': []}
    for i, I in enumerate(I_values):
        for cond in ['Control', 'GEBR7b']:
            sol = results[cond][i]
            _, n = detect_spikes(sol.y[0], sol.t)
            t_run = sol.t[-1] - stim_start
            rate = n / (t_run / 1000)
            fI[cond].append(rate)
    return fI

def compute_AHP(V, t, spike_times, search_window=20):
    ahp_vals = []
    for st in spike_times:
        idx = np.searchsorted(t, st)
        end_idx = np.searchsorted(t, st + search_window)
        if end_idx > len(V):
            end_idx = len(V)
        ahp_vals.append(np.min(V[idx:end_idx]))
    return ahp_vals

# =============================================================================
# 4. SIMULATION (Control vs GEBR-7b at multiple currents)
# =============================================================================
ATOL, RTOL = 1e-8, 1e-6
t_span_eq = [0, 400]
t_span_run = [0, 400]
current_amplitude = 0.45
current_period = 20000
IApp_test_values = [115, 200, 250, 300]

def run_gebr7b_comparison():
    results = {'Control': [], 'GEBR7b': []}
    for condition in ['Control', 'GEBR7b']:
        print(f"Simulating {condition}...")
        p = get_parameters(condition)
        p.IApp = 0
        eq = solve_ivp(cAMP_model_ion, t_span_eq, initial_conditions,
                       method='LSODA', args=(p, current_amplitude, current_period),
                       atol=ATOL, rtol=RTOL)
        step_start = eq.y[:, -1]
        for i_app in IApp_test_values:
            p.IApp = i_app
            sol = solve_ivp(cAMP_model_ion, t_span_run, step_start,
                            method='LSODA', args=(p, current_amplitude, current_period),
                            atol=ATOL, rtol=RTOL)
            results[condition].append(sol)
    return results

# =============================================================================
# 5. COMPUTE AND DISPLAY RESULTS
# =============================================================================
results = run_gebr7b_comparison()

print("\n===== Spike Count & First Spike Latency =====")
for i, I in enumerate(IApp_test_values):
    sol_c = results['Control'][i]
    sol_g = results['GEBR7b'][i]
    _, n_c = detect_spikes(sol_c.y[0], sol_c.t)
    _, n_g = detect_spikes(sol_g.y[0], sol_g.t)
    lat_c = first_spike_latency(sol_c.y[0], sol_c.t)
    lat_g = first_spike_latency(sol_g.y[0], sol_g.t)
    print(f"I={I} pA: Control spikes={n_c}, lat={lat_c:.1f} ms | GEBR-7b spikes={n_g}, lat={lat_g:.1f} ms")

fI = compute_fI_curve(results, IApp_test_values)
plt.figure()
plt.plot(IApp_test_values, fI['Control'], 'ko-', label='Control')
plt.plot(IApp_test_values, fI['GEBR7b'], 'ro-', label='GEBR-7b')
plt.xlabel('Current (pA)')
plt.ylabel('Firing rate (Hz)')
plt.title('f‑I curve: PDE4 inhibition (GEBR-7b)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

idx = 2  # 250 pA
sol_c = results['Control'][idx]
sol_g = results['GEBR7b'][idx]
spike_c, _ = detect_spikes(sol_c.y[0], sol_c.t)
spike_g, _ = detect_spikes(sol_g.y[0], sol_g.t)
ahp_c = compute_AHP(sol_c.y[0], sol_c.t, spike_c)
ahp_g = compute_AHP(sol_g.y[0], sol_g.t, spike_g)
print(f"\nAHP at 250 pA: Control mean = {np.mean(ahp_c):.2f} mV, GEBR-7b = {np.mean(ahp_g):.2f} mV")
plt.figure()
plt.hist(ahp_c, bins=10, alpha=0.5, label='Control')
plt.hist(ahp_g, bins=10, alpha=0.5, label='GEBR-7b')
plt.xlabel('AHP amplitude (mV)')
plt.ylabel('Count')
plt.legend()
plt.title('AHP distribution at 250 pA')
plt.show()

# =============================================================================
# 6. SENSITIVITY ANALYSIS
# =============================================================================
print("\n===== Sensitivity Analysis: varying PDE4 inhibition (factor on kcat_D7) =====")
factors = [1.0, 0.8, 0.6, 0.4, 0.2, 0.16]
firing_rates = []
I_test = 250
for factor in factors:
    p = set_parameters_cAMP()
    p = get_parameters('Control')  # start from control
    p.kcat_D7 *= factor   # reduce kcat_D7 by factor
    p.kcat_D7_ERK *= factor
    p.kcat_D7_PKA *= factor
    p.kcat_D7_DP *= factor
    # Equilibrate
    p_eq = p  # same parameters for equilibration
    p_eq.IApp = 0
    eq = solve_ivp(cAMP_model_ion, t_span_eq, initial_conditions,
                   method='LSODA', args=(p_eq, current_amplitude, current_period),
                   atol=ATOL, rtol=RTOL)
    step_start = eq.y[:, -1]
    p_step = p
    p_step.IApp = I_test
    sol = solve_ivp(cAMP_model_ion, t_span_run, step_start,
                    method='LSODA', args=(p_step, current_amplitude, current_period),
                    atol=ATOL, rtol=RTOL)
    _, n = detect_spikes(sol.y[0], sol.t)
    rate = n / (sol.t[-1] / 1000)
    firing_rates.append(rate)
    print(f"Inhibition factor = {factor:.2f} → firing rate = {rate:.2f} Hz")

plt.figure()
plt.plot(factors, firing_rates, 'bo-')
plt.xlabel('PDE4 inhibition factor (kcat multiplier)')
plt.ylabel('Firing rate (Hz)')
plt.title('Sensitivity: effect of PDE4 inhibition on firing rate (250 pA)')
plt.gca().invert_xaxis()  
plt.grid(alpha=0.3)
plt.show()
