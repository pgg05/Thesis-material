import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cAMP_model_definition import cAMP_model_ion, set_parameters_cAMP, update_params_from_dict

# =============================================================================
# 1. INITIAL CONDITIONS
# =============================================================================
core_ic_values = [
    1.0,        # cAMP
    0.0,        # AMP
    0.488,      # Epac
    0.0,        # Epac_ON
    0.2,        # RAP1_GDP
    0.0,        # RAP1_GTP
    0.8,        # ERK
    0.0,        # pERK
    0.0,        # ERK_dimer
    0.0,        # C1
    0.173,      # PKA_tet
    0.023,      # PKA_reg
    0.023,      # PKA_cat
    0.0,        # PKA_cat1
    0.011493185,# PDE4D7
    0.0,        # PDE4D7_ERK
    0.0,        # PDE4D7_PKA
    0.0,        # PDE4D7_DP
    0.011493185,# PDE4D4
    0.0,        # PDE4D4_ERK
    0.0,        # PDE4D4_PKA
    0.0,        # PDE4D4_DP
    0.011493185,# PDE4D5
    0.0,        # PDE4D5_ERK
    0.0,        # PDE4D5_PKA
    0.0,        # PDE4D5_DP
    0.011493185,# PDE4D3
    0.0,        # PDE4D3_ERK
    0.0,        # PDE4D3_PKA
    0.0,        # PDE4D3_DP
    0.011493185,# PDE4D8
    0.0,        # PDE4D8_ERK
    0.0,        # PDE4D8_PKA
    0.0,        # PDE4D8_DP
    0.011493185,# PDE4D9
    0.0,        # PDE4D9_ERK
    0.0,        # PDE4D9_PKA
    0.0,        # PDE4D9_DP
    0.010836431,# PDE4D1
    0.0,        # PDE4D1_ERK
    0.0,        # PDE4D1_PKA
    0.0,        # PDE4D1_DP
    0.009358736,# PDE4D2
    0.0,        # PDE4D2_ERK
    0.0,        # PDE4D2_PKA
    0.0,        # PDE4D2_DP
    0.009358736,# PDE4D6
    0.0,        # PDE4D6_ERK
    0.0,        # PDE4D6_PKA
    0.0,        # PDE4D6_DP
    0.0,        # PDE4D7_rolipram
    0.0,        # PDE4D7_ERK_rolipram
    0.0,        # PDE4D7_PKA_rolipram
    0.0,        # PDE4D7_DP_rolipram
    0.0,        # PDE4D4_rolipram
    0.0,        # PDE4D4_ERK_rolipram
    0.0,        # PDE4D4_PKA_rolipram
    0.0,        # PDE4D4_DP_rolipram
    0.0,        # PDE4D5_rolipram
    0.0,        # PDE4D5_ERK_rolipram
    0.0,        # PDE4D5_PKA_rolipram
    0.0,        # PDE4D5_DP_rolipram
    0.0,        # PDE4D3_rolipram
    0.0,        # PDE4D3_ERK_rolipram
    0.0,        # PDE4D3_PKA_rolipram
    0.0,        # PDE4D3_DP_rolipram
    0.0,        # PDE4D8_rolipram
    0.0,        # PDE4D8_ERK_rolipram
    0.0,        # PDE4D8_PKA_rolipram
    0.0,        # PDE4D8_DP_rolipram
    0.0,        # PDE4D9_rolipram
    0.0,        # PDE4D9_ERK_rolipram
    0.0,        # PDE4D9_PKA_rolipram
    0.0,        # PDE4D9_DP_rolipram
    0.0,        # PDE4D1_rolipram
    0.0,        # PDE4D1_ERK_rolipram
    0.0,        # PDE4D2_rolipram
    0.0,        # PDE4D2_ERK_rolipram
    0.0,        # PDE4D6_rolipram
    0.0,        # PDE4D6_ERK_rolipram
    0.085,      # CREB
    0.015,      # pCREB
    0.087360595,# PDE4A
    0.0,        # PDE4A_rolipram
    0.214126394,# PDE4B
    0.0,        # PDE4B_rolipram
    0.0         # rolipram
]
# V, h, s, r, w
vhsrw_ics = [-65.0, 0.5, 0.5, 0.5, 0.5]

# Ca²⁺/CaM system 
ca_cam_ics = [1.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0]

# New gating variables (m_CaL, h_CaL, n_BK, w_RyR)
extra_gates = [0.0, 1.0, 0.0, 0.0]

initial_conditions = vhsrw_ics + core_ic_values + ca_cam_ics + extra_gates
print(f"Initial conditions length: {len(initial_conditions)} (should be 106)")

# =============================================================================
# 2. PARAMETER FUNCTION 
# =============================================================================
def get_parameters(gCaL_value, gBK_value):
    p = set_parameters_cAMP()
    
    missing_params = {
        # Temperature and scaling
        'Tmp': 37.0,
        'Cm': 0.9,               # membrane capacitance (pF)
        # Maximal conductances (pS or nS)
        'gNaF': 1000.0,
        'gNaP': 1.0,
        'gKS': 40.0,
        'gL': 11.3,
        'DgHCN': 12.0,           # cAMP‑dependent HCN increase
        'DgM': 50.0,             # PKA‑dependent M‑current increase
        'gM': 50.0,              # baseline M‑current conductance
        'gHCN': 23.0,            # baseline HCN conductance
        # Reversal potentials (mV)
        'ENa': 50.0,
        'EK': -84.0,
        'ELK': -83.38,
        'EHCN': -50.0,
        # Q10 scaling (will be recomputed)
        'Ps': 3.0 ** ((37.0 - 20) / 10),
        'Ph': 2.9 ** ((37.0 - 20) / 10),
        'Pr': 3.0 ** ((37.0 - 35) / 10),
        # L‑type and BK (user provided)
        'gCaL': gCaL_value,
        'gBK': gBK_value,
        'ECa': 60.0,
        'Kd_BK': 5.0,
        'n_BK': 2.0,
        'caL_pka_gain': 2.0,
        'Kd_PKA_CaL': 0.5,
        'bk_pka_gain': 3.0,
        'Kd_PKA_BK': 0.5,
        # NMDA
        'gNMDA_max': 0.29,
        'ENMDA': 0.0,
        'k': 0.28,
        'b': 0.06,
        'CaM_gain_NMDA': 1.5,
        # RyR
        'v_rel': 0.5,
        'C_er_fixed': 100.0,
        'K_a_RyR': 0.2,
        'K_b_RyR': 0.3,
        'K_c_RyR': 0.4,
        'K_d_w_RyR': 10.0,
        'v_leak_RyR': 0.01,
        'PKA_switch_RyR': 0.5,
        # Ca²⁺/CaM system (all constants)
        'k_Ca2_CaM': 100.0,
        'k_inv_Ca2_CaM': 0.1,
        'k_Ca4_CaM': 100.0,
        'k_inv_Ca4_CaM': 0.04,
        'k_AC1_Ca4_CaM': 0.5,
        'k_inv_AC1_Ca4_CaM': 0.01,
        'Km4': 0.5,
        'k_AC1_Ca2_CaM': 0.01,
        'k_inv_AC1_Ca2_CaM': 0.001,
        'k_PDE1_Ca2_CaM': 0.005,
        'k_inv_PDE1_Ca2_CaM': 0.005,
        'Km6': 1.8,
        'k_PDE1_Ca4_CaM': 10.0,
        'k_inv_PDE1_Ca4_CaM': 0.005,
        'k_c_Ca2_to_Ca4_AC1': 0.5,
        'K_m_Ca2_to_Ca4_AC1': 1.0,
        'k_deg_Ca4_to_Ca2_AC1': 0.05,
        'K_m_Ca4_to_Ca2_AC1': 1.0,
        'k_PDE1_Ca2_to_Ca4': 0.2,
        'Km_PDE1_Ca2_to_Ca4': 1.0,
        'k_inv_PDE1_Ca4_to_Ca2': 0.005,
        'Km_PDE1_Ca4_to_Ca2': 1.0,
        'kb6': 0.05,
    }
    update_params_from_dict(p, missing_params)
    return p

# =============================================================================
# 3. SIMULATION SETTINGS
# =============================================================================
ATOL = 1e-8
RTOL = 1e-6
t_span_eq = [0, 400]   # equilibration
t_span_run = [0, 400]  # current step
current_amplitude = 0.45
current_period = 20000

def run_nimodipine_comparison():
    results = {}
    for condition, (gCaL, gBK) in [('Control', (20.0, 60.0)), ('Nimodipine', (6.0, 60.0))]:
        print(f"Simulating {condition}...")
        p = get_parameters(gCaL, gBK)
        # Equilibration with IApp = 0
        p.IApp = 0
        eq = solve_ivp(cAMP_model_ion, t_span_eq, initial_conditions,
                       method='LSODA', args=(p, current_amplitude, current_period),
                       atol=ATOL, rtol=RTOL)
        # Current step at 250 pA
        p.IApp = 250
        sol = solve_ivp(cAMP_model_ion, t_span_run, eq.y[:, -1],
                        method='LSODA', args=(p, current_amplitude, current_period),
                        atol=ATOL, rtol=RTOL)
        results[condition] = sol
    return results

def plot_results(results):
    plt.figure(figsize=(10,6))
    plt.plot(results['Control'].t, results['Control'].y[0], 'k-', label='Control (gCaL=20 nS)', linewidth=2)
    plt.plot(results['Nimodipine'].t, results['Nimodipine'].y[0], 'r-', label='Nimodipine (gCaL=6 nS)', linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.title('L-type calcium channel block at 250 pA')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    res = run_nimodipine_comparison()
    plot_results(res)
