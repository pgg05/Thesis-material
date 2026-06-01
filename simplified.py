import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# 0. IMPORTS (USER MODULES)
# =============================================================================
from ion_channel_model_definition_test import cAMP_model_ion, set_paramaters_cAMP, update_params_from_dict

# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================

# ODE solver tolerances — tighten if you see numerical artifacts,
# loosen to speed up long simulations
ATOL = 1e-8  # Absolute tolerance
RTOL = 1e-6  # Relative tolerance

file_path = "new_equilibrium_results.xlsx"

print("Loading Initial Conditions...")

def get_ic(sheet):
    data = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")["value"].dropna().tolist()
    print(f"Sheet '{sheet}' has {len(data)} values after dropna()")
    return data

try:
    initial_conditions      = get_ic('IC_literature')  # Control / Normal PDE
    initial_conditions_exp2 = get_ic('IC_literature')  # Stimulated / Normal PDE (simulates L-type channel inhibition)
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()
    
# Time spans (in ms)
t_span_eq  = [0, 400]  # Equilibration run — lets the model settle before the current step
t_span_run = [0, 400]  # Current-step run — the window we actually analyse / plot

# Applied current parameters passed into the model
current_amplitude = 0.45   # Amplitude of the oscillatory drive (model units)
current_period    = 20000  # Period of the oscillatory drive (ms)

# The four DC current-step amplitudes to sweep (pA)
IApp_test_values = [115, 200, 250, 300]


# =============================================================================
# 2. PARAMETERS
# =============================================================================

def get_base_parameters(block_level=0.0):
    """
    Build a fresh parameter object with physiological defaults.
    
    Parameters:
    -----------
    block_level : float, default 0.0
        Fractional block of L-type calcium channels (nimodipine).
        0.0 = no block, 0.7 = 70% block, 1.0 = full block.
    """
    p = set_paramaters_cAMP()
    update_params_from_dict(p, {
        'Tmp':   37,     # Temperature (°C)
        'Cm':    0.9,    # Membrane capacitance (pF)
        # Maximal conductances (pS or nS depending on convention)
        'gNaF':  1000,   # Fast sodium
        'gNaP':  1,      # Persistent sodium
        'gKS':   40,     # Slow potassium
        'gL':    11.3,   # Leak
        'DgHCN': 12,     # cAMP-dependent HCN conductance increment
        'DgM':   50,     # cAMP-dependent M-current conductance increment
        'gM':    50,     # Baseline M-current conductance
        'gHCN':  23,     # Baseline HCN conductance
        # L‑type calcium channel
        'gCaL': 20.0 * (1 - block_level),     # Apply nimodipine block here
        'ECa':  60.0,     # Calcium reversal potential (mV)
        # BK channel
        'gBK':  60.0,     # Maximal BK conductance (nS)
        'Kd_BK': 5.0,     # Ca²⁺ affinity (µM)
        'n_BK':  2.0,     # Hill coefficient for Ca²⁺ dependence
        
        # ===== PKA modulation parameters =====
        # L-type (Cav1.2) modulation
        'caL_pka_gain': 2.0,      # Max fold increase in gCaL (1.0 = no effect)
        'Kd_PKA_CaL': 0.5,        # PKA concentration for half-max effect (μM)
        
        # BK channel modulation
        'bk_pka_gain': 3.0,       # Max fold reduction in Kd_BK (1.0 = no effect)
        'Kd_PKA_BK': 0.5,         # PKA concentration for half-max effect (μM)
        # ==========================================
        
        # Reversal potentials (mV)
        'ENa':   50,
        'EK':   -84,
        'ELK':  -83.38,
        'EHCN': -50,
    })
    
    # Apply Q10 temperature scaling
    p.Ps = 3.0 ** ((p.Tmp - 20) / 10)
    p.Ph = 2.9 ** ((p.Tmp - 20) / 10)
    p.Pr = 3.0 ** ((p.Tmp - 35) / 10)

    return p


# =============================================================================
# 3. SIMULATION
# =============================================================================

def run_experiment_1():
    """
    Run Experiment 1 (Control vs L-type blockage) across all IApp values.
    
    Control:    block_level = 0.0 (no nimodipine)
    Blocked:    block_level = 0.7 (70% block of L-type channels)
    """
    results = {'Control': [], 'Ltype_blocked': []}
    
    # Control condition (no block)
    p_ctrl = get_base_parameters(block_level=0.0)
    p_ctrl.IApp = 0
    
    print("   [Control] Finding equilibrium...")
    eq_sim_ctrl = solve_ivp(
        cAMP_model_ion, t_span_eq, initial_conditions,
        method='LSODA',
        args=(p_ctrl, current_amplitude, current_period),
        atol=ATOL, rtol=RTOL
    )
    step_start_IC_ctrl = eq_sim_ctrl.y[:, -1]
    
    # Blocked condition (70% block)
    p_block = get_base_parameters(block_level=0.7)
    p_block.IApp = 0
    
    print("   [Ltype_blocked] Finding equilibrium...")
    eq_sim_block = solve_ivp(
        cAMP_model_ion, t_span_eq, initial_conditions,
        method='LSODA',
        args=(p_block, current_amplitude, current_period),
        atol=ATOL, rtol=RTOL
    )
    step_start_IC_block = eq_sim_block.y[:, -1]
    
    # Sweep IApp values for Control
    step_results_ctrl = []
    for i_val in IApp_test_values:
        p_ctrl.IApp = i_val
        sol = solve_ivp(
            cAMP_model_ion, t_span_run, step_start_IC_ctrl,
            method='LSODA',
            args=(p_ctrl, current_amplitude, current_period),
            atol=ATOL, rtol=RTOL
        )
        step_results_ctrl.append(sol)
    results['Control'] = step_results_ctrl
    
    # Sweep IApp values for Blocked
    step_results_block = []
    for i_val in IApp_test_values:
        p_block.IApp = i_val
        sol = solve_ivp(
            cAMP_model_ion, t_span_run, step_start_IC_block,
            method='LSODA',
            args=(p_block, current_amplitude, current_period),
            atol=ATOL, rtol=RTOL
        )
        step_results_block.append(sol)
    results['Ltype_blocked'] = step_results_block
    
    return results


# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def plot_figure_1(results):
    """
    2×4 grid comparing Control (black) vs CGS/Stimulated (red)
    across the four IApp values, zoomed into the steady-state window.
    """
    ZOOM_X = [350, 400]  # ms — crop to the last 50 ms to show steady-state firing

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    fig.suptitle('Figure 1: Experiment 1 — Control vs L-type blockage', fontsize=14)

    for i, i_app in enumerate(IApp_test_values):
        sol_ctrl = results['Control'][i]
        sol_stim = results['Stimulated'][i]

        # Row 0: Control condition (black trace)
        axes[0, i].plot(sol_ctrl.t, sol_ctrl.y[0], 'k')
        axes[0, i].set_title(f'{i_app} pA')
        axes[0, i].set_xlim(ZOOM_X)
        axes[0, i].grid(alpha=0.3)

        # Row 1: Stimulated / CGS condition (red trace)
        axes[1, i].plot(sol_stim.t, sol_stim.y[0], 'r')
        axes[1, i].set_xlabel('Time (ms)')
        axes[1, i].set_xlim(ZOOM_X)
        axes[1, i].grid(alpha=0.3)

        # Y-axis labels on the leftmost column only
        if i == 0:
            axes[0, i].set_ylabel('Control (mV)')
            axes[1, i].set_ylabel('L-type blockage / Stimulated (mV)')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. ENTRY POINT
# =============================================================================
results = run_experiment_1()
plot_figure_1(results)
