import numpy as np
import pandas as pd

class Params:
    pass

def set_parameters_cAMP():
    df = pd.read_excel("initial_conditions and parameters.xlsx",
                       sheet_name="parameters_literature", engine="openpyxl")
    params = dict(zip(df['name'], df['value']))
    p = Params()
    for key, value in params.items():
        setattr(p, key, value)
    return p

def update_params_from_dict(p, param_dict):
    for key, value in param_dict.items():
        setattr(p, key, value)
    return p

def hard_switch(x, threshold=0.5):
    return np.where(x > threshold, 1, 0)

def soft_switch(x, threshold=0.5, steepness=4):
    return 1 / (1 + np.exp(-steepness * (x - threshold)))

def vtrap(x, y):
    if abs(x) < 1e-6:
        return y * (1 - x/y/2)
    else:
        return x / (1 - np.exp(-x / y))

def cAMP_model_ion(t, my_x, p, amplitude, period):
    (V, h, s, r, w,
     cAMP, AMP, Epac, Epac_ON, RAP1_GDP, RAP1_GTP,
     ERK, pERK, ERK_dimer, C1, PKA_tet, PKA_reg, PKA_cat, PKA_cat1,
     PDE4D7, PDE4D7_ERK, PDE4D7_PKA, PDE4D7_DP,
     PDE4D4, PDE4D4_ERK, PDE4D4_PKA, PDE4D4_DP,
     PDE4D5, PDE4D5_ERK, PDE4D5_PKA, PDE4D5_DP,
     PDE4D3, PDE4D3_ERK, PDE4D3_PKA, PDE4D3_DP,
     PDE4D8, PDE4D8_ERK, PDE4D8_PKA, PDE4D8_DP,
     PDE4D9, PDE4D9_ERK, PDE4D9_PKA, PDE4D9_DP,
     PDE4D1, PDE4D1_ERK, PDE4D1_PKA, PDE4D1_DP,
     PDE4D2, PDE4D2_ERK, PDE4D2_PKA, PDE4D2_DP,
     PDE4D6, PDE4D6_ERK, PDE4D6_PKA, PDE4D6_DP,
     PDE4D7_rol, PDE4D7_ERK_rol, PDE4D7_PKA_rol, PDE4D7_DP_rol,
     PDE4D4_rol, PDE4D4_ERK_rol, PDE4D4_PKA_rol, PDE4D4_DP_rol,
     PDE4D5_rol, PDE4D5_ERK_rol, PDE4D5_PKA_rol, PDE4D5_DP_rol,
     PDE4D3_rol, PDE4D3_ERK_rol, PDE4D3_PKA_rol, PDE4D3_DP_rol,
     PDE4D8_rol, PDE4D8_ERK_rol, PDE4D8_PKA_rol, PDE4D8_DP_rol,
     PDE4D9_rol, PDE4D9_ERK_rol, PDE4D9_PKA_rol, PDE4D9_DP_rol,
     PDE4D1_rol, PDE4D1_ERK_rol,
     PDE4D2_rol, PDE4D2_ERK_rol,
     PDE4D6_rol, PDE4D6_ERK_rol,
     CREB, pCREB, PDE4A, PDE4A_rol, PDE4B, PDE4B_rol, rolipram,
     Ca, CaM, Ca2_CaM, Ca4_CaM, AC1, AC1_Ca2_CaM, AC1_Ca4_CaM,
     PDE1, PDE1_Ca2_CaM, PDE1_Ca4_CaM,
     m_CaL, h_CaL, n_BK, w_RyR) = my_x

    
    dcAMP = 0.0
    dAMP = 0.0
    dEpac = 0.0
    dEpac_ON = 0.0
    dRAP1_GDP = 0.0
    dRAP1_GTP = 0.0
    dERK = 0.0
    dpERK = 0.0
    dERK_dimer = 0.0
    dC1 = 0.0
    dPKA_tet = 0.0
    dPKA_reg = 0.0
    dPKA_cat = 0.0
    dPKA_cat1 = 0.0
    # All PDE4 derivatives similarly
    dPDE4D7 = 0.0; dPDE4D7_ERK = 0.0; dPDE4D7_PKA = 0.0; dPDE4D7_DP = 0.0
    dPDE4D4 = 0.0; dPDE4D4_ERK = 0.0; dPDE4D4_PKA = 0.0; dPDE4D4_DP = 0.0
    dPDE4D5 = 0.0; dPDE4D5_ERK = 0.0; dPDE4D5_PKA = 0.0; dPDE4D5_DP = 0.0
    dPDE4D3 = 0.0; dPDE4D3_ERK = 0.0; dPDE4D3_PKA = 0.0; dPDE4D3_DP = 0.0
    dPDE4D8 = 0.0; dPDE4D8_ERK = 0.0; dPDE4D8_PKA = 0.0; dPDE4D8_DP = 0.0
    dPDE4D9 = 0.0; dPDE4D9_ERK = 0.0; dPDE4D9_PKA = 0.0; dPDE4D9_DP = 0.0
    dPDE4D1 = 0.0; dPDE4D1_ERK = 0.0; dPDE4D1_PKA = 0.0; dPDE4D1_DP = 0.0
    dPDE4D2 = 0.0; dPDE4D2_ERK = 0.0; dPDE4D2_PKA = 0.0; dPDE4D2_DP = 0.0
    dPDE4D6 = 0.0; dPDE4D6_ERK = 0.0; dPDE4D6_PKA = 0.0; dPDE4D6_DP = 0.0
    # Rolipram derivatives
    drolipram = 0.0
    dPDE4D7_rol = 0.0; dPDE4D7_ERK_rol = 0.0; dPDE4D7_PKA_rol = 0.0; dPDE4D7_DP_rol = 0.0
    dPDE4D4_rol = 0.0; dPDE4D4_ERK_rol = 0.0; dPDE4D4_PKA_rol = 0.0; dPDE4D4_DP_rol = 0.0
    dPDE4D5_rol = 0.0; dPDE4D5_ERK_rol = 0.0; dPDE4D5_PKA_rol = 0.0; dPDE4D5_DP_rol = 0.0
    dPDE4D3_rol = 0.0; dPDE4D3_ERK_rol = 0.0; dPDE4D3_PKA_rol = 0.0; dPDE4D3_DP_rol = 0.0
    dPDE4D8_rol = 0.0; dPDE4D8_ERK_rol = 0.0; dPDE4D8_PKA_rol = 0.0; dPDE4D8_DP_rol = 0.0
    dPDE4D9_rol = 0.0; dPDE4D9_ERK_rol = 0.0; dPDE4D9_PKA_rol = 0.0; dPDE4D9_DP_rol = 0.0
    dPDE4D1_rol = 0.0; dPDE4D1_ERK_rol = 0.0
    dPDE4D2_rol = 0.0; dPDE4D2_ERK_rol = 0.0
    dPDE4D6_rol = 0.0; dPDE4D6_ERK_rol = 0.0
    dCREB = 0.0; dpCREB = 0.0
    dPDE4A = 0.0; dPDE4A_rol = 0.0
    dPDE4B = 0.0; dPDE4B_rol = 0.0
    # ============================================================

    # ============================================================
    # 2. Ca²⁺/CaM SYSTEM
    # ============================================================
    dCa = (-2 * p.k_Ca2_CaM * (Ca**2) * CaM + 2 * p.k_inv_Ca2_CaM * Ca2_CaM) \
          + (-2 * p.k_Ca4_CaM * (Ca**2) * Ca2_CaM) + (2 * p.k_inv_Ca4_CaM * Ca4_CaM) \
          + ((-p.k_AC1_Ca4_CaM * AC1_Ca2_CaM * Ca) / (p.Km4 + Ca)) \
          + p.k_inv_AC1_Ca4_CaM * AC1_Ca4_CaM \
          + ((-p.k_PDE1_Ca2_CaM * PDE1_Ca2_CaM * (Ca**2)) / (p.Km6 + Ca)) \
          + p.k_inv_PDE1_Ca2_CaM * PDE1_Ca4_CaM \
          + ((-p.k_AC1_Ca4_CaM * AC1_Ca2_CaM * Ca) / (p.Km4 + Ca)) \
          + p.k_inv_AC1_Ca4_CaM * AC1_Ca4_CaM
    dCaM = -p.k_Ca2_CaM * CaM * Ca + p.k_inv_Ca2_CaM * Ca2_CaM
    dCa2_CaM = (+p.k_Ca2_CaM * (Ca**2) * CaM - p.k_inv_Ca2_CaM * Ca2_CaM) \
               + (-p.k_Ca4_CaM * Ca2_CaM * (Ca**2)) + p.k_inv_Ca4_CaM * Ca4_CaM \
               + (-p.k_AC1_Ca4_CaM * AC1 * Ca2_CaM * (Ca**2) + p.k_inv_AC1_Ca2_CaM * AC1_Ca2_CaM) \
               + (-p.k_PDE1_Ca2_CaM * PDE1 * Ca2_CaM + p.k_inv_PDE1_Ca2_CaM * PDE1_Ca2_CaM)
    dCa4_CaM = p.k_Ca4_CaM * (Ca**2) * Ca2_CaM - p.k_inv_Ca4_CaM * Ca4_CaM
    dAC1 = -p.k_AC1_Ca2_CaM * AC1 * Ca2_CaM + p.k_inv_AC1_Ca2_CaM * AC1_Ca2_CaM
    dAC1_Ca2_CaM = (+p.k_AC1_Ca2_CaM * AC1 * Ca2_CaM - p.k_inv_AC1_Ca2_CaM * AC1_Ca2_CaM) \
                   + (-p.k_AC1_Ca4_CaM * AC1_Ca2_CaM * Ca) / (p.Km4 + Ca) \
                   + p.k_inv_AC1_Ca4_CaM * AC1_Ca4_CaM \
                   - (p.k_c_Ca2_to_Ca4_AC1 * Ca2_CaM * AC1_Ca2_CaM) / (p.K_m_Ca2_to_Ca4_AC1 + AC1_Ca2_CaM) \
                   + (p.k_deg_Ca4_to_Ca2_AC1 * AC1_Ca4_CaM) / (p.K_m_Ca4_to_Ca2_AC1 + AC1_Ca4_CaM)
    dAC1_Ca4_CaM = (p.k_c_Ca2_to_Ca4_AC1 * Ca2_CaM * AC1_Ca2_CaM) / (p.K_m_Ca2_to_Ca4_AC1 + AC1_Ca2_CaM) \
                   - (p.k_deg_Ca4_to_Ca2_AC1 * AC1_Ca4_CaM) / (p.K_m_Ca4_to_Ca2_AC1 + AC1_Ca4_CaM)
    dPDE1 = -p.k_PDE1_Ca2_CaM * PDE1 * Ca2_CaM + p.k_inv_PDE1_Ca2_CaM * PDE1_Ca2_CaM
    dPDE1_Ca2_CaM = (+p.k_PDE1_Ca2_CaM * PDE1 * Ca2_CaM - p.k_inv_PDE1_Ca2_CaM * PDE1_Ca2_CaM) \
                    - (p.k_PDE1_Ca2_to_Ca4 * Ca2_CaM * PDE1_Ca2_CaM) / (p.Km_PDE1_Ca2_to_Ca4 + PDE1_Ca2_CaM) \
                    + (p.k_inv_PDE1_Ca4_to_Ca2 * PDE1_Ca4_CaM) / (p.Km_PDE1_Ca4_to_Ca2 + PDE1_Ca4_CaM)
    dPDE1_Ca4_CaM = (p.k_PDE1_Ca4_CaM * PDE1_Ca2_CaM * Ca) / (p.Km6 + Ca) - p.kb6 * PDE1_Ca4_CaM

    # ============================================================
    # 3. SWITCHES
    # ============================================================
    cAMP_switch = hard_switch(cAMP, threshold=2.0)
    PKA_switch = hard_switch(PKA_cat1, threshold=0.1575)
    CaM_switch = soft_switch(Ca4_CaM, threshold=0.5, steepness=5)

    # ============================================================
    # 4. VOLTAGE-GATED CURRENTS
    # ============================================================
    Am = (1.86 * (V + 25.4)) / (1 - np.exp(-(V + 25.4) / 10.3))
    Bm = (-0.086 * (V + 29.7)) / (1 - np.exp((V + 29.7) / 9.16))
    mInf = Am / (Am + Bm)
    Ah = (-0.0336 * (V + 118)) / (1 - np.exp((V + 118) / 11))
    Bh = 2.3 / (1 + np.exp(-(V + 35.8) / 13.4))
    hInf = Ah / (Ah + Bh)
    hTau = 1 / (p.Ph * (Ah + Bh))
    An = (0.186 * (V + 48.4)) / (1 - np.exp(-(V + 48.4) / 10.3))
    Bn = (-0.0086 * (V + 42.7)) / (1 - np.exp((V + 42.7) / 9.16))
    nInf = An / (An + Bn)
    As = (0.00122 * (V + 19.5)) / (1 - np.exp(-(V + 19.5) / 23.6))
    Bs = (-0.000739 * (V + 87.1)) / (1 - np.exp((V + 87.1) / 21.8))
    sInf = As / (As + Bs)
    sTau = 1 / (p.Ps * (As + Bs))
    Ar = (0.007 * np.exp(-(V + 95 * cAMP_switch + 103.5 * (1 - cAMP_switch)) / 19))
    Br = (0.007 * np.exp((V + 95 * cAMP_switch + 103.5 * (1 - cAMP_switch)) / 22))
    rInf = Ar / (Ar + Br)
    rTau = 1 / (p.Pr * (Ar + Br))
    wInf = 1 / (1 + np.exp(-(V + 35) / 10))
    wTau = 400 / (3.3 * np.exp((V + 35) / 20) + np.exp(-(V + 35) / 20))

    INaF = p.gNaF * (mInf**3) * h * (V - p.ENa)
    INaP = p.gNaP * (nInf**3) * (V - p.ENa)
    IKS = p.gKS * s * (V - p.EK)
    IL = p.gL * (V - p.ELK)
    IHCN = (p.gHCN + cAMP_switch * p.DgHCN) * r * (V - p.EHCN)
    IM = (p.gM + PKA_switch * p.DgM) * w * (V - p.EK)

    # ============================================================
    # 5. NEW ION CHANNELS (L-type, BK, NMDA)
    # ============================================================
    # L-type calcium channel
    m_CaL_inf = 1 / (1 + np.exp(-(V + 38) / 4))
    h_CaL_inf = 1 / (1 + np.exp((V + 63) / 8))
    tau_mCaL = 5.0
    tau_hCaL = 20.0
    phi_PKA = 1 + p.caL_pka_gain * (PKA_cat1 / (PKA_cat1 + p.Kd_PKA_CaL))
    ICaL = p.gCaL * phi_PKA * m_CaL * h_CaL * (V - p.ECa)

    # BK channel
    V_part_BK = 1 / (1 + np.exp(-(V + 10) / 10))
    Kd_eff = p.Kd_BK / (1 + p.bk_pka_gain * (PKA_cat1 / (PKA_cat1 + p.Kd_PKA_BK)))
    Ca_part_BK = (Ca**2) / (Ca**2 + Kd_eff**2)
    n_BK_inf = V_part_BK * Ca_part_BK
    tau_nBK = 1.5
    IBK = p.gBK * n_BK * (V - p.EK)

    # NMDA receptor
    kinetics = 1.0   
    B_V = 1 / (1 + p.k * np.exp(-p.b * V))
    INMDA = p.gNMDA_max * kinetics * B_V * (V - p.ENMDA)
    INMDA *= (1 + p.CaM_gain_NMDA * CaM_switch)

    # Calcium influx from currents
    gamma = 1.236e-6
    J_in = gamma * (ICaL + INMDA)

    # RyR (Keizer & Levine)
    v_rel = p.v_rel
    C_er_fixed = p.C_er_fixed
    K_a = p.K_a_RyR
    K_b = p.K_b_RyR
    K_c = p.K_c_RyR
    K_d_w = p.K_d_w_RyR
    v_leak = p.v_leak_RyR
    PKA_switch_RyR = p.PKA_switch_RyR
    Ca4 = Ca**4
    Ca3 = Ca**3
    K_a_eff = K_a * (1 - PKA_switch_RyR * PKA_switch)
    term_act = K_a_eff / (Ca4 + 1e-12)
    w_inf = (term_act + 1 + Ca3/K_b) / (1/(K_c + term_act) + 1 + Ca3/K_b)
    tau_w = w_inf / K_d_w
    P_open = (w_RyR * (1 + Ca3/K_b)) / (term_act + 1 + Ca3/K_b)
    J_er = (v_rel * P_open + v_leak) * (C_er_fixed - Ca)

    # Update dCa
    dCa = dCa + J_in + J_er

    # Gate ODEs
    dm_CaL = (m_CaL_inf - m_CaL) / tau_mCaL
    dh_CaL = (h_CaL_inf - h_CaL) / tau_hCaL
    dn_BK = (n_BK_inf - n_BK) / tau_nBK
    dw_RyR = (w_inf - w_RyR) / tau_w

    # Membrane potential ODE
    dV = (1 / p.Cm) * (-INaF - INaP - IKS - IL - IHCN - IM - ICaL - IBK - INMDA + p.IApp)

    # Original gating ODEs
    dh = (hInf - h) / hTau
    ds = (sInf - s) / sTau
    dr = (rInf - r) / rTau
    dw = (wInf - w) / wTau

    # ============================================================
    # 6. BUILD RESULTS LIST
    # ============================================================
    results = [dV, dh, ds, dr, dw,
               dcAMP, dAMP, dEpac, dEpac_ON, dRAP1_GDP, dRAP1_GTP,
               dERK, dpERK, dERK_dimer, dC1, dPKA_tet, dPKA_reg, dPKA_cat, dPKA_cat1,
               dPDE4D7, dPDE4D7_ERK, dPDE4D7_PKA, dPDE4D7_DP,
               dPDE4D4, dPDE4D4_ERK, dPDE4D4_PKA, dPDE4D4_DP,
               dPDE4D5, dPDE4D5_ERK, dPDE4D5_PKA, dPDE4D5_DP,
               dPDE4D3, dPDE4D3_ERK, dPDE4D3_PKA, dPDE4D3_DP,
               dPDE4D8, dPDE4D8_ERK, dPDE4D8_PKA, dPDE4D8_DP,
               dPDE4D9, dPDE4D9_ERK, dPDE4D9_PKA, dPDE4D9_DP,
               dPDE4D1, dPDE4D1_ERK, dPDE4D1_PKA, dPDE4D1_DP,
               dPDE4D2, dPDE4D2_ERK, dPDE4D2_PKA, dPDE4D2_DP,
               dPDE4D6, dPDE4D6_ERK, dPDE4D6_PKA, dPDE4D6_DP,
               dPDE4D7_rol, dPDE4D7_ERK_rol, dPDE4D7_PKA_rol, dPDE4D7_DP_rol,
               dPDE4D4_rol, dPDE4D4_ERK_rol, dPDE4D4_PKA_rol, dPDE4D4_DP_rol,
               dPDE4D5_rol, dPDE4D5_ERK_rol, dPDE4D5_PKA_rol, dPDE4D5_DP_rol,
               dPDE4D3_rol, dPDE4D3_ERK_rol, dPDE4D3_PKA_rol, dPDE4D3_DP_rol,
               dPDE4D8_rol, dPDE4D8_ERK_rol, dPDE4D8_PKA_rol, dPDE4D8_DP_rol,
               dPDE4D9_rol, dPDE4D9_ERK_rol, dPDE4D9_PKA_rol, dPDE4D9_DP_rol,
               dPDE4D1_rol, dPDE4D1_ERK_rol,
               dPDE4D2_rol, dPDE4D2_ERK_rol,
               dPDE4D6_rol, dPDE4D6_ERK_rol,
               dCREB, dpCREB, dPDE4A, dPDE4A_rol, dPDE4B, dPDE4B_rol, drolipram,
               dCa, dCaM, dCa2_CaM, dCa4_CaM,
               dAC1, dAC1_Ca2_CaM, dAC1_Ca4_CaM,
               dPDE1, dPDE1_Ca2_CaM, dPDE1_Ca4_CaM,
               dm_CaL, dh_CaL, dn_BK, dw_RyR]
    return results
