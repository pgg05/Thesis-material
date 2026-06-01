import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# ====================================================================
# 1. PARAMETER LOADING
# ====================================================================
class Params:
    pass

def set_parameters_cAMP():
    try:
        df = pd.read_excel("initial_conditions and parameters.xlsx",
                           sheet_name="parameters_literature", engine="openpyxl")
        params = dict(zip(df['name'], df['value']))
        p = Params()
        for key, value in params.items():
            setattr(p, key, value)
        return p
    except:
        print("Warning: Could not load Excel file. Using default parameters.")
        p = Params()
        return p

def update_params_from_dict(p, d):
    for k, v in d.items():
        setattr(p, k, v)

# ====================================================================
# 2. HELPER FUNCTIONS
# ====================================================================
def hard_switch(x, th=0.5):
    return np.where(x > th, 1, 0)

def soft_switch(x, th=0.5, steep=4):
    return 1/(1+np.exp(-steep*(x-th)))

def vtrap(x, y):
    if abs(x) < 1e-6:
        return y*(1 - x/y/2)
    else:
        return x/(1-np.exp(-x/y))

# ====================================================================
# 3. ODEs
# ====================================================================
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

    # -----------------------------------------------------------------
    # 3.1 cAMP dynamics
    # -----------------------------------------------------------------
    # AC1 production (Ca²⁺/CaM dependent)
    kcat_AC1 = getattr(p, 'kcat_AC1_Ca4', 30.0)
    AC1_prod = kcat_AC1 * AC1_Ca4_CaM

    def mm(rate, km, enz, sub):
        return rate * enz * sub / (km + sub) if enz > 0 else 0

    deg_D7 = mm(getattr(p,'kcat_D7',10), getattr(p,'Km_D7',1.8), PDE4D7, cAMP)
    deg_D7_ERK = mm(getattr(p,'kcat_D7_ERK',0.01), getattr(p,'Km_D7_ERK',1.8), PDE4D7_ERK, cAMP)
    deg_D7_PKA = mm(getattr(p,'kcat_D7_PKA',27), getattr(p,'Km_D7_PKA',1.8), PDE4D7_PKA, cAMP)
    deg_D7_DP = mm(getattr(p,'kcat_D7_DP',10), getattr(p,'Km_D7_DP',1.8), PDE4D7_DP, cAMP)

    deg_D4 = mm(getattr(p,'kcat_D4',10), getattr(p,'Km_D4',1.8), PDE4D4, cAMP)
    deg_D4_ERK = mm(getattr(p,'kcat_D4_ERK',0.01), getattr(p,'Km_D4_ERK',1.8), PDE4D4_ERK, cAMP)
    deg_D4_PKA = mm(getattr(p,'kcat_D4_PKA',27), getattr(p,'Km_D4_PKA',1.8), PDE4D4_PKA, cAMP)
    deg_D4_DP = mm(getattr(p,'kcat_D4_DP',10), getattr(p,'Km_D4_DP',1.8), PDE4D4_DP, cAMP)

    deg_D5 = mm(getattr(p,'kcat_D5',10), getattr(p,'Km_D5',1.8), PDE4D5, cAMP)
    deg_D5_ERK = mm(getattr(p,'kcat_D5_ERK',0.01), getattr(p,'Km_D5_ERK',1.8), PDE4D5_ERK, cAMP)
    deg_D5_PKA = mm(getattr(p,'kcat_D5_PKA',27), getattr(p,'Km_D5_PKA',1.8), PDE4D5_PKA, cAMP)
    deg_D5_DP = mm(getattr(p,'kcat_D5_DP',10), getattr(p,'Km_D5_DP',1.8), PDE4D5_DP, cAMP)

    deg_D3 = mm(getattr(p,'kcat_D3',10), getattr(p,'Km_D3',1.8), PDE4D3, cAMP)
    deg_D3_ERK = mm(getattr(p,'kcat_D3_ERK',0.01), getattr(p,'Km_D3_ERK',1.8), PDE4D3_ERK, cAMP)
    deg_D3_PKA = mm(getattr(p,'kcat_D3_PKA',27), getattr(p,'Km_D3_PKA',1.8), PDE4D3_PKA, cAMP)
    deg_D3_DP = mm(getattr(p,'kcat_D3_DP',10), getattr(p,'Km_D3_DP',1.8), PDE4D3_DP, cAMP)

    deg_D8 = mm(getattr(p,'kcat_D8',10), getattr(p,'Km_D8',1.8), PDE4D8, cAMP)
    deg_D8_ERK = mm(getattr(p,'kcat_D8_ERK',0.01), getattr(p,'Km_D8_ERK',1.8), PDE4D8_ERK, cAMP)
    deg_D8_PKA = mm(getattr(p,'kcat_D8_PKA',27), getattr(p,'Km_D8_PKA',1.8), PDE4D8_PKA, cAMP)
    deg_D8_DP = mm(getattr(p,'kcat_D8_DP',10), getattr(p,'Km_D8_DP',1.8), PDE4D8_DP, cAMP)

    deg_D9 = mm(getattr(p,'kcat_D9',10), getattr(p,'Km_D9',1.8), PDE4D9, cAMP)
    deg_D9_ERK = mm(getattr(p,'kcat_D9_ERK',0.01), getattr(p,'Km_D9_ERK',1.8), PDE4D9_ERK, cAMP)
    deg_D9_PKA = mm(getattr(p,'kcat_D9_PKA',27), getattr(p,'Km_D9_PKA',1.8), PDE4D9_PKA, cAMP)
    deg_D9_DP = mm(getattr(p,'kcat_D9_DP',10), getattr(p,'Km_D9_DP',1.8), PDE4D9_DP, cAMP)

    deg_D1 = mm(getattr(p,'kcat_D1',10), getattr(p,'Km_D1',1.8), PDE4D1, cAMP)
    deg_D1_ERK = mm(getattr(p,'kcat_D1_ERK',0.01), getattr(p,'Km_D1_ERK',1.8), PDE4D1_ERK, cAMP)

    deg_D2 = mm(getattr(p,'kcat_D2',10), getattr(p,'Km_D2',1.8), PDE4D2, cAMP)
    deg_D2_ERK = mm(getattr(p,'kcat_D2_ERK',0.0085), getattr(p,'Km_D2_ERK',1.8), PDE4D2_ERK, cAMP)

    deg_D6 = mm(getattr(p,'kcat_D6',10), getattr(p,'Km_D6',1.8), PDE4D6, cAMP)
    deg_D6_ERK = mm(getattr(p,'kcat_D6_ERK',0.013), getattr(p,'Km_D6_ERK',1.8), PDE4D6_ERK, cAMP)

    deg_A = mm(getattr(p,'kcat_A',6.7), getattr(p,'Km_A',5.1), PDE4A, cAMP)
    deg_B = mm(getattr(p,'kcat_B',1.56), getattr(p,'Km_B',4.5), PDE4B, cAMP)

    total_degradation = (deg_D7 + deg_D7_ERK + deg_D7_PKA + deg_D7_DP +
                         deg_D4 + deg_D4_ERK + deg_D4_PKA + deg_D4_DP +
                         deg_D5 + deg_D5_ERK + deg_D5_PKA + deg_D5_DP +
                         deg_D3 + deg_D3_ERK + deg_D3_PKA + deg_D3_DP +
                         deg_D8 + deg_D8_ERK + deg_D8_PKA + deg_D8_DP +
                         deg_D9 + deg_D9_ERK + deg_D9_PKA + deg_D9_DP +
                         deg_D1 + deg_D1_ERK +
                         deg_D2 + deg_D2_ERK +
                         deg_D6 + deg_D6_ERK +
                         deg_A + deg_B)

    k_on_EPAC = getattr(p, 'k_on_EPAC', 0.031)
    K_m_EPAC = getattr(p, 'K_m_EPAC', 30.0)
    epac_cons = k_on_EPAC * (Epac * cAMP) / (K_m_EPAC + cAMP)

   
    k_on_C1 = getattr(p, 'k_on_C1', 0.0261)
    k_m_PKA = getattr(p, 'k_m_PKA', 5.2)
    pka_cons = 4 * k_on_C1 * (PKA_tet * (cAMP**1.6)) / ((k_m_PKA**1.6) + (cAMP**1.6))

    k_off_EPAC = getattr(p, 'k_off_EPAC', 0.00651)
    k_deg_C1 = getattr(p, 'k_deg_C1', 0.21)

    dcAMP = (AC1_prod
             - total_degradation
             - epac_cons
             - pka_cons
             + k_off_EPAC * Epac_ON
             + 4 * k_deg_C1 * C1) + 0.3   # constant offset

    # -----------------------------------------------------------------
    # 3.2 Additional ODEs
    # -----------------------------------------------------------------
    dAMP = 0.0
    dEpac = -k_on_EPAC * (Epac * cAMP) / (K_m_EPAC + cAMP) + k_off_EPAC * Epac_ON
    dEpac_ON = -dEpac
    # RAP
    k_on_RAP = getattr(p, 'k_on_RAP', 0.05)
    k_off_RAP = getattr(p, 'k_off_RAP', 1.666e-4)
    dRAP1_GDP = -k_on_RAP * Epac_ON * RAP1_GDP + k_off_RAP * RAP1_GTP
    dRAP1_GTP = -dRAP1_GDP
    # ERK
    k_on_ERK = getattr(p, 'k_on_ERK', 0.88)
    k_off_ERK = getattr(p, 'k_off_ERK', 0.088)
    dERK = k_off_ERK * pERK - k_on_ERK * RAP1_GTP * ERK
    dpERK = -k_off_ERK * pERK + k_on_ERK * RAP1_GTP * ERK - 2*getattr(p,'k_on_dimer',0.2)*(pERK**2) + 2*getattr(p,'k_off_dimer',0.0015)*ERK_dimer
    dERK_dimer = 2*getattr(p,'k_on_dimer',0.2)*(pERK**2) - 2*getattr(p,'k_off_dimer',0.0015)*ERK_dimer
    # C1 and PKA
    dC1 = k_on_C1 * (PKA_tet * (cAMP**1.6)) / ((k_m_PKA**1.6) + (cAMP**1.6)) - k_deg_C1 * C1
    k_deg_1 = getattr(p, 'k_deg_1', 0.0051)
    dPKA_cat1 = 2 * k_deg_C1 * C1 - k_deg_1 * PKA_cat1
    k_on_PKA = getattr(p, 'k_on_PKA', 10.0)
    k_off_PKA = getattr(p, 'k_off_PKA', 6e-4)
    dPKA_cat = (k_deg_1 * PKA_cat1 + 2*k_off_PKA*PKA_tet - 2*k_on_PKA*(PKA_cat**2)*(PKA_reg**2))
    dPKA_reg = (2*k_off_PKA*PKA_tet - 2*k_on_PKA*(PKA_cat**2)*(PKA_reg**2) + 2*k_deg_C1*C1)
    dPKA_tet = (-k_on_C1*(PKA_tet*(cAMP**1.6))/((k_m_PKA**1.6)+(cAMP**1.6)) + k_on_PKA*(PKA_cat**2)*(PKA_reg**2) - k_off_PKA*PKA_tet)
    # rolipram and bound forms (set to 0)
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
    dPDE4A_rol = 0.0; dPDE4B_rol = 0.0

    # PDE4 phosphorylation/dephosphorylation
    dPDE4D7 = 0.0; dPDE4D7_ERK = 0.0; dPDE4D7_PKA = 0.0; dPDE4D7_DP = 0.0
    dPDE4D4 = 0.0; dPDE4D4_ERK = 0.0; dPDE4D4_PKA = 0.0; dPDE4D4_DP = 0.0
    dPDE4D5 = 0.0; dPDE4D5_ERK = 0.0; dPDE4D5_PKA = 0.0; dPDE4D5_DP = 0.0
    dPDE4D3 = 0.0; dPDE4D3_ERK = 0.0; dPDE4D3_PKA = 0.0; dPDE4D3_DP = 0.0
    dPDE4D8 = 0.0; dPDE4D8_ERK = 0.0; dPDE4D8_PKA = 0.0; dPDE4D8_DP = 0.0
    dPDE4D9 = 0.0; dPDE4D9_ERK = 0.0; dPDE4D9_PKA = 0.0; dPDE4D9_DP = 0.0
    dPDE4D1 = 0.0; dPDE4D1_ERK = 0.0; dPDE4D1_PKA = 0.0; dPDE4D1_DP = 0.0
    dPDE4D2 = 0.0; dPDE4D2_ERK = 0.0; dPDE4D2_PKA = 0.0; dPDE4D2_DP = 0.0
    dPDE4D6 = 0.0; dPDE4D6_ERK = 0.0; dPDE4D6_PKA = 0.0; dPDE4D6_DP = 0.0
    dPDE4A = 0.0; dPDE4B = 0.0

    # CREB
    k_creb_PKA = getattr(p, 'k_creb_PKA', 8.0)
    k_creb_ERK = getattr(p, 'k_creb_ERK', 6.5)
    k_creb_d = getattr(p, 'k_creb_d', 5.0)
    dpCREB = k_creb_PKA * CREB * PKA_cat1 + k_creb_ERK * CREB * ERK_dimer - k_creb_d * pCREB
    dCREB = -dpCREB

    # -----------------------------------------------------------------
    # 3.3 Ca²⁺/CaM system
    # -----------------------------------------------------------------
    k_Ca2_CaM = getattr(p, 'k_Ca2_CaM', 100.0)
    k_inv_Ca2_CaM = getattr(p, 'k_inv_Ca2_CaM', 0.1)
    k_Ca4_CaM = getattr(p, 'k_Ca4_CaM', 100.0)
    k_inv_Ca4_CaM = getattr(p, 'k_inv_Ca4_CaM', 0.04)
    k_AC1_Ca4_CaM = getattr(p, 'k_AC1_Ca4_CaM', 0.5)
    k_inv_AC1_Ca4_CaM = getattr(p, 'k_inv_AC1_Ca4_CaM', 0.01)
    Km4 = getattr(p, 'Km4', 0.5)
    k_AC1_Ca2_CaM = getattr(p, 'k_AC1_Ca2_CaM', 0.01)
    k_inv_AC1_Ca2_CaM = getattr(p, 'k_inv_AC1_Ca2_CaM', 0.001)
    k_PDE1_Ca2_CaM = getattr(p, 'k_PDE1_Ca2_CaM', 0.005)
    k_inv_PDE1_Ca2_CaM = getattr(p, 'k_inv_PDE1_Ca2_CaM', 0.005)
    Km6 = getattr(p, 'Km6', 1.8)
    k_PDE1_Ca4_CaM = getattr(p, 'k_PDE1_Ca4_CaM', 10.0)
    k_inv_PDE1_Ca4_CaM = getattr(p, 'k_inv_PDE1_Ca4_CaM', 0.005)
    k_c_Ca2_to_Ca4_AC1 = getattr(p, 'k_c_Ca2_to_Ca4_AC1', 0.5)
    K_m_Ca2_to_Ca4_AC1 = getattr(p, 'K_m_Ca2_to_Ca4_AC1', 1.0)
    k_deg_Ca4_to_Ca2_AC1 = getattr(p, 'k_deg_Ca4_to_Ca2_AC1', 0.05)
    K_m_Ca4_to_Ca2_AC1 = getattr(p, 'K_m_Ca4_to_Ca2_AC1', 1.0)
    k_PDE1_Ca2_to_Ca4 = getattr(p, 'k_PDE1_Ca2_to_Ca4', 0.2)
    Km_PDE1_Ca2_to_Ca4 = getattr(p, 'Km_PDE1_Ca2_to_Ca4', 1.0)
    k_inv_PDE1_Ca4_to_Ca2 = getattr(p, 'k_inv_PDE1_Ca4_to_Ca2', 0.005)
    Km_PDE1_Ca4_to_Ca2 = getattr(p, 'Km_PDE1_Ca4_to_Ca2', 1.0)
    kb6 = getattr(p, 'kb6', 0.05)

    dCa = ( -2*k_Ca2_CaM*(Ca**2)*CaM + 2*k_inv_Ca2_CaM*Ca2_CaM
            -2*k_Ca4_CaM*(Ca**2)*Ca2_CaM + 2*k_inv_Ca4_CaM*Ca4_CaM
            + (-k_AC1_Ca4_CaM * AC1_Ca2_CaM * Ca)/(Km4 + Ca)
            + k_inv_AC1_Ca4_CaM * AC1_Ca4_CaM
            + (-k_PDE1_Ca2_CaM * PDE1_Ca2_CaM * (Ca**2))/(Km6 + Ca)
            + k_inv_PDE1_Ca2_CaM * PDE1_Ca4_CaM
            + (-k_AC1_Ca4_CaM * AC1_Ca2_CaM * Ca)/(Km4 + Ca)
            + k_inv_AC1_Ca4_CaM * AC1_Ca4_CaM )
    dCaM = -k_Ca2_CaM * CaM * Ca + k_inv_Ca2_CaM * Ca2_CaM
    dCa2_CaM = ( +k_Ca2_CaM*(Ca**2)*CaM - k_inv_Ca2_CaM*Ca2_CaM
                 -k_Ca4_CaM*Ca2_CaM*(Ca**2) + k_inv_Ca4_CaM*Ca4_CaM
                 -k_AC1_Ca4_CaM*AC1*Ca2_CaM*(Ca**2) + k_inv_AC1_Ca2_CaM*AC1_Ca2_CaM
                 -k_PDE1_Ca2_CaM*PDE1*Ca2_CaM + k_inv_PDE1_Ca2_CaM*PDE1_Ca2_CaM )
    dCa4_CaM = +k_Ca4_CaM*(Ca**2)*Ca2_CaM - k_inv_Ca4_CaM*Ca4_CaM
    dAC1 = -k_AC1_Ca2_CaM * AC1 * Ca2_CaM + k_inv_AC1_Ca2_CaM * AC1_Ca2_CaM
    dAC1_Ca2_CaM = ( +k_AC1_Ca2_CaM*AC1*Ca2_CaM - k_inv_AC1_Ca2_CaM*AC1_Ca2_CaM
                     + (-k_AC1_Ca4_CaM*AC1_Ca2_CaM*Ca)/(Km4+Ca)
                     + k_inv_AC1_Ca4_CaM*AC1_Ca4_CaM
                     - (k_c_Ca2_to_Ca4_AC1 * Ca2_CaM * AC1_Ca2_CaM)/(K_m_Ca2_to_Ca4_AC1 + AC1_Ca2_CaM)
                     + (k_deg_Ca4_to_Ca2_AC1 * AC1_Ca4_CaM)/(K_m_Ca4_to_Ca2_AC1 + AC1_Ca4_CaM) )
    dAC1_Ca4_CaM = ( (k_c_Ca2_to_Ca4_AC1 * Ca2_CaM * AC1_Ca2_CaM)/(K_m_Ca2_to_Ca4_AC1 + AC1_Ca2_CaM)
                     - (k_deg_Ca4_to_Ca2_AC1 * AC1_Ca4_CaM)/(K_m_Ca4_to_Ca2_AC1 + AC1_Ca4_CaM) )
    dPDE1 = -k_PDE1_Ca2_CaM * PDE1 * Ca2_CaM + k_inv_PDE1_Ca2_CaM * PDE1_Ca2_CaM
    dPDE1_Ca2_CaM = ( +k_PDE1_Ca2_CaM*PDE1*Ca2_CaM - k_inv_PDE1_Ca2_CaM*PDE1_Ca2_CaM
                      - (k_PDE1_Ca2_to_Ca4 * Ca2_CaM * PDE1_Ca2_CaM)/(Km_PDE1_Ca2_to_Ca4 + PDE1_Ca2_CaM)
                      + (k_inv_PDE1_Ca4_to_Ca2 * PDE1_Ca4_CaM)/(Km_PDE1_Ca4_to_Ca2 + PDE1_Ca4_CaM) )
    dPDE1_Ca4_CaM = (k_PDE1_Ca4_CaM * PDE1_Ca2_CaM * Ca)/(Km6 + Ca) - kb6 * PDE1_Ca4_CaM

    # -----------------------------------------------------------------
    # 3.4 Switches
    # -----------------------------------------------------------------
    cAMP_switch = hard_switch(cAMP, 2.0)
    PKA_switch = hard_switch(PKA_cat1, 0.1575)
    CaM_switch = soft_switch(Ca4_CaM, 0.5, 5)

    # -----------------------------------------------------------------
    # 3.5 Hodgkin‑Huxley currents
    # -----------------------------------------------------------------
    gNaF = getattr(p, 'gNaF', 1000.0); gNaP = getattr(p, 'gNaP', 1.0); gKS = getattr(p, 'gKS', 40.0); gL = getattr(p, 'gL', 11.3)
    DgHCN = getattr(p, 'DgHCN', 12.0); DgM = getattr(p, 'DgM', 50.0); gM = getattr(p, 'gM', 50.0); gHCN = getattr(p, 'gHCN', 23.0)
    ENa = getattr(p, 'ENa', 50.0); EK = getattr(p, 'EK', -84.0); ELK = getattr(p, 'ELK', -83.38); EHCN = getattr(p, 'EHCN', -50.0)
    Ps = getattr(p, 'Ps', 3.0**((37-20)/10)); Ph = getattr(p, 'Ph', 2.9**((37-20)/10)); Pr = getattr(p, 'Pr', 3.0**((37-35)/10))
    Cm = getattr(p, 'Cm', 0.9)

    Am = (1.86*(V+25.4))/(1-np.exp(-(V+25.4)/10.3))
    Bm = (-0.086*(V+29.7))/(1-np.exp((V+29.7)/9.16))
    mInf = Am/(Am+Bm)
    Ah = (-0.0336*(V+118))/(1-np.exp((V+118)/11))
    Bh = 2.3/(1+np.exp(-(V+35.8)/13.4))
    hInf = Ah/(Ah+Bh)
    hTau = 1/(Ph*(Ah+Bh))
    An = (0.186*(V+48.4))/(1-np.exp(-(V+48.4)/10.3))
    Bn = (-0.0086*(V+42.7))/(1-np.exp((V+42.7)/9.16))
    nInf = An/(An+Bn)
    As = (0.00122*(V+19.5))/(1-np.exp(-(V+19.5)/23.6))
    Bs = (-0.000739*(V+87.1))/(1-np.exp((V+87.1)/21.8))
    sInf = As/(As+Bs)
    sTau = 1/(Ps*(As+Bs))
    Ar = 0.007*np.exp(-(V+95*cAMP_switch+103.5*(1-cAMP_switch))/19)
    Br = 0.007*np.exp((V+95*cAMP_switch+103.5*(1-cAMP_switch))/22)
    rInf = Ar/(Ar+Br)
    rTau = 1/(Pr*(Ar+Br))
    wInf = 1/(1+np.exp(-(V+35)/10))
    wTau = 400/(3.3*np.exp((V+35)/20)+np.exp(-(V+35)/20))

    INaF = gNaF*(mInf**3)*h*(V-ENa)
    INaP = gNaP*(nInf**3)*(V-ENa)
    IKS = gKS*s*(V-EK)
    IL = gL*(V-ELK)
    IHCN = (gHCN + cAMP_switch*DgHCN)*r*(V-EHCN)
    IM = (gM + PKA_switch*DgM)*w*(V-EK)

    # -----------------------------------------------------------------
    # 3.6 L‑type Ca²⁺ channel
    # -----------------------------------------------------------------
    gCaL = getattr(p, 'gCaL', 20.0); ECa = getattr(p, 'ECa', 60.0)
    caL_pka_gain = getattr(p, 'caL_pka_gain', 2.0); Kd_PKA_CaL = getattr(p, 'Kd_PKA_CaL', 0.5)
    m_CaL_inf = 1/(1+np.exp(-(V+38)/4))
    h_CaL_inf = 1/(1+np.exp((V+63)/8))
    tau_mCaL = 5.0; tau_hCaL = 20.0
    phi_PKA = 1 + caL_pka_gain * (PKA_cat1/(PKA_cat1+Kd_PKA_CaL))
    ICaL = gCaL * phi_PKA * m_CaL * h_CaL * (V-ECa)

    # -----------------------------------------------------------------
    # 3.7 BK channel
    # -----------------------------------------------------------------
    gBK = getattr(p, 'gBK', 60.0); Kd_BK = getattr(p, 'Kd_BK', 5.0); n_BK = getattr(p, 'n_BK', 2.0)
    bk_pka_gain = getattr(p, 'bk_pka_gain', 3.0); Kd_PKA_BK = getattr(p, 'Kd_PKA_BK', 0.5)
    V_part_BK = 1/(1+np.exp(-(V+10)/10))
    Kd_eff = Kd_BK / (1 + bk_pka_gain * (PKA_cat1/(PKA_cat1+Kd_PKA_BK)))
    Ca_part_BK = (Ca**2)/(Ca**2 + Kd_eff**2)
    n_BK_inf = V_part_BK * Ca_part_BK
    tau_nBK = 1.5
    IBK = gBK * n_BK * (V-EK)

    # -----------------------------------------------------------------
    # 3.8 NMDA (simplified)
    # -----------------------------------------------------------------
    gNMDA_max = getattr(p, 'gNMDA_max', 0.29); ENMDA = getattr(p, 'ENMDA', 0.0)
    k_Mg = getattr(p, 'k', 0.28); b_Mg = getattr(p, 'b', 0.06)
    CaM_gain_NMDA = getattr(p, 'CaM_gain_NMDA', 1.5)
    kinetics = 1.0
    B_V = 1/(1 + k_Mg * np.exp(-b_Mg * V))
    INMDA = gNMDA_max * kinetics * B_V * (V - ENMDA)
    INMDA *= (1 + CaM_gain_NMDA * CaM_switch)

    # -----------------------------------------------------------------
    # 3.9 Calcium influx and RyR
    # -----------------------------------------------------------------
    gamma = 1.236e-6
    J_in = gamma * (ICaL + INMDA)
    # RyR
    v_rel = getattr(p, 'v_rel', 0.5); C_er_fixed = getattr(p, 'C_er_fixed', 100.0)
    K_a = getattr(p, 'K_a_RyR', 0.2); K_b = getattr(p, 'K_b_RyR', 0.3)
    K_c = getattr(p, 'K_c_RyR', 0.4); K_d_w = getattr(p, 'K_d_w_RyR', 10.0)
    v_leak = getattr(p, 'v_leak_RyR', 0.01); PKA_switch_RyR = getattr(p, 'PKA_switch_RyR', 0.5)
    Ca4 = Ca**4; Ca3 = Ca**3
    K_a_eff = K_a * (1 - PKA_switch_RyR * PKA_switch)
    term_act = K_a_eff / (Ca4 + 1e-12)
    w_inf = (term_act + 1 + Ca3/K_b) / (1/(K_c + term_act) + 1 + Ca3/K_b)
    tau_w = w_inf / K_d_w
    P_open = (w_RyR * (1 + Ca3/K_b)) / (term_act + 1 + Ca3/K_b)
    J_er = (v_rel * P_open + v_leak) * (C_er_fixed - Ca)
    dCa += J_in + J_er

    # -----------------------------------------------------------------
    # 3.10 New gating ODEs
    # -----------------------------------------------------------------
    dm_CaL = (m_CaL_inf - m_CaL) / tau_mCaL
    dh_CaL = (h_CaL_inf - h_CaL) / tau_hCaL
    dn_BK = (n_BK_inf - n_BK) / tau_nBK
    dw_RyR = (w_inf - w_RyR) / tau_w

    # -----------------------------------------------------------------
    # 3.11 Membrane potential and gating ODEs
    # -----------------------------------------------------------------
    dV = (1/Cm) * (-INaF - INaP - IKS - IL - IHCN - IM - ICaL - IBK - INMDA + getattr(p,'IApp',0))
    dh = (hInf - h) / hTau
    ds = (sInf - s) / sTau
    dr = (rInf - r) / rTau
    dw = (wInf - w) / wTau

    # -----------------------------------------------------------------
    # 3.12 Results list
    # -----------------------------------------------------------------
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

# ====================================================================
# 4. INITIAL CONDITIONS (106 variables)
# ====================================================================
vhsrw = [-65.0, 0.5, 0.5, 0.5, 0.5]
core = [
    1.0, 0.0, 0.488, 0.0, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0,
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
    0.214126394, 0.0, 0.0
]
ca_cam = [1.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0]
gates = [0.0, 1.0, 0.0, 0.0]
ics = vhsrw + core + ca_cam + gates
if len(ics) != 106:
    if len(ics) > 106:
        ics = ics[:106]
    else:
        ics += [0.0]*(106-len(ics))
initial_conditions = np.array(ics)
print(f"Initial conditions length: {len(initial_conditions)}")

# ====================================================================
# 5. PARAMETER FUNCTION WITH GEBR‑7b INHIBITION
# ====================================================================
def get_parameters(condition):
    p = set_parameters_cAMP()
    defaults = {
        'Tmp':37.0, 'Cm':0.9,
        'Ps':3**((37-20)/10), 'Ph':2.9**((37-20)/10), 'Pr':3**((37-35)/10),
        'gNaF':1000, 'gNaP':1, 'gKS':40, 'gL':11.3,
        'DgHCN':12, 'DgM':50, 'gM':50, 'gHCN':23,
        'ENa':50, 'EK':-84, 'ELK':-83.38, 'EHCN':-50,
        'gCaL':20, 'gBK':60, 'ECa':60,
        'Kd_BK':5, 'n_BK':2,
        'caL_pka_gain':2, 'Kd_PKA_CaL':0.5,
        'bk_pka_gain':3, 'Kd_PKA_BK':0.5,
        'gNMDA_max':0.29, 'ENMDA':0, 'k':0.28, 'b':0.06, 'CaM_gain_NMDA':1.5,
        'v_rel':0.5, 'C_er_fixed':100, 'K_a_RyR':0.2, 'K_b_RyR':0.3,
        'K_c_RyR':0.4, 'K_d_w_RyR':10, 'v_leak_RyR':0.01, 'PKA_switch_RyR':0.5,
        'kcat_AC1_Ca4':30, 'Km_AC1_Ca4':200,
        'k_Ca2_CaM':100, 'k_inv_Ca2_CaM':0.1,
        'k_Ca4_CaM':100, 'k_inv_Ca4_CaM':0.04,
        'k_AC1_Ca4_CaM':0.5, 'k_inv_AC1_Ca4_CaM':0.01,
        'Km4':0.5, 'k_AC1_Ca2_CaM':0.01, 'k_inv_AC1_Ca2_CaM':0.001,
        'k_PDE1_Ca2_CaM':0.005, 'k_inv_PDE1_Ca2_CaM':0.005,
        'Km6':1.8, 'k_PDE1_Ca4_CaM':10, 'k_inv_PDE1_Ca4_CaM':0.005,
        'k_c_Ca2_to_Ca4_AC1':0.5, 'K_m_Ca2_to_Ca4_AC1':1,
        'k_deg_Ca4_to_Ca2_AC1':0.05, 'K_m_Ca4_to_Ca2_AC1':1,
        'k_PDE1_Ca2_to_Ca4':0.2, 'Km_PDE1_Ca2_to_Ca4':1,
        'k_inv_PDE1_Ca4_to_Ca2':0.005, 'Km_PDE1_Ca4_to_Ca2':1,
        'kb6':0.05,
    }
    update_params_from_dict(p, defaults)

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
        for iso in ['D4','D5','D6','D7','D8','D9']:
            setattr(p, f'kcat_{iso}', getattr(p, f'kcat_{iso}',10) * 0.16)
            setattr(p, f'kcat_{iso}_ERK', getattr(p, f'kcat_{iso}_ERK',0.01) * 0.16)
            setattr(p, f'kcat_{iso}_PKA', getattr(p, f'kcat_{iso}_PKA',27) * 0.16)
            setattr(p, f'kcat_{iso}_DP', getattr(p, f'kcat_{iso}_DP',10) * 0.16)
    return p

# ====================================================================
# 6. RUN SIMULATION
# ====================================================================
ATOL, RTOL = 1e-8, 1e-6
t_span = [0, 400]
current_amplitude = 0.45
current_period = 20000
Iapp = 250

def run_sim(cond):
    p = get_parameters(cond)
    p.IApp = Iapp
    sol = solve_ivp(cAMP_model_ion, t_span, initial_conditions,
                    method='LSODA', args=(p, current_amplitude, current_period),
                    atol=ATOL, rtol=RTOL)
    return sol

print("Running Control...")
sol_c = run_sim('Control')
print("Running GEBR-7b...")
sol_g = run_sim('GEBR7b')

# ====================================================================
# 7. PLOT
# ====================================================================
fig, axs = plt.subplots(3,1,figsize=(10,8), sharex=True)
axs[0].plot(sol_c.t, sol_c.y[0], label='Control')
axs[0].plot(sol_g.t, sol_g.y[0], label='GEBR-7b')
axs[0].set_ylabel('V (mV)')
axs[0].legend()
axs[1].plot(sol_c.t, sol_c.y[5], label='Control')
axs[1].plot(sol_g.t, sol_g.y[5], label='GEBR-7b')
axs[1].set_ylabel('cAMP (μM)')
axs[1].legend()
axs[2].plot(sol_c.t, sol_c.y[17], label='Control')
axs[2].plot(sol_g.t, sol_g.y[17], label='GEBR-7b')
axs[2].set_ylabel('PKA_cat1 (μM)')
axs[2].legend()
axs[2].set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()

print(f"Final cAMP: Control = {sol_c.y[5,-1]:.4f}, GEBR-7b = {sol_g.y[5,-1]:.4f}")
print(f"Final PKA_cat1: Control = {sol_c.y[17,-1]:.4f}, GEBR-7b = {sol_g.y[17,-1]:.4f}")
