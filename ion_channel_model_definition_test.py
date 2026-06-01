from traceback import print_tb

import numpy as np
from scipy.signal import square
import pandas as pd
import matplotlib.pyplot as plt

class Params:
    pass

data_file_path = "initial_conditions and parameters copia.xlsx"
sheet_name_params = "parameters_literature"
column_name = "value"

# Read the specific sheet and column
df = pd.read_excel(data_file_path, sheet_name=sheet_name_params, engine="openpyxl")

# Convert to dictionary (assuming 'name' and 'value' columns)
def set_paramaters_cAMP():
    params = dict(zip(df['name'], df['value']))
    p = Params()
    for key, value in params.items():
        setattr(p, key, value)
    return p

def update_params_from_dict(p, param_dict):
    """
    Updates an existing Params object with values from a dictionary.

    Parameters:
    - p: existing Params object.
    - param_dict: dictionary with keys as attribute names and values to set.

    Returns:
    - The updated Params object.
    """
    for key, value in param_dict.items():
        setattr(p, key, value)
    return p
def vtrap(x, y):
    """
    Computes x / (1 - exp(-x/y)) with protection for x -> 0.
    Using L'Hopital's rule, the limit as x->0 is y.
    """
    if abs(x) < 1e-6:
        return y * (1 - x/y/2)  # Taylor expansion approximation
    else:
        return x / (1 - np.exp(-x / y))

def cAMP_model_ion(t, my_x, p, amplitude, period):
    # DYNAMICAL VARIABLES
    [V, h, s, r, w, cAMP, Epac, Epac_ON, RAP1_GDP, RAP1_GTP, ERK, pERK, ERK_dimer,
     C1, PKA_tet, PKA_reg, PKA_cat, PKA_cat1, PDE4D7, PDE4D7_ERK,
     PDE4D7_PKA, PDE4D7_DP , PDE4D4, PDE4D4_ERK, PDE4D4_PKA,
     PDE4D4_DP , PDE4D5, PDE4D5_ERK, PDE4D5_PKA, PDE4D5_DP ,
     PDE4D3, PDE4D3_ERK, PDE4D3_PKA, PDE4D3_DP , PDE4D8, PDE4D8_ERK,
     PDE4D8_PKA, PDE4D8_DP , PDE4D9, PDE4D9_ERK, PDE4D9_PKA,
     PDE4D9_DP , PDE4D1, PDE4D1_ERK, PDE4D1_PKA, PDE4D1_DP , PDE4D2,
     PDE4D2_ERK, PDE4D2_PKA, PDE4D2_DP , PDE4D6, PDE4D6_ERK,
     PDE4D6_PKA, PDE4D6_DP, PDE4D7_rolipram, PDE4D7_ERK_rolipram,
     PDE4D7_PKA_rolipram, PDE4D7_DP_rolipram, PDE4D4_rolipram, PDE4D4_ERK_rolipram,
     PDE4D4_PKA_rolipram, PDE4D4_DP_rolipram, PDE4D5_rolipram, PDE4D5_ERK_rolipram,
     PDE4D5_PKA_rolipram, PDE4D5_DP_rolipram, PDE4D3_rolipram, PDE4D3_ERK_rolipram,
     PDE4D3_PKA_rolipram, PDE4D3_DP_rolipram, PDE4D8_rolipram, PDE4D8_ERK_rolipram,
     PDE4D8_PKA_rolipram, PDE4D8_DP_rolipram, PDE4D9_rolipram, PDE4D9_ERK_rolipram,
     PDE4D9_PKA_rolipram, PDE4D9_DP_rolipram, PDE4D1_rolipram, PDE4D1_ERK_rolipram,
     PDE4D2_rolipram, PDE4D2_ERK_rolipram, PDE4D6_rolipram, PDE4D6_ERK_rolipram, CREB,
     pCREB, PDE4A, PDE4A_rolipram, PDE4B, PDE4B_rolipram, rolipram, Ca, CaM, Ca2_CaM, Ca4_CaM, AC1, AC1_Ca2_CaM,
     AC1_Ca4_CaM, PDE1, PDE1_Ca2_CaM, PDE1_Ca4_CaM] = my_x

    # PARAMETERS - Integrated cAMP signalling
    dcAMP = (p.kcat_AC1_Ca4 * AC1_Ca4_CaM * ATP) / (p.Km_AC1_Ca4 + ATP)
    - (p.kcat_PDE1_Ca4 * PDE1_Ca4_CaM * cAMP) / (p.Km_PDE1_Ca4 + cAMP)
    - (p.kcat_D7 * PDE4D7 * cAMP) / (p.Km_D7 + cAMP)
    - (p.kcat_D7_ERK * PDE4D7_ERK * cAMP) / (p.Km_D7_ERK + cAMP)
    - (p.kcat_D7_PKA * PDE4D7_PKA * cAMP) / (p.Km_D7_PKA + cAMP)
    - (p.kcat_D7_DP * PDE4D7_DP * cAMP) / (p.Km_D7_DP + cAMP)
    - (p.kcat_D4 * PDE4D4 * cAMP) / (p.Km_D4 + cAMP)
    - (p.kcat_D4_ERK * PDE4D4_ERK * cAMP) / (p.Km_D4_ERK + cAMP)
    - (p.kcat_D4_PKA * PDE4D4_PKA * cAMP) / (p.Km_D4_PKA + cAMP)
    - (p.kcat_D4_DP * PDE4D4_DP * cAMP) / (p.Km_D4_DP + cAMP)
    - (p.kcat_D5 * PDE4D5 * cAMP) / (p.Km_D5 + cAMP)
    - (p.kcat_D5_ERK * PDE4D5_ERK * cAMP) / (p.Km_D5_ERK + cAMP)
    - (p.kcat_D5_PKA * PDE4D5_PKA * cAMP) / (p.Km_D5_PKA + cAMP)
    - (p.kcat_D5_DP * PDE4D5_DP * cAMP) / (p.Km_D5_DP + cAMP)
    - (p.kcat_D3 * PDE4D3 * cAMP) / (p.Km_D3 + cAMP)
    - (p.kcat_D3_ERK * PDE4D3_ERK * cAMP) / (p.Km_D3_ERK + cAMP)
    - (p.kcat_D3_PKA * PDE4D3_PKA * cAMP) / (p.Km_D3_PKA + cAMP)
    - (p.kcat_D3_DP * PDE4D3_DP * cAMP) / (p.Km_D3_DP + cAMP)
    - (p.kcat_D8 * PDE4D8 * cAMP) / (p.Km_D8 + cAMP)
    - (p.kcat_D8_ERK * PDE4D8_ERK * cAMP) / (p.Km_D8_ERK + cAMP)
    - (p.kcat_D8_PKA * PDE4D8_PKA * cAMP) / (p.Km_D8_PKA + cAMP)
    - (p.kcat_D8_DP * PDE4D8_DP * cAMP) / (p.Km_D8_DP + cAMP)
    - (p.kcat_D9 * PDE4D9 * cAMP) / (p.Km_D9 + cAMP)
    - (p.kcat_D9_ERK * PDE4D9_ERK * cAMP) / (p.Km_D9_ERK + cAMP)
    - (p.kcat_D9_PKA * PDE4D9_PKA * cAMP) / (p.Km_D9_PKA + cAMP)
    - (p.kcat_D9_DP * PDE4D9_DP * cAMP) / (p.Km_D9_DP + cAMP)
    - (p.kcat_D1 * PDE4D1 * cAMP) / (p.Km_D1 + cAMP)
    - (p.kcat_D1_ERK * PDE4D1_ERK * cAMP) / (p.Km_D1_ERK + cAMP)
    - (p.kcat_D2 * PDE4D2 * cAMP) / (p.Km_D2 + cAMP)
    - (p.kcat_D2_ERK * PDE4D2_ERK * cAMP) / (p.Km_D2_ERK + cAMP)
    - (p.kcat_D6 * PDE4D6 * cAMP) / (p.Km_D6 + cAMP)
    - (p.kcat_D6_ERK * PDE4D6_ERK * cAMP) / (p.Km_D6_ERK + cAMP)
    - (p.kcat_A * PDE4A * cAMP) / (p.Km_A + cAMP)
    - (p.kcat_B * PDE4B * cAMP) / (p.Km_B + cAMP)
    - p.k_on_EPAC * ((Epac * cAMP) / (p.K_m_EPAC + cAMP))
    - 4 * p.k_on_C1 * ((PKA_tet * (cAMP**1.6)) / ((p.k_m_PKA**1.6) + (cAMP**1.6)))
    + p.k_off_EPAC * Epac_ON
    + 4 * p.k_deg_C1 * C1
    + 0.3

    dEpac = (- p.k_on_EPAC * ((Epac * cAMP) / (p.K_m_EPAC + cAMP))
             + p.k_off_EPAC * Epac_ON)

    dEpac_ON = (p.k_on_EPAC * ((Epac * cAMP) / (p.K_m_EPAC + cAMP))
                - p.k_off_EPAC * Epac_ON)  # - p.k_on_RAP * Epac_ON*RAP1_GDP + p.k_off_RAP * RAP1_GTP

    dRAP1_GDP = (- p.k_on_RAP * Epac_ON * RAP1_GDP
                 + p.k_off_RAP * RAP1_GTP)

    dRAP1_GTP = (- p.k_off_RAP * RAP1_GTP
                 + p.k_on_RAP * Epac_ON * RAP1_GDP)  # - p.k_on_ERK * RAP1_GTP * ERK + p.k_off_ERK * pERK

    dERK = p.k_off_ERK * pERK - p.k_on_ERK * RAP1_GTP * ERK

    dpERK = (- p.k_off_ERK * pERK
             + p.k_on_ERK * RAP1_GTP * ERK
             - 2 * p.k_on_dimer * np.float_power(pERK, 2)
             + 2 * p.k_off_dimer * ERK_dimer)

    dERK_dimer = (2 * p.k_on_dimer * np.float_power(pERK, 2)
                  - 2 * p.k_off_dimer * ERK_dimer)

    dC1 = p.k_on_C1 * ((PKA_tet * np.float_power(cAMP, 1.6)) / (np.float_power(p.k_m_PKA, 1.6)
                                                                + np.float_power(cAMP, 1.6))) - p.k_deg_C1 * C1

    dPKA_cat1 = 2 * p.k_deg_C1 * C1 - p.k_deg_1 * PKA_cat1

    dPKA_cat = (p.k_deg_1 * PKA_cat1
                + 2 * p.k_off_PKA * PKA_tet
                - 2 * p.k_on_PKA * np.float_power(PKA_cat, 2) * np.float_power(PKA_reg, 2))

    dPKA_reg = (2 * p.k_off_PKA * PKA_tet
                - 2 * p.k_on_PKA * np.float_power(PKA_cat, 2) * np.float_power(PKA_reg, 2)
                + 2 * p.k_deg_C1 * C1)

    dPKA_tet = (- p.k_on_C1 * (
                (PKA_tet * np.float_power(cAMP, 1.6)) / (np.float_power(p.k_m_PKA, 1.6) + np.float_power(cAMP, 1.6)))
                + p.k_on_PKA * np.float_power(PKA_cat, 2) * np.float_power(PKA_reg, 2)
                - p.k_off_PKA * PKA_tet)

    drolipram = (-p.k2 * rolipram
                 + p.rolipram_max_peak / (p.sigma * np.sqrt(2 * np.pi)) * np.exp(-(t - 40000) ** 2 / (2 * p.sigma ** 2))
                 - (p.Ki_D7_rolipram * rolipram * PDE4D7) / (p.Km_D7_rolipram + rolipram)
                 - (p.Ki_D7_ERK_rolipram * rolipram * PDE4D7_ERK) / (p.Km_D7_ERK_rolipram + rolipram)
                 - (p.Ki_D7_PKA_rolipram * rolipram * PDE4D7_PKA) / (p.Km_D7_PKA_rolipram + rolipram)
                 - (p.Ki_D7_DP_rolipram * rolipram * PDE4D7_DP) / (p.Km_D7_DP_rolipram + rolipram)
                 + p.k_D7_deg_rolipram * PDE4D7_rolipram
                 + p.k_D7_deg_rolipram_ERK * PDE4D7_ERK_rolipram
                 + p.k_D7_deg_rolipram_PKA * PDE4D7_PKA_rolipram
                 + p.k_D7_deg_rolipram_DP * PDE4D7_DP_rolipram
                 - (p.Ki_A_rolipram * rolipram * PDE4A) / (p.Km_A_rolipram + rolipram)
                 + p.k_A_deg_rolipram * PDE4A_rolipram
                 - (p.Ki_B_rolipram * rolipram * PDE4B) / (p.Km_B_rolipram + rolipram)
                 + p.k_B_deg_rolipram * PDE4B_rolipram)

    dPDE4D7 = (- (p.k_c_ERK_D7 * ERK_dimer * PDE4D7) / (p.K_m_ERK_D7 + PDE4D7)
               - (p.k_c_PKA_D7 * PKA_cat1 * PDE4D7) / (p.K_m_PKA_D7 + PDE4D7)
               + (p.k_deg_ERK_D7 * p.PP1 * PDE4D7_ERK) / (p.K_m_deg_ERK_D7 + PDE4D7_ERK)
               + (p.k_deg_PKA_D7 * p.PP1 * PDE4D7_PKA) / (p.K_m_deg_PKA_D7 + PDE4D7_PKA)
               + (p.k_deg_DP_D7 * p.PP1 * PDE4D7_DP) / (p.K_m_deg_DP_D7 + PDE4D7_DP)
               - (p.Ki_D7_rolipram * rolipram * PDE4D7) / (p.Km_D7_rolipram + rolipram)
               + p.k_D7_deg_rolipram * PDE4D7_rolipram)

    dPDE4D7_PKA = ((p.k_c_PKA_D7 * PKA_cat1 * PDE4D7) / (p.K_m_PKA_D7 + PDE4D7)
                   - (p.k_deg_PKA_D7 * p.PP1 * PDE4D7_PKA) / (p.K_m_deg_PKA_D7 + PDE4D7_PKA)
                   - (p.k_c_PKA_ERK_D7 * ERK_dimer * PDE4D7_PKA) / (p.K_m_PKA_ERK_D7 + PDE4D7_PKA)
                   - (p.Ki_D7_PKA_rolipram * rolipram * PDE4D7_PKA) / (p.Km_D7_PKA_rolipram + rolipram)
                   + p.k_D7_deg_rolipram_PKA * PDE4D7_PKA_rolipram)

    dPDE4D7_ERK = ((p.k_c_ERK_D7 * ERK_dimer * PDE4D7) / (p.K_m_ERK_D7 + PDE4D7)
                   - (p.k_deg_ERK_D7 * p.PP1 * PDE4D7_ERK) / (p.K_m_deg_ERK_D7 + PDE4D7_ERK)
                   - (p.k_c_ERK_PKA_D7 * PKA_cat1 * PDE4D7_ERK) / (p.K_m_ERK_PKA_D7 + PDE4D7_ERK)
                   - (p.Ki_D7_ERK_rolipram * rolipram * PDE4D7_ERK) / (p.Km_D7_ERK_rolipram + rolipram)
                   + p.k_D7_deg_rolipram_ERK * PDE4D7_ERK_rolipram)

    dPDE4D7_DP = (- (p.k_deg_DP_D7 * p.PP1 * PDE4D7_DP) / (p.K_m_deg_DP_D7 + PDE4D7_DP)
                  + (p.k_c_ERK_PKA_D7 * PKA_cat1 * PDE4D7_ERK) / (p.K_m_ERK_PKA_D7 + PDE4D7_ERK)
                  + (p.k_c_PKA_ERK_D7 * ERK_dimer * PDE4D7_PKA) / (p.K_m_PKA_ERK_D7 + PDE4D7_PKA)
                  - (p.Ki_D7_DP_rolipram * rolipram * PDE4D7_DP) / (p.Km_D7_DP_rolipram + rolipram)
                  + p.k_D7_deg_rolipram_DP * PDE4D7_DP_rolipram)

    dPDE4D7_rolipram = ((p.Ki_D7_rolipram * rolipram * PDE4D7) / (p.Km_D7_rolipram + rolipram)
                        - p.k_D7_deg_rolipram * PDE4D7_rolipram)

    dPDE4D7_ERK_rolipram = ((p.Ki_D7_ERK_rolipram * rolipram * PDE4D7_ERK) / (p.Km_D7_ERK_rolipram + rolipram)
                            - p.k_D7_deg_rolipram_ERK * PDE4D7_ERK_rolipram)

    dPDE4D7_PKA_rolipram = ((p.Ki_D7_PKA_rolipram * rolipram * PDE4D7_PKA) / (p.Km_D7_PKA_rolipram + rolipram)
                            - p.k_D7_deg_rolipram_PKA * PDE4D7_PKA_rolipram)

    dPDE4D7_DP_rolipram = ((p.Ki_D7_DP_rolipram * rolipram * PDE4D7_DP) / (p.Km_D7_DP_rolipram + rolipram)
                           - p.k_D7_deg_rolipram_DP * PDE4D7_DP_rolipram)

    dPDE4D4 = (- (p.k_c_ERK_D4 * ERK_dimer * PDE4D4) / (p.K_m_ERK_D4 + PDE4D4)
               - (p.k_c_PKA_D4 * PKA_cat1 * PDE4D4) / (p.K_m_PKA_D4 + PDE4D4)
               + (p.k_deg_ERK_D4 * p.PP1 * PDE4D4_ERK) / (p.K_m_deg_ERK_D4 + PDE4D4_ERK)
               + (p.k_deg_PKA_D4 * p.PP1 * PDE4D4_PKA) / (p.K_m_deg_PKA_D4 + PDE4D4_PKA)
               + (p.k_deg_DP_D4 * p.PP1 * PDE4D4_DP) / (p.K_m_deg_DP_D4 + PDE4D4_DP)
               - (p.Ki_D4_rolipram * rolipram * PDE4D4) / (p.Km_D4_rolipram + rolipram)
               + p.k_D4_deg_rolipram * PDE4D4_rolipram)

    dPDE4D4_PKA = ((p.k_c_PKA_D4 * PKA_cat1 * PDE4D4) / (p.K_m_PKA_D4 + PDE4D4)
                   - (p.k_deg_PKA_D4 * p.PP1 * PDE4D4_PKA) / (p.K_m_deg_PKA_D4 + PDE4D4_PKA)
                   - (p.k_c_PKA_ERK_D4 * ERK_dimer * PDE4D4_PKA) / (p.K_m_PKA_ERK_D4 + PDE4D4_PKA)
                   - (p.Ki_D4_PKA_rolipram * rolipram * PDE4D4_PKA) / (p.Km_D4_PKA_rolipram + rolipram)
                   + p.k_D4_deg_rolipram_PKA * PDE4D4_PKA_rolipram)

    dPDE4D4_ERK = ((p.k_c_ERK_D4 * ERK_dimer * PDE4D4) / (p.K_m_ERK_D4 + PDE4D4)
                   - (p.k_deg_ERK_D4 * p.PP1 * PDE4D4_ERK) / (p.K_m_deg_ERK_D4 + PDE4D4_ERK)
                   - (p.k_c_ERK_PKA_D4 * PKA_cat1 * PDE4D4_ERK) / (p.K_m_ERK_PKA_D4 + PDE4D4_ERK)
                   - (p.Ki_D4_ERK_rolipram * rolipram * PDE4D4_ERK) / (p.Km_D4_ERK_rolipram + rolipram)
                   + p.k_D4_deg_rolipram_ERK * PDE4D4_ERK_rolipram)

    dPDE4D4_DP = (- (p.k_deg_DP_D4 * p.PP1 * PDE4D4_DP) / (p.K_m_deg_DP_D4 + PDE4D4_DP)
                  + (p.k_c_ERK_PKA_D4 * PKA_cat1 * PDE4D4_ERK) / (p.K_m_ERK_PKA_D4 + PDE4D4_ERK)
                  + (p.k_c_PKA_ERK_D4 * ERK_dimer * PDE4D4_PKA) / (p.K_m_PKA_ERK_D4 + PDE4D4_PKA)
                  - (p.Ki_D4_DP_rolipram * rolipram * PDE4D4_DP) / (p.Km_D4_DP_rolipram + rolipram)
                  + p.k_D4_deg_rolipram_DP * PDE4D4_DP_rolipram)

    dPDE4D4_rolipram = ((p.Ki_D4_rolipram * rolipram * PDE4D4) / (p.Km_D4_rolipram + rolipram)
                        - p.k_D4_deg_rolipram * PDE4D4_rolipram)

    dPDE4D4_ERK_rolipram = ((p.Ki_D4_ERK_rolipram * rolipram * PDE4D4_ERK) / (p.Km_D4_ERK_rolipram + rolipram)
                            - p.k_D4_deg_rolipram_ERK * PDE4D4_ERK_rolipram)

    dPDE4D4_PKA_rolipram = ((p.Ki_D4_PKA_rolipram * rolipram * PDE4D4_PKA) / (p.Km_D4_PKA_rolipram + rolipram)
                            - p.k_D4_deg_rolipram_PKA * PDE4D4_PKA_rolipram)

    dPDE4D4_DP_rolipram = ((p.Ki_D4_DP_rolipram * rolipram * PDE4D4_DP) / (p.Km_D4_DP_rolipram + rolipram)
                           - p.k_D4_deg_rolipram_DP * PDE4D4_DP_rolipram)

    dPDE4D5 = (- (p.k_c_ERK_D5 * ERK_dimer * PDE4D5) / (p.K_m_ERK_D5 + PDE4D5)
               - (p.k_c_PKA_D5 * PKA_cat1 * PDE4D5) / (p.K_m_PKA_D5 + PDE4D5)
               + (p.k_deg_ERK_D5 * p.PP1 * PDE4D5_ERK) / (p.K_m_deg_ERK_D5 + PDE4D5_ERK)
               + (p.k_deg_PKA_D5 * p.PP1 * PDE4D5_PKA) / (p.K_m_deg_PKA_D5 + PDE4D5_PKA)
               + (p.k_deg_DP_D5 * p.PP1 * PDE4D5_DP) / (p.K_m_deg_DP_D5 + PDE4D5_DP)
               - (p.Ki_D5_rolipram * rolipram * PDE4D5) / (p.Km_D5_rolipram + rolipram)
               + p.k_D5_deg_rolipram * PDE4D5_rolipram)

    dPDE4D5_PKA = ((p.k_c_PKA_D5 * PKA_cat1 * PDE4D5) / (p.K_m_PKA_D5 + PDE4D5)
                   - (p.k_deg_PKA_D5 * p.PP1 * PDE4D5_PKA) / (p.K_m_deg_PKA_D5 + PDE4D5_PKA)
                   - (p.k_c_PKA_ERK_D5 * ERK_dimer * PDE4D5_PKA) / (p.K_m_PKA_ERK_D5 + PDE4D5_PKA)
                   - (p.Ki_D5_PKA_rolipram * rolipram * PDE4D5_PKA) / (p.Km_D5_PKA_rolipram + rolipram)
                   + p.k_D5_deg_rolipram_PKA * PDE4D5_PKA_rolipram)

    dPDE4D5_ERK = ((p.k_c_ERK_D5 * ERK_dimer * PDE4D5) / (p.K_m_ERK_D5 + PDE4D5)
                   - (p.k_deg_ERK_D5 * p.PP1 * PDE4D5_ERK) / (p.K_m_deg_ERK_D5 + PDE4D5_ERK)
                   - (p.k_c_ERK_PKA_D5 * PKA_cat1 * PDE4D5_ERK) / (p.K_m_ERK_PKA_D5 + PDE4D5_ERK)
                   - (p.Ki_D5_ERK_rolipram * rolipram * PDE4D5_ERK) / (p.Km_D5_ERK_rolipram + rolipram)
                   + p.k_D5_deg_rolipram_ERK * PDE4D5_ERK_rolipram)

    dPDE4D5_DP = (- (p.k_deg_DP_D5 * p.PP1 * PDE4D5_DP) / (p.K_m_deg_DP_D5 + PDE4D5_DP)
                  + (p.k_c_ERK_PKA_D5 * PKA_cat1 * PDE4D5_ERK) / (p.K_m_ERK_PKA_D5 + PDE4D5_ERK)
                  + (p.k_c_PKA_ERK_D5 * ERK_dimer * PDE4D5_PKA) / (p.K_m_PKA_ERK_D5 + PDE4D5_PKA)
                  - (p.Ki_D5_DP_rolipram * rolipram * PDE4D5_DP) / (p.Km_D5_DP_rolipram + rolipram)
                  + p.k_D5_deg_rolipram_DP * PDE4D5_DP_rolipram)

    dPDE4D5_rolipram = ((p.Ki_D5_rolipram * rolipram * PDE4D5) / (p.Km_D5_rolipram + rolipram)
                        - p.k_D5_deg_rolipram * PDE4D5_rolipram)

    dPDE4D5_ERK_rolipram = ((p.Ki_D5_ERK_rolipram * rolipram * PDE4D5_ERK) / (p.Km_D5_ERK_rolipram + rolipram)
                            - p.k_D5_deg_rolipram_ERK * PDE4D5_ERK_rolipram)

    dPDE4D5_PKA_rolipram = ((p.Ki_D5_PKA_rolipram * rolipram * PDE4D5_PKA) / (p.Km_D5_PKA_rolipram + rolipram)
                            - p.k_D5_deg_rolipram_PKA * PDE4D5_PKA_rolipram)

    dPDE4D5_DP_rolipram = ((p.Ki_D5_DP_rolipram * rolipram * PDE4D5_DP) / (p.Km_D5_DP_rolipram + rolipram)
                           - p.k_D5_deg_rolipram_DP * PDE4D5_DP_rolipram)

    dPDE4D3 = (- (p.k_c_ERK_D3 * ERK_dimer * PDE4D3) / (p.K_m_ERK_D3 + PDE4D3)
               - (p.k_c_PKA_D3 * PKA_cat1 * PDE4D3) / (p.K_m_PKA_D3 + PDE4D3)
               + (p.k_deg_ERK_D3 * p.PP1 * PDE4D3_ERK) / (p.K_m_deg_ERK_D3 + PDE4D3_ERK)
               + (p.k_deg_PKA_D3 * p.PP1 * PDE4D3_PKA) / (p.K_m_deg_PKA_D3 + PDE4D3_PKA)
               + (p.k_deg_DP_D3 * p.PP1 * PDE4D3_DP) / (p.K_m_deg_DP_D3 + PDE4D3_DP)
               - (p.Ki_D3_rolipram * rolipram * PDE4D3) / (p.Km_D3_rolipram + rolipram)
               + p.k_D3_deg_rolipram * PDE4D3_rolipram)

    dPDE4D3_PKA = ((p.k_c_PKA_D3 * PKA_cat1 * PDE4D3) / (p.K_m_PKA_D3 + PDE4D3)
                   - (p.k_deg_PKA_D3 * p.PP1 * PDE4D3_PKA) / (p.K_m_deg_PKA_D3 + PDE4D3_PKA)
                   - (p.k_c_PKA_ERK_D3 * ERK_dimer * PDE4D3_PKA) / (p.K_m_PKA_ERK_D3 + PDE4D3_PKA)
                   - (p.Ki_D3_PKA_rolipram * rolipram * PDE4D3_PKA) / (p.Km_D3_PKA_rolipram + rolipram)
                   + p.k_D3_deg_rolipram_PKA * PDE4D3_PKA_rolipram)

    dPDE4D3_ERK = ((p.k_c_ERK_D3 * ERK_dimer * PDE4D3) / (p.K_m_ERK_D3 + PDE4D3)
                   - (p.k_deg_ERK_D3 * p.PP1 * PDE4D3_ERK) / (p.K_m_deg_ERK_D3 + PDE4D3_ERK)
                   - (p.k_c_ERK_PKA_D3 * PKA_cat1 * PDE4D3_ERK) / (p.K_m_ERK_PKA_D3 + PDE4D3_ERK)
                   - (p.Ki_D3_ERK_rolipram * rolipram * PDE4D3_ERK) / (p.Km_D3_ERK_rolipram + rolipram)
                   + p.k_D3_deg_rolipram_ERK * PDE4D3_ERK_rolipram)

    dPDE4D3_DP = (- (p.k_deg_DP_D3 * p.PP1 * PDE4D3_DP) / (p.K_m_deg_DP_D3 + PDE4D3_DP)
                  + (p.k_c_ERK_PKA_D3 * PKA_cat1 * PDE4D3_ERK) / (p.K_m_ERK_PKA_D3 + PDE4D3_ERK)
                  + (p.k_c_PKA_ERK_D3 * ERK_dimer * PDE4D3_PKA) / (p.K_m_PKA_ERK_D3 + PDE4D3_PKA)
                  - (p.Ki_D3_DP_rolipram * rolipram * PDE4D3_DP) / (p.Km_D3_DP_rolipram + rolipram)
                  + p.k_D3_deg_rolipram_DP * PDE4D3_DP_rolipram)

    dPDE4D3_rolipram = ((p.Ki_D3_rolipram * rolipram * PDE4D3) / (p.Km_D3_rolipram + rolipram)
                        - p.k_D3_deg_rolipram * PDE4D3_rolipram)

    dPDE4D3_ERK_rolipram = ((p.Ki_D3_ERK_rolipram * rolipram * PDE4D3_ERK) / (p.Km_D3_ERK_rolipram + rolipram)
                            - p.k_D3_deg_rolipram_ERK * PDE4D3_ERK_rolipram)

    dPDE4D3_PKA_rolipram = ((p.Ki_D3_PKA_rolipram * rolipram * PDE4D3_PKA) / (p.Km_D3_PKA_rolipram + rolipram)
                            - p.k_D3_deg_rolipram_PKA * PDE4D3_PKA_rolipram)

    dPDE4D3_DP_rolipram = ((p.Ki_D3_DP_rolipram * rolipram * PDE4D3_DP) / (p.Km_D3_DP_rolipram + rolipram)
                           - p.k_D3_deg_rolipram_DP * PDE4D3_DP_rolipram)

    dPDE4D8 = (- (p.k_c_ERK_D8 * ERK_dimer * PDE4D8) / (p.K_m_ERK_D8 + PDE4D8)
               - (p.k_c_PKA_D8 * PKA_cat1 * PDE4D8) / (p.K_m_PKA_D8 + PDE4D8)
               + (p.k_deg_ERK_D8 * p.PP1 * PDE4D8_ERK) / (p.K_m_deg_ERK_D8 + PDE4D8_ERK)
               + (p.k_deg_PKA_D8 * p.PP1 * PDE4D8_PKA) / (p.K_m_deg_PKA_D8 + PDE4D8_PKA)
               + (p.k_deg_DP_D8 * p.PP1 * PDE4D8_DP) / (p.K_m_deg_DP_D8 + PDE4D8_DP)
               - (p.Ki_D8_rolipram * rolipram * PDE4D8) / (p.Km_D8_rolipram + rolipram)
               + p.k_D8_deg_rolipram * PDE4D8_rolipram)

    dPDE4D8_PKA = ((p.k_c_PKA_D8 * PKA_cat1 * PDE4D8) / (p.K_m_PKA_D8 + PDE4D8)
                   - (p.k_deg_PKA_D8 * p.PP1 * PDE4D8_PKA) / (p.K_m_deg_PKA_D8 + PDE4D8_PKA)
                   - (p.k_c_PKA_ERK_D8 * ERK_dimer * PDE4D8_PKA) / (p.K_m_PKA_ERK_D8 + PDE4D8_PKA)
                   - (p.Ki_D8_PKA_rolipram * rolipram * PDE4D8_PKA) / (p.Km_D8_PKA_rolipram + rolipram)
                   + p.k_D8_deg_rolipram_PKA * PDE4D8_PKA_rolipram)

    dPDE4D8_ERK = ((p.k_c_ERK_D8 * ERK_dimer * PDE4D8) / (p.K_m_ERK_D8 + PDE4D8)
                   - (p.k_deg_ERK_D8 * p.PP1 * PDE4D8_ERK) / (p.K_m_deg_ERK_D8 + PDE4D8_ERK)
                   - (p.k_c_ERK_PKA_D8 * PKA_cat1 * PDE4D8_ERK) / (p.K_m_ERK_PKA_D8 + PDE4D8_ERK)
                   - (p.Ki_D8_ERK_rolipram * rolipram * PDE4D8_ERK) / (p.Km_D8_ERK_rolipram + rolipram)
                   + p.k_D8_deg_rolipram_ERK * PDE4D8_ERK_rolipram)

    dPDE4D8_DP = (- (p.k_deg_DP_D8 * p.PP1 * PDE4D8_DP) / (p.K_m_deg_DP_D8 + PDE4D8_DP)
                  + (p.k_c_ERK_PKA_D8 * PKA_cat1 * PDE4D8_ERK) / (p.K_m_ERK_PKA_D8 + PDE4D8_ERK)
                  + (p.k_c_PKA_ERK_D8 * ERK_dimer * PDE4D8_PKA) / (p.K_m_PKA_ERK_D8 + PDE4D8_PKA)
                  - (p.Ki_D8_DP_rolipram * rolipram * PDE4D8_DP) / (p.Km_D8_DP_rolipram + rolipram)
                  + p.k_D8_deg_rolipram_DP * PDE4D8_DP_rolipram)

    dPDE4D8_rolipram = ((p.Ki_D8_rolipram * rolipram * PDE4D8) / (p.Km_D8_rolipram + rolipram)
                        - p.k_D8_deg_rolipram * PDE4D8_rolipram)

    dPDE4D8_ERK_rolipram = ((p.Ki_D8_ERK_rolipram * rolipram * PDE4D8_ERK) / (p.Km_D8_ERK_rolipram + rolipram)
                            - p.k_D8_deg_rolipram_ERK * PDE4D8_ERK_rolipram)

    dPDE4D8_PKA_rolipram = ((p.Ki_D8_PKA_rolipram * rolipram * PDE4D8_PKA) / (p.Km_D8_PKA_rolipram + rolipram)
                            - p.k_D8_deg_rolipram_PKA * PDE4D8_PKA_rolipram)

    dPDE4D8_DP_rolipram = ((p.Ki_D8_DP_rolipram * rolipram * PDE4D8_DP) / (p.Km_D8_DP_rolipram + rolipram)
                           - p.k_D8_deg_rolipram_DP * PDE4D8_DP_rolipram)

    dPDE4D9 = (- (p.k_c_ERK_D9 * ERK_dimer * PDE4D9) / (p.K_m_ERK_D9 + PDE4D9)
               - (p.k_c_PKA_D9 * PKA_cat1 * PDE4D9) / (p.K_m_PKA_D9 + PDE4D9)
               + (p.k_deg_ERK_D9 * p.PP1 * PDE4D9_ERK) / (p.K_m_deg_ERK_D9 + PDE4D9_ERK)
               + (p.k_deg_PKA_D9 * p.PP1 * PDE4D9_PKA) / (p.K_m_deg_PKA_D9 + PDE4D9_PKA)
               + (p.k_deg_DP_D9 * p.PP1 * PDE4D9_DP) / (p.K_m_deg_DP_D9 + PDE4D9_DP)
               - (p.Ki_D9_rolipram * rolipram * PDE4D9) / (p.Km_D9_rolipram + rolipram)
               + p.k_D9_deg_rolipram * PDE4D9_rolipram)

    dPDE4D9_PKA = ((p.k_c_PKA_D9 * PKA_cat1 * PDE4D9) / (p.K_m_PKA_D9 + PDE4D9)
                   - (p.k_deg_PKA_D9 * p.PP1 * PDE4D9_PKA) / (p.K_m_deg_PKA_D9 + PDE4D9_PKA)
                   - (p.k_c_PKA_ERK_D9 * ERK_dimer * PDE4D9_PKA) / (p.K_m_PKA_ERK_D9 + PDE4D9_PKA)
                   - (p.Ki_D9_PKA_rolipram * rolipram * PDE4D9_PKA) / (p.Km_D9_PKA_rolipram + rolipram)
                   + p.k_D9_deg_rolipram_PKA * PDE4D9_PKA_rolipram)

    dPDE4D9_ERK = ((p.k_c_ERK_D9 * ERK_dimer * PDE4D9) / (p.K_m_ERK_D9 + PDE4D9)
                   - (p.k_deg_ERK_D9 * p.PP1 * PDE4D9_ERK) / (p.K_m_deg_ERK_D9 + PDE4D9_ERK)
                   - (p.k_c_ERK_PKA_D9 * PKA_cat1 * PDE4D9_ERK) / (p.K_m_ERK_PKA_D9 + PDE4D9_ERK)
                   - (p.Ki_D9_ERK_rolipram * rolipram * PDE4D9_ERK) / (p.Km_D9_ERK_rolipram + rolipram)
                   + p.k_D9_deg_rolipram_ERK * PDE4D9_ERK_rolipram)

    dPDE4D9_DP = (- (p.k_deg_DP_D9 * p.PP1 * PDE4D9_DP) / (p.K_m_deg_DP_D9 + PDE4D9_DP)
                  + (p.k_c_ERK_PKA_D9 * PKA_cat1 * PDE4D9_ERK) / (p.K_m_ERK_PKA_D9 + PDE4D9_ERK)
                  + (p.k_c_PKA_ERK_D9 * ERK_dimer * PDE4D9_PKA) / (p.K_m_PKA_ERK_D9 + PDE4D9_PKA)
                  - (p.Ki_D9_DP_rolipram * rolipram * PDE4D9_DP) / (p.Km_D9_DP_rolipram + rolipram)
                  + p.k_D9_deg_rolipram_DP * PDE4D9_DP_rolipram)

    dPDE4D9_rolipram = ((p.Ki_D9_rolipram * rolipram * PDE4D9) / (p.Km_D9_rolipram + rolipram)
                        - p.k_D9_deg_rolipram * PDE4D9_rolipram)

    dPDE4D9_ERK_rolipram = ((p.Ki_D9_ERK_rolipram * rolipram * PDE4D9_ERK) / (p.Km_D9_ERK_rolipram + rolipram)
                            - p.k_D9_deg_rolipram_ERK * PDE4D9_ERK_rolipram)

    dPDE4D9_PKA_rolipram = ((p.Ki_D9_PKA_rolipram * rolipram * PDE4D9_PKA) / (p.Km_D9_PKA_rolipram + rolipram)
                            - p.k_D9_deg_rolipram_PKA * PDE4D9_PKA_rolipram)

    dPDE4D9_DP_rolipram = ((p.Ki_D9_DP_rolipram * rolipram * PDE4D9_DP) / (p.Km_D9_DP_rolipram + rolipram)
                           - p.k_D9_deg_rolipram_DP * PDE4D9_DP_rolipram)

    dPDE4D1 = (- (p.k_c_ERK_D1 * ERK_dimer * PDE4D1) / (p.K_m_ERK_D1 + PDE4D1)
               + (p.k_deg_ERK_D1 * p.PP1 * PDE4D1_ERK) / (p.K_m_deg_ERK_D1 + PDE4D1_ERK)
               - (p.Ki_D1_rolipram * rolipram * PDE4D1) / (p.Km_D1_rolipram + rolipram)
               + p.k_D1_deg_rolipram * PDE4D1_rolipram)

    dPDE4D1_PKA = 0  # no phosphorylation possible

    dPDE4D1_ERK = ((p.k_c_ERK_D1 * ERK_dimer * PDE4D1) / (p.K_m_ERK_D1 + PDE4D1)
                   - (p.k_deg_ERK_D1 * p.PP1 * PDE4D1_ERK) / (p.K_m_deg_ERK_D1 + PDE4D1_ERK)
                   - (p.Ki_D1_ERK_rolipram * rolipram * PDE4D1_ERK) / (p.Km_D1_ERK_rolipram + rolipram)
                   + p.k_D1_deg_rolipram_ERK * PDE4D1_ERK_rolipram)

    dPDE4D1_DP = 0  # no PKA phosphorylation thus no double phosphorylation

    dPDE4D1_rolipram = ((p.Ki_D1_rolipram * rolipram * PDE4D1) / (p.Km_D1_rolipram + rolipram)
                        - p.k_D1_deg_rolipram * PDE4D1_rolipram)

    dPDE4D1_ERK_rolipram = ((p.Ki_D1_ERK_rolipram * rolipram * PDE4D1_ERK) / (p.Km_D1_ERK_rolipram + rolipram)
                            - p.k_D1_deg_rolipram_ERK * PDE4D1_ERK_rolipram)

    dPDE4D2 = (- (p.k_c_ERK_D2 * ERK_dimer * PDE4D2) / (p.K_m_ERK_D2 + PDE4D2)
               + (p.k_deg_ERK_D2 * p.PP1 * PDE4D2_ERK) / (p.K_m_deg_ERK_D2 + PDE4D2_ERK)
               - (p.Ki_D2_rolipram * rolipram * PDE4D2) / (p.Km_D2_rolipram + rolipram)
               + p.k_D2_deg_rolipram * PDE4D2_rolipram)

    dPDE4D2_PKA = 0  # no phosphorylation possible

    dPDE4D2_ERK = ((p.k_c_ERK_D2 * ERK_dimer * PDE4D2) / (p.K_m_ERK_D2 + PDE4D2)
                   - (p.k_deg_ERK_D2 * p.PP1 * PDE4D2_ERK) / (p.K_m_deg_ERK_D2 + PDE4D2_ERK)
                   - (p.Ki_D2_ERK_rolipram * rolipram * PDE4D2_ERK) / (p.Km_D2_ERK_rolipram + rolipram)
                   + p.k_D2_deg_rolipram_ERK * PDE4D2_ERK_rolipram)

    dPDE4D2_DP = 0  # no PKA phosphorylation thus no double phosphorylation

    dPDE4D2_rolipram = ((p.Ki_D2_rolipram * rolipram * PDE4D2) / (p.Km_D2_rolipram + rolipram)
                        - p.k_D2_deg_rolipram * PDE4D2_rolipram)

    dPDE4D2_ERK_rolipram = ((p.Ki_D2_ERK_rolipram * rolipram * PDE4D2_ERK) / (p.Km_D2_ERK_rolipram + rolipram)
                            - p.k_D2_deg_rolipram_ERK * PDE4D2_ERK_rolipram)

    dPDE4D6 = (- (p.k_c_ERK_D6 * ERK_dimer * PDE4D6) / (p.K_m_ERK_D6 + PDE4D6)
               + (p.k_deg_ERK_D6 * p.PP1 * PDE4D6_ERK) / (p.K_m_deg_ERK_D6 + PDE4D6_ERK)
               - (p.Ki_D6_rolipram * rolipram * PDE4D6) / (p.Km_D6_rolipram + rolipram)
               + p.k_D6_deg_rolipram * PDE4D6_rolipram)

    dPDE4D6_PKA = 0  # no phosphorylation possible

    dPDE4D6_ERK = ((p.k_c_ERK_D6 * ERK_dimer * PDE4D6) / (p.K_m_ERK_D6 + PDE4D6)
                   - (p.k_deg_ERK_D6 * p.PP1 * PDE4D6_ERK) / (p.K_m_deg_ERK_D6 + PDE4D6_ERK)
                   - (p.Ki_D6_ERK_rolipram * rolipram * PDE4D6_ERK) / (p.Km_D6_ERK_rolipram + rolipram)
                   + p.k_D6_deg_rolipram_ERK * PDE4D6_ERK_rolipram)

    dPDE4D6_DP = 0  # no PKA phosphorylation thus no double phosphorylation

    dPDE4D6_rolipram = ((p.Ki_D6_rolipram * rolipram * PDE4D6) / (p.Km_D6_rolipram + rolipram)
                        - p.k_D6_deg_rolipram * PDE4D6_rolipram)

    dPDE4D6_ERK_rolipram = ((p.Ki_D6_ERK_rolipram * rolipram * PDE4D6_ERK) / (p.Km_D6_ERK_rolipram + rolipram)
                            - p.k_D6_deg_rolipram_ERK * PDE4D6_ERK_rolipram)

    dPDE4A = (-(p.Ki_A_rolipram * rolipram * PDE4A) / (p.Km_A_rolipram + rolipram)
              + p.k_A_deg_rolipram * PDE4A_rolipram)
    dPDE4A_rolipram = ((p.Ki_A_rolipram * rolipram * PDE4A) / (p.Km_A_rolipram + rolipram)
                       - p.k_A_deg_rolipram * PDE4A_rolipram)

    dPDE4B = (-(p.Ki_B_rolipram * rolipram * PDE4B) / (p.Km_B_rolipram + rolipram)
              + p.k_B_deg_rolipram * PDE4B_rolipram)
    dPDE4B_rolipram = ((p.Ki_B_rolipram * rolipram * PDE4B) / (p.Km_B_rolipram + rolipram)
                       - p.k_B_deg_rolipram * PDE4B_rolipram)

    dpCREB = (p.k_creb_PKA * CREB * PKA_cat1
              + p.k_creb_ERK * CREB * ERK_dimer
              - p.k_creb_d * pCREB)

    dCREB = (- p.k_creb_PKA * CREB * PKA_cat1
             - p.k_creb_ERK * CREB * ERK_dimer
             + p.k_creb_d * pCREB)

    
    dCa = (- 2 * p.k_Ca2_CaM * (Ca**2) * CaM + 2 * p.k_inv_Ca2_CaM * Ca2_CaM) + (-2 * p.k_Ca4_CaM * (Ca**2) * Ca2_CaM)
    + (2 * p.k_inv_Ca4_CaM * Ca4_CaM)+ ((-p.k_AC1_Ca4_CaM * AC1_Ca2_CaM * Ca) / (Km4 + Ca))
    + p.k_inv_AC1_Ca4_CaM * AC1_Ca4_CaM + ((-p.k_PDE1_Ca2_CaM * PDE1_Ca2_CaM * (Ca**2)/(Km6 + Ca))
    + p.k_inv_PDE1_Ca2_CaM * PDE1_Ca4_CaM) + ((-p.k_AC1_Ca4_CaM * AC1_Ca2_CaM * Ca) / (Km4 + Ca))
    + (p.k_inv_AC1_Ca4_CaM * AC1_Ca4_CaM)

    dCaM = - (p.k_Ca2_CaM * CaM * Ca) + (p.k_inv_Ca2_CaM * Ca2_CaM)

    dCa2_CaM = (+ p.k_Ca2_CaM * (Ca**2) * CaM - (p.k_inv_Ca2_CaM * Ca2_CaM))
    + (-p.k_Ca4_CaM * Ca2_CaM * (Ca**2)) + (p.k_inv_Ca4_CaM * Ca4_CaM)
    + (-p.k_AC1_Ca4_CaM * AC1 * Ca2_CaM * (Ca**2) + p.k_inv_AC1_Ca2_CaM * AC1_Ca2_CaM)
    + (-p.k_PDE1_Ca2_CaM * PDE1 * Ca2_CaM + p.k_inv_PDE1_Ca2_CaM * PDE1_Ca2_CaM)

    dCa4_CaM = + (p.k_Ca4_CaM * (Ca**2) * Ca2_CaM) - (p.k_inv_Ca4_CaM * Ca4_CaM)

    dAC1 = - (p.k_AC1_Ca2_CaM * AC1 * Ca2_CaM) + (p.k_inv_AC1_Ca2_CaM * AC1_Ca2_CaM)
       
    dAC1_Ca2_CaM = + (p.k_AC1_Ca2_CaM * AC1 * Ca2_CaM) + (- p.k_inv_AC1_Ca2_CaM * AC1_Ca2_CaM)
    + ((-p.k_AC1_Ca4_CaM * AC1_Ca2_CaM * Ca) / (Km4 + Ca)) + p.k_inv_AC1_Ca4_CaM * AC1_Ca4_CaM
    - (p.k_Ca2_CaM_AC1 * Ca2_CaM * AC1_Ca2_CaM) / (p.K_m_Ca2_to_Ca4_AC1 + AC1_Ca2_CaM)  
    + (p.k_deg_Ca4_to_Ca2_AC1 * AC1_Ca4_CaM) / (p.K_m_Ca4_to_Ca2_AC1 + AC1_Ca4_CaM)

    dAC1_Ca4_CaM = (p.k_c_Ca2_to_Ca4_AC1 * Ca2_CaM * AC1_Ca2_CaM) / (p.K_m_Ca2_to_Ca4_AC1 + AC1_Ca2_CaM)
    - (p.k_deg_Ca4_to_Ca2_AC1 * AC1_Ca4_CaM) / (p.K_m_Ca4_to_Ca2_AC1 + AC1_Ca4_CaM)

    dPDE1 = - (p.k_PDE1_Ca2_CaM * PDE1 * Ca2_CaM) + (p.k_inv_PDE1_Ca2_CaM * PDE1_Ca2_CaM)
              - (p.k_c_PKA_1 * PKA_cat1 * PDE41) / (p.K_m_PKA_1 + PDE1)
               + (p.k_deg_ERK_1* p.PP1 * PDE1_ERK) / (p.K_m_deg_ERK_1 + PDE1_ERK)
               + (p.k_deg_PKA_1 * p.PP1 * PDE1_PKA) / (p.K_m_deg_PKA_1 + PDE1_PKA)
               + (p.k_deg_DP_1 * p.PP1 * PDE1_DP) / (p.K_m_deg_DP_1 + PDE1_DP)

    dPDE1_Ca2_CaM = + (p.k_PDE1_Ca2_CaM * PDE1 * Ca2_CaM) - (p.k_inv_PDE1_Ca2_CaM * PDE1_Ca2_CaM)
    - (p.k_PDE1_Ca2_to_Ca4 * Ca2_CaM * PDE1_Ca2_CaM) / (p.Km_PDE1_Ca2_to_Ca4 + PDE1_Ca2_CaM)
    + (p.k_inv_PDE1_Ca4_to_Ca2 * PDE1_Ca4_CaM) / (p.Km_PDE1_Ca4_to_Ca2 + PDE1_Ca4_CaM)

    dPDE1_Ca4_CaM = + ((p.k_PDE1_Ca4_CaM * PDE1_Ca2_CaM * Ca)/(Km6 + Ca)) - (p.kb6 * PDE1_Ca4_CaM)

    def hard_switch(x, threshold=0.5):
        return np.where(x > threshold, 1, 0)

    def soft_switch(x, threshold=0.5, steepness=4):
        switch = 1 / (1 + np.exp(-steepness * (x - threshold)))
        return switch

    #cAMP_switch = soft_switch(cAMP, threshold=2.0, steepness=5)
    #PKA_switch = soft_switch(PKA_cat1, threshold=0.1575, steepness=5)
    cAMP_switch = hard_switch(cAMP, threshold=2.0)
    PKA_switch = hard_switch(PKA_cat1, threshold=0.1575)

    # REACTION RATES - M Activation Variable
    Am = (1.86 * (V + 25.4)) / (1 - np.exp(-(V + 25.4) / 10.3))
    Bm = (-0.086 * (V + 29.7)) / (1 - np.exp((V + 29.7) / 9.16))
    mInf = Am / (Am + Bm)

    # REACTION RATES - H Activation Variable
    Ah = (-0.0336 * (V + 118)) / (1 - np.exp((V + 118) / 11))
    Bh = 2.3 / (1 + np.exp(-(V + 35.8) / 13.4))
    hInf = Ah / (Ah + Bh)
    hTau = 1 / (p.Ph * (Ah + Bh))

    # REACTION RATES - N Activation Variable
    An = (0.186 * (V + 48.4)) / (1 - np.exp(-(V + 48.4) / 10.3))
    Bn = (-0.0086 * (V + 42.7)) / (1 - np.exp((V + 42.7) / 9.16))
    nInf = An / (An + Bn)

    # REACTION RATES - S Activation Variable
    As = (0.00122 * (V + 19.5)) / (1 - np.exp(-(V + 19.5) / 23.6))
    Bs = (-0.000739 * (V + 87.1)) / (1 - np.exp((V + 87.1) / 21.8))
    sInf = As / (As + Bs)
    sTau = 1 / (p.Ps * (As + Bs))

    # REACTION RATE - R Activation Variable
    Ar = (0.007 * np.exp(-(V + 95 * cAMP_switch + 103.5 * (1 - cAMP_switch)) / 19))
    Br = (0.007 * np.exp((V + 95 * cAMP_switch + 103.5 * (1 - cAMP_switch)) / 22))
    rInf = Ar / (Ar + Br)
    rTau = 1 / (p.Pr * (Ar + Br))

    # LTYPE CALCIUM CHANNEL (Cav1.2) - from Martin et al. 2026
    #Activation gate (dL)
    dL_inf = 1.0 / (1.0 + np.exp(-(V + 38.0) / 4.0))
    tau_dL = 5.0
    # Inactivation gate (fL)
    fL_inf = 1.0 / (1.0 + np.exp((V + 63.0) / 8.0))
    tau_fL = 20.0  # ms
    # PKA modulation factor (1 = no modulation, >1 = enhanced)
    pka_factor_CaL = 1.0 + p.caL_pka_gain * (PKA_cat1 / (PKA_cat1 + p.Kd_PKA_CaL))

    # BK CHANNEL (Large Potassium, Ca-dependent) - from Martin et al. 2026
    # BK activation depends on BOTH voltage AND intracellular calcium
    # Voltage-dependent part (sigmoid)
    V_part_BK = 1.0 / (1.0 + np.exp(-(V + 10.0) / 10.0))
    # Calcium-dependent part (Hill function)
    # PKA reduces the Kd (makes channel more sensitive to Ca²⁺)
    effective_Kd_BK = p.Kd_BK / (1.0 + p.bk_pka_gain * (PKA_cat1 / (PKA_cat1 + p.Kd_PKA_BK)))
    # Then Ca_part uses this effective_Kd
    Ca_part_BK = (Ca**n_BK) / (Ca**n_BK + effective_Kd_BK**n_BK)
    # Steady state activation
    bk_inf = V_part_BK * Ca_part_BK
    tau_bk = 1.5  # ms (very fast)

    # IONIC CURRENTS - Computation
    INaF = p.gNaF * (mInf**3) * h * (V - p.ENa)
    INaP = p.gNaP * (nInf**3) * (V - p.ENa)
    IKS = p.gKS * s * (V - p.EK)
    IL = p.gL * (V - p.ELK)
    IHCN = (p.gHCN + cAMP_switch * p.DgHCN) * r * (V - p.EHCN)
    IM = (p.gM + PKA_switch * p.DgM) * w * (V - p.EK)
    ICaL = p.gCaL * pka_factor_CaL * dL * fL * (V - p.ECa) # Ltype calcium current (inward, will be negative in dV/dt)
    IBK = p.gBK * bk * (V - p.EK) # BK current (outward potassium, negative in dV/dt)

    # ===== NMDA RECEPTOR =====
# Voltage-dependent
# Constants
k = 0.28          # dimensionless (combined constant)
b = 0.06          # mV⁻¹ (voltage sensitivity)
B_V = 1.0 / (1.0 + k * np.exp(-b * V))
# CaM switch (soft, graded)

CaM_switch = soft_switch(Ca4_CaM, threshold=0.5, steepness=5)
g_NMDA_eff = p.gNMDA_max * (1 + p.CaM_gain_NMDA * CaM_switch)
update_params_from_dict(p, {
    # NMDA parameters (no Mg²⁺)
    'gNMDA_max': 0.29,        # nS
    'ENMDA': 0.0,             # mV
    'k': 0.28,                # dimensionless combined constant
    'b': 0.06,                # mV⁻¹ voltage sensitivity
    'CaM_gain_NMDA': 1.5,     # dimensionless, max fold increase from CaM switch
})
# Voltage-dependent block (no Mg²⁺)
B_V = 1.0 / (1.0 + p.k * np.exp(-p.b * V))

# NMDA current
INMDA = g_NMDA_eff * kinetics(t) * B_V * (V - p.ENMDA)

# ===== RyR (CICR) =====
# Ca²⁺-induced Ca²⁺ release from ER
Cer = 100.0  # ER calcium concentration (μM) - assume constant for now
C = Ca       # cytosolic calcium (μM)
VRyR = (p.k1_RyR + p.k2_RyR * (C**3 / (p.K0_RyR**3 + C**3))) * (Cer - C)

    # REACTION RATES - W Activation Variable
    wInf = 1 / (1 + np.exp(-(V + 35) / 10))
    wTau = 400 / (3.3 * np.exp((V + 35) / 20) + np.exp(-(V + 35) / 20))
    # ODEs
    dV = (1 / p.Cm) * (-INaF - INaP - IKS - IL - IHCN - IM - ICaL - IBK + p.IApp)
    dh = (hInf - h) / hTau
    ds = (sInf - s) / sTau
    dr = (rInf - r) / rTau
    dw = (wInf - w) / wTau
    ddL = (dL_inf - dL) / tau_dL
    dfL = (fL_inf - fL) / tau_fL
    dbk = (bk_inf - bk) / tau_bk

    results = [dV, dh, ds, dr, dw, dcAMP, dEpac, dEpac_ON, dRAP1_GDP, dRAP1_GTP, dERK, dpERK, dERK_dimer,
 dC1, dPKA_tet, dPKA_reg, dPKA_cat, dPKA_cat1, dPDE4D7, dPDE4D7_ERK,
 dPDE4D7_PKA, dPDE4D7_DP, dPDE4D4, dPDE4D4_ERK, dPDE4D4_PKA,
 dPDE4D4_DP, dPDE4D5, dPDE4D5_ERK, dPDE4D5_PKA, dPDE4D5_DP,
 dPDE4D3, dPDE4D3_ERK, dPDE4D3_PKA, dPDE4D3_DP, dPDE4D8, dPDE4D8_ERK,
 dPDE4D8_PKA, dPDE4D8_DP, dPDE4D9, dPDE4D9_ERK, dPDE4D9_PKA,
 dPDE4D9_DP, dPDE4D1, dPDE4D1_ERK, dPDE4D1_PKA, dPDE4D1_DP, dPDE4D2,
 dPDE4D2_ERK, dPDE4D2_PKA, dPDE4D2_DP, dPDE4D6, dPDE4D6_ERK,
 dPDE4D6_PKA, dPDE4D6_DP, dPDE4D7_rolipram, dPDE4D7_ERK_rolipram,
 dPDE4D7_PKA_rolipram, dPDE4D7_DP_rolipram, dPDE4D4_rolipram, dPDE4D4_ERK_rolipram,
 dPDE4D4_PKA_rolipram, dPDE4D4_DP_rolipram, dPDE4D5_rolipram, dPDE4D5_ERK_rolipram,
 dPDE4D5_PKA_rolipram, dPDE4D5_DP_rolipram, dPDE4D3_rolipram, dPDE4D3_ERK_rolipram,
 dPDE4D3_PKA_rolipram, dPDE4D3_DP_rolipram, dPDE4D8_rolipram, dPDE4D8_ERK_rolipram,
 dPDE4D8_PKA_rolipram, dPDE4D8_DP_rolipram, dPDE4D9_rolipram, dPDE4D9_ERK_rolipram,
 dPDE4D9_PKA_rolipram, dPDE4D9_DP_rolipram, dPDE4D1_rolipram, dPDE4D1_ERK_rolipram,
 dPDE4D2_rolipram, dPDE4D2_ERK_rolipram, dPDE4D6_rolipram, dPDE4D6_ERK_rolipram, dCREB,
 dpCREB, dPDE4A, dPDE4A_rolipram, dPDE4B, dPDE4B_rolipram, drolipram, dCa, dCaM, dCa2_CaM, dCa4_CaM, dAC1, dAC1_Ca2_CaM,
 dAC1_Ca4_CaM, dPDE1, dPDE1_Ca2_CaM, dPDE1_Ca4_CaM]
    return results  # dcAMP
