import numpy as np
from .AbstractModel import AbstractModel
from tqdm import tqdm

class MathModel(AbstractModel):
    """
    Neutralization dialysis one-dimensional model
    """
    deltaSmall = 1e-8
    Tmax = 11800  # Duration of experiment, seconds(3 hours = 10800 sec)
    dt = 20
    M = int(np.floor(Tmax / dt))  # Number of time steps

    # Flux through the CEM side, time > 0
    def f_J_c(self, J_c, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM, cNa_A_new, cH_A_new, k2_CEM, DNa_CEM):
        P = cH_D_new + cNa_D_new - J_c * DBL2 / d31
        up_CHs = 2 * d31 * (cH_D_new + cNa_D_new) * (DH_S_DBL2 * cH_D_new + J_c * DBL2) - J_c * DBL2 * J_c * DBL2
        down_CHs = 2 * DH_S_DBL2 * (d31 * (cH_D_new + cNa_D_new) - J_c * DBL2)
        CH_DBL2 = up_CHs / down_CHs
        CNa_DBL2 = P - CH_DBL2
        J_c_up = k1_CEM * (1 / (1 + (cNa_A_new / cH_A_new)) - 1 / (1 + (CNa_DBL2 / CH_DBL2)))
        J_c_down = k2_CEM * (1 / (1 + (cNa_A_new / cH_A_new)) + 1 / (1 + (CNa_DBL2 / CH_DBL2))) + 2 * DNa_CEM
        f_J_c = J_c_up / J_c_down - J_c
        return f_J_c

    # derivative of f(J_c)
    def df_J_c(self, J_c, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM, cNa_A_new, cH_A_new, k2_CEM, DNa_CEM):
        return (self.f_J_c(J_c + self.deltaSmall, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM, cNa_A_new,
                           cH_A_new, k2_CEM,
                           DNa_CEM) - self.f_J_c(J_c - self.deltaSmall, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2,
                                                 k1_CEM, cNa_A_new,
                                                 cH_A_new, k2_CEM, DNa_CEM)) / (2 * self.deltaSmall)

    # Flux through the AEM side, time > 0
    def f_J_a(self, J_a, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM, cCl_B_new, cOH_B_new, k2_AEM, DCl_AEM):
        Z = cOH_D_new + cCl_D_new - J_a * DBL3 / d42
        up_COHs = 2 * d42 * (cOH_D_new + cCl_D_new) * (DOH_S_DBL3 * cOH_D_new + J_a * DBL3) - J_a * DBL3 * J_a * DBL3
        down_COHs = 2 * DOH_S_DBL3 * (d42 * (cOH_D_new + cCl_D_new) - J_a * DBL3)
        COH_DBL3 = up_COHs / down_COHs
        CCl_DBL3 = Z - COH_DBL3
        J_a_up = k1_AEM * (1 / (1 + (cCl_B_new / cOH_B_new)) - 1 / (1 + (CCl_DBL3 / COH_DBL3)))
        J_a_down = k2_AEM * (1 / (1 + (cCl_B_new / cOH_B_new)) + 1 / (1 + (CCl_DBL3 / COH_DBL3))) + 2 * DCl_AEM
        f_J_a = J_a_up / J_a_down - J_a
        return f_J_a

    # derivative of f(J_a)
    def df_J_a(self, J_a, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM, cCl_B_new, cOH_B_new, k2_AEM, DCl_AEM):
        df_J_a = (self.f_J_a(J_a + self.deltaSmall, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM, cCl_B_new,
                             cOH_B_new,
                             k2_AEM,
                             DCl_AEM) - self.f_J_a(J_a - self.deltaSmall, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3,
                                                   k1_AEM,
                                                   cCl_B_new,
                                                   cOH_B_new, k2_AEM, DCl_AEM)) / (2 * self.deltaSmall)
        return df_J_a

    def calculate(self, x):
        cNaCl, cHCl, cNaOH, DBLs = x
        # model paramters
        DH_CEM = 2.7e-6  # Diffusion coefficient of H + into a CEM, cm2 / s
        DNa_CEM = 3.6e-7  # Diffusion coefficient of Na + into a CEM, cm2 / s
        DOH_AEM = 9.6e-7  # Diffusion coefficient of OH + into an AEM, cm2 / s
        DCl_AEM = 3.0e-7  # Diffusion coefficient of Cl - into an AEM, cm2 / s
        # DBLs = 80
        DBL2_mkm = DBLs  # DBL thickness(from the right - hand side of CEM), microns
        DBL3_mkm = DBLs  # DBL thickness(from the left - hand side of AEM), microns
        DBL2 = DBL2_mkm / 10000  # DBL thickness(from the right - hand side of CEM), cm
        DBL3 = DBL3_mkm / 10000  # DBL thickness(from the left - hand side of AEM), cm
        # Parameters for the numerical solution
        dAEM_mkm = 140  # AEM thickness, microns
        dCEM_mkm = 170  # CEM thickness, microns
        dAEM = dAEM_mkm / 10000  # AEM thickness, cm
        dCEM = dCEM_mkm / 10000  # CEM thickness, cm
        # Diffusion coefficients of ions in solution at infinite dilution, cm2 / s
        DH_S = 9.31e-5
        DOH_S = 5.26e-5
        DNa_S = 1.34e-5
        DCl_S = 2.03e-5
        # precision
        eps = 1e-8
        # max iterations for Newton-Raphson
        maxIter = 1000
        # Initial concentrations(mmol / cm3) in Desalination(D), Base(B) and Acid(A)compartments
        cH_A = cHCl
        cOH_B = cNaOH
        cNa_D = cNaCl
        cCl_D = cNaCl
        Kw = 1e-14
        cH_D = 1e-7
        cOH_D = Kw / cH_D
        # Exchange capacity, mmol / cm3
        Xa = 1.28
        Xc = 1.43
        # Volume of alkali, base, desalination compartment, cm3 Membrane active surface area, cm2
        VA = 1500
        VB = 1500
        VD = 500
        S = 64
        # Faraday constant, C / mmol Gas constant, J / (mmol * K) Temperature, K
        F = 96485.34  # C / mol
        R = 8.314  # J / (mol * K)
        T = 298  # K
        # numeric solution
        cH_D_new = cH_D
        cOH_D_new = cOH_D
        cNa_D_new = cNa_D
        cCl_D_new = cCl_D
        cH_A_new = cH_A
        cOH_B_new = cOH_B
        cNa_A_new = 0
        cCl_B_new = 0
        k1_CEM = 2 * DH_CEM * DNa_CEM * Xc / dCEM
        k2_CEM = DH_CEM - DNa_CEM
        k1_AEM = 2 * DOH_AEM * DCl_AEM * Xa / dAEM
        k2_AEM = DOH_AEM - DCl_AEM
        # time - dependent problem
        tauk = np.zeros(self.M)
        kappa = np.zeros(self.M)
        pH = np.zeros(self.M)
        J_ct = np.zeros(self.M)
        J_at = np.zeros(self.M)
        J_c = 3.0e-5
        J_a = 2.7e-5
        E_D = self.dt * S / VD
        E_A = self.dt * S / VA
        E_B = self.dt * S / VB
        CH_DBL2_num = 0.0036
        CNa_DBL2_num = 0.0084
        COH_DBL3_num = 0.0046
        CCl_DBL3_num = 0.0121
        for k in range(0, self.M):
            tauk[k] = k * self.dt
            # Diff coef dependent on concentration
            # Activity factor at DBLs / membrane borders PROBLEM FOR C>0.02
            g_H_DBL2 = 0.7396 * CH_DBL2_num - 0.5184 * pow(CH_DBL2_num, 0.5) + 0.9977
            g_Na_DBL2 = 0.7396 * CNa_DBL2_num - 0.5184 * pow(CNa_DBL2_num, 0.5) + 0.9977
            g_OH_DBL3 = 0.7396 * COH_DBL3_num - 0.5184 * pow(COH_DBL3_num, 0.5) + 0.9977
            g_Cl_DBL3 = 0.7396 * CCl_DBL3_num - 0.5184 * pow(CCl_DBL3_num, 0.5) + 0.9977
            # Activity factor in Desalination chamber
            g_H_D = 0.7396 * cH_D_new - 0.5184 * pow(cH_D_new, 0.5) + 0.9977
            g_Na_D = 0.7396 * cNa_D_new - 0.5184 * pow(cNa_D_new, 0.5) + 0.9977
            g_OH_D = 0.7396 * cOH_D_new - 0.5184 * pow(cOH_D_new, 0.5) + 0.9977
            g_Cl_D = 0.7396 * cCl_D_new - 0.5184 * pow(cCl_D_new, 0.5) + 0.9977
            # New diffusion coefficients at DBLs / membrane borders
            # New diffusion coefficients in DBLs
            DH_S_DBL2 = DH_S * g_H_DBL2
            DNa_S_DBL2 = DNa_S * g_Na_DBL2
            DOH_S_DBL3 = DOH_S * g_OH_DBL3
            DCl_S_DBL3 = DCl_S * g_Cl_DBL3
            # New diffusion coefficients in Desalination chamber
            DH_S_D = DH_S * g_H_D
            DNa_S_D = DNa_S * g_Na_D
            DOH_S_D = DOH_S * g_OH_D
            DCl_S_D = DCl_S * g_Cl_D
            d31 = 2 * DH_S_DBL2 * DNa_S_DBL2 / (DH_S_DBL2 - DNa_S_DBL2)
            d42 = 2 * DOH_S_DBL3 * DCl_S_DBL3 / (DOH_S_DBL3 - DCl_S_DBL3)
            x0 = J_c
            y0 = J_a
            # initial point for Newton - Raphson
            J_c = x0
            J_c_old = x0
            for i in range(1, maxIter):
                J_c = J_c - self.f_J_c(J_c, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM, cNa_A_new, cH_A_new,
                                       k2_CEM,
                                       DNa_CEM) / self.df_J_c(J_c, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM,
                                                              cNa_A_new,
                                                              cH_A_new, k2_CEM, DNa_CEM)
                if abs(J_c - J_c_old) < eps:
                    break
                J_c_old = J_c
            J_ct[k] = J_c * 100000
            P = cH_D_new + cNa_D_new - J_c * DBL2 / d31
            up_CHs = 2 * d31 * (cH_D_new + cNa_D_new) * (DH_S_DBL2 * cH_D_new + J_c * DBL2) - J_c * DBL2 * J_c * DBL2
            down_CHs = 2 * DH_S_DBL2 * (d31 * (cH_D_new + cNa_D_new) - J_c * DBL2)
            CH_DBL2_num = up_CHs / down_CHs
            CNa_DBL2_num = P - CH_DBL2_num
            # initial point for Newton - Raphson
            J_a = y0
            J_a_old = y0
            for i in range(1, maxIter):
                J_a = J_a - self.f_J_a(J_a, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM, cCl_B_new, cOH_B_new,
                                       k2_AEM,
                                       DCl_AEM) / self.df_J_a(J_a, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM,
                                                              cCl_B_new,
                                                              cOH_B_new, k2_AEM, DCl_AEM)
                if abs(J_a - J_a_old) < eps:
                    break
                J_a_old = J_a
            J_at[k] = J_a * 100000
            Z = cOH_D_new + cCl_D_new - J_a * DBL3 / d42
            up_COHs = 2 * d42 * (cOH_D_new + cCl_D_new) * (
                    DOH_S_DBL3 * cOH_D_new + J_a * DBL3) - J_a * DBL3 * J_a * DBL3
            down_COHs = 2 * DOH_S_DBL3 * (d42 * (cOH_D_new + cCl_D_new) - J_a * DBL3)
            COH_DBL3_num = up_COHs / down_COHs
            CCl_DBL3_num = Z - COH_DBL3_num
            # New concentrations into compartments
            # Acid compartment
            cH_A_new = cH_A_new - E_A * J_c
            cNa_A_new = cH_A - cH_A_new
            # Alkali compartment
            cOH_B_new = cOH_B_new - E_B * J_a
            cCl_B_new = cOH_B - cOH_B_new
            # Desalination compartment
            cCl_D_new = cCl_D_new - E_D * J_a
            B = cH_D_new - cOH_D_new + E_D * (J_c - J_a)
            cH_D_new = (B + pow((pow(B, 2) + 4 * Kw), 0.5)) / 2
            cOH_D_new = Kw / cH_D_new
            cNa_D_new = cCl_D_new + cOH_D_new - cH_D_new
            kappa[k] = ((pow(F, 2)) / (R * T)) * (
                    DNa_S_D * cNa_D_new + DCl_S_D * cCl_D_new + DH_S_D * cH_D_new + DOH_S_D * cOH_D_new)  # mS / cm
            pH[k] = -np.log10(cH_D_new)

        tau_opt_kappa = tauk[0]
        opt_kappa = kappa[0]

        for i in range(1, self.M):
            if abs(kappa[i] - 1) < abs(opt_kappa - 1):
                opt_kappa = kappa[i]
                tau_opt_kappa = tauk[i]

        tau_opt_pH1 = tauk[0]
        opt_pH1 = pH[0]
        tau_opt_pH2 = tauk[1]
        opt_pH2 = pH[1]
        for i in range(2, self.M):
            if abs(pH[i] - 7.5) < abs(opt_pH1 - 7.5):
                opt_pH2 = opt_pH1
                tau_opt_pH2 = tau_opt_pH1
                opt_pH1 = pH[i]
                tau_opt_pH1 = tauk[i]

        opt_pH = (opt_pH1 + opt_pH2) / 2
        tau_opt_pH = (tau_opt_pH1 + tau_opt_pH2) / 2
        if abs(opt_pH1 - 7.5) < abs(opt_pH - 7.5):
            opt_pH = opt_pH1
            tau_opt_pH = tau_opt_pH1

        if abs(opt_pH2 - 7.5) < abs(opt_pH - 7.5):
            opt_pH = opt_pH2
            tau_opt_pH = tau_opt_pH2

        target_params = np.zeros(4)
        target_params[0] = tau_opt_kappa / 60  # min
        target_params[1] = opt_kappa
        target_params[2] = tau_opt_pH / 60  # min
        target_params[3] = opt_pH

        process_log = np.zeros((self.M, 3))
        for i in range(0, self.M):
            process_log[i][0] = round(tauk[i] / 60, 3)
            process_log[i][1] = round(kappa[i], 3)
            process_log[i][2] = round(pH[i], 3)

        return target_params, process_log
