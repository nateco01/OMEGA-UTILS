import math

class Constantes:
    """
    Repositório de Constantes Matemáticas e Físicas Universais do OMEGA.
    Os valores seguem o padrão internacional do Sistema Internacional de Unidades (SI).
    """
    def __init__(self):
        # ==========================================
        # 1. CONSTANTES MATEMÁTICAS
        # ==========================================
        self.PI = math.pi                   # Proporção circular (3.14159...)
        self.E = math.e                     # Base do logaritmo natural (2.71828...)
        self.PHI = (1 + math.sqrt(5)) / 2   # Proporção Áurea (1.61803...)
        self.GAMMA = 0.57721566490153286    # Constante de Euler-Mascheroni

        # ==========================================
        # 2. MECÂNICA CLÁSSICA E GRAVITAÇÃO
        # ==========================================
        self.G_TERRA = 9.80665              # Aceleração da gravidade padrão na superfície da Terra (m/s²)
        self.G_UNIVERSAL = 6.67430e-11      # Constante gravitacional universal de Newton (N·m²/kg²)
        self.P_ATM = 101325.0               # Pressão atmosférica padrão ao nível do mar (Pa)
        
        # ==========================================
        # 3. ELETROMAGNETISMO E FÍSICA QUÂNTICA
        # ==========================================
        self.C = 299792458                  # Velocidade da luz no vácuo (m/s) - Valor exato
        self.H = 6.62607015e-34             # Constante de Planck (J·s)
        self.H_BAR = self.H / (2 * self.PI) # Constante de Planck Reduzida (J·s)
        self.E_CARGA = 1.602176634e-19      # Carga elementar (do elétron/próton) (C)
        self.M_ELETRON = 9.1093837015e-31   # Massa de repouso do elétron (kg)
        self.M_PROTON = 1.67262192369e-27   # Massa de repouso do próton (kg)
        self.EPSILON_0 = 8.8541878128e-12   # Permissividade elétrica do vácuo (F/m)
        self.MU_0 = 1.25663706212e-6        # Permeabilidade magnética do vácuo (H/m)

        # ==========================================
        # 4. TERMODINÂMICA E FÍSICO-QUÍMICA
        # ==========================================
        self.K_BOLTZMANN = 1.380649e-23     # Constante de Boltzmann (J/K)
        self.R_GASES = 8.314462618          # Constante universal dos gases ideais (J/(mol·K))
        self.N_AVOGADRO = 6.02214076e23     # Número de Avogadro (mol^-1)
        self.SIGMA_STEFAN = 5.670374419e-8  # Constante de Stefan-Boltzmann (W/(m²·K⁴))
        self.ZERO_ABSOLUTO = -273.15        # Zero Absoluto em graus Celsius (°C)

        # ==========================================
        # 5. ASTROFÍSICA E COSMOLOGIA
        # ==========================================
        self.M_TERRA = 5.9722e24            # Massa da Terra (kg)
        self.R_TERRA = 6371000              # Raio médio da Terra (m)
        self.M_SOL = 1.98847e30             # Massa do Sol (kg)
        self.UA = 149597870700              # Unidade Astronômica (distância média Terra-Sol) (m)
        self.PARSEC = 3.085677581e16        # Parsec (m)
        self.ANO_LUZ = 9.4607e15            # Distância percorrida pela luz em um ano no vácuo (m)