import numpy as np
import math

class Schwarzschild():
  def __init__(self, r_0, v_0, L, h, N, phi, t, tempo_sim):
    """
    r_0: raio inicial, em unidades geometricas, r_0 > 2
    v_0: velocidade radial inicial dr/dtau
    L: momento angular por unidade de massa, constante
    h: passo proprio maximo de integracao
    N: numero maximo de passos
    phi, t: coordenadas angulares e temporais iniciais
    tempo_sim: duracao total em tempo proprio

    EXEMPLO DE USO:

    c = Schwarzschild(
        r_0 = 10.0,       # Raio seguro (acima de 3)
        v_0 = 0.2,        # Sem velocidade radial inicial (começa em repouso radial)
        L = np.sqrt(10.0**2 / (10.0 - 3)), # Momento angular exato para órbita circular
        h = 0.01,         # Passo de integração menor e mais preciso (ex: 0.01)
        N = 50000,
        t = 100.1,
        phi = 0.2,
        tempo_sim = 500.0
    )

    print("Teste Órbita Circular (L, E):", c.teste_orbita_circular())
    print(c.Rk4())
    c.plot_dados()

    """
    self.r_0 = r_0
    self.v_0 = v_0
    self.L = L
    self.h = h
    self.N = N
    self.phi = phi
    self.t = t
    self.tempo_sim = tempo_sim

    valores = (self.r_0, self.v_0, self.L, self.h, self.phi, self.t, self.tempo_sim)
    if not all(math.isfinite(float(valor)) for valor in valores):
      raise ValueError("Os parametros orbitais devem ser finitos.")
    if self.r_0 <= 2.0:
      raise ValueError("r_0 deve ser maior que o horizonte r=2.")
    if self.h <= 0.0 or self.tempo_sim < 0.0:
      raise ValueError("h deve ser positivo e tempo_sim nao negativo.")
    if isinstance(self.N, bool) or not isinstance(self.N, int) or self.N < 1:
      raise ValueError("N deve ser um inteiro positivo (numero maximo de passos).")

  def E(self):
    """
    E é a energia por unidade de massa.
    Ela é constante e só entra em dt/dτ e na checagem de conservação.
    """
    E = np.sqrt(self.v_0**2 + (1 - 2/self.r_0) * (1 + self.L**2/self.r_0**2))

    return E

  def y(self):
    """
    definido vetores de estado, onde:
    y = [ r, v, φ, t]

    r: raio
    v = ṙ: velocidade radial
    φ: ângulo na órbita (começa em 0)
    t: tempo coordenado (começa em 0, opcional)
    """
    return self._derivadas(np.array([self.r_0, self.v_0, self.phi, self.t]))

  def _derivadas(self, estado):
    r, velocidade_radial, _phi, _tempo = estado
    if not math.isfinite(float(r)) or r <= 2.0:
      raise ValueError("A geodesica atingiu ou cruzou o horizonte r=2.")
    energia = self.E()
    aceleracao_radial = (
        -1.0 / r**2
        + self.L**2 / r**3
        - 3.0 * self.L**2 / r**4
    )
    return np.array(
        [
            velocidade_radial,
            aceleracao_radial,
            self.L / r**2,
            energia / (1.0 - 2.0 / r),
        ],
        dtype=float,
    )

  def Rk4(self):
    """Integra o estado [r, dr/dtau, phi, t] e devolve todos os estados."""
    passos = max(1, int(math.ceil(self.tempo_sim / self.h))) if self.tempo_sim else 0
    if passos > self.N:
      raise ValueError(
          f"A duracao e o passo exigem {passos} passos, acima do limite N={self.N}."
      )
    estado = np.array([self.r_0, self.v_0, self.phi, self.t], dtype=float)
    self.lista = [estado.copy()]
    if passos == 0:
      return np.asarray(self.lista)

    passo = self.tempo_sim / passos
    for _ in range(passos):
      k1 = self._derivadas(estado)
      k2 = self._derivadas(estado + 0.5 * passo * k1)
      k3 = self._derivadas(estado + 0.5 * passo * k2)
      k4 = self._derivadas(estado + passo * k3)
      estado = estado + (passo / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
      if not np.all(np.isfinite(estado)):
        raise FloatingPointError("O estado integrado contem valores nao finitos.")
      if estado[0] <= 2.0:
        raise ValueError("A geodesica atingiu ou cruzou o horizonte r=2.")
      self.lista.append(estado.copy())
    return np.asarray(self.lista)

  def teste_orbita_circular(self):
      L = np.sqrt(self.r_0**2 / (self.r_0 - 3))
      E = (1 - 2/self.r_0) / np.sqrt(1 - 3/self.r_0)
      return L, E

  def plot_dados(self):
    import matplotlib.pyplot as plt
    import numpy as np
    if not getattr(self, "lista", None):
      raise ValueError("Execute Rk4() antes de plotar a orbita.")
    estados = np.asarray(self.lista, dtype=float)
    R_vals = estados[:, 0]
    Phi_vals = estados[:, 2]
    T_vals = estados[:, 3]

    # Convertendo coordenadas polares (r, phi) para Cartesianas (X, Y) no plano orbital
    X_cart = [r * np.cos(phi) for r, phi in zip(R_vals, Phi_vals)]
    Y_cart = [r * np.sin(phi) for r, phi in zip(R_vals, Phi_vals)]
    Z_tempo = T_vals  # O tempo avança no eixo vertical Z

    # Cria a figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # Plota a órbita tridimensional no espaço-tempo (Espiral de Schwarzschild)
    ax.plot(X_cart, Y_cart, Z_tempo, label='Órbita Relativística no Espaço-Tempo', color='red', lw=2)

    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_zlabel('Tempo Coordenado (t)')
    ax.set_title('Evolução Geométrica 3D da Órbita ao Longo do Tempo')

    plt.legend()
    plt.show()

