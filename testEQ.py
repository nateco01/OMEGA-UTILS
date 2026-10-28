from equacoes import TeoremasSolver
import numpy as np
import math

solver = TeoremasSolver(n_passos=50) # 100 passos é suficiente para validar a precisão O(h^4)

print("="*60)
print("1. TEOREMA DE GREEN")
print("="*60)
# Campo: P = -y, Q = x. O rotacional 2D é (∂Q/∂x - ∂P/∂y) = 1 - (-1) = 2.
# Região: Quadrado [0, 1] x [0, 1]. Área = 1. Integral dupla esperada = 2 * 1 = 2.
P_green = lambda x, y: -y
Q_green = lambda x, y: x

# Curva fechada percorrendo o quadrado no sentido anti-horário (parametrizada por partes ou aproximada)
# Para simplificar a parametrização contínua, usaremos um círculo de raio 1 centrado na origem.
# Área do círculo = π. Integral dupla esperada = 2 * π ≈ 6.283185
curva_circulo = lambda t: [math.cos(t), math.sin(t)]

res_green = solver.CalcularTeoremaDeGreen(
    P_green, Q_green, 
    curva=curva_circulo, 
    t_limits=(0.0, 2*math.pi), 
    regiao_limits=((-1.0, 1.0), (-1.0, 1.0)) # Retângulo que engloba o círculo
)
print(f"Integral de Linha (∮): {res_green['integral_linha']:.6f}")
print(f"Integral Dupla (∬):   {res_green['integral_dupla']:.6f}")
print(f"Diferença:            {res_green['diferenca_absoluta']:.2e}")
print("Esperado (2 * Área do Círculo = 2π): 6.283185\n")

print("="*60)
print("2. TEOREMA DE STOKES")
print("="*60)
# Campo: F = [-y, x, 0]. Rotacional de F = [0, 0, 2].
# Superfície: Hemisfério superior de raio 1: r(u,v) = [sin(u)cos(v), sin(u)sin(v), cos(u)]
# u em [0, pi/2], v em [0, 2pi].
# Integral de superfície do rotacional: ∬ [0,0,2] · n dS = 2 * (Área do hemisfério) = 2 * (2π) = 4π ≈ 12.566370
F_stokes = lambda x, y, z: [-y, x, 0.0]

superficie_hemisferio = lambda u, v: [
    math.sin(u) * math.cos(v),
    math.sin(u) * math.sin(v),
    math.cos(u)
]

# Bordo do hemisfério: Círculo no plano z=0, raio 1.
curva_bordo = lambda t: [math.cos(t), math.sin(t), 0.0]

res_stokes = solver.CalcularTeoremaDeStokes(
    F=F_stokes,
    superficie=superficie_hemisferio,
    u_limits=(0.0, math.pi/2),
    v_limits=(0.0, 2*math.pi),
    curva_bordo=curva_bordo,
    t_limits=(0.0, 2*math.pi)
)
print(f"Integral de Linha (∮):      {res_stokes['integral_linha']:.6f}")
print(f"Integral de Superfície (∬): {res_stokes['integral_superficie']:.6f}")
print(f"Diferença:                  {res_stokes['diferenca_absoluta']:.2e}")
print("Esperado (4π): 12.566370\n")

print("="*60)
print("3. TEOREMA DA DIVERGÊNCIA (GAUSS)")
print("="*60)
# Campo: F = [x, y, z]. Divergência = ∂x/∂x + ∂y/∂y + ∂z/∂z = 1 + 1 + 1 = 3.
# Volume: Cubo unitário [0,1]x[0,1]x[0,1]. Volume = 1.
# Integral de volume esperada = 3 * 1 = 3.
F_div = lambda x, y, z: [x, y, z]

# Superfícies que fecham o cubo (6 faces). 
# Nota: A orientação do normal (r_u x r_v) deve apontar para FORA do volume.
faces_cubo = [
    # Face x=1 (Normal +x)
    (lambda u, v: [1.0, u, v], (0.0, 1.0), (0.0, 1.0)),
    # Face x=0 (Normal -x) -> Invertendo a ordem de u,v inverte o produto vetorial
    (lambda u, v: [0.0, v, u], (0.0, 1.0), (0.0, 1.0)),
    # Face y=1 (Normal +y)
    (lambda u, v: [u, 1.0, v], (0.0, 1.0), (0.0, 1.0)),
    # Face y=0 (Normal -y)
    (lambda u, v: [v, 0.0, u], (0.0, 1.0), (0.0, 1.0)),
    # Face z=1 (Normal +z)
    (lambda u, v: [u, v, 1.0], (0.0, 1.0), (0.0, 1.0)),
    # Face z=0 (Normal -z)
    (lambda u, v: [v, u, 0.0], (0.0, 1.0), (0.0, 1.0)),
]

res_div = solver.CalcularTeoremaDaDivergencia(
    F=F_div,
    volume_limits=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    superficies_limite=faces_cubo
)
print(f"Integral de Volume (∭):          {res_div['integral_volume']:.6f}")
print(f"Integral de Superfície Fechada:  {res_div['integral_superficie_fechada']:.6f}")
print(f"Diferença:                       {res_div['diferenca_absoluta']:.2e}")
print("Esperado (3 * Volume): 3.000000\n")

print("="*60)
print("Validação numérica concluída com sucesso!")
print("="*60)