# OMEGA_UTILS

Biblioteca de ferramentas numéricas e científicas em Python. O projeto reúne
rotinas de matemática, física, integração numérica, álgebra linear, mecânica dos
fluidos, geometria diferencial, simulações, análise estrutural e materiais.
Os módulos podem ser usados diretamente a partir da raiz do projeto; não há
configuração de empacotamento (`pyproject.toml`/`setup.py`) nesta versão.

> **Importante:** os resultados são calculados conforme as hipóteses descritas
em cada modelo. As rotinas não substituem validação independente, dados
experimentais, análise de engenharia certificada, CFD ou elementos finitos.
Confira unidades, convenções de sinais, domínio e condições de contorno antes
de usar qualquer resultado fora de estudo exploratório.

## Requisitos

- Python 3.10 ou superior. O código usa anotações de união com `|`; o workspace
  está configurado no VS Code para Python 3.12, mas não existe uma matriz formal
  de versões testadas.
- Dependências externas:
  - `numpy`: operações numéricas e arrays;
  - `tensorflow`: operações do solver de matrizes;
  - `matplotlib`: gráficos, visualização e o exemplo de casco.
- `math`, `json`, `os`, `pathlib`, `dataclasses`, `numbers`, `typing`,
  `functools`, `inspect`, `warnings` e `unittest` são da biblioteca padrão.

Instale as dependências a partir da raiz do projeto:

```powershell
python -m pip install -r requirements.txt
```

No Windows, se necessário, crie e ative um ambiente virtual antes da instalação:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

O TensorFlow deve ser instalado em uma versão compatível com a versão de Python
e com o sistema operacional em uso. O projeto não fixa versões das dependências;
para builds reproduzíveis, registre versões aprovadas no ambiente de implantação.

## Início rápido

Execute os comandos a partir da pasta raiz, para que os pacotes locais sejam
encontrados pelo Python:

```python
from constantes import Constantes
from derivadas import DerivadasSolver
from equacoes import EDOsSolver
from integrais import IntegralSolverDefinidas

constantes = Constantes()
derivadas = DerivadasSolver()
edo = EDOsSolver()
integrais = IntegralSolverDefinidas()

print(constantes.G_UNIVERSAL)
print(derivadas.Derivada(lambda x: x**2, 3.0))
tempos, estados = edo.Rk4(lambda t, y: -y, y0=1.0, h=0.1, passos=10)
print(estados[-1])
print(integrais.RegraDoTrapezioComposta(0.0, 1.0, lambda x: x**2, 100, 1e-8))
```

Os exemplos demonstrativos da raiz podem ser executados como scripts:

```powershell
python .\main\main.py
python .\testEQ.py
python .\testFuncoes.py
python .\testDerivada.py
python .\test_T.py
```

`main.main` calcula um escoamento idealizado em bocal Venturi e grava um CSV.
`test_T.py` executa a demonstração de tensões do casco e abre gráficos; ela não é
um teste automatizado de unidade.

## Estrutura do projeto

| Caminho | Conteúdo |
| --- | --- |
| `constantes/` | Constantes matemáticas e físicas em unidades SI. |
| `derivadas/` | Derivadas numéricas, diferenciação vetorial e operadores diferenciais. |
| `equacoes/` | Fórmulas de matemática/física, EDOs, EDPs e geometria diferencial. |
| `estruturas/` | Modelos analíticos de estabilidade, fadiga e mecânica da fratura. |
| `funcoes/` | Funções polinomiais e métodos numéricos para raízes e sistemas não lineares. |
| `geometria_diferencial/` | Integração de geodésicas de Schwarzschild. |
| `integrais/` | Primitivas, integração numérica e integrais de linha/superfície/volume. |
| `main/` | Demonstração e exportação de campo para escoamento em Venturi 3D. |
| `materiais/` | Teoria clássica de laminados compósitos. |
| `matrizes/` | Álgebra linear e integração das equações TOV. |
| `simulacoes/` | Integração Monte Carlo, incerteza, trajetórias estocásticas e otimização. |
| `Elementos_Materiais/` | Consultas a dados JSON locais sobre elementos, fluidos e cristais. |
| `test*.py` | Testes `unittest` e scripts de demonstração/validação. |
| `casco_stress_3d.png` | Imagem incluída na raiz, associada à demonstração estrutural. |

Cada diretório de módulo contém um `__init__.py` com a API reexportada, exceto
`main`, cuja entrada principal é `main/main.py`. Os arquivos `__pycache__`
presentes localmente são caches gerados pelo Python e não fazem parte da API.

## Módulos e APIs

### `constantes`

`Constantes` fornece valores como `PI`, `E`, `PHI`, constante de Euler-Mascheroni,
gravidade terrestre, gravitação universal, pressão atmosférica, velocidade da
luz, constantes de Planck, carga elementar, massas de partículas, constantes
eletromagnéticas e termodinâmicas, além de propriedades astronômicas da Terra e
do Sol. Os valores têm nomes em português e são atributos de instância.

### `derivadas`

`DerivadasSolver` concentra aproximações por diferenças finitas e operações
vetoriais. A API inclui derivadas numéricas de primeira e ordens superiores,
derivadas parciais/mistas, gradiente e Jacobiana, rotacional e operações
relacionadas a campos multidimensionais. Os argumentos de função e o passo
numérico devem ser escolhidos de acordo com a escala e regularidade do problema.

### `equacoes`

O pacote reexporta quatro classes:

- `EquacoesSolver`: equações de primeiro e segundo grau, relações geométricas,
  trigonometria, combinatória, cinemática, leis básicas de mecânica,
  densidade/arrasto, força de Lorentz e iterações associadas a órbitas.
- `EDOsSolver`: solução numérica de EDOs (incluindo RK4 e Dormand–Prince RK45)
  e fórmulas/modelos para oscilações, condução de calor, ondas, escoamentos
  viscosos, Navier–Stokes e aplicações físicas especializadas.
- `EDPsSolver`: diferenças finitas lineares/não lineares e EDPs discretizadas;
  inclui rotinas para métrica, símbolos de Christoffel, tensores de Riemann e
  Ricci, escalar de Ricci, tensor de Einstein, derivada covariante, geodésicas e
  equações de Ricci no vácuo.
- `TeoremasSolver`: integrações numéricas para comparar as formas de linha,
  superfície e volume dos teoremas de Green, Stokes e da divergência.

Os nomes públicos seguem em grande parte o estilo PascalCase legado. Alguns
métodos podem devolver texto para comunicar casos sem solução ou domínio
inválido, enquanto outros lançam exceções; consulte o método específico antes
de automatizar o tratamento de resultados.

### `funcoes`

- `FuncoesSolver`: avaliação de polinômios e determinação de raízes reais para
  polinômios de graus 1 a 4, além de famílias de funções elementares.
- `FuncoesSolverNaoLinear`: métodos numéricos para localizar raízes e resolver
  sistemas não lineares, incluindo métodos de intervalos/iterativos e
  variantes com Jacobiana.

As funções recebem callables Python. Resultados, limites de iteração e
convergência dependem da função, das estimativas iniciais e das tolerâncias.

### `integrais`

- `IntegralSolverIndefinidas`: primitivas conhecidas de funções polinomiais,
  racionais, exponenciais, logarítmicas, trigonométricas e hiperbólicas; inclui
  integração numérica de linha, superfície e volume. Os métodos simbólicos
  recebem explicitamente a constante `C`.
- `IntegralSolverDefinidas`: regras fechadas para integrais comuns, transformada
  de Laplace de casos implementados, integrais de Biot–Savart e métodos
  numéricos como Simpson (inclusive adaptativo) e trapézios.

As regras numéricas dependem do integrando e do intervalo. Verifique
singularidades, orientação das parametrizações, limites e convergência.

### `matrizes`

- `MatrizesSolver`: operações matriciais, solução de sistemas lineares,
  determinante, inversa, autovalores/autovetores, normas, SVD, Gauss–Jordan e
  regressão linear. As operações de tensor usam `tensorflow`; certas rotinas
  também recorrem ao `numpy`.
- `T_uv`: integração RK4 das equações de Tolman–Oppenheimer–Volkoff (TOV) em
  unidades geométricas \(G=c=1\). Suporta densidade de energia constante e uma
  equação de estado politrópica; a integração pode parar ao localizar a
  superfície estelar \(p=0\). `plot_dados()` usa Matplotlib.

`MatrizesSolver.OperacoesMatricial` aceita os modos `soma`, `sub`, `mult` e
`div`; a divisão é implementada como resolução linear, não como uma operação
matemática geral de divisão entre matrizes.

### `geometria_diferencial`

`Schwarzschild` integra, por RK4, o estado orbital \([r, dr/d\tau, \phi, t]\)
em unidades geométricas. Valida os parâmetros e rejeita trajetórias que atinjam
ou cruzem o horizonte \(r=2\). `teste_orbita_circular()` retorna valores de
referência e `plot_dados()` mostra uma visualização 3D; esta última requer
Matplotlib.

### `estruturas`

`AnaliseEstruturalSolver` reúne funções estáticas para:

- flambagem elástica de colunas de Euler e flambagem de placas isotrópicas;
- flambagem de placas ortotrópicas simplesmente apoiadas via aproximação
  multimodal de Ritz;
- amplitude equivalente de Goodman, vida de fadiga de Basquin e dano de Miner;
- fator de intensidade de tensão no modo I e crescimento de trinca de Paris;
- dilatação térmica e tensão de restrição térmica uniaxial.

As docstrings descrevem as idealizações e as unidades de retorno. Alguns
resultados dependem de coeficientes, propriedades, geometria e condições de
contorno fornecidos pelo chamador.

### `materiais`

O módulo implementa teoria clássica de laminados (CLT) para lâminas ortotrópicas
lineares em estado plano de tensão:

- `LaminaOrtotropica`: propriedades elásticas, térmicas e resistências da lâmina;
- `Lamina`: material, orientação e espessura de uma camada;
- `MateriaisCompositosSolver`: matrizes \(Q\) e \(\bar Q\), transformação de
  tensões, critérios de tensão máxima/Tsai–Hill/Tsai–Wu e análise de laminados
  com rigidezes e respostas térmicas.

As classes validam propriedades finitas e positivas e a estabilidade elástica.
O modelo documenta como hipóteses a aderência perfeita, pequenas deformações e
deformação normal constante por camada; não inclui tensões interlaminares,
delaminação, dano progressivo ou viscoelasticidade.

### `simulacoes`

O pacote reexporta `SimulacoesSolver`, `Condicao`, `Parametro` e as constantes
`AMINOACIDOS`, `DNA` e `RNA`. As funções públicas cobrem:

- integração e otimização por Monte Carlo em 1D e múltiplas dimensões;
- propagação de incerteza a partir de geradores de entrada;
- simulação estocástica de trajetórias 3D;
- otimização evolutiva com espaços de busca descritos por parâmetros reais,
  inteiros, categóricos, binários, sequências, permutações, vetores e composições;
- condições, objetivos adicionais, pesos e políticas para resultados inválidos.

`Parametro` fornece construtores para tipos de variáveis; `Condicao` representa
restrições avaliadas pelo solver. A natureza aleatória torna importante
configurar sementes e limites de avaliações quando se deseja reprodutibilidade.
Os detalhes de chamadas e do formato dos resultados estão nas docstrings de
`simulacoes/simulacoes.py`.

### `main`

`SimulacaoTempoRealVenturi3D` representa um bocal axisimétrico com modelo
compressível quase unidimensional. Calcula área, regime e propriedades axiais,
incluindo a possibilidade de choque normal, e estende o escoamento para um
campo 3D aproximado. `calcular_campo_3d()` retorna coordenadas, Mach, pressão,
temperatura, densidade e componentes de velocidade; `exportar_campo_csv()`
grava esses campos em CSV.

Por hipótese, o escoamento é de gás ideal caloricamente perfeito, adiabático e
sem atrito. A extensão 3D não equivale a CFD viscosa ou turbulenta. A execução
`python .\main\main.py` também invoca o laço demonstrativo em tempo real antes
de exportar o arquivo.

### `Elementos_Materiais`

`ElementosMateriais` faz consultas offline a JSONs no próprio diretório do
módulo. Os métodos estáticos `carregar_elemento(*chaves)`,
`carregar_fluido(*chaves)` e `carregar_materiais(*chaves)` percorrem chaves
aninhadas e retornam o valor/dicionário selecionado:

- `elementos_tabela_periodica.json`: dados atômicos e propriedades
  físico-químicas organizados pelo símbolo químico;
- `fluidos_termodinamicos.json`: massa molar, temperaturas/pressões críticas,
  ponto triplo e faixas térmicas de substâncias;
- `estruturas_cristalinas.json`: estrutura, parâmetros/vetores de rede,
  posições atômicas e periodicidade;
- `x.json`: pequeno objeto JSON de exemplo (`chico`), não consumido pelos três
  métodos de consulta.

Exemplo:

```python
from Elementos_Materiais import ElementosMateriais

massa_atomica = ElementosMateriais.carregar_elemento("H", "atomic_weight")
temperatura_critica = ElementosMateriais.carregar_fluido(
    "CycloHexane", "critical_temperature_K"
)
estrutura = ElementosMateriais.carregar_materiais("Si", "crystal_structure")
```

As chaves são sensíveis a maiúsculas/minúsculas e erros de chave/arquivo são
propagados; os métodos não convertem unidades.

## Testes e demonstrações

Os módulos `test_correcoes_numericas.py`, `test_analise_estrutural.py`,
`test_ricci_vacuo.py` e `test_tov.py` contêm testes `unittest` automatizados.
Execute-os explicitamente na raiz:

```powershell
python -m unittest -v test_correcoes_numericas test_analise_estrutural test_ricci_vacuo test_tov
```

Os demais scripts `testDerivada.py`, `testEQ.py`, `testFuncoes.py`,
`test_importacao.py` e `test_T.py` são exemplos/validações executáveis, não uma
suite uniforme de testes automatizados. `testEQ.py`, `testDerivada.py`,
`testFuncoes.py` e `test_importacao.py` executam ações no nível do módulo durante
a importação; prefira executá-los individualmente em vez de importá-los como
parte de uma coleta genérica de testes. `test_T.py` plota a triagem de tensões
do casco e pode abrir janelas gráficas.

Cobertura automatizada existente:

- correções numéricas, integração, diferenciação, EDO/EDP e solvers não lineares;
- operações matriciais e sistemas lineares;
- geometria diferencial, métricas, tensores e geodésicas;
- análise estrutural, critérios de fadiga e materiais compósitos;
- integração TOV e localização da superfície da estrela.

## Convenções e limites de uso

1. **Unidades:** muitas rotinas assumem SI, mas os métodos de relatividade/TOV
   trabalham em unidades geométricas. A unidade esperada deve ser verificada na
   documentação do método.
2. **Precisão:** métodos numéricos têm tolerâncias e limites de iteração; sucesso
   em um exemplo não garante convergência para outros dados.
3. **Retornos:** o projeto é legado e alguns métodos retornam strings em casos
   de erro/domínio, enquanto outros levantam `ValueError`, `RuntimeError` ou
   exceções numéricas. Valide o tipo e o significado do resultado.
4. **Modelos de engenharia:** as rotinas estruturais, materiais e de escoamento
   são idealizadas. Confirme hipóteses, valores de material, fatores de segurança
   e requisitos normativos com um profissional qualificado antes de decisões
   reais de projeto.
5. **Importações:** execute exemplos e comandos a partir da raiz. Importar
   `matrizes.matrizes` requer TensorFlow, pois a dependência é carregada no topo
   do módulo.
