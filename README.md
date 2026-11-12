# Linha de base: agrupadores profundos do aeon no UCR

Etapa 4 do TCC. Seis autoencoders de `aeon.clustering.deep_learning`
treinados nos conjuntos univariados do UCR, com busca do número de grupos K.

## Ambiente

| | Windows (`.venv/`) | WSL2 Ubuntu 20.04 (`~/venvs/tcc`) |
|---|---|---|
| Python | 3.11.3 | 3.11.18 (instalado com `uv`) |
| Pacotes | aeon 1.6.0, TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.9.1, numpy 2.3.5 | iguais, com `tensorflow[and-cuda]` |
| GPU | não (TF >= 2.11 não usa GPU no Windows nativo) | RTX 3050 Laptop, 4 GB |
| Versões fixadas | `ambiente/requirements-lock-windows.txt` | `ambiente/requirements-lock-wsl-gpu.txt` |

`experimentos/run_wsl.sh` põe as bibliotecas CUDA do pip no `LD_LIBRARY_PATH`
(o TF 2.21 não encontra `libcusolver.so.11` sozinho).

Criar o ambiente WSL do zero:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11 && uv venv --python 3.11 ~/venvs/tcc
uv pip install --python ~/venvs/tcc/bin/python -r ambiente/requirements-lock-wsl-gpu.txt
```


## Versão para supercomputador (Jupyter)

`experimentos/supercomputador/aeon_dl_supercomputador.ipynb` é a mesma linha de base adaptada
para rodar num servidor com Jupyter e uma ou mais GPUs: um par (conjunto, modelo) por GPU em
paralelo, em segundo plano, retomável, com limite de tempo por par. Rode as células em ordem
(primeiro com `TESTE_RAPIDO = True`). O notebook embute `aeon_dl_par.py` (treina um par) e
`aeon_dl_agendador.py` (fila, GPUs e consolidação dos CSVs); para alterar, edite os `.py` e rode
`python experimentos/supercomputador/gerar_notebook.py`. O dicionário das colunas está no
próprio notebook. `validar_wsl.sh` executa o notebook de ponta a ponta no modo de teste.

## Comandos

```powershell
# rodada completa: 5 modelos na GPU (WSL) e depois o AEDRNN em CPU (Windows)
powershell -File experimentos\rodada.ps1 -Epochs 200 -Out results/aeon_dl -GpuHours 7 -CpuHours 2

# teste rápido
powershell -File experimentos\rodada.ps1 -Epochs 5 -Out results/aeon_dl_teste -Datasets "Coffee Beef GunPoint"

# tabelas-resumo e diagramas de diferença crítica
.venv\Scripts\python experimentos\resumo.py --out results/aeon_dl

# testes (pré-processamento, grade de K, métricas, seleção)
.venv\Scripts\python -m pytest tests -q
```

Argumentos do `experimentos/aeon_dl_baseline.py`:
- `--datasets`, `--models`, `--epochs` (200), `--seed` (0);
- `--k-min/--k-max/--k-step` (2/70/2);
- `--k-mode grid|n_classes|both`;
- `--device auto|cpu`, `--stop-after-hours`, `--max-size`, `--out`.

## Decisões de pré-processamento (as mesmas em todos os conjuntos)

1. Treino e teste concatenados (Paparrizos e Bogireddy, 2025). Rótulos só nas métricas externas.
2. Valores ausentes: interpolação linear dentro de cada série; nos extremos, o valor válido
   mais próximo; série inteira ausente vira zeros (registrado).
3. Comprimentos diferentes: `Resizer(resized_length="max")` (reamostragem linear para o
   maior comprimento). **Não** se usa `load_equal_length=True` (que trunca).
4. Normalização z por série, depois da reamostragem (desvio 0 -> 1).
5. A coluna `preprocessing` do CSV registra os passos aplicados a cada conjunto.
   Os dados tratados ficam em cache em `data/processed/<conjunto>.npz`.

## Protocolo de treino e avaliação

- Hiperparâmetros de arquitetura padrão do aeon; `n_epochs=200`, `batch_size=32`, `random_state=0`.
- Callbacks iguais ao padrão do aeon (ReduceLROnPlateau em `loss` e checkpoint do menor `loss`),
  mais um cronômetro por época.
- **Cada (conjunto, modelo) é treinado uma vez.** O latente (`model_.layers[1]`, o codificador)
  é salvo em `latents/e{épocas}_s{semente}/<modelo>/<conjunto>.npy`. Para cada K, roda-se
  `TimeSeriesKMeans(euclidiano, média, random_state=semente)` sobre o latente.
- Toda execução confere que `layers[1]` é o codificador: `decoder(encoder(x)) == model(x)`
  e o `predict` do aeon coincide com o k-means aplicado ao latente salvo
  (coluna `encoder_check` de `execucoes.csv`).
- Métricas por K: silhueta e Davies-Bouldin (no latente), RI, ARI, NMI (sklearn, média
  aritmética), `recon_loss` (menor `loss` de treino, como no protocolo), `recon_mse` (MSE pura do
  modelo restaurado sobre todo o conjunto; comparável entre modelos, porque o `loss` do AEDCNN
  inclui a penalidade L2 do decodificador); `elbo` fica vazio (não variacional).
- Cada (conjunto, modelo) roda num subprocesso próprio: a memória volta ao sistema e uma morte
  por falta de memória fica registrada como `erro_processo` em `execucoes.csv`.
- Se o k-means falhar ou devolver menos de 2 grupos, as métricas ficam NaN e o motivo vai
  para a coluna `message` de `execucoes.csv`.

## Saídas (`results/aeon_dl/`)

| Arquivo | Conteúdo |
|---|---|
| `resultados_longo.csv` | uma linha por (conjunto, modelo, K, k_mode, métrica) |
| `selecao_k_silhueta.csv` | K de maior silhueta por (conjunto, modelo); empate: menor K |
| `execucoes.csv` | status, tempos (carga, treino, 1ª época, mediana por época, latente, busca de K), perdas, conferência do codificador, erros |
| `latents/` | espaços latentes (permitem trocar o critério de seleção sem retreinar) |
| `resumo/` | gerado por `resumo.py` |

## Limitações conhecidas

Detalhes e evidências em `docs/notas/achados-linha-de-base.md`.

- **AEDRNNClusterer:** o decodificador tem 3 GRUs de 400 unidades com ReLU, sem kernel cuDNN.
  Já no Coffee (286 pontos) ele pede mais de 3,4 GB e não cabe na GPU de 4 GB. Por isso roda
  em CPU no Windows, e falhas por memória ficam registradas em `execucoes.csv`.
- Os modelos recorrentes (BiGRU, AttentionBiGRU, DRNN) usam ReLU e por isso não usam cuDNN;
  são os mais lentos.
- Semente única, aguardando confirmação do Prof. César. 200 épocas, contra o padrão de 1.500
  a 2.000 do aeon, então a comparação com os resultados publicados pelo aeon é só de ordem de grandeza.
