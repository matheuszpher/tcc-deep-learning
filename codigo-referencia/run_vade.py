import os
import sys
import warnings
import gc

print(sys.version)

print(hasattr(sys, "get_int_max_str_digits"))
print(hasattr(sys, "set_int_max_str_digits"))

if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300

if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda value: None


import time
import pandas as pd
import numpy as np

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import multivariate, univariate
from metrics.metrics import compute_metrics

from models.vade import VaDE

# CONFIG ======================================================================
K_RANGE = range(2, 70, 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAIN_EPOCHS = 20
TRAIN_EPOCHS = 50
BATCH_SIZE = 64
HID_DIM = 10


# SAVE FUNCTIONS ==============================================================
def save_result(row, selection_mode):
    filename = "results/vade/vade_ground_truth.csv" if selection_mode == "ground_truth" else "results/vade/vade_search.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df = pd.DataFrame([row])
    df.to_csv(filename, mode="a", index=False, header=not os.path.exists(filename))


def save_k_history(dataset_name, history):
    output_dir = "results/vade/history"
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, f"{dataset_name}.csv"),
        index=False,
    )


def save_learning_curve(dataset_name, k_val, pretrain_losses, main_losses, selection_mode):
    output_dir = "results/vade/plots"
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Pre-training loss (MSE)
    ax1.plot(range(1, len(pretrain_losses) + 1), pretrain_losses, marker="o", color="tab:blue")
    ax1.set_title("Pre-training (Autoencoder MSE Loss)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Main training loss (ELBO)
    ax2.plot(range(1, len(main_losses) + 1), main_losses, marker="s", color="tab:orange")
    ax2.set_title("Main Training (VaDE ELBO Loss)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Negative ELBO Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(f"Convergence Curves - {dataset_name} (K={k_val})", fontsize=14)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{dataset_name}_{selection_mode}_k{k_val}_loss.png")
    plt.savefig(filename, dpi=300)
    plt.close()


# PREPROCESSING ===============================================================
def prepare_dataloader(X, y=None, batch_size=32, shuffle=True):
    if X.ndim == 3:
        X = X.squeeze(axis=1)

    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std

    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    if y is not None:
        _, y_true = np.unique(y, return_inverse=True)
        tensor_y = torch.tensor(y_true, dtype=torch.long)
    else:
        tensor_y = torch.zeros(len(X), dtype=torch.long)

    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), X_scaled


# EXPERIMENTS =================================================================
def get_processed_datasets(selection_mode):
    filename = (
        "results/vade/vade_ground_truth.csv"
        if selection_mode == "ground_truth"
        else "results/vade/vade_search.csv"
    )

    if not os.path.exists(filename):
        return set()

    try:
        df = pd.read_csv(filename)
        if "dataset" not in df.columns:
            return set()
        return set(df["dataset"].astype(str))
    except Exception:
        return set()


def run_aeon_vade(cluster_mode="ground_truth", ts_type="uni"):
    datasets = univariate if ts_type == "uni" else multivariate
    # datasets = ['Beef']
    processed = get_processed_datasets(cluster_mode)

    for ds_name in datasets:
        if ds_name in processed:
            print(f"!!! Pulando {ds_name} (já processado)")
            continue

        print(f"\n{'=' * 70}")
        print(ds_name)
        print("=" * 70)

        base_results = {
            "dataset": ds_name,
            "model": "VaDE",
            "k_real": None,
            "k_selected": None,
            "k_bic": None,
            "k_aic": None,
            "best_bic": None,
            "best_aic": None,
            "ARI": None,
            "NMI": None,
            "RI": None,
            "time_sec": None,
            "error": None,
        }

        try:
            X_train, y_train = load_classification(
                ds_name, split="train", load_equal_length=True, load_no_missing=True
            )
            X_test, y_test = load_classification(
                ds_name, split="test", load_equal_length=True, load_no_missing=True
            )

            train_loader, X_train_scaled = prepare_dataloader(X_train, y_train, batch_size=BATCH_SIZE)
            test_loader, X_test_scaled = prepare_dataloader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
            
            input_dim = X_train_scaled.shape[1]
            n_samples = len(X_train_scaled)

        except Exception as e:
            print(f"Erro carregando dataset: {e}")
            results = base_results.copy()
            results["error"] = f"Erro carga: {str(e)}"
            save_result(results, cluster_mode)
            continue

        #######################################################################
        # GROUND TRUTH
        #######################################################################
        if cluster_mode == "ground_truth":
            k = len(np.unique(y_train))
            print(f"Using K={k}")

            try:
                start = time.perf_counter()

                vade = VaDE(input_dim=input_dim, n_clusters=k, hid_dim=HID_DIM).to(DEVICE)
                vade.pre_train(train_loader, pre_epoch=PRETRAIN_EPOCHS)

                optimizer = Adam(vade.parameters(), lr=2e-3)
                vade.train()
                for _ in range(TRAIN_EPOCHS):
                    for x_batch, _ in train_loader:
                        x_batch = x_batch.to(DEVICE)
                        loss = vade.ELBO_Loss(x_batch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Predição no teste
                preds = []
                vade.eval()
                with torch.no_grad():
                    for x_batch, _ in test_loader:
                        p = vade.predict(x_batch.to(DEVICE))
                        preds.extend(p)

                total_time = time.perf_counter() - start
                _, y_test_true = np.unique(y_test, return_inverse=True)
                metrics = compute_metrics(y_test_true, np.array(preds))

                results = base_results.copy()
                results.update({
                    "k_real": len(np.unique(y_test)),
                    "k_selected": k,
                    "ARI": metrics["ARI"],
                    "NMI": metrics["NMI"],
                    "RI": metrics["RI"],
                    "time_sec": total_time,
                })

                save_result(results, cluster_mode)
            except Exception as e:
                print(f"  ! Erro no treino K={k}: {e}")
                results = base_results.copy()
                results.update({"k_selected": k, "error": str(e)})
                save_result(results, cluster_mode)

        #######################################################################
        # RANGE SEARCH ( BIC / AIC )
        #######################################################################
        elif cluster_mode == "range_search":
            history = []
            for k in K_RANGE:
                if k >= n_samples:
                    break

                print(f"Testing K={k}")
                try:
                    fit_start = time.perf_counter()

                    vade = VaDE(input_dim=input_dim, n_clusters=k, hid_dim=HID_DIM).to(DEVICE)
                    _ = vade.pre_train(train_loader, pre_epoch=PRETRAIN_EPOCHS)

                    optimizer = Adam(vade.parameters(), lr=2e-3)
                    vade.train()
                    for _ in range(TRAIN_EPOCHS):
                        for x_batch, _ in train_loader:
                            x_batch = x_batch.to(DEVICE)
                            loss = vade.ELBO_Loss(x_batch)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    fit_time = time.perf_counter() - fit_start

                    # Métricas Internas e de Teoria da Informação
                    metric_start = time.perf_counter()

                    preds_train = []
                    vade.eval()
                    with torch.no_grad():
                        for x_batch, _ in train_loader:
                            preds_train.extend(vade.predict(x_batch.to(DEVICE)))
                    preds_train = np.array(preds_train)

                    # BIC e AIC
                    bic, aic = vade.compute_information_criteria(train_loader, n_samples)

                    metric_time = time.perf_counter() - metric_start

                    history.append({
                        "dataset": ds_name,
                        "k": k,
                        "bic": bic,
                        "aic": aic,
                        "fit_time_sec": fit_time,
                        "metric_time_sec": metric_time,
                        "total_time_sec": fit_time + metric_time,
                        "error": None,
                    })

                    print(f"   K={k} | BIC={bic:.2f} | AIC={aic:.2f} ")

                except Exception as e:
                    import traceback
                    print(f"  ! Erro no treino K={k}: {e}")
                    traceback.print_exc()
                    history.append({
                        "dataset": ds_name,
                        "k": k,
                        "bic": np.nan,
                        "aic": np.nan,
                        "silhouette": np.nan,
                        "dbi": np.nan,
                        "fit_time_sec": np.nan,
                        "metric_time_sec": np.nan,
                        "total_time_sec": np.nan,
                        "error": str(e),
                    })
                finally:
                    # Limpeza explícita para evitar o erro de dispatch map do PyTorch Dynamo
                    if 'vade' in locals():
                        del vade
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            save_k_history(ds_name, history)
            hist_df = pd.DataFrame(history).dropna(subset=["bic", "aic"])

            if len(hist_df) == 0:
                print("Nenhum K válido.")
                results = base_results.copy()
                results["error"] = "No valid K"
                save_result(results, cluster_mode)
                continue

            ###################################################################
            # SELEÇÃO DO MELHOR K (escolha através do BIC)
            ###################################################################
            best_bic = hist_df.loc[hist_df["bic"].idxmin()]
            best_aic = hist_df.loc[hist_df["aic"].idxmin()]

            k_bic = int(best_bic["k"])
            k_aic = int(best_aic["k"])

            k_final = k_bic

            ###################################################################
            # TREINO FINAL COM O K_BIC SELECIONADO
            ###################################################################
            start = time.perf_counter()

            vade_final = VaDE(input_dim=input_dim, n_clusters=k_final, hid_dim=HID_DIM).to(DEVICE)
            pretrain_losses = vade_final.pre_train(train_loader, pre_epoch=PRETRAIN_EPOCHS)

            optimizer = Adam(vade_final.parameters(), lr=2e-3)
            main_losses = []

            vade_final.train()
            for _ in range(TRAIN_EPOCHS):
                running_loss = 0.0
                for x_batch, _ in train_loader:
                    x_batch = x_batch.to(DEVICE)
                    loss = vade_final.ELBO_Loss(x_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * x_batch.size(0)
                main_losses.append(running_loss / n_samples)

            # Salva o gráfico do melhor modelo encontrado
            save_learning_curve(ds_name, k_final, pretrain_losses, main_losses, cluster_mode)

            preds_test = []
            vade_final.eval()
            with torch.no_grad():
                for x_batch, _ in test_loader:
                    preds_test.extend(vade_final.predict(x_batch.to(DEVICE)))

            total_time = time.perf_counter() - start
            _, y_test_true = np.unique(y_test, return_inverse=True)
            metrics = compute_metrics(y_test_true, np.array(preds_test))

            results = base_results.copy()
            results.update({
                "k_real": len(np.unique(y_test)),
                "k_selected": k_final,
                "k_bic": k_bic,
                "k_aic": k_aic,
                "best_bic": best_bic["bic"],
                "best_aic": best_aic["aic"],
                "ARI": metrics["ARI"],
                "NMI": metrics["NMI"],
                "RI": metrics["RI"],
                "time_sec": total_time,
            })
            save_result(results, cluster_mode)


# =============================================================================
if __name__ == "__main__":
    print("RUN VADE UNI_DATA: range_search =====================================")
    run_aeon_vade(cluster_mode="range_search")
