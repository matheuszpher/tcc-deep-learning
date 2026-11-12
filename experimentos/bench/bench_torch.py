"""Benchmark PyTorch: mesma carga do bench_tf.py, um caso por processo.

fcn_ae replica a AEFCNNetwork do aeon (Conv1D 128/256/128, kernels 8/5/3, BN+ReLU,
Flatten->Dense(128) ; Dense->Reshape, ConvTranspose 128/256/128 invertido, projecao 1x1).
A contagem de parametros treinaveis e gravada para conferir a equivalencia com o TF.
bigru_relu nao existe aqui: nn.GRU so oferece tanh (registrado como 'nao_suportado').
"""

import argparse
import json
import time

import numpy as np


def build(model, L, torch):
    nn = torch.nn
    if model == "fcn_ae":
        f, k = [128, 256, 128], [8, 5, 3]

        class Crop(nn.Module):
            """ConvTranspose1d 'same' (stride 1): corta para o comprimento original."""

            def __init__(self, ks):
                super().__init__()
                self.l = (ks - 1) // 2

            def forward(self, x):
                return x[..., self.l:self.l + L]

        enc, cin = [], 1
        for fi, ki in zip(f, k):
            enc += [nn.Conv1d(cin, fi, ki, padding="same"), nn.BatchNorm1d(fi), nn.ReLU()]
            cin = fi
        enc += [nn.Flatten(), nn.Linear(f[-1] * L, 128)]
        dec = [nn.Linear(128, f[-1] * L), nn.Unflatten(1, (f[-1], L))]
        cin = f[-1]
        for fi, ki in list(zip(f, k))[::-1]:
            dec += [nn.ConvTranspose1d(cin, fi, ki), Crop(ki), nn.BatchNorm1d(fi), nn.ReLU()]
            cin = fi
        dec += [nn.ConvTranspose1d(cin, 1, 1)]
        return nn.Sequential(*enc, *dec), "ncl"

    class BiGRUAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.GRU(1, 64, batch_first=True, bidirectional=True)
            self.z = nn.Linear(128, 128)
            self.dec = nn.GRU(128, 128, batch_first=True)
            self.out = nn.Linear(128, 1)

        def forward(self, x):
            _, h = self.enc(x)                       # h: (2, B, 64)
            z = self.z(torch.cat([h[0], h[1]], dim=1))
            r, _ = self.dec(z.unsqueeze(1).expand(-1, L, -1).contiguous())
            return self.out(r)

    return BiGRUAE(), "nlc"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rec = {"framework": "pytorch", "version": torch.__version__,
           "cudnn": torch.backends.cudnn.version() if dev == "cuda" else None,
           "device": "GPU" if dev == "cuda" else "CPU", "model": a.model,
           "length": a.length, "n": a.n, "epochs": a.epochs, "batch": 32}
    if a.model == "bigru_relu":
        rec.update(status="nao_suportado",
                   error="torch.nn.GRU nao tem parametro de ativacao (so tanh); exigiria celula propria sem cuDNN")
    else:
        try:
            torch.manual_seed(0)
            X = np.random.default_rng(0).normal(size=(a.n, a.length, 1)).astype(np.float32)
            t0 = time.perf_counter()
            m, layout = build(a.model, a.length, torch)
            m = m.to(dev)
            Xt = torch.from_numpy(X if layout == "nlc" else X.transpose(0, 2, 1)).to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            lossf = torch.nn.MSELoss()
            rec["build_s"] = time.perf_counter() - t0
            rec["params_trainable"] = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            times, last = [], None
            g = torch.Generator(device="cpu").manual_seed(0)
            for _ in range(a.epochs):
                if dev == "cuda":
                    torch.cuda.synchronize()
                s = time.perf_counter()
                m.train()
                perm = torch.randperm(a.n, generator=g)
                tot = 0.0
                for i in range(0, a.n, 32):
                    xb = Xt[perm[i:i + 32].to(dev)]
                    opt.zero_grad(set_to_none=True)
                    loss = lossf(m(xb), xb)
                    loss.backward()
                    opt.step()
                    tot += loss.item() * len(xb)
                last = tot / a.n
                if dev == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - s)
            rec["first_epoch_s"] = times[0]
            rec["epoch_median_s"] = float(np.median(times[1:]))
            rec["total_s"] = time.perf_counter() - t0
            rec["final_loss"] = float(last)
            if dev == "cuda":
                rec["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
            rec["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            rec["status"] = "erro"
            rec["error"] = " ".join(f"{type(e).__name__}: {e}".split())[:500]
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
