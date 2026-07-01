"""Reusable PyTorch building blocks for the energy-demand forecasting notebooks.

Contains the TiDE architecture (configurable, supports point or multi-quantile
output), a window builder, a generic trainer with LR scheduling + early stopping,
and a thin forecaster wrapper that adapts a trained model to the
``forecast_utils`` backtest interface (``predict_fn(df, origin, h)``).
"""
from __future__ import annotations
import copy, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# --------------------------------------------------------------- architecture
class ResBlock(nn.Module):
    """TiDE residual block: MLP + linear skip + LayerNorm."""
    def __init__(self, d_in, d_out, dropout=0.2):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(),
                                 nn.Linear(d_out, d_out), nn.Dropout(dropout))
        self.skip = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        return self.norm(self.lin(x) + self.skip(x))


class TiDE(nn.Module):
    """Time-series Dense Encoder (arXiv:2304.08424), with known future covariates.

    ``n_out=1`` gives a point forecast of shape (B, H); ``n_out>1`` gives a
    quantile forecast of shape (B, n_out, H).
    """
    def __init__(self, n_feats, L, H, proj=8, hidden=512, dec_dim=8,
                 n_enc=2, dropout=0.2, n_out=1):
        super().__init__()
        self.L, self.H, self.dec_dim, self.n_out = L, H, dec_dim, n_out
        self.feat_proj = ResBlock(n_feats, proj, dropout)
        enc_in = L + (L + H) * proj
        layers = [ResBlock(enc_in, hidden, dropout)]
        layers += [ResBlock(hidden, hidden, dropout) for _ in range(n_enc - 1)]
        self.encoder = nn.Sequential(*layers)
        self.decoder = ResBlock(hidden, H * dec_dim, dropout)
        self.temporal = ResBlock(dec_dim + proj, n_out, dropout)
        self.skip = nn.Linear(L, H)

    def forward(self, xp, xf):
        B = xp.size(0)
        past_y, past_cov = xp[:, :, 0], xp[:, :, 1:]
        cov_p = self.feat_proj(torch.cat([past_cov, xf], dim=1))     # (B, L+H, proj)
        e = self.encoder(torch.cat([past_y, cov_p.reshape(B, -1)], dim=1))
        d = self.decoder(e).reshape(B, self.H, self.dec_dim)
        o = self.temporal(torch.cat([d, cov_p[:, self.L:, :]], dim=-1))   # (B,H,n_out)
        o = o.permute(0, 2, 1) + self.skip(past_y).unsqueeze(1)           # (B,n_out,H)
        return o[:, 0, :] if self.n_out == 1 else o


class Seq2Seq(nn.Module):
    """RNN/GRU/LSTM encoder-decoder driven by known future covariates.

    Encoder reads past [target, covariates]; decoder consumes the future
    covariate sequence (initialised with the encoder state) and emits one
    demand value per future step. ``cell`` in {"RNN", "GRU", "LSTM"}.
    """
    def __init__(self, n_feats, cell="GRU", hidden=64, n_out=1):
        super().__init__()
        rnn = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell]
        self.n_out = n_out
        self.enc = rnn(1 + n_feats, hidden, batch_first=True)
        self.dec = rnn(n_feats, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, xp, xf):
        _, hidden = self.enc(xp)
        dout, _ = self.dec(xf, hidden)
        o = self.head(dout)                       # (B, H, n_out)
        return o.squeeze(-1) if self.n_out == 1 else o.permute(0, 2, 1)  # ->(B,n_out,H)


# ------------------------------------------------------------------ windowing
def make_windows(ys, Xs, starts, L, H):
    """Build (Xp, Xf, Y) tensors. Xp=(N,L,1+F) past [y,cov]; Xf=(N,H,F) future cov."""
    F = Xs.shape[1]
    Xp = np.empty((len(starts), L, 1 + F), "float32")
    Xf = np.empty((len(starts), H, F), "float32")
    Y = np.empty((len(starts), H), "float32")
    for i, o in enumerate(starts):
        Xp[i, :, 0] = ys[o - L + 1: o + 1]
        Xp[i, :, 1:] = Xs[o - L + 1: o + 1]
        Xf[i] = Xs[o + 1: o + 1 + H]
        Y[i] = ys[o + 1: o + 1 + H]
    return torch.from_numpy(Xp), torch.from_numpy(Xf), torch.from_numpy(Y)


def quantile_loss(pred, target, quantiles):
    """Pinball loss. pred=(B,Q,H), target=(B,H)."""
    target = target.unsqueeze(1)
    diff = target - pred
    q = torch.tensor(quantiles, device=pred.device).view(1, -1, 1)
    return torch.mean(torch.maximum(q * diff, (q - 1) * diff))


# -------------------------------------------------------------------- trainer
def train_torch(model, train_ds, val_tensors, device, epochs=120, patience=12,
                lr=2e-3, batch=256, lossfn=None, verbose=False):
    """Train with Adam + ReduceLROnPlateau + early stopping; restore best weights."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    lossfn = lossfn or nn.MSELoss()
    dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    Xp_v, Xf_v, Y_v = (t.to(device) for t in val_tensors)
    best, best_state, wait = float("inf"), None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xp, xf, y in dl:
            opt.zero_grad()
            loss = lossfn(model(xp.to(device), xf.to(device)), y.to(device))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossfn(model(Xp_v, Xf_v), Y_v).item()
        sched.step(vl)
        if vl < best - 1e-5:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return {"model": model, "val": best, "epochs": ep + 1, "secs": time.time() - t0}


# ----------------------------------------------------------------- forecaster
class TorchForecaster:
    """Adapt a trained TiDE to the forecast_utils backtest interface.

    Scaling (ymu/ysd) is applied on the way in/out so predictions are in GWh.
    """
    def __init__(self, model, ys, Xs, L, device, ymu, ysd):
        self.model, self.ys, self.Xs = model, ys, Xs
        self.L, self.device, self.ymu, self.ysd = L, device, ymu, ysd
        self.F = Xs.shape[1]

    def _inputs(self, o, h, Xfut):
        xp = np.empty((1, self.L, 1 + self.F), "float32")
        xp[0, :, 0] = self.ys[o - self.L + 1: o + 1]
        xp[0, :, 1:] = self.Xs[o - self.L + 1: o + 1]
        xf = Xfut[o + 1: o + 1 + h].reshape(1, h, -1).astype("float32")
        return (torch.from_numpy(xp).to(self.device),
                torch.from_numpy(xf).to(self.device))

    def point_fn(self, Xfut=None):
        Xfut = self.Xs if Xfut is None else Xfut
        def f(d, o, h):
            self.model.eval()
            with torch.no_grad():
                out = self.model(*self._inputs(o, h, Xfut))
            return out.cpu().numpy().ravel()[:h] * self.ysd + self.ymu
        return f

    def quantile_fn(self, Xfut=None):
        Xfut = self.Xs if Xfut is None else Xfut
        def f(d, o, h):
            self.model.eval()
            with torch.no_grad():
                out = self.model(*self._inputs(o, h, Xfut))     # (1,Q,h)
            return out.cpu().numpy()[0, :, :h] * self.ysd + self.ymu
        return f
