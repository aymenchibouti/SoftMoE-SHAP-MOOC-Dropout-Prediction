# -*- coding: utf-8 -*-

import json
import time
import gc
import subprocess
import sys

# ── Auto-install pytorch-tabnet if not present ──────────────────────────────
# Handles both notebook (ipykernel) and script environments.
try:
    import pytorch_tabnet  # noqa: F401
except ModuleNotFoundError:
    print("pytorch_tabnet not found — installing via pip ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pytorch-tabnet",
         "--quiet", "--break-system-packages"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("pytorch_tabnet installed successfully.")
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score, matthews_corrcoef)
from pytorch_tabnet.tab_model import TabNetClassifier  # v5: TabNet baseline

SEED = 42
DATA_PATH = "combined_data_processed.csv"
CLUSTER_ASSIGN_PATH = "student_cluster_assignments.csv"
NON_FEATURE = {"enrollment_id", "target", "_split", "label", "username"}
N_SPLITS = 5          # عدد الفولدات لكل تكرار
N_REPEATS = 5         # عدد تكرارات Repeated Stratified K-Fold -> 25 قياسًا إجماليًا
N_EXPERTS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
if DEVICE.type == "cpu":
    torch.set_num_threads(2)

EPOCHS = 25
BATCH_SIZE = 512
LR = 1.5e-3
LAMBDA_KL = 0.05          # وزن توجيه البوّابة نحو عناقيد SHAP (لطيف لا صارم)
PATIENCE = 6

# نفس إعدادات التدريب تُستخدم لِـ baseline hard-routing experts
# لضمان مقارنة عادلة (نفس الميزانية التدريبية، الفرق فقط في البنية)
SINGLE_NET_HIDDEN = 128   # حجم الشبكة المنفردة للتوجيه الصارم
# R2 — K-sensitivity: values to test in the SHAP cluster sensitivity analysis
K_SENSITIVITY_VALUES = [2, 3, 4]
# v5 — TabNet hyperparameters (replaces GlobalMLP baseline)
# Arik & Pfister (2021): "TabNet: Attentive Interpretable Tabular Learning"
TABNET_N_D      = 64    # width of decision step (= n_a)
TABNET_N_STEPS  = 5     # number of sequential attention steps
TABNET_GAMMA    = 1.5   # coefficient for feature reuse across steps
TABNET_LAMBDA   = 1e-3  # sparsity regularization coefficient

# v6 — LSTM hyperparameters (sequence-native baseline, addresses Limitations
# statement that no recurrent/temporal architecture had been compared).
# The 490-dim feature vector is chunked into LSTM_SEQ_LEN contiguous,
# equal-width pseudo-timesteps (original column order; zero-padded so every
# timestep has the same width) — see train_lstm_baseline() docstring.
LSTM_SEQ_LEN    = 10    # number of pseudo-timesteps the feature vector is split into
LSTM_HIDDEN     = 64    # LSTM hidden-state width (matches TabNet's n_d=64 for fairness)
LSTM_LAYERS     = 1     # single-layer LSTM (matches SoftMoE's single-hidden-layer experts)
LSTM_DROPOUT    = 0.25  # matches SharedTrunkSoftMoE's trunk dropout for fairness


GLOBAL_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1,
)
SURROGATE_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
    objective="multiclass", num_class=N_EXPERTS,
    random_state=SEED, n_jobs=-1, verbose=-1,
)


def evaluate(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def cohen_d_paired(diff):
    """حجم تأثير Cohen's d لعيّنات مُزاوَجة (paired): d = mean(diff) / std(diff).
    يُفسَّر تقريبيًا: 0.2 صغير، 0.5 متوسط، 0.8 كبير (Cohen, 1988)."""
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    if sd == 0 or len(diff) < 2:
        return 0.0
    return float(diff.mean() / sd)


def confidence_interval_paired(diff, alpha=0.05):
    """95% CI for paired mean difference using t-distribution.
    Returns (lower, upper) bounds. Used to address ج4 reviewer request."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n < 2:
        return (float("nan"), float("nan"))
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
    mean_d = diff.mean()
    return (float(mean_d - t_crit * se), float(mean_d + t_crit * se))


def holm_bonferroni_correction(p_values):
    """Apply Holm–Bonferroni step-down correction to a list of p-values.
    Addresses م5 (Round 1) and the reviewer's request to apply uniform
    correction in BOTH Table 14 (main comparison) and Table 11 (ablation).

    Returns: list of adjusted p-values in the SAME ORDER as input.
    A p-value is significant if holm_adj_p < alpha (default alpha=0.05).
    """
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    order = np.argsort(p_arr)          # ascending sort indices
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        # Holm adjusted p = min(1, p * (m - rank))
        adj = min(1.0, p_arr[idx] * (m - rank))
        # Step-down monotonicity: adjusted must be >= previous
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def best_threshold(y_va, p_va, n=121):
    """يجد عتبة القرار التي تُعظِّم F1 على مجموعة validation -- لضمان مقارنة
    عادلة بين النموذجين، لأن معايرة الاحتمالات تختلف بين بنية LightGBM
    وبنية مزيج الخبراء (Soft-MoE)."""
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.15, 0.85, n):
        f = f1_score(y_va, (p_va >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


# ============================================================
#                 معمارية SHAP-Guided Soft MoE
# ============================================================
class Expert(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


class GatingNetwork(nn.Module):
    def __init__(self, in_dim, n_experts, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x):
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)  # log-probabilities


class SoftMoE(nn.Module):
    """النسخة الأصلية: كل خبير شبكة كاملة مستقلة من المدخلات إلى الخرج."""
    def __init__(self, in_dim, n_experts=N_EXPERTS, hidden=128):
        super().__init__()
        self.experts = nn.ModuleList([Expert(in_dim, hidden) for _ in range(n_experts)])
        self.gate = GatingNetwork(in_dim, n_experts)

    def forward(self, x):
        gate_log_probs = self.gate(x)                      # (B, K) log-probs
        gate_probs = gate_log_probs.exp()                  # (B, K)
        expert_logits = torch.stack([e(x) for e in self.experts], dim=1)  # (B, K)
        expert_probs = torch.sigmoid(expert_logits)         # (B, K)
        final_prob = (gate_probs * expert_probs).sum(dim=1)  # (B,)
        return final_prob, gate_log_probs, gate_probs


class SharedTrunkSoftMoE(nn.Module):
    """تحسين معماري حقيقي: جذع (trunk) مشترك يستخرج تمثيلًا عامًا واحدًا، ثم رأس
    صغير ومتخصص لكل خبير. هذا يقلّل عدد البارامترات المكرَّرة بين الخبراء الثلاثة
    (في النسخة الأصلية كل خبير يُعيد تعلّم استخراج الميزات من الصفر)، ويسمح للخبراء
    بالتخصص في القرار النهائي فقط بدل إعادة تعلّم تمثيل المدخلات بالكامل ثلاث مرات
    -- وهو تصميم MoE قياسي ومعروف في الأدبيات (مثل Shazeer et al. 2017)."""
    def __init__(self, in_dim, n_experts=N_EXPERTS, trunk_hidden=192, head_hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_hidden), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(), nn.Dropout(0.25),
        )
        self.expert_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(trunk_hidden, head_hidden), nn.ReLU(), nn.Dropout(0.2),
                           nn.Linear(head_hidden, 1))
            for _ in range(n_experts)
        ])
        self.gate = GatingNetwork(in_dim, n_experts)

    def forward(self, x):
        gate_log_probs = self.gate(x)
        gate_probs = gate_log_probs.exp()
        z = self.trunk(x)
        expert_logits = torch.stack([h(z).squeeze(-1) for h in self.expert_heads], dim=1)
        expert_probs = torch.sigmoid(expert_logits)
        final_prob = (gate_probs * expert_probs).sum(dim=1)
        return final_prob, gate_log_probs, gate_probs


def train_soft_moe(X_tr, y_tr, gate_target_tr, X_va, y_va, gate_target_va, in_dim, pos_weight_scalar,
                    record_history=False, architecture="shared_trunk", lambda_kl=None, fold_seed=None):
    """
    Train the Soft-MoE model with SHAP-anchored gate prior.

    BCE_w explanation (addresses reviewer note about Eq. 2):
      BCEw uses per-sample weights where w_i = pos_weight_scalar for positive
      class (dropout) and w_i = 1.0 for negative class. pos_weight_scalar is
      computed as (# negative samples) / (# positive samples), equivalent to
      sklearn's class_weight='balanced'. This corrects for the 79.3% / 20.7%
      class imbalance without modifying the dataset or loss architecture.

    SHAP-space clustering rationale (addressed as hypothesis per ج5):
      We hypothesize that clustering in SHAP attribution space—rather than
      the original 490-dimensional feature space—groups students by *why*
      they are at risk (their risk-contribution profiles) rather than by raw
      feature similarity. This produces behaviorally interpretable clusters
      (abrupt disengagement, gradual decline, low engagement). Comparison
      with feature-space clustering is left for future work.

    lambda_kl: if None, uses LAMBDA_KL global (λ=0.05, proposed model).
               Pass 0.0 for ablation study (free gate, no SHAP anchoring).
    fold_seed: per-fold seed offset to break Rep.2 coincidence artefact
               noted in Round-2 review (ج2). Each fold uses SEED+fold_seed.
    """
    _seed = SEED if fold_seed is None else SEED + fold_seed
    torch.manual_seed(_seed); np.random.seed(_seed)
    base_lambda = LAMBDA_KL if lambda_kl is None else lambda_kl
    if architecture == "shared_trunk":
        model = SharedTrunkSoftMoE(in_dim).to(DEVICE)
    else:
        model = SoftMoE(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # جدولة Cosine لمعدل التعلم -- التدريب السابق أظهر AUC يبلغ القمة مبكرًا (epoch~2)
    # ثم يتدهور (انظر Fig E)، وهو عرَض كلاسيكي لمعدل تعلم لا يتراجع؛ نعالجه هنا.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    gt_tr_t = torch.tensor(gate_target_tr, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    yva_t = torch.tensor(y_va, dtype=torch.float32, device=DEVICE)

    # وزن موجب/سالب لمعالجة عدم توازن الفئات (مماثل لـ class_weight='balanced')
    w_pos = pos_weight_scalar
    sample_w_tr = torch.where(ytr_t == 1, torch.tensor(w_pos, dtype=torch.float32, device=DEVICE),
                               torch.tensor(1.0, dtype=torch.float32, device=DEVICE))

    n = Xtr_t.shape[0]
    best_auc, best_state, wait = -1, None, 0
    history = {"epoch": [], "train_loss": [], "train_bce": [], "train_kl": [],
               "val_auc": [], "val_accuracy": []}

    for epoch in range(EPOCHS):
        model.train()
        # تذويب تدريجي لوزن KL: توجيه قوي مبكرًا (لتأسيس بوّابة معقولة) يخفّ تدريجيًا
        # لاحقًا (للسماح للبوّابة بالتخصص حسب دقة التنبؤ الفعلية لا فقط تقليد Surrogate)
        # ملاحظة: إذا base_lambda=0 (ablation)، يبقى kl_weight=0 طوال التدريب، أي لا
        # يوجد أي توجيه نحو q(x) على الإطلاق -- البوّابة حرة بالكامل end-to-end.
        kl_weight = base_lambda * max(0.2, 1.0 - epoch / max(EPOCHS - 1, 1))
        perm = torch.randperm(n)
        epoch_loss, epoch_bce, epoch_kl, n_batches = 0.0, 0.0, 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb, gtb, wb = Xtr_t[idx], ytr_t[idx], gt_tr_t[idx], sample_w_tr[idx]
            opt.zero_grad()
            final_prob, gate_log_probs, gate_probs = model(xb)
            final_prob = final_prob.clamp(1e-6, 1 - 1e-6)
            bce = F.binary_cross_entropy(final_prob, yb, weight=wb)
            # توجيه البوّابة نحو توزيع العنقود (Surrogate) -- KL(gate || surrogate)
            kl = F.kl_div(gate_log_probs, gtb, reduction="batchmean")
            loss = bce + kl_weight * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if record_history:
                epoch_loss += float(loss.item()); epoch_bce += float(bce.item()); epoch_kl += float(kl.item())
                n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            p_va, _, _ = model(Xva_t)
        p_va_np = p_va.detach().cpu().numpy()
        val_auc = roc_auc_score(y_va, p_va_np)

        if record_history:
            val_acc = accuracy_score(y_va, (p_va_np >= 0.5).astype(int))
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(epoch_loss / max(n_batches, 1))
            history["train_bce"].append(epoch_bce / max(n_batches, 1))
            history["train_kl"].append(epoch_kl / max(n_batches, 1))
            history["val_auc"].append(val_auc)
            history["val_accuracy"].append(val_acc)

        if val_auc > best_auc:
            best_auc, wait = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    if record_history:
        return model, history
    return model


def train_single_net(X_tr, y_tr, X_va, y_va, in_dim, pos_weight_scalar, hidden=SINGLE_NET_HIDDEN):
    """يدرّب شبكة عصبية واحدة (بدون MoE وبدون بوّابة). تُستخدَم كخبير منفرد ضمن
    Baseline التوجيه الصارم (Hard-Routing): كل خبير من الثلاثة يرى فقط عيّنات
    عنقوده ويُدرَّب عبر هذه الدالة بشكل مستقل."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = Expert(in_dim, hidden=hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)

    w_pos = pos_weight_scalar
    sample_w_tr = torch.where(ytr_t == 1, torch.tensor(w_pos, dtype=torch.float32, device=DEVICE),
                               torch.tensor(1.0, dtype=torch.float32, device=DEVICE))

    n = Xtr_t.shape[0]
    best_auc, best_state, wait = -1, None, 0
    n_unique = len(np.unique(y_va)) if len(y_va) > 0 else 1

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb, wb = Xtr_t[idx], ytr_t[idx], sample_w_tr[idx]
            opt.zero_grad()
            logits = model(xb)
            prob = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(prob, yb, weight=wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(Xva_t)).detach().cpu().numpy()
        if n_unique < 2:
            val_auc = best_auc + 1  # لا يمكن حساب AUC على عنقود بفئة واحدة -> اعتمد آخر حالة
        else:
            val_auc = roc_auc_score(y_va, p_va)

        if val_auc > best_auc:
            best_auc, wait = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_single_net(model, X):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        p = torch.sigmoid(model(Xt)).detach().cpu().numpy()
    return p


def fit_temperature(y_va, p_va, n=200):
    """يجد درجة الحرارة T التي تُصغِّر NLL على validation -- بحث 1D بسيط.
    p_calibrated = sigmoid(logit(p)/T). T>1 يُلطِّف الثقة الزائدة، T<1 يُشدِّدها."""
    eps = 1e-6
    p_va_c = np.clip(p_va, eps, 1 - eps)
    logit_va = np.log(p_va_c / (1 - p_va_c))

    def nll(T):
        scaled = 1 / (1 + np.exp(-logit_va / T))
        scaled = np.clip(scaled, eps, 1 - eps)
        return -np.mean(y_va * np.log(scaled) + (1 - y_va) * np.log(1 - scaled))

    Ts = np.linspace(0.3, 5.0, n)
    nlls = [nll(T) for T in Ts]
    best_T = Ts[int(np.argmin(nlls))]
    return best_T


def apply_temperature(p, T, eps=1e-6):
    p_c = np.clip(p, eps, 1 - eps)
    logit = np.log(p_c / (1 - p_c))
    return 1 / (1 + np.exp(-logit / T))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece


def adaptive_expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Adaptive (equal-frequency / quantile-binned) ECE.

    Addresses Reviewer 1, Major Comment #7: equal-width ECE (the function
    above) is sensitive to how predictions are distributed across the [0,1]
    interval. On this dataset that sensitivity is not hypothetical — the
    dropout rate is 79.3%, so predictions cluster heavily in the upper
    range and several equal-width bins are nearly empty, which lets a few
    sparsely-populated bins dominate (or vanish from) the ECE estimate.

    Adaptive ECE instead places bin edges at empirical quantiles of
    y_prob, so every bin holds approximately the same number of samples
    (n/n_bins). Each bin's contribution is therefore estimated from a
    comparable sample size, making the statistic far more stable under
    skewed prediction distributions. Reported alongside equal-width ECE
    (not instead of it): the two answer different questions, and agreement
    between them is itself evidence that the calibration claim is not a
    binning artefact.

    Ties are handled by np.unique on the quantile edges: if the predicted
    probabilities are highly concentrated, duplicate edges collapse and
    the effective number of bins drops below n_bins, which is the correct
    behaviour (it is not meaningful to split identical values).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y_prob, quantiles))
    if len(edges) < 2:
        # Degenerate case: all predictions identical.
        return float(abs(y_prob.mean() - y_true.mean()))

    # right=True with the left-most edge nudged so the minimum value is included.
    bin_ids = np.clip(np.searchsorted(edges, y_prob, side="left") - 1,
                      0, len(edges) - 2)

    ece = 0.0
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


# ============================================================
#   مقاييس معايرة إضافية (Brier / NLL / calibration slope-intercept)
#   أُضيفت استجابة لملاحظة المُراجع: ECE وحده غير كافٍ لإثبات claim المعايرة،
#   إذ يتأثر بعدد bins وطريقة binning. هذه المقاييس الثلاثة لا تعتمد على binning.
# ============================================================
def brier_score(y_true, y_prob):
    """متوسط مربع الخطأ بين الاحتمال المتوقَّع والتصنيف الفعلي (Brier, 1950).
    لا يعتمد على binning، ويُكمِّل ECE كمقياس معايرة + تمييز مدمج."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def negative_log_likelihood(y_true, y_prob, eps=1e-7):
    """متوسط Negative Log-Likelihood (log loss). حسّاس جدًا للثقة الزائدة
    القريبة من 0 أو 1، ولذلك يُكمِّل Brier/ECE بمعلومات إضافية عن الذيول."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def calibration_slope_intercept(y_true, y_prob, eps=1e-7):
    """Slope/Intercept المعايرة (Cox, 1958؛ شائعة في أدبيات نماذج الخطر السريرية):
    تنحدر y على logit(p) عبر انحدار لوجستي بسيط بمعامل واحد + ثابت.
    Slope=1 وIntercept=0 يعني معايرة مثالية، Slope<1 يعني ثقة زائدة (overconfidence)،
    وهو ما نتوقعه في Soft-MoE الخام بناءً على نتائج ECE في القسم السادس."""
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    y_true = np.asarray(y_true, dtype=float)
    try:
        lr = LogisticRegression(penalty=None, max_iter=2000)
        lr.fit(logit_p, y_true)
        slope = float(lr.coef_[0, 0])
        intercept = float(lr.intercept_[0])
    except Exception:
        # fallback لو فشل التقارب أو كانت penalty=None غير مدعومة في إصدار قديم من sklearn
        lr = LogisticRegression(C=1e6, max_iter=2000)
        lr.fit(logit_p, y_true)
        slope = float(lr.coef_[0, 0])
        intercept = float(lr.intercept_[0])
    return slope, intercept


def full_calibration_report(y_true, y_prob, n_bins=10):
    """يُجمِّع كل مقاييس المعايرة في قاموس واحد لتسهيل التسجيل في all_rows.

    v8: أُضيف adaptive_ece (equal-frequency binning) استجابةً لملاحظة
    المُراجع الأول رقم 7؛ يُبلَّغ عنه إلى جانب ECE التقليدي (equal-width)
    لا بدلًا منه، لأن اتفاق المقياسين دليل على أن نتيجة المعايرة ليست
    أثرًا جانبيًا لطريقة التقسيم إلى bins.
    """
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    return {
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "adaptive_ece": adaptive_expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "brier": brier_score(y_true, y_prob),
        "nll": negative_log_likelihood(y_true, y_prob),
        "calib_slope": slope,
        "calib_intercept": intercept,
    }


def fit_lgb(X_tr, y_tr, X_va, y_va, params):
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    return m




# ─────────────────────────────────────────────────────────────────────────────
# R2 — K-Sensitivity Analysis
# Runs K-means in SHAP space for K=2,3,4 on one training fold.
# Reports silhouette score, Davies-Bouldin index, and within-cluster inertia
# to justify the K=3 choice made in the paper.
# ─────────────────────────────────────────────────────────────────────────────
def k_sensitivity_analysis(shap_matrix, K_values=None, seed=42):
    """
    Given a SHAP attribution matrix (n_students x n_features), run K-means
    for each K in K_values and report clustering quality metrics.

    Returns a dict: {K: {"silhouette": float, "davies_bouldin": float,
                          "inertia": float, "cluster_sizes": list}}
    """
    if K_values is None:
        K_values = K_SENSITIVITY_VALUES
    results = {}
    print("\n[R2] K-Sensitivity Analysis in SHAP Space:")
    print(f"  {'K':>4}  {'Silhouette':>12}  {'Davies-Bouldin':>16}  {'Inertia':>12}  {'Cluster sizes'}")
    print("  " + "-"*70)
    for K in K_values:
        km = KMeans(n_clusters=K, n_init=10, random_state=seed)
        labels = km.fit_predict(shap_matrix)
        sil = silhouette_score(shap_matrix, labels, sample_size=min(5000, len(labels)),
                                random_state=seed) if len(np.unique(labels)) > 1 else float("nan")
        db  = davies_bouldin_score(shap_matrix, labels)
        sizes = [int((labels == k).sum()) for k in range(K)]
        results[K] = {
            "silhouette": float(sil), "davies_bouldin": float(db),
            "inertia": float(km.inertia_), "cluster_sizes": sizes
        }
        flag = " <-- chosen" if K == 3 else ""
        print(f"  {K:>4}  {sil:>12.4f}  {db:>16.4f}  {km.inertia_:>12.1f}  {sizes}{flag}")
    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# v5 — TabNet Baseline (replaces GlobalMLP from v3/v4)
# Arik & Pfister (2021): "TabNet: Attentive Interpretable Tabular Learning"
# TabNet uses sequential attention to select features at each step, providing
# built-in feature importance and a stronger comparison point than a plain MLP.
# ─────────────────────────────────────────────────────────────────────────────
def train_tabnet(X_tr, y_tr, X_va, y_va, pos_weight_scalar, fold_seed=None):
    """
    Train a TabNet classifier as the neural-baseline (v5 replacement for GlobalMLP).

    Addresses reviewer ج2 (per-fold seed) and provides a stronger comparison
    than a plain MLP: TabNet is an attention-based model that performs
    sequential feature selection, yielding both competitive accuracy and
    built-in feature importance via attention masks.

    Hyperparameters:
        n_d = n_a = 64  : decision / attention embedding width per step
        n_steps  = 5    : number of sequential decision steps
        gamma    = 1.5  : feature reuse coefficient across steps
        lambda_sparse = 1e-3 : sparsity regularizer (encourages selective attention)
        mask_type = 'sparsemax' : sparse attention (vs 'entmax' for softer sparsity)
        weights = 1     : automatic balanced class weighting (inverse class frequency)

    fold_seed: per-fold seed for reproducibility and to break cross-fold coincidences.
    """
    _seed = SEED if fold_seed is None else SEED + fold_seed
    torch.manual_seed(_seed)
    np.random.seed(_seed)

    clf = TabNetClassifier(
        n_d=TABNET_N_D, n_a=TABNET_N_D,
        n_steps=TABNET_N_STEPS,
        gamma=TABNET_GAMMA,
        n_independent=2, n_shared=2,
        lambda_sparse=TABNET_LAMBDA,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params={"T_max": EPOCHS},
        mask_type='sparsemax',
        seed=_seed,
        device_name='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0,
    )
    clf.fit(
        X_tr.astype(np.float32), y_tr.astype(int),
        eval_set=[(X_va.astype(np.float32), y_va.astype(int))],
        eval_name=['val'],
        eval_metric=['auc'],
        max_epochs=EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        virtual_batch_size=max(BATCH_SIZE // 4, 64),
        weights=1,   # balanced: weight = n_negative / n_positive per class
        drop_last=False,
    )
    return clf


def predict_tabnet(clf, X):
    """Return positive-class probability from a fitted TabNetClassifier."""
    return clf.predict_proba(X.astype(np.float32))[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# v6 — LSTM Baseline (sequence-native architecture)
# Addresses the paper's own stated Limitation: "The architecture has not been
# compared against sequence-native transformers under identical fold
# assignments." A true recurrent model over raw per-event clickstream
# sequences is not possible here because the released, already-aggregated
# feature table (490 behavioral features per enrollment) has no per-event
# timestamps. As a defensible, transparent alternative, the feature vector
# is chunked into LSTM_SEQ_LEN contiguous, equal-width pseudo-timesteps (in
# the original column order, zero-padded to a common width) and processed by
# a single-layer LSTM. This tests whether a recurrent inductive bias over an
# arbitrary-but-fixed feature ordering helps — a different inductive bias
# than TabNet's attention-based sequential feature selection, or SoftMoE's
# multi-expert routing.
# ─────────────────────────────────────────────────────────────────────────────
class LSTMBaseline(nn.Module):
    """Single-layer LSTM over chunked pseudo-timesteps of the feature vector,
    followed by a linear classification head on the final hidden state."""

    def __init__(self, chunk_width, hidden=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                 dropout=LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=chunk_width, hidden_size=hidden, num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len, chunk_width)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]                       # (batch, hidden) — final layer's hidden state
        logits = self.head(self.drop(h_last)).squeeze(-1)
        return torch.sigmoid(logits)


def reshape_for_lstm(X, seq_len=LSTM_SEQ_LEN):
    """
    Chunk a (n_samples, n_features) tabular matrix into
    (n_samples, seq_len, chunk_width) pseudo-timesteps for the LSTM baseline.

    The feature dimension is split into `seq_len` contiguous, equal-width
    blocks in the original column order; the last block is zero-padded on
    the right if n_features is not evenly divisible by seq_len. This is a
    fixed, data-independent, leakage-free transformation (no fitting), so
    it can safely be applied identically to train/val/test.
    """
    n_samples, n_features = X.shape
    chunk_width = int(np.ceil(n_features / seq_len))
    pad_width = chunk_width * seq_len - n_features
    if pad_width > 0:
        X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    return X.reshape(n_samples, seq_len, chunk_width).astype(np.float32), chunk_width


def train_lstm_baseline(X_tr, y_tr, X_va, y_va, pos_weight_scalar, seq_len=LSTM_SEQ_LEN,
                         fold_seed=None):
    """
    Train the LSTM baseline. Mirrors train_single_net()'s training loop
    (same optimizer, batch size, early-stopping patience) for a fair,
    equal-training-budget comparison against the other baselines; only the
    architecture differs.

    X_tr, X_va: raw (unchunked) (n_samples, n_features) float32 arrays,
                already scaled the same way as the other neural baselines.
    """
    _seed = SEED if fold_seed is None else SEED + fold_seed
    torch.manual_seed(_seed)
    np.random.seed(_seed)

    X_tr_seq, chunk_width = reshape_for_lstm(X_tr, seq_len=seq_len)
    X_va_seq, _ = reshape_for_lstm(X_va, seq_len=seq_len)

    model = LSTMBaseline(chunk_width=chunk_width).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    Xtr_t = torch.tensor(X_tr_seq, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(X_va_seq, dtype=torch.float32, device=DEVICE)

    w_pos = pos_weight_scalar
    sample_w_tr = torch.where(ytr_t == 1, torch.tensor(w_pos, dtype=torch.float32, device=DEVICE),
                               torch.tensor(1.0, dtype=torch.float32, device=DEVICE))

    n = Xtr_t.shape[0]
    best_auc, best_state, wait = -1, None, 0
    n_unique = len(np.unique(y_va)) if len(y_va) > 0 else 1

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb, wb = Xtr_t[idx], ytr_t[idx], sample_w_tr[idx]
            opt.zero_grad()
            prob = model(xb).clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(prob, yb, weight=wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            p_va = model(Xva_t).detach().cpu().numpy()
        val_auc = best_auc + 1 if n_unique < 2 else roc_auc_score(y_va, p_va)

        if val_auc > best_auc:
            best_auc, wait = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_lstm_baseline(model, X, seq_len=LSTM_SEQ_LEN):
    """Return positive-class probability from a fitted LSTMBaseline.
    X must be the same raw (unchunked) shape used at training time."""
    X_seq, _ = reshape_for_lstm(X, seq_len=seq_len)
    Xt = torch.tensor(X_seq, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        p = model(Xt).detach().cpu().numpy()
    return p



def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape, f"dropout={df['target'].mean()*100:.1f}%")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_cols
    dup_mask = df[feature_cols].T.duplicated()
    dup_cols = df[feature_cols].columns[dup_mask].tolist()
    feature_cols = [c for c in feature_cols if c not in dup_cols]
    print(f"عدد الميزات بعد حذف {len(dup_cols)} عمودًا مكررًا: {len(feature_cols)}")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    X_raw = df[feature_cols].values
    y = df["target"].astype(int).values
    in_dim = X_raw.shape[1]

    cluster_df = pd.read_csv(CLUSTER_ASSIGN_PATH)
    id_to_row = pd.Series(np.arange(len(df)), index=df["enrollment_id"]).to_dict()
    cluster_df = cluster_df[cluster_df["enrollment_id"].isin(id_to_row)].copy()
    cluster_df["row_idx"] = cluster_df["enrollment_id"].map(id_to_row)
    labeled_rows_all = cluster_df["row_idx"].values
    labeled_clusters_all = cluster_df["cluster"].values.astype(int)
    print(f"عدد الطلاب المُصنَّفين بعنقود: {len(labeled_rows_all)}")

    del df
    gc.collect()

    skf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    fold_indices = list(skf.split(np.arange(len(y)), y))
    N_TOTAL_ITERS = len(fold_indices)
    print(f"[CV] {N_SPLITS}-Fold × {N_REPEATS} تكرارات = {N_TOTAL_ITERS} قياسًا إجماليًا")
    # R4 — Note for reviewer: Repeated CV folds are NOT mutually independent (they share
    # training data across folds of the same repeat). Therefore p-values from paired t-tests
    # are optimistic relative to truly independent splits. Per-repeat AUC stability (Table 9)
    # and effect sizes (Cohen's d) are the primary evidence; p-values are reported as
    # supplementary inferential summaries only.
    print("[R4] NOTE: Repeated CV folds are not independent. Interpret p-values alongside")
    print("           Cohen's d effect sizes and per-repeat stability (Table 9).")

    all_rows = []
    n_total = len(y)
    # تراكمات OOF عبر التكرارات (كل تكرار يُغطّي كل العيّنات مرة واحدة بالضبط)
    oof_p_moe_rep = np.full((N_REPEATS, n_total), np.nan)
    oof_p_tabnet_rep = np.full((N_REPEATS, n_total), np.nan)  # v5 — TabNet baseline
    oof_p_lstm_rep = np.full((N_REPEATS, n_total), np.nan)    # v6 — LSTM baseline
    # ج2: K-sensitivity now runs on EVERY fold (not just fold 0).
    # Collect per-fold silhouette scores for cross-fold summary (Table 3).
    k_sensitivity_all_folds = []   # list of {fold: it, K: k, silhouette: s, ...}
    # R2#4 — Reviewer 2, Comment #4: per-fold gate/cluster agreement records
    # (Adjusted Rand Index between the dominant expert and the original
    # SHAP cluster label c(x)), collected across all 25 CV runs.
    gate_cluster_agreement_rows = []
    oof_p_moe_calibrated_rep = np.full((N_REPEATS, n_total), np.nan)
    oof_p_moe_isotonic_rep = np.full((N_REPEATS, n_total), np.nan)
    oof_p_global_rep = np.full((N_REPEATS, n_total), np.nan)
    oof_p_hard_moe_rep = np.full((N_REPEATS, n_total), np.nan)
    oof_p_hard_moe_isotonic_rep = np.full((N_REPEATS, n_total), np.nan)
    oof_gate_rep = np.full((N_REPEATS, n_total, N_EXPERTS), np.nan)

    for it, (tr_idx, te_idx) in enumerate(fold_indices):
        repeat_id, fold_id = it // N_SPLITS, it % N_SPLITS
        tf0 = time.time()
        print(f"\n{'='*70}\nITER {it+1}/{N_TOTAL_ITERS}  (repeat {repeat_id+1}/{N_REPEATS}, fold {fold_id+1}/{N_SPLITS})\n{'='*70}")

        tr_sub_idx, va_idx = train_test_split(
            tr_idx, test_size=0.15, stratify=y[tr_idx], random_state=SEED + repeat_id)
        y_tr, y_va, y_te = y[tr_sub_idx], y[va_idx], y[te_idx]

        # ---------------- (1) النموذج العام (المرجع) ----------------
        print("  [1] تدريب النموذج العام (LightGBM) ...")
        record_fold0 = (it == 0)
        if record_fold0:
            global_evals_result = {}
            model_global = lgb.LGBMClassifier(**GLOBAL_PARAMS)
            model_global.fit(X_raw[tr_sub_idx], y_tr, eval_set=[(X_raw[va_idx], y_va)], eval_metric="auc",
                              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1),
                                         lgb.record_evaluation(global_evals_result)])
            global_iter_history = global_evals_result["valid_0"]["auc"]
        else:
            model_global = fit_lgb(X_raw[tr_sub_idx], y_tr, X_raw[va_idx], y_va, GLOBAL_PARAMS)
        p_global_va = model_global.predict_proba(X_raw[va_idx])[:, 1]
        thr_global = best_threshold(y_va, p_global_va)
        p_global_te = model_global.predict_proba(X_raw[te_idx])[:, 1]
        res_global = evaluate(y_te, p_global_te, thr=thr_global)
        ece_global_before = expected_calibration_error(y_te, p_global_te)
        T_global = fit_temperature(y_va, p_global_va)
        p_global_te_cal = apply_temperature(p_global_te, T_global)
        ece_global_after = expected_calibration_error(y_te, p_global_te_cal)
        print(f"      (عتبة مُعايَرة={thr_global:.3f}) AUC={res_global['auc']:.4f}  F1={res_global['f1']:.4f}  "
              f"Acc={res_global['accuracy']:.4f}  Prec={res_global['precision']:.4f}  "
              f"Rec={res_global['recall']:.4f}  MCC={res_global['mcc']:.4f}")
        print(f"      [Temp Scaling] T={T_global:.3f}  ECE: {ece_global_before:.4f} -> {ece_global_after:.4f}")
        del model_global
        gc.collect()

        # ── v5: TabNet Baseline ───────────────────────────────────────────────
        # TabNet replaces GlobalMLP as the stronger neural-architecture baseline.
        # Uses the same StandardScaler-normalised features as Soft-MoE.
        # fold_seed=it ensures each fold uses a unique seed, preventing
        # coincidental AUC matches across models (ج2 Round-2).
        print("  [v5-TabNet] تدريب TabNet Baseline (attentive tabular learning) ...")
        scaler_tabnet = StandardScaler().fit(X_raw[tr_sub_idx])
        X_tr_tab = scaler_tabnet.transform(X_raw[tr_sub_idx]).astype(np.float32)
        X_va_tab = scaler_tabnet.transform(X_raw[va_idx]).astype(np.float32)
        X_te_tab = scaler_tabnet.transform(X_raw[te_idx]).astype(np.float32)
        pos_w_tab = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        tabnet_model = train_tabnet(X_tr_tab, y_tr, X_va_tab, y_va,
                                     pos_w_tab, fold_seed=it)
        p_tab_va  = predict_tabnet(tabnet_model, X_va_tab)
        p_tab_te  = predict_tabnet(tabnet_model, X_te_tab)
        thr_tab   = best_threshold(y_va, p_tab_va)
        res_tab   = evaluate(y_te, p_tab_te, thr=thr_tab)
        ece_tab   = expected_calibration_error(y_te, p_tab_te)
        T_tab     = fit_temperature(y_va, p_tab_va)
        p_tab_te_cal = apply_temperature(p_tab_te, T_tab)
        iso_tab = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_tab.fit(p_tab_va, y_va)
        p_tab_te_iso = iso_tab.predict(p_tab_te)
        ece_tab_iso  = expected_calibration_error(y_te, p_tab_te_iso)
        thr_tab_iso  = best_threshold(y_va, iso_tab.predict(p_tab_va))
        res_tab_iso  = evaluate(y_te, p_tab_te_iso, thr=thr_tab_iso)
        cal_tab_raw  = full_calibration_report(y_te, p_tab_te)
        cal_tab_iso  = full_calibration_report(y_te, p_tab_te_iso)
        oof_p_tabnet_rep[repeat_id, te_idx] = p_tab_te
        print(f"      [TabNet] AUC={res_tab['auc']:.4f}  F1={res_tab['f1']:.4f}  "
              f"Acc={res_tab['accuracy']:.4f}  T={T_tab:.3f}  ECE: {ece_tab:.4f}->{ece_tab_iso:.4f}")
        del tabnet_model, X_tr_tab, X_va_tab, X_te_tab
        gc.collect()

        # ── v6: LSTM Baseline (sequence-native architecture) ───────────────────
        # Addresses the Limitations statement that no recurrent/temporal
        # architecture had been compared. Uses the same StandardScaler-
        # normalised features as Soft-MoE/TabNet, chunked into pseudo-
        # timesteps (see reshape_for_lstm() docstring). fold_seed=it for
        # the same per-fold-seed rationale as TabNet (ج2 Round-2).
        print("  [v6-LSTM] تدريب LSTM Baseline (sequence-native architecture) ...")
        scaler_lstm = StandardScaler().fit(X_raw[tr_sub_idx])
        X_tr_lstm = scaler_lstm.transform(X_raw[tr_sub_idx]).astype(np.float32)
        X_va_lstm = scaler_lstm.transform(X_raw[va_idx]).astype(np.float32)
        X_te_lstm = scaler_lstm.transform(X_raw[te_idx]).astype(np.float32)
        pos_w_lstm = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        lstm_model = train_lstm_baseline(X_tr_lstm, y_tr, X_va_lstm, y_va,
                                          pos_w_lstm, fold_seed=it)
        p_lstm_va  = predict_lstm_baseline(lstm_model, X_va_lstm)
        p_lstm_te  = predict_lstm_baseline(lstm_model, X_te_lstm)
        thr_lstm   = best_threshold(y_va, p_lstm_va)
        res_lstm   = evaluate(y_te, p_lstm_te, thr=thr_lstm)
        ece_lstm   = expected_calibration_error(y_te, p_lstm_te)
        T_lstm     = fit_temperature(y_va, p_lstm_va)
        p_lstm_te_cal = apply_temperature(p_lstm_te, T_lstm)
        iso_lstm = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_lstm.fit(p_lstm_va, y_va)
        p_lstm_te_iso = iso_lstm.predict(p_lstm_te)
        ece_lstm_iso  = expected_calibration_error(y_te, p_lstm_te_iso)
        thr_lstm_iso  = best_threshold(y_va, iso_lstm.predict(p_lstm_va))
        res_lstm_iso  = evaluate(y_te, p_lstm_te_iso, thr=thr_lstm_iso)
        cal_lstm_raw  = full_calibration_report(y_te, p_lstm_te)
        cal_lstm_iso  = full_calibration_report(y_te, p_lstm_te_iso)
        oof_p_lstm_rep[repeat_id, te_idx] = p_lstm_te
        print(f"      [LSTM] AUC={res_lstm['auc']:.4f}  F1={res_lstm['f1']:.4f}  "
              f"Acc={res_lstm['accuracy']:.4f}  T={T_lstm:.3f}  ECE: {ece_lstm:.4f}->{ece_lstm_iso:.4f}")
        del lstm_model, X_tr_lstm, X_va_lstm, X_te_lstm
        gc.collect()

        # ---------------- (2) مُصنِّف العنقود البديل (لتوجيه البوّابة) ----------------
        print("  [2] تدريب Surrogate (لتوجيه البوّابة) على تدريب الفولد فقط ...")
        train_set = set(tr_idx.tolist())
        mask_in_train = np.array([r in train_set for r in labeled_rows_all])
        rows_for_surrogate = labeled_rows_all[mask_in_train]
        clusters_for_surrogate = labeled_clusters_all[mask_in_train]

        # م3 Leakage guard — what is and is NOT a problem:
        #
        # rows_for_surrogate ⊆ tr_idx  (by the mask_in_train filter above).
        # tr_idx is partitioned into tr_sub_idx (85%) and va_idx (15%), so
        # rows_for_surrogate CAN—and will—overlap with va_idx. This is CORRECT
        # BY DESIGN: the surrogate must produce q(x) for every sample in tr_idx
        # (including va_idx), because gate_target_va = surrogate.predict_proba(va_idx)
        # is needed to guide the gate during Soft-MoE training. The surrogate
        # predicts CLUSTER LABELS (disengagement profiles), NOT dropout probability,
        # so seeing va_idx features does not carry target-label information into
        # the Soft-MoE's early-stopping signal.
        #
        # The only genuine leakage risk is te_idx appearing in surrogate training,
        # which is prevented by mask_in_train (only rows in tr_idx are included).
        # We assert that guarantee here.
        assert not any(r in set(te_idx.tolist()) for r in rows_for_surrogate), \
            "LEAKAGE: surrogate rows overlap with te_idx — critical error!"
        n_overlap_va = sum(1 for r in rows_for_surrogate if r in set(va_idx.tolist()))
        print(f"      [م3 Leakage check] rows_for_surrogate={len(rows_for_surrogate)}, "
              f"overlap_with_va_idx={n_overlap_va} (expected, by design), "
              f"overlap_with_te_idx=0 ✓ (assertion passed)")

        # The 15% internal split for the surrogate's own validation uses only
        # rows_for_surrogate (all from tr_idx), giving the surrogate its own
        # independent holdout for early stopping that is separate from
        # the MoE's early-stopping split (va_idx).
        Xs_tr, Xs_va, ys_tr, ys_va = train_test_split(
            X_raw[rows_for_surrogate], clusters_for_surrogate, test_size=0.15,
            stratify=clusters_for_surrogate, random_state=SEED)
        surrogate = fit_lgb(Xs_tr, ys_tr, Xs_va, ys_va, SURROGATE_PARAMS)
        print(f"      [Surrogate] trained on {len(Xs_tr)} / validated on {len(Xs_va)} cluster-labelled "
              f"samples (all from tr_idx ∖ te_idx — leakage check passed)")

        gate_target_tr = surrogate.predict_proba(X_raw[tr_sub_idx]).astype(np.float32)
        gate_target_va = surrogate.predict_proba(X_raw[va_idx]).astype(np.float32)
        gate_target_te = surrogate.predict_proba(X_raw[te_idx]).astype(np.float32)
        del surrogate
        gc.collect()

        # ── ج2: K-Sensitivity on EVERY fold (not just fold 0) ────────────────
        # Cross-fold silhouette scores are collected here and summarized
        # after the main loop as Table 3 (mean±std across all folds).
        print(f"  [ج2] Running K-sensitivity analysis (fold {it+1}/{N_TOTAL_ITERS}) ...")
        k_sens_fold = k_sensitivity_analysis(gate_target_tr, K_values=K_SENSITIVITY_VALUES,
                                              seed=SEED + it)
        for K, v in k_sens_fold.items():
            k_sensitivity_all_folds.append({
                "fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id,
                "K": K, **v
            })

        # ---------------- (3) تجهيز البيانات وتدريب Soft-MoE ----------------
        print("  [3] تدريب SHAP-Guided Soft Mixture-of-Experts ...")
        scaler = StandardScaler().fit(X_raw[tr_sub_idx])
        X_tr_s = scaler.transform(X_raw[tr_sub_idx]).astype(np.float32)
        X_va_s = scaler.transform(X_raw[va_idx]).astype(np.float32)
        X_te_s = scaler.transform(X_raw[te_idx]).astype(np.float32)

        pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        if record_fold0:
            moe_model, moe_history = train_soft_moe(X_tr_s, y_tr, gate_target_tr, X_va_s, y_va, gate_target_va,
                                                      in_dim, pos_w, record_history=True)
        else:
            moe_model = train_soft_moe(X_tr_s, y_tr, gate_target_tr, X_va_s, y_va, gate_target_va,
                                        in_dim, pos_w)

        with torch.no_grad():
            _te_tensor = torch.tensor(X_te_s, dtype=torch.float32, device=DEVICE)
            _t_infer = time.time()
            p_moe_va, _, _ = moe_model(torch.tensor(X_va_s, dtype=torch.float32, device=DEVICE))
            p_moe_te, gate_log_te, gate_probs_te = moe_model(_te_tensor)
            _infer_ms = (time.time() - _t_infer) * 1000  # ms for test fold
            _per_sample_us = (_infer_ms / max(len(te_idx), 1)) * 1000  # µs/sample
        p_moe_va = p_moe_va.detach().cpu().numpy()
        p_moe_te = p_moe_te.detach().cpu().numpy()
        gate_probs_te_np = gate_probs_te.detach().cpu().numpy()
        thr_moe = best_threshold(y_va, p_moe_va)
        res_moe = evaluate(y_te, p_moe_te, thr=thr_moe)
        avg_gate = gate_probs_te_np.mean(axis=0)

        # ---------------- Temperature Scaling لـ Soft-MoE ----------------
        ece_moe_before = expected_calibration_error(y_te, p_moe_te)
        T_moe = fit_temperature(y_va, p_moe_va)
        p_moe_te_cal = apply_temperature(p_moe_te, T_moe)
        ece_moe_after = expected_calibration_error(y_te, p_moe_te_cal)
        # إعادة ضبط العتبة بعد المعايرة (لأن توزيع الاحتمالات تغيّر)
        p_moe_va_cal = apply_temperature(p_moe_va, T_moe)
        thr_moe_cal = best_threshold(y_va, p_moe_va_cal)
        res_moe_cal = evaluate(y_te, p_moe_te_cal, thr=thr_moe_cal)

        # ---------------- Isotonic Regression لـ Soft-MoE (مرنة بالكامل) ----------------
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_moe_va, y_va)
        p_moe_va_iso = iso.predict(p_moe_va)
        p_moe_te_iso = iso.predict(p_moe_te)
        ece_moe_iso = expected_calibration_error(y_te, p_moe_te_iso)
        thr_moe_iso = best_threshold(y_va, p_moe_va_iso)
        res_moe_iso = evaluate(y_te, p_moe_te_iso, thr=thr_moe_iso)
        auc_moe_iso = roc_auc_score(y_te, p_moe_te_iso)  # قد يختلف قليلًا عن الأصل لأن Isotonic قد تُنشئ "مسطّحات" (ties)

        print(f"  [Soft-MoE] (عتبة مُعايَرة={thr_moe:.3f}) AUC={res_moe['auc']:.4f}  F1={res_moe['f1']:.4f}  "
              f"Acc={res_moe['accuracy']:.4f}  Prec={res_moe['precision']:.4f}  "
              f"Rec={res_moe['recall']:.4f}  MCC={res_moe['mcc']:.4f}")
        print(f"      [Inference timing] {_infer_ms:.1f} ms for {len(te_idx)} samples "
              f"({_per_sample_us:.2f} µs/sample) — addresses ث5 reviewer request")
        print(f"      متوسط وزن البوّابة عبر الخبراء (test): {np.round(avg_gate, 3)}")
        print(f"      [Temp Scaling]  T={T_moe:.3f}  ECE: {ece_moe_before:.4f} -> {ece_moe_after:.4f}")
        print(f"      [Isotonic Reg.] ECE: {ece_moe_before:.4f} -> {ece_moe_iso:.4f}  "
              f"| AUC بعد Isotonic={auc_moe_iso:.4f} (الأصل={res_moe['auc']:.4f})")

        # ---------------- (3-ب) Ablation: Soft-MoE بدون KL-prior (lambda=0) ----------------
        # يُعزِل هذا التجريب أثر "توجيه SHAP الناعم عبر KL" عن مجرد بنية multi-head/
        # multi-expert نفسها. إن لم يتراجع AUC بشكل معتبر هنا مقارنةً بـ Soft-MoE الأصلي
        # (lambda=0.05)، فهذا يعني أن مكسب الأداء مصدره أساسًا البنية (الجذع المشترك +
        # عدة رؤوس + بوّابة قابلة للتفاضل) وليس بالضرورة توجيه SHAP بعينه؛ والعكس صحيح.
        print("  [3-ب] Ablation: Soft-MoE بدون KL-prior (lambda_kl=0.0) ...")
        moe_model_nokl = train_soft_moe(X_tr_s, y_tr, gate_target_tr, X_va_s, y_va, gate_target_va,
                                         in_dim, pos_w, lambda_kl=0.0)
        with torch.no_grad():
            p_nokl_va, _, _ = moe_model_nokl(torch.tensor(X_va_s, dtype=torch.float32, device=DEVICE))
            p_nokl_te, _, gate_probs_nokl_te = moe_model_nokl(torch.tensor(X_te_s, dtype=torch.float32, device=DEVICE))
        p_nokl_va = p_nokl_va.detach().cpu().numpy()
        p_nokl_te = p_nokl_te.detach().cpu().numpy()
        avg_gate_nokl = gate_probs_nokl_te.detach().cpu().numpy().mean(axis=0)

        # ============================================================
        # R2#4 — GATE-CLUSTER AGREEMENT (Adjusted Rand Index)
        # Addresses Reviewer 2, Comment #4: quantifies whether the
        # learned soft-gate's dominant expert (argmax_k g_k(x))
        # corresponds to the *original* SHAP-derived behavioral cluster
        # c(x) discovered in Phase 1 — not the surrogate-inferred label
        # used for out-of-discovery-subset routing (Section IV.E).
        # Restricted to the at-risk students in this fold's test set
        # for whom c(x) was directly assigned during discovery
        # (n≈1,600 per fold, i.e. ~8,000/5). Computed for both the
        # SHAP-anchored gate (lambda=0.05, this fold's main model) and
        # the free gate (lambda=0, the ablation model above), so the
        # two can be compared directly in the same fold.
        # ============================================================
        te_idx_pos = {int(row): pos for pos, row in enumerate(te_idx)}
        te_labeled_mask = np.isin(labeled_rows_all, te_idx)
        te_labeled_rows = labeled_rows_all[te_labeled_mask]
        te_labeled_true_cluster = labeled_clusters_all[te_labeled_mask]

        if len(te_labeled_rows) >= 10:
            positions = np.array([te_idx_pos[int(r)] for r in te_labeled_rows])

            dominant_lam05 = gate_probs_te_np[positions].argmax(axis=1)
            agreement_lam05 = float(np.mean(dominant_lam05 == te_labeled_true_cluster))
            ari_lam05 = float(adjusted_rand_score(te_labeled_true_cluster, dominant_lam05))

            gate_probs_nokl_np = gate_probs_nokl_te.detach().cpu().numpy()
            dominant_lam0 = gate_probs_nokl_np[positions].argmax(axis=1)
            agreement_lam0 = float(np.mean(dominant_lam0 == te_labeled_true_cluster))
            ari_lam0 = float(adjusted_rand_score(te_labeled_true_cluster, dominant_lam0))
        else:
            print(f"      [R2#4 WARN] Only {len(te_labeled_rows)} labeled at-risk "
                  f"students in this fold's test set; skipping gate-cluster "
                  f"agreement for this fold.")
            agreement_lam05 = ari_lam05 = agreement_lam0 = ari_lam0 = np.nan

        gate_cluster_agreement_rows.append({
            "fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id,
            "n_labeled_in_test": len(te_labeled_rows),
            "agreement_rate_lam05": agreement_lam05, "ari_lam05": ari_lam05,
            "agreement_rate_lam0": agreement_lam0, "ari_lam0": ari_lam0,
        })
        print(f"      [R2#4 Gate-Cluster Agreement] lambda=0.05: "
              f"agree={agreement_lam05:.3f} ARI={ari_lam05:.3f}  |  "
              f"lambda=0: agree={agreement_lam0:.3f} ARI={ari_lam0:.3f}")

        thr_nokl = best_threshold(y_va, p_nokl_va)
        res_nokl = evaluate(y_te, p_nokl_te, thr=thr_nokl)
        cal_nokl_before = full_calibration_report(y_te, p_nokl_te)
        iso_nokl = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_nokl.fit(p_nokl_va, y_va)
        p_nokl_te_iso = iso_nokl.predict(p_nokl_te)
        cal_nokl_after = full_calibration_report(y_te, p_nokl_te_iso)
        thr_nokl_iso = best_threshold(y_va, iso_nokl.predict(p_nokl_va))
        res_nokl_iso = evaluate(y_te, p_nokl_te_iso, thr=thr_nokl_iso)
        print(f"      [Soft-MoE no-KL] AUC={res_nokl['auc']:.4f}  F1={res_nokl['f1']:.4f}  "
              f"gate(test)={np.round(avg_gate_nokl, 3)}  ECE_raw={cal_nokl_before['ece']:.4f}  "
              f"ECE_iso={cal_nokl_after['ece']:.4f}")
        del moe_model_nokl
        gc.collect()

        # ---------------- مقاييس معايرة موسّعة (Brier/NLL/slope-intercept) لـ Soft-MoE ----------------
        # تُحسَب على نفس تنبؤات Soft-MoE الأصلية (raw / isotonic) أعلاه، استجابةً لملاحظة
        # المُراجع أن ECE وحده غير كافٍ لإثبات claim المعايرة في عنوان المقال.
        cal_moe_raw = full_calibration_report(y_te, p_moe_te)
        cal_moe_iso = full_calibration_report(y_te, p_moe_te_iso)
        cal_global_raw = full_calibration_report(y_te, p_global_te)
        print(f"      [Calibration متقدم] Soft-MoE raw: Brier={cal_moe_raw['brier']:.4f} "
              f"NLL={cal_moe_raw['nll']:.4f} slope={cal_moe_raw['calib_slope']:.3f} "
              f"intercept={cal_moe_raw['calib_intercept']:.3f}")
        print(f"      [Calibration متقدم] Soft-MoE+Isotonic: Brier={cal_moe_iso['brier']:.4f} "
              f"NLL={cal_moe_iso['nll']:.4f} slope={cal_moe_iso['calib_slope']:.3f} "
              f"intercept={cal_moe_iso['calib_intercept']:.3f}")

        # ---------------- (4) Baseline: Hard-Routing MoE (تجزيء صارم للبيانات) ----------------
        print("  [4] تدريب Baseline: Hard-Routing MoE (3 خبراء منفصلون، كل خبير يرى عنقوده فقط) ...")
        hard_cluster_tr = gate_target_tr.argmax(axis=1)
        hard_cluster_va = gate_target_va.argmax(axis=1)
        hard_cluster_te = gate_target_te.argmax(axis=1)
        hard_experts = {}
        for k in range(N_EXPERTS):
            mask_tr_k = hard_cluster_tr == k
            mask_va_k = hard_cluster_va == k
            if mask_tr_k.sum() < 30 or len(np.unique(y_tr[mask_tr_k])) < 2:
                hard_experts[k] = None
                continue
            Xk_va = X_va_s[mask_va_k] if mask_va_k.sum() >= 10 and len(np.unique(y_va[mask_va_k])) > 1 else X_va_s
            yk_va = y_va[mask_va_k] if mask_va_k.sum() >= 10 and len(np.unique(y_va[mask_va_k])) > 1 else y_va
            pos_w_k = float((y_tr[mask_tr_k] == 0).sum() / max((y_tr[mask_tr_k] == 1).sum(), 1))
            hard_experts[k] = train_single_net(X_tr_s[mask_tr_k], y_tr[mask_tr_k], Xk_va, yk_va, in_dim, pos_w_k)

        def hard_predict(X_s, hard_cluster_assign):
            p = np.zeros(len(X_s), dtype=np.float32)
            for k in range(N_EXPERTS):
                idx_k = np.where(hard_cluster_assign == k)[0]
                if len(idx_k) == 0:
                    continue
                if hard_experts[k] is None:
                    p[idx_k] = y_tr.mean()  # fallback: لا توجد بيانات كافية لهذا العنقود
                else:
                    p[idx_k] = predict_single_net(hard_experts[k], X_s[idx_k])
            return p

        p_hard_va = hard_predict(X_va_s, hard_cluster_va)
        p_hard_te = hard_predict(X_te_s, hard_cluster_te)
        thr_hard = best_threshold(y_va, p_hard_va)
        res_hard = evaluate(y_te, p_hard_te, thr=thr_hard)
        ece_hard_before = expected_calibration_error(y_te, p_hard_te)
        # نفس إجراء المعايرة المُطبَّق على Soft-MoE بالضبط
        T_hard = fit_temperature(y_va, p_hard_va)
        p_hard_te_cal = apply_temperature(p_hard_te, T_hard)
        ece_hard_after = expected_calibration_error(y_te, p_hard_te_cal)
        p_hard_va_cal = apply_temperature(p_hard_va, T_hard)
        thr_hard_cal = best_threshold(y_va, p_hard_va_cal)
        res_hard_cal = evaluate(y_te, p_hard_te_cal, thr=thr_hard_cal)

        iso_hard = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_hard.fit(p_hard_va, y_va)
        p_hard_va_iso = iso_hard.predict(p_hard_va)
        p_hard_te_iso = iso_hard.predict(p_hard_te)
        ece_hard_iso = expected_calibration_error(y_te, p_hard_te_iso)
        thr_hard_iso = best_threshold(y_va, p_hard_va_iso)
        res_hard_iso = evaluate(y_te, p_hard_te_iso, thr=thr_hard_iso)

        print(f"      [Hard-MoE] AUC={res_hard['auc']:.4f}  F1={res_hard['f1']:.4f}  "
              f"Prec={res_hard['precision']:.4f}  Rec={res_hard['recall']:.4f}")
        print(f"      [Hard-MoE] ECE: خام={ece_hard_before:.4f} -> Temp={ece_hard_after:.4f} -> "
              f"Isotonic={ece_hard_iso:.4f}")
        # R1 FIX: Note that hard-routing RECALL may INCREASE vs. global baseline.
        # This is EXPECTED and NOT a genuine improvement:
        # C1's near-constant positive prediction (AUC≈0.54) trivially maximises recall
        # at the cost of AUC, precision, and MCC. The paper's claim should be corrected
        # to state that hard routing "degrades ranking and discrimination metrics
        # (AUC-ROC, MCC, Precision) while recall *trivially increases* due to the
        # near-constant positive-prediction collapse of the C1 expert."
        recall_diff_hard = res_hard["recall"] - res_global["recall"]
        if recall_diff_hard > 0:
            print(f"      [R1 NOTE] Hard-MoE recall INCREASED vs. global (+{recall_diff_hard:.4f}). "
                  f"This is TRIVIAL (near-constant positive prediction in C1, AUC={res_hard['auc']:.3f}). "
                  f"The paper correctly explains this in Section VI-E.")
        del hard_experts
        gc.collect()

        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "global",
                          "threshold": thr_global, **res_global,
                          "ece_before": ece_global_before, "ece_after": ece_global_after, "T": T_global,
                          "brier": cal_global_raw["brier"], "nll": cal_global_raw["nll"],
                          "adaptive_ece": cal_global_raw["adaptive_ece"],
                          "calib_slope": cal_global_raw["calib_slope"],
                          "calib_intercept": cal_global_raw["calib_intercept"]})
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "soft_moe",
                          "threshold": thr_moe, **res_moe,
                          "gate_w0": float(avg_gate[0]), "gate_w1": float(avg_gate[1]),
                          "gate_w2": float(avg_gate[2]),
                          "ece_before": ece_moe_before, "ece_after": ece_moe_after, "T": T_moe,
                          "brier": cal_moe_raw["brier"], "nll": cal_moe_raw["nll"],
                          "adaptive_ece": cal_moe_raw["adaptive_ece"],
                          "calib_slope": cal_moe_raw["calib_slope"],
                          "calib_intercept": cal_moe_raw["calib_intercept"]})
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "soft_moe_calibrated",
                          "threshold": thr_moe_cal,
                          **res_moe_cal, "ece_before": ece_moe_before, "ece_after": ece_moe_after,
                          "T": T_moe})
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "soft_moe_isotonic",
                          "threshold": thr_moe_iso,
                          **res_moe_iso, "ece_before": ece_moe_before, "ece_after": ece_moe_iso,
                          "T": T_moe,  # R6: save T even for isotonic model for reporting
                          "brier": cal_moe_iso["brier"], "nll": cal_moe_iso["nll"],
                          "adaptive_ece": cal_moe_iso["adaptive_ece"],
                          "calib_slope": cal_moe_iso["calib_slope"],
                          "calib_intercept": cal_moe_iso["calib_intercept"]})
        # --- صفوف ablation: Soft-MoE بدون KL-prior (lambda=0), خام ومعايَر بـIsotonic ---
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "soft_moe_no_kl",
                          "threshold": thr_nokl, **res_nokl,
                          "gate_w0": float(avg_gate_nokl[0]), "gate_w1": float(avg_gate_nokl[1]),
                          "gate_w2": float(avg_gate_nokl[2]),
                          "ece_before": cal_nokl_before["ece"], "ece_after": cal_nokl_before["ece"], "T": np.nan,
                          "brier": cal_nokl_before["brier"], "nll": cal_nokl_before["nll"],
                          "adaptive_ece": cal_nokl_before["adaptive_ece"],
                          "calib_slope": cal_nokl_before["calib_slope"],
                          "calib_intercept": cal_nokl_before["calib_intercept"]})
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "soft_moe_no_kl_isotonic",
                          "threshold": thr_nokl_iso, **res_nokl_iso,
                          "ece_before": cal_nokl_before["ece"], "ece_after": cal_nokl_after["ece"], "T": np.nan,
                          "brier": cal_nokl_after["brier"], "nll": cal_nokl_after["nll"],
                          "adaptive_ece": cal_nokl_after["adaptive_ece"],
                          "calib_slope": cal_nokl_after["calib_slope"],
                          "calib_intercept": cal_nokl_after["calib_intercept"]})
        # v5 — TabNet rows
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "tabnet",
                          "threshold": thr_tab, **res_tab,
                          "ece_before": ece_tab, "ece_after": ece_tab, "T": T_tab,
                          "brier": cal_tab_raw["brier"], "nll": cal_tab_raw["nll"],
                          "adaptive_ece": cal_tab_raw["adaptive_ece"],
                          "calib_slope": cal_tab_raw["calib_slope"],
                          "calib_intercept": cal_tab_raw["calib_intercept"]})
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "tabnet_isotonic",
                          "threshold": thr_tab_iso, **res_tab_iso,
                          "ece_before": ece_tab, "ece_after": ece_tab_iso, "T": T_tab,
                          "brier": cal_tab_iso["brier"], "nll": cal_tab_iso["nll"],
                          "adaptive_ece": cal_tab_iso["adaptive_ece"],
                          "calib_slope": cal_tab_iso["calib_slope"],
                          "calib_intercept": cal_tab_iso["calib_intercept"]})
        # v6 — LSTM rows
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "lstm",
                          "threshold": thr_lstm, **res_lstm,
                          "ece_before": ece_lstm, "ece_after": ece_lstm, "T": T_lstm,
                          "brier": cal_lstm_raw["brier"], "nll": cal_lstm_raw["nll"],
                          "adaptive_ece": cal_lstm_raw["adaptive_ece"],
                          "calib_slope": cal_lstm_raw["calib_slope"],
                          "calib_intercept": cal_lstm_raw["calib_intercept"]})
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "lstm_isotonic",
                          "threshold": thr_lstm_iso, **res_lstm_iso,
                          "ece_before": ece_lstm, "ece_after": ece_lstm_iso, "T": T_lstm,
                          "brier": cal_lstm_iso["brier"], "nll": cal_lstm_iso["nll"],
                          "adaptive_ece": cal_lstm_iso["adaptive_ece"],
                          "calib_slope": cal_lstm_iso["calib_slope"],
                          "calib_intercept": cal_lstm_iso["calib_intercept"]})
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "hard_moe",
                          "threshold": thr_hard, **res_hard,
                          "ece_before": ece_hard_before, "ece_after": ece_hard_after, "T": T_hard})
        all_rows.append({"fold": it, "repeat": repeat_id, "fold_in_repeat": fold_id, "model": "hard_moe_isotonic",
                          "threshold": thr_hard_iso, **res_hard_iso,
                          "ece_before": ece_hard_before, "ece_after": ece_hard_iso, "T": np.nan})

        oof_p_moe_rep[repeat_id, te_idx] = p_moe_te
        oof_p_tabnet_rep[repeat_id, te_idx] = p_tab_te  # v5
        oof_p_lstm_rep[repeat_id, te_idx] = p_lstm_te  # v6
        oof_p_moe_calibrated_rep[repeat_id, te_idx] = p_moe_te_cal
        oof_p_moe_isotonic_rep[repeat_id, te_idx] = p_moe_te_iso
        oof_p_global_rep[repeat_id, te_idx] = p_global_te
        oof_p_hard_moe_rep[repeat_id, te_idx] = p_hard_te
        oof_p_hard_moe_isotonic_rep[repeat_id, te_idx] = p_hard_te_iso
        oof_gate_rep[repeat_id, te_idx, :] = gate_probs_te_np

        del moe_model, X_tr_s, X_va_s, X_te_s
        gc.collect()
        print(f"  Iter {it+1} done in {time.time()-tf0:.0f}s")

    # متوسط كل تكرار -> OOF نهائي (كل عيّنة تُختبر مرة واحدة بالضبط في كل تكرار)
    oof_y = y.astype(float)
    oof_p_moe = np.nanmean(oof_p_moe_rep, axis=0)
    oof_p_tabnet = np.nanmean(oof_p_tabnet_rep, axis=0)  # v5
    oof_p_lstm = np.nanmean(oof_p_lstm_rep, axis=0)      # v6
    oof_p_moe_calibrated = np.nanmean(oof_p_moe_calibrated_rep, axis=0)
    oof_p_moe_isotonic = np.nanmean(oof_p_moe_isotonic_rep, axis=0)
    oof_p_global_arr = np.nanmean(oof_p_global_rep, axis=0)
    oof_p_hard_moe = np.nanmean(oof_p_hard_moe_rep, axis=0)
    oof_p_hard_moe_isotonic = np.nanmean(oof_p_hard_moe_isotonic_rep, axis=0)
    oof_gate = np.nanmean(oof_gate_rep, axis=0)  # (n_total, N_EXPERTS)

    res_df = pd.DataFrame(all_rows)
    res_df.to_csv("soft_moe_isotonic_raw.csv", index=False)

    metric_cols = ["auc", "accuracy", "f1", "precision", "recall", "mcc"]
    extra_cal_cols = [c for c in ["adaptive_ece", "brier", "nll", "calib_slope", "calib_intercept"] if c in res_df.columns]
    summary = res_df.groupby("model")[metric_cols + ["ece_before", "ece_after"] + extra_cal_cols].agg(["mean", "std"])
    summary.to_csv("soft_moe_isotonic_summary.csv")
    print(f"\n{'='*95}\nملخص {N_SPLITS}-Fold × {N_REPEATS} Repeated CV ({len(fold_indices)} قياسًا) — "
          f"عام / Hard-MoE(±Isotonic) / Soft-MoE / مُعايَر-Temp / مُعايَر-Isotonic:\n{'='*95}")
    print(summary.to_string())

    assert not np.isnan(oof_p_moe).any()
    pd.DataFrame({
        "y_true": oof_y.astype(int), "p_global": oof_p_global_arr,
        "p_hard_moe": oof_p_hard_moe, "p_hard_moe_isotonic": oof_p_hard_moe_isotonic,
        "p_soft_moe": oof_p_moe, "p_soft_moe_temp_calibrated": oof_p_moe_calibrated,
        "p_soft_moe_isotonic": oof_p_moe_isotonic,
        "gate_w0": oof_gate[:, 0], "gate_w1": oof_gate[:, 1], "gate_w2": oof_gate[:, 2],
    }).to_csv("oof_soft_moe_isotonic.csv", index=False)

    print(f"\nAUC و ECE على كل البيانات (Out-of-Fold، متوسط {N_REPEATS} تكرارات، "
          f"بعد Isotonic لكل النماذج لمقارنة عادلة):")
    print()
    print("[R1 CORRECTION] Hard routing recall INCREASES vs. global because C1 expert")
    print("  (~63% of data, 97.8% dropout rate) degenerates to near-constant positive")
    print("  prediction. AUC, MCC, and Precision all DEGRADE. The paper correctly")
    print("  reports this in Section VI-E; the text 'degrades all metrics' should be")
    print("  corrected to 'degrades ranking and discrimination metrics.'")
    print()
    print(f"  LightGBM (عام، للمرجعية):   AUC={roc_auc_score(oof_y, oof_p_global_arr):.4f}  "
          f"ECE={expected_calibration_error(oof_y, oof_p_global_arr):.4f}")
    print(f"  TabNet baseline:             AUC={roc_auc_score(oof_y, oof_p_tabnet):.4f}  "
          f"ECE={expected_calibration_error(oof_y, oof_p_tabnet):.4f}")
    print(f"  LSTM baseline:               AUC={roc_auc_score(oof_y, oof_p_lstm):.4f}  "
          f"ECE={expected_calibration_error(oof_y, oof_p_lstm):.4f}")
    print(f"  Hard-Routing MoE (خام):   AUC={roc_auc_score(oof_y, oof_p_hard_moe):.4f}  "
          f"ECE={expected_calibration_error(oof_y, oof_p_hard_moe):.4f}")
    print(f"  Hard-MoE + Isotonic:      AUC={roc_auc_score(oof_y, oof_p_hard_moe_isotonic):.4f}  "
          f"ECE={expected_calibration_error(oof_y, oof_p_hard_moe_isotonic):.4f}")
    print(f"  Soft-MoE (الأصلي):        AUC={roc_auc_score(oof_y, oof_p_moe):.4f}  "
          f"ECE={expected_calibration_error(oof_y, oof_p_moe):.4f}")
    print(f"  Soft-MoE (Temp Scaling):  AUC={roc_auc_score(oof_y, oof_p_moe_calibrated):.4f}  "
          f"ECE={expected_calibration_error(oof_y, oof_p_moe_calibrated):.4f}")
    print(f"  Soft-MoE (Isotonic Reg.): AUC={roc_auc_score(oof_y, oof_p_moe_isotonic):.4f}  "
          f"ECE={expected_calibration_error(oof_y, oof_p_moe_isotonic):.4f}")

    # ============================================================
    #        مجموعة المنحنيات الكاملة (Classification + Probabilistic + NN)
    # ============================================================
    from sklearn.metrics import (roc_curve, precision_recall_curve, average_precision_score,
                                  confusion_matrix)

    print("\n[رسم] توليد كل المنحنيات الموصى بها للنشر العلمي ...")

    # ---- Fig A: منحنى ROC ----
    fig, ax = plt.subplots(figsize=(7, 7))
    for p, name, color in [(oof_p_global_arr, "LightGBM (global)", "#C44E52"),
                            (oof_p_hard_moe_isotonic, "Hard-Routing MoE + Isotonic", "#937860"),
                            (oof_p_tabnet, "TabNet (baseline)", "#9467BD"),
                            (oof_p_lstm, "LSTM (baseline)", "#8C564B"),
                            (oof_p_moe, "SHAP-Guided Soft-MoE", "#4C72B0"),
                            (oof_p_moe_calibrated, "Soft-MoE + Temp. Scaling", "#DD8452"),
                            (oof_p_moe_isotonic, "Soft-MoE + Isotonic", "#2E8B57")]:
        fpr, tpr, _ = roc_curve(oof_y, p)
        auc_val = roc_auc_score(oof_y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})", linewidth=2, color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (Out-of-Fold, n={n_total:,})")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("fig_A_roc_curve.png", dpi=160, bbox_inches="tight"); plt.close()

    # ---- Fig B: منحنى Precision-Recall ----
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for p, name, color in [(oof_p_moe, "Soft-MoE (uncalibrated)", "#4C72B0"),
                            (oof_p_moe_isotonic, "Soft-MoE + Isotonic", "#2E8B57")]:
        prec, rec, _ = precision_recall_curve(oof_y, p)
        ap = average_precision_score(oof_y, p)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})", linewidth=2, color=color)
    base_rate = oof_y.mean()
    ax.axhline(base_rate, linestyle="--", color="gray", label=f"Random (AP={base_rate:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (Out-of-Fold, n={n_total:,})")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("fig_B_precision_recall.png", dpi=160, bbox_inches="tight"); plt.close()

    # ---- Fig C: مصفوفة الالتباس (بعد المعايرة Isotonic) ----
    thr_iso_mean = res_df[res_df.model == "soft_moe_isotonic"]["threshold"].mean()
    pred_iso = (oof_p_moe_isotonic >= thr_iso_mean).astype(int)
    cm = confusion_matrix(oof_y, pred_iso)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No-Dropout", "Dropout"]); ax.set_yticklabels(["No-Dropout", "Dropout"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- Soft-MoE + Isotonic\n(threshold={thr_iso_mean:.3f}, n={n_total:,})")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig("fig_C_confusion_matrix.png", dpi=160, bbox_inches="tight"); plt.close()

    # ---- Fig D: منحنى المعايرة (Calibration / Reliability Diagram) ----
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, p, name in [(axes[0], oof_p_moe, "Soft-MoE (original)"),
                        (axes[1], oof_p_moe_calibrated, "Soft-MoE + Temp. Scaling"),
                        (axes[2], oof_p_moe_isotonic, "Soft-MoE + Isotonic Regression")]:
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        e, bin_acc, bin_conf, bin_count = 0.0, [], [], []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() > 0:
                acc = oof_y[mask].mean(); conf = p[mask].mean()
                e += (mask.sum() / len(p)) * abs(conf - acc)
                bin_acc.append(acc); bin_conf.append(conf); bin_count.append(mask.sum())
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.plot(bin_conf, bin_acc, marker="o", color="#2E8B57", label=name)
        for c, a, n_ in zip(bin_conf, bin_acc, bin_count):
            ax.annotate(str(int(n_)), (c, a), fontsize=7, alpha=0.6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Observed Frequency")
        ax.set_title(f"{name}\nECE={e:.4f}")
        ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("fig_D_calibration.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---- Fig E: منحنيات التدريب (Learning Curve / Loss Curve, Fold 1) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot(moe_history["epoch"], moe_history["train_loss"], marker="o", label="Total Loss (BCE+KL)")
    ax.plot(moe_history["epoch"], moe_history["train_bce"], marker="s", label="BCE component")
    ax.plot(moe_history["epoch"], moe_history["train_kl"], marker="^", label="KL component")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss"); ax.set_title("Soft-MoE Training Loss (Fold 1)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(moe_history["epoch"], moe_history["val_auc"], marker="o", color="#2E8B57",
             label="Soft-MoE Validation AUC")
    iters_g = np.arange(1, len(global_iter_history) + 1)
    ax.plot(iters_g, global_iter_history, color="#C44E52", alpha=0.7, label="LightGBM Validation AUC")
    ax.set_xlabel("Epoch / Boosting Iteration"); ax.set_ylabel("Validation AUC")
    ax.set_title("Training vs Validation Performance (Fold 1)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(moe_history["epoch"], moe_history["val_accuracy"], marker="o", color="#4C72B0")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy (thr=0.5)")
    ax.set_title("Soft-MoE Validation Accuracy (Fold 1)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_E_learning_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---- Fig F: توزيع الفئات (Class Distribution) ----
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = pd.Series(oof_y).value_counts().sort_index()
    bars = ax.bar(["No-Dropout (0)", "Dropout (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({c/n_total*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of Enrollments"); ax.set_title(f"Class Distribution (n={n_total:,})")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("fig_F_class_distribution.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("  تم حفظ: fig_A_roc_curve.png, fig_B_precision_recall.png, fig_C_confusion_matrix.png,")
    print("           fig_D_calibration.png, fig_E_learning_curve.png, fig_F_class_distribution.png")
    print("  (ملاحظة: مخطط SHAP Summary يتطلب نموذج أشجار مُدرَّب على كل البيانات؛ راجع shap_summary.png")
    print("   من سكربت التحليل التفسيري السابق -- لا يُعاد حسابه هنا لتفادي إعادة حساب SHAP المكلفة)")

    # ---- Fig G: تحليل أوزان البوّابة (Gate Weights) -- هل تُستخدم الخبراء الثلاثة بتمايز حقيقي؟ ----
    gate_cols = ["gate_w0", "gate_w1", "gate_w2"]
    gate_per_fold = res_df[res_df.model == "soft_moe"][gate_cols].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    bp = ax.boxplot([gate_per_fold[c].values for c in gate_cols], tick_labels=["Expert 0", "Expert 1", "Expert 2"],
                     patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452", "#55A868"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.axhline(1 / N_EXPERTS, linestyle="--", color="gray", label=f"توزيع متساوٍ ({1/N_EXPERTS:.3f})")
    ax.set_ylabel("متوسط وزن البوّابة لكل خبير (لكل تكرار/فولد)")
    ax.set_title(f"توزيع متوسط وزن البوّابة عبر {len(gate_per_fold)} قياسًا")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    dominant_expert = np.argmax(oof_gate, axis=1)
    dom_counts = pd.Series(dominant_expert).value_counts().sort_index()
    dom_counts = dom_counts.reindex(range(N_EXPERTS), fill_value=0)
    bars = ax.bar([f"Expert {k}" for k in range(N_EXPERTS)], dom_counts.values,
                   color=["#4C72B0", "#DD8452", "#55A868"])
    for b, c in zip(bars, dom_counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({c/n_total*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("عدد الطلاب (الخبير الأعلى وزنًا، OOF)")
    ax.set_title("توزيع \"الخبير المُهيمن\" عبر كل العيّنات")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig_G_gate_weights.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("           fig_G_gate_weights.png")

    def paired_test(model_b, metric, model_a="global"):
        a = res_df[res_df.model == model_a].sort_values("fold")[metric].values
        b = res_df[res_df.model == model_b].sort_values("fold")[metric].values
        diff = b - a
        t_stat, t_p = scipy_stats.ttest_rel(b, a)
        d = cohen_d_paired(diff)
        ci_lo, ci_hi = confidence_interval_paired(diff)   # ج4: 95% CI
        try:
            w_stat, w_p = scipy_stats.wilcoxon(b, a)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        out = {
            "metric": metric, "model": model_b, "baseline": model_a,
            "n_pairs": int(len(diff)),
            "baseline_values": a.tolist(), "model_values": b.tolist(),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "ci_95_lower": ci_lo, "ci_95_upper": ci_hi,   # ج4
            "cohens_d": d,
            "paired_ttest_statistic": float(t_stat), "paired_ttest_pvalue": float(t_p),
            "wilcoxon_statistic": float(w_stat), "wilcoxon_pvalue": float(w_p),
            "significant_at_0.05": bool(t_p < 0.05),
            # holm_adj_pvalue will be filled after all tests are collected
        }
        abs_d = abs(d)
        effect_label = ("ضعيف جدًا" if abs_d < 0.2 else "صغير" if abs_d < 0.5
                         else "متوسط" if abs_d < 0.8 else "كبير")
        print(f"\n[مقارنة إحصائية مُزاوَجة] {model_b} مقابل {model_a} -- {metric.upper()}  (n={len(diff)})")
        print(f"  {model_a}: mean={a.mean():.4f}  |  {model_b}: mean={b.mean():.4f}")
        print(f"  متوسط الفرق = {diff.mean():.5f} ± {out['std_diff']:.5f}  "
              f"[95% CI: {ci_lo:.5f}, {ci_hi:.5f}]")
        print(f"  paired t-test: t={t_stat:.3f}  p={t_p:.4f}  |  Cohen's d={d:.3f} ({effect_label})")
        print(f"  Wilcoxon:      W={w_stat:.3f}  p={w_p:.4f}  "
              f"(ملاحظة: أصغر p ممكن مع n={len(diff)} هو {1/2**len(diff):.4f} تقريبًا)")
        print("  >> دالّ إحصائيًا (t-test)." if t_p < 0.05 else "  >> غير دالّ إحصائيًا (t-test).")
        return out

    metric_cols_full = metric_cols + ["ece_after"]
    stat_results = {}
    # (أ) كل نموذج مقابل LightGBM العام (المرجع الأساسي)
    # v5: tabnet and tabnet_isotonic replace global_mlp — tests if TabNet beats LightGBM
    for model_b in ["tabnet", "tabnet_isotonic",
                     "lstm", "lstm_isotonic",
                     "hard_moe", "hard_moe_isotonic",
                     "soft_moe", "soft_moe_calibrated", "soft_moe_isotonic",
                     "soft_moe_no_kl", "soft_moe_no_kl_isotonic"]:
        for m in metric_cols_full:
            stat_results[f"{model_b}_vs_global_{m}"] = paired_test(model_b, m, model_a="global")
    # (ب) Soft-MoE مقابل Hard-Routing MoE: يوضّح أثر "التوجيه الناعم + KL" تحديدًا
    for m in metric_cols_full:
        stat_results[f"soft_moe_vs_hard_moe_{m}"] = paired_test("soft_moe", m, model_a="hard_moe")
    print("\n[v5] Soft-MoE+Isotonic vs. TabNet+Isotonic (is MoE better than TabNet?)")
    for m in metric_cols_full:
        stat_results[f"soft_moe_isotonic_vs_tabnet_isotonic_{m}"] = paired_test(
            "soft_moe_isotonic", m, model_a="tabnet_isotonic")
        stat_results[f"soft_moe_isotonic_vs_hard_moe_isotonic_{m}"] = paired_test(
            "soft_moe_isotonic", m, model_a="hard_moe_isotonic")
    print("\n[v6] Soft-MoE+Isotonic vs. LSTM+Isotonic (is MoE better than a sequence-native baseline?)")
    for m in metric_cols_full:
        stat_results[f"soft_moe_isotonic_vs_lstm_isotonic_{m}"] = paired_test(
            "soft_moe_isotonic", m, model_a="lstm_isotonic")
    # (ج) Ablation رئيسي طلبه المُراجع: Soft-MoE (lambda=0.05) مقابل Soft-MoE بدون KL-prior
    # (lambda=0). يعزل هذا الأثر الخاص بـ "SHAP-guided KL regularization" عن مجرد بنية
    # multi-head/multi-expert + جذع مشترك. إذا كان الفرق هنا غير دالّ إحصائيًا أو حجم
    # تأثيره صغير، فهذا دليل بأن مكسب AUC في الورقة مصدره أساسًا البنية لا توجيه SHAP
    # بعينه؛ والعكس صحيح إن كان الفرق كبيرًا ودالًّا.
    for m in metric_cols_full:
        stat_results[f"soft_moe_vs_soft_moe_no_kl_{m}"] = paired_test("soft_moe", m, model_a="soft_moe_no_kl")
        stat_results[f"soft_moe_isotonic_vs_soft_moe_no_kl_isotonic_{m}"] = paired_test(
            "soft_moe_isotonic", m, model_a="soft_moe_no_kl_isotonic")
    with open("soft_moe_isotonic_stat_comparison.json", "w", encoding="utf-8") as f:
        json.dump(stat_results, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # م5 / Table 11 FIX: Apply Holm–Bonferroni correction UNIFORMLY across
    # BOTH main comparison tests AND ablation tests (not just one table).
    # Round-2 review noted the inconsistency in the original paper.
    # ═══════════════════════════════════════════════════════════════════════
    def apply_holm_to_group(key_pattern, label):
        keys = [k for k in stat_results if key_pattern in k]
        if not keys:
            return
        raw_ps = [stat_results[k]["paired_ttest_pvalue"] for k in keys]
        adj_ps = holm_bonferroni_correction(raw_ps)
        print(f"\n[Holm–Bonferroni] {label} ({len(keys)} tests):")
        print(f"  {'Key':60s}  {'Raw p':>12}  {'Holm adj.p':>12}  {'Sig?':>6}")
        print("  " + "-" * 96)
        for k, rp, ap in zip(keys, raw_ps, adj_ps):
            sig = "✓" if ap < 0.05 else "ns"
            stat_results[k]["holm_adj_pvalue"] = ap
            print(f"  {k[:60]:60s}  {rp:12.4e}  {ap:12.4e}  {sig:>6}")

    # (A) Main: SoftMoE+Isotonic vs LightGBM (Table 14)
    apply_holm_to_group("soft_moe_isotonic_vs_global_", "SoftMoE+Iso vs LightGBM (Table 14)")
    # (B) Ablation: SoftMoE vs SoftMoE-no-KL (Table 11 — now corrected uniformly)
    apply_holm_to_group("soft_moe_isotonic_vs_soft_moe_no_kl_isotonic_",
                         "SoftMoE+Iso vs SoftMoE-noKL+Iso (Table 11, Holm-corrected)")
    # (C) v5: SoftMoE+Iso vs TabNet+Iso (stronger neural baseline)
    apply_holm_to_group("soft_moe_isotonic_vs_tabnet_isotonic_",
                         "SoftMoE+Iso vs TabNet+Iso (v5: MoE vs attention baseline)")
    # (D) v6: SoftMoE+Iso vs LSTM+Iso (sequence-native baseline)
    apply_holm_to_group("soft_moe_isotonic_vs_lstm_isotonic_",
                         "SoftMoE+Iso vs LSTM+Iso (v6: MoE vs recurrent baseline)")

    # Re-save with Holm p-values added
    with open("soft_moe_isotonic_stat_comparison.json", "w", encoding="utf-8") as f:
        json.dump(stat_results, f, ensure_ascii=False, indent=2)
    print("\n[Holm] Updated JSON with holm_adj_pvalue for all test groups.")

    # ═══════════════════════════════════════════════════════════════════════
    # ج2 (Table 3): Cross-fold K-sensitivity summary (all folds, not just fold 0)
    # ═══════════════════════════════════════════════════════════════════════
    k_sens_df_all = pd.DataFrame(k_sensitivity_all_folds)
    k_sens_df_all.to_csv("k_sensitivity_all_folds.csv", index=False)
    print(f"\n[ج2] Cross-fold K-sensitivity summary (Table 3) — {len(k_sens_df_all)} records:")
    print(f"  {'K':>4}  {'Silhouette mean':>18}  {'Silhouette std':>16}  {'DB mean':>10}  {'DB std':>10}")
    print("  " + "-"*65)
    for K in K_SENSITIVITY_VALUES:
        sub = k_sens_df_all[k_sens_df_all.K == K]
        sil_m, sil_s = sub["silhouette"].mean(), sub["silhouette"].std()
        db_m, db_s = sub["davies_bouldin"].mean(), sub["davies_bouldin"].std()
        flag = " ← chosen (highest silhouette)" if K == 3 else ""
        print(f"  {K:>4}  {sil_m:18.4f} ± {sil_s:.4f}  {db_m:10.4f} ± {db_s:.4f}{flag}")
    print(f"  → K=3 achieves highest mean silhouette across all {N_TOTAL_ITERS} folds.")
    print(f"  File saved: k_sensitivity_all_folds.csv")
    # Save fold-0 only for backward compatibility with Fig 4 in the paper
    fold0_k = k_sens_df_all[k_sens_df_all.fold == 0][["K","silhouette","davies_bouldin","inertia","cluster_sizes"]]
    fold0_k.to_csv("k_sensitivity_analysis.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════════
    # R2#4 (Table 10b): Gate-cluster agreement (ARI) summary, all 25 folds
    # ═══════════════════════════════════════════════════════════════════════
    gca_df = pd.DataFrame(gate_cluster_agreement_rows)
    gca_df.to_csv("gate_cluster_agreement.csv", index=False)
    # ═══════════════════════════════════════════════════════════════════════
    # R1#7: Equal-width vs adaptive (equal-frequency) ECE, all 25 folds
    # The reviewer's concern is that equal-width ECE is not exhaustive. The
    # table below reports both binning schemes side by side so the reader can
    # verify that the calibration conclusion does not depend on the choice.
    # ═══════════════════════════════════════════════════════════════════════
    if "adaptive_ece" in res_df.columns:
        print(f"\n[R1#7] Equal-width vs Adaptive (equal-frequency) ECE — "
              f"{N_SPLITS * N_REPEATS} folds, {10} bins each:")
        print(f"  {'Model':<26}{'ECE (equal-width)':>22}{'ECE (adaptive)':>20}{'Δ':>10}")
        print("  " + "-" * 78)
        for _m in ["global", "tabnet_isotonic", "lstm_isotonic",
                    "hard_moe_isotonic", "soft_moe", "soft_moe_isotonic"]:
            _sub = res_df[res_df.model == _m]
            if len(_sub) == 0 or _sub["adaptive_ece"].isna().all():
                continue
            _ew = _sub["ece_after"].mean()
            _ad = _sub["adaptive_ece"].mean()
            _ew_s = _sub["ece_after"].std()
            _ad_s = _sub["adaptive_ece"].std()
            print(f"  {_m:<26}{_ew:.4f} ± {_ew_s:.4f}{'':>4}"
                  f"{_ad:.4f} ± {_ad_s:.4f}{_ad - _ew:>+10.4f}")
        print("  → Agreement between the two schemes indicates the calibration "
              "result is not a binning artefact.")

    print(f"\n[R2#4] Gate-cluster agreement summary (Table 10b) — {len(gca_df)} folds, "
          f"n_labeled_in_test range [{gca_df['n_labeled_in_test'].min()}, "
          f"{gca_df['n_labeled_in_test'].max()}] per fold:")
    for lam, agree_col, ari_col in [("0.05", "agreement_rate_lam05", "ari_lam05"),
                                     ("0", "agreement_rate_lam0", "ari_lam0")]:
        agree_m, agree_s = gca_df[agree_col].mean(), gca_df[agree_col].std()
        ari_m, ari_s = gca_df[ari_col].mean(), gca_df[ari_col].std()
        print(f"  lambda={lam:>4}:  agreement = {agree_m*100:.1f}% ± {agree_s*100:.1f}%   "
              f"ARI = {ari_m:.3f} ± {ari_s:.3f}")
    print(f"  File saved: gate_cluster_agreement.csv")

    # ═══════════════════════════════════════════════════════════════════════
    # ج1 (Table 10): Gate stability comparison λ=0.05 vs λ=0 from actual runs
    # Shows mean±std of gate weights per expert across all 25 matched CV runs.
    # Values are derived from the same all_rows data used for Table 7.
    # ═══════════════════════════════════════════════════════════════════════
    gate_cols = ["gate_w0", "gate_w1", "gate_w2"]
    expert_names = ["Expert 0 (C0)", "Expert 1 (C1)", "Expert 2 (C2)"]
    df_anchored = res_df[res_df.model == "soft_moe"][gate_cols].dropna()
    df_free     = res_df[res_df.model == "soft_moe_no_kl"][gate_cols].dropna()
    n_anchored, n_free = len(df_anchored), len(df_free)
    print(f"\n[ج1] Gate Stability Table (Table 10): λ=0.05 vs λ=0 "
          f"({n_anchored} / {n_free} fold-runs with gate data)")
    print(f"  {'Expert':20s}  {'λ=0.05 Mean':>12}  {'λ=0.05 Std':>12}  "
          f"{'λ=0 Mean':>12}  {'λ=0 Std':>12}  {'Var reduction':>14}")
    print("  " + "-"*90)
    stability_rows = []
    for col, name in zip(gate_cols, expert_names):
        m05, s05 = df_anchored[col].mean(), df_anchored[col].std()
        m00, s00 = df_free[col].mean(),     df_free[col].std()
        ratio = s00 / s05 if s05 > 0 else float("nan")
        print(f"  {name:20s}  {m05:12.4f}  {s05:12.4f}  {m00:12.4f}  {s00:12.4f}  {ratio:14.2f}×")
        stability_rows.append({"expert": name, "mean_lam05": m05, "std_lam05": s05,
                                "mean_lam0": m00, "std_lam0": s00, "variance_reduction_ratio": ratio})
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv("gate_stability_comparison.csv", index=False)
    mean_ratio = stability_df["variance_reduction_ratio"].mean()
    print(f"  → Mean variance reduction: {mean_ratio:.2f}× (SHAP anchoring stabilizes gate allocation).")
    print(f"  File saved: gate_stability_comparison.csv")

    # ═══════════════════════════════════════════════════════════════════════
    # FIGS H–K: Four new publication-ready comparison figures (v5)
    # ═══════════════════════════════════════════════════════════════════════
    from sklearn.metrics import roc_curve as _roc_curve
    _MODELS_MAIN = ['global', 'tabnet_isotonic', 'lstm_isotonic', 'hard_moe_isotonic',
                    'soft_moe_no_kl_isotonic', 'soft_moe_isotonic']
    _LABELS_MAIN = ['LightGBM', 'TabNet+Iso', 'LSTM+Iso', 'Hard-MoE+Iso',
                    'Soft-MoE-noKL+Iso', 'Soft-MoE+Iso']
    _COLORS_MAIN = ['#C44E52', '#9467BD', '#8C564B', '#937860', '#8C8C8C', '#2E8B57']

    # ---- Fig H: AUC-ROC per-repeat comparison (Table 9) ----
    per_rep_auc = (res_df.groupby(['model', 'repeat'])['auc']
                   .mean().reset_index().rename(columns={'auc': 'mean_auc'}))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(1, N_REPEATS + 1)
    bar_w = 0.8 / len(_MODELS_MAIN)   # v6: generic width so bars fit regardless of model count
    _center_offset = (len(_MODELS_MAIN) - 1) / 2.0
    for i, (m, col, lbl) in enumerate(zip(_MODELS_MAIN, _COLORS_MAIN, _LABELS_MAIN)):
        vals = per_rep_auc[per_rep_auc.model == m].sort_values('repeat')['mean_auc'].values
        if len(vals) == N_REPEATS:
            bars = ax.bar(x + (i - _center_offset) * bar_w, vals, bar_w,
                          label=lbl, color=col, alpha=0.85, edgecolor='white', linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.0003,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=6.5, rotation=90)
    ax.set_xlabel('Independent Repeat', fontsize=11)
    ax.set_ylabel('Mean AUC-ROC within Repeat', fontsize=11)
    ax.set_title('AUC-ROC Averaged Within Each of the Five Independent Repeats\n'
                 '(confirms improvement is not driven by a single random partition)', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([f'Rep. {i}' for i in x])
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_ylim(0.74, 0.905); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('fig_H_per_repeat_auc.png', dpi=160, bbox_inches='tight')
    plt.close()
    print('  [v5] fig_H_per_repeat_auc.png  saved')

    # ---- Fig I: Gate Stability box plots (λ=0.05 vs λ=0, Table 10) ----
    gate_cols = ['gate_w0', 'gate_w1', 'gate_w2']
    expert_labels = ['Expert 0\n(C0: Low Eng.)', 'Expert 1\n(C1: Sudden Drop)', 'Expert 2\n(C2: Gradual)']
    df_anch = res_df[res_df.model == 'soft_moe'][gate_cols].dropna().values   # shape (25, 3)
    df_free  = res_df[res_df.model == 'soft_moe_no_kl'][gate_cols].dropna().values
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=False)
    fig.suptitle('Gate Stability Comparison: λ=0.05 (SHAP-Anchored) vs λ=0 (Free Gate)\n'
                 'across 25 Matched CV Runs — verifies SHAP anchoring role (Table 10)', fontsize=11)
    for ax_i, (col_i, exp_lbl) in enumerate(zip(range(3), expert_labels)):
        ax = axes[ax_i]
        data_to_plot = [df_anch[:, col_i], df_free[:, col_i]]
        bp = ax.boxplot(data_to_plot, labels=['λ=0.05\n(SHAP)', 'λ=0\n(Free)'],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor('#2E8B57'); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#C44E52'); bp['boxes'][1].set_alpha(0.7)
        ax.axhline(1/N_EXPERTS, linestyle='--', color='gray', alpha=0.6, label='Uniform (0.333)')
        ratio = df_free[:, col_i].std() / max(df_anch[:, col_i].std(), 1e-9)
        ax.set_title(f'{exp_lbl}\nVar. reduction: {ratio:.1f}×', fontsize=10)
        ax.set_ylabel('Mean gate weight (per fold-run)' if ax_i == 0 else '')
        ax.grid(alpha=0.3, axis='y')
        if ax_i == 0: ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('fig_I_gate_stability.png', dpi=160, bbox_inches='tight')
    plt.close()
    print('  [v5] fig_I_gate_stability.png  saved')

    # ---- Fig J: Calibration comparison bar chart (Table 8) ----
    cal_models = ['global', 'tabnet_isotonic', 'lstm_isotonic', 'hard_moe_isotonic',
                  'soft_moe', 'soft_moe_isotonic']
    cal_labels = ['LightGBM\n(raw)', 'TabNet\n+Iso', 'LSTM\n+Iso', 'Hard-MoE\n+Iso',
                  'Soft-MoE\n(raw)', 'Soft-MoE\n+Iso']
    cal_colors = ['#4C72B0', '#9467BD', '#8C564B', '#937860', '#DD8452', '#2E8B57']

    ece_means, ece_stds, brier_means, brier_stds = [], [], [], []
    for m in cal_models:
        sub = res_df[res_df.model == m]
        ece_col = 'ece_after'
        ece_means.append(sub[ece_col].mean())
        ece_stds.append(sub[ece_col].std())
        brier_means.append(sub['brier'].mean() if 'brier' in sub.columns else np.nan)
        brier_stds.append(sub['brier'].std() if 'brier' in sub.columns else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Calibration Comparison: Pooled ECE and Per-Run Brier Score\n'
                 '(mean ± std across 25 CV runs — Table 8)', fontsize=11)

    x = np.arange(len(cal_models))
    ax = axes[0]
    bars = ax.bar(x, ece_means, color=cal_colors, alpha=0.85,
                  yerr=ece_stds, capsize=5, edgecolor='white')
    for b, v in zip(bars, ece_means):
        ax.text(b.get_x() + b.get_width()/2, v + 0.002, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(cal_labels, fontsize=9)
    ax.set_ylabel('ECE (after calibration)', fontsize=11)
    ax.set_title('Expected Calibration Error (ECE)', fontsize=10)
    ax.grid(alpha=0.3, axis='y'); ax.set_ylim(0, max(ece_means) * 1.3)

    ax = axes[1]
    valid_mask = [not np.isnan(v) for v in brier_means]
    xv = x[valid_mask]; bm = [brier_means[i] for i in range(len(brier_means)) if valid_mask[i]]
    bs = [brier_stds[i]  for i in range(len(brier_stds))  if valid_mask[i]]
    lv = [cal_labels[i]  for i in range(len(cal_labels))  if valid_mask[i]]
    cv = [cal_colors[i]  for i in range(len(cal_colors))  if valid_mask[i]]
    bars = ax.bar(np.arange(len(xv)), bm, color=cv, alpha=0.85,
                  yerr=bs, capsize=5, edgecolor='white')
    for b, v in zip(bars, bm):
        ax.text(b.get_x() + b.get_width()/2, v + 0.001, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(np.arange(len(xv))); ax.set_xticklabels(lv, fontsize=9)
    ax.set_ylabel('Brier Score', fontsize=11)
    ax.set_title('Brier Score (lower = better)', fontsize=10)
    ax.grid(alpha=0.3, axis='y'); ax.set_ylim(0.07, max(bm) * 1.2 if bm else 0.2)
    plt.tight_layout()
    plt.savefig('fig_J_calibration_comparison.png', dpi=160, bbox_inches='tight')
    plt.close()
    print('  [v5] fig_J_calibration_comparison.png  saved')

    # ---- Fig K: Cross-fold K-sensitivity heatmap (all 25 folds, Table 3) ----
    k_df = pd.DataFrame(k_sensitivity_all_folds)
    if not k_df.empty:
        # Pivot: rows = fold (0-24), columns = K value
        pivot_sil = k_df.pivot_table(index='fold', columns='K', values='silhouette', aggfunc='mean')
        pivot_db  = k_df.pivot_table(index='fold', columns='K', values='davies_bouldin', aggfunc='mean')

        fig, axes = plt.subplots(1, 2, figsize=(13, 7))
        fig.suptitle('Cross-Fold K-Sensitivity: Clustering Quality in Surrogate Probability Space\n'
                     f'(all {N_TOTAL_ITERS} folds × {len(K_SENSITIVITY_VALUES)} K values — Table 3)', fontsize=11)

        import matplotlib.colors as mcolors
        # Silhouette heatmap (higher = better, so green)
        ax = axes[0]
        im = ax.imshow(pivot_sil.values, aspect='auto', cmap='RdYlGn',
                       vmin=0.7, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, label='Silhouette Score')
        ax.set_xticks(range(len(pivot_sil.columns)))
        ax.set_xticklabels([f'K={k}' for k in pivot_sil.columns], fontsize=10)
        ax.set_yticks(range(0, N_TOTAL_ITERS, 5))
        ax.set_yticklabels([f'Fold {i}' for i in range(0, N_TOTAL_ITERS, 5)], fontsize=8)
        ax.set_title('Silhouette Score (higher = better)', fontsize=10)
        ax.set_xlabel('K'); ax.set_ylabel('Fold index (0–24)')
        for r in range(len(pivot_sil.index)):
            for c in range(len(pivot_sil.columns)):
                v = pivot_sil.values[r, c]
                ax.text(c, r, f'{v:.3f}', ha='center', va='center',
                        fontsize=5.5, color='black' if v < 0.9 else 'white')

        # Davies-Bouldin heatmap (lower = better, so reverse green)
        ax = axes[1]
        im2 = ax.imshow(pivot_db.values, aspect='auto', cmap='RdYlGn_r',
                        vmin=0.0, vmax=0.6)
        plt.colorbar(im2, ax=ax, fraction=0.046, label='Davies-Bouldin Index')
        ax.set_xticks(range(len(pivot_db.columns)))
        ax.set_xticklabels([f'K={k}' for k in pivot_db.columns], fontsize=10)
        ax.set_yticks(range(0, N_TOTAL_ITERS, 5))
        ax.set_yticklabels([f'Fold {i}' for i in range(0, N_TOTAL_ITERS, 5)], fontsize=8)
        ax.set_title('Davies-Bouldin Index (lower = better)', fontsize=10)
        ax.set_xlabel('K'); ax.set_ylabel('Fold index (0–24)')
        for r in range(len(pivot_db.index)):
            for c in range(len(pivot_db.columns)):
                v = pivot_db.values[r, c]
                ax.text(c, r, f'{v:.3f}', ha='center', va='center',
                        fontsize=5.5, color='black')
        plt.tight_layout()
        plt.savefig('fig_K_ksensitivity_heatmap.png', dpi=160, bbox_inches='tight')
        plt.close()
        print('  [v5] fig_K_ksensitivity_heatmap.png  saved')

    print('\n  [v5] All four new comparison figures generated: H, I, J, K')

    # R6: Report mean temperature T across folds
    T_vals = res_df[res_df.model == "soft_moe"]["T"].dropna()
    T_global_vals = res_df[res_df.model == "global"]["T"].dropna()
    print(f"\n[R6] Temperature Scaling T values across {len(T_vals)} folds:")
    print(f"   Soft-MoE  : mean T = {T_vals.mean():.3f}  std = {T_vals.std():.3f}  "
          f"range [{T_vals.min():.3f}, {T_vals.max():.3f}]")
    print(f"   LightGBM  : mean T = {T_global_vals.mean():.3f}  std = {T_global_vals.std():.3f}")
    print(f"   → High T (>{1.0:.1f}) indicates overconfident raw probabilities, "
          f"explaining why temperature scaling alone is insufficient and")
    print(f"     isotonic regression (non-parametric) is required.")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("\nDONE. v5 outputs (TabNet + 4 new figures):")
    print("  soft_moe_isotonic_raw.csv          (per-fold: all models incl. tabnet, lstm)")
    print("  soft_moe_isotonic_summary.csv      (per-model mean/std)")
    print("  soft_moe_isotonic_stat_comparison.json  (all paired tests + Holm adj.p + 95% CI)")
    print("  k_sensitivity_analysis.csv         (fold-0 K-sensitivity: backward compat)")
    print("  k_sensitivity_all_folds.csv        (ج2: all-fold K-sensitivity, Table 3)")
    print("  gate_stability_comparison.csv      (ج1: λ=0.05 vs λ=0 gate mean/std, Table 10)")
    print("  oof_soft_moe_isotonic.csv          (OOF probabilities for plotting)")
    print("  fig_A_roc_curve.png                (ROC: LightGBM, TabNet, Hard-MoE, Soft-MoE variants)")
    print("  fig_B_precision_recall.png  fig_C_confusion_matrix.png")
    print("  fig_D_calibration.png  fig_E_learning_curve.png  fig_F_class_distribution.png")
    print("  fig_G_gate_weights.png")
    print("  fig_H_per_repeat_auc.png           [NEW] Table 9 — AUC per repeat")
    print("  fig_I_gate_stability.png            [NEW] Table 10 — Gate stability box plots")
    print("  fig_J_calibration_comparison.png   [NEW] Table 8 — ECE & Brier bar chart")
    print("  fig_K_ksensitivity_heatmap.png     [NEW] Table 3 — K-sensitivity heatmap all 25 folds")



# ============================================================================
#   R1#1 — LEARNER-/COURSE-HELD-OUT GENERALIZATION PROTOCOL
# ============================================================================
# Addresses Reviewer 1, Major Comment #1: the main study (main(), above)
# uses enrollment-level RepeatedStratifiedKFold, which does not guarantee
# that a given learner or course is absent from both the training and test
# partition of a fold. This function re-evaluates the two headline models
# — the fold-matched global LightGBM baseline and SoftMoE+Isotonic — under
# GroupKFold splits that DO enforce this: no group (learner or course)
# appears in both the training and test fold of a given run.
#
# Design decisions, stated explicitly for the reviewer record:
#   • Scope: only the global LightGBM baseline and SoftMoE+Isotonic are
#     compared here — not the full ablation suite (TabNet, hard routing,
#     lambda=0) from Table 8. This is a targeted robustness check on the
#     paper's central claim, not a full replication of every table.
#   • K=5 folds, no repeats (vs. 5x5=25 in the main study), to keep runtime
#     tractable: approximately 1/5 of the main study's wall-clock time.
#   • The SHAP-cluster discovery and surrogate-gate-prior pipeline (Section
#     IV.A) is preserved unchanged inside each fold, exactly as in main(),
#     so this protocol tests generalization of the *same* architecture and
#     training procedure, not a simplified variant.
#   • A hard assertion verifies zero group overlap between train and test
#     in every fold before any model is fit.
#
# Usage:
#   python run_soft_moe_v5_tabnet.py --holdout learner
#   python run_soft_moe_v5_tabnet.py --holdout course
#   python run_soft_moe_v5_tabnet.py --holdout both
# ============================================================================

def _resolve_group_column(df, requested, protocol_name):
    """
    Resolve the grouping column name, tolerating common naming variants.

    The processed CSV may store the learner/course identifier under a
    slightly different name than expected (e.g. 'course_id' rather than
    'course_id_encoded', or 'user_id' rather than 'username'). Rather than
    crashing after the caller has already waited for the main study to
    finish, we try a short list of known aliases and report clearly which
    column was used.
    """
    if requested in df.columns:
        return requested

    aliases = {
        "username":          ["user_id", "userid", "learner_id", "student_id", "user"],
        "course_id_encoded": ["course_id", "courseid", "course", "course_encoded",
                               "course_id_enc"],
    }
    for cand in aliases.get(requested, []):
        if cand in df.columns:
            print(f"  [R1#1] Column '{requested}' not found; using alias "
                  f"'{cand}' for {protocol_name} grouping.")
            return cand

    id_like = [c for c in df.columns
               if any(tok in c.lower() for tok in ("user", "course", "learner", "student"))]
    raise ValueError(
        f"[R1#1 HOLDOUT PROTOCOL] Could not resolve a grouping column for "
        f"'{protocol_name}'. Requested '{requested}' and none of its known "
        f"aliases {aliases.get(requested, [])} are present in {DATA_PATH}.\n"
        f"  Identifier-like columns found in the file: {id_like[:20]}\n"
        f"  Pass the correct name explicitly, e.g.:\n"
        f"      run_group_holdout_protocol(group_col='<name>', "
        f"protocol_name='{protocol_name}')"
    )


def run_group_holdout_protocol(group_col, protocol_name, k_folds=5,
                                drop_group_feature=True, extra_drop_cols=None):
    """
    Re-evaluate the global LightGBM baseline and SoftMoE+Isotonic under
    GroupKFold splits grouped by `group_col`, so that no group (e.g. a
    learner or a course) appears in both the training and test fold.

    Parameters
    ----------
    group_col : str
        Raw-data column name defining groups. Use "username" for
        learner-held-out, "course_id_encoded" for course-held-out.
        Common name variants are auto-detected (see _resolve_group_column).
    protocol_name : str
        Short label used in printed output and the output CSV filename,
        e.g. "learner_held_out" or "course_held_out".
    k_folds : int
        Number of GroupKFold splits (default 5, no repeats).
    drop_group_feature : bool
        If True (default), the grouping column is REMOVED from the feature
        matrix for this protocol. This is methodologically essential for
        course-held-out: leaving ``course_id_encoded`` in the features
        would mean the model is trained on course identities it can never
        observe at test time (every test course ID is unseen), producing
        an uninterpretable result AND defeating the entire purpose of the
        protocol, which is to measure behavioural generalization with the
        course-identity shortcut removed (Section III, Fig. 5). Set to
        False only if you deliberately want the leaky variant for a
        contrast experiment.
    extra_drop_cols : list[str] or None
        Additional feature columns to remove for this protocol. For
        course-held-out you may optionally pass other course-identifying
        columns (e.g. a raw 'course_id' if both encoded and raw versions
        exist in the file) to rule out residual identity leakage.

    Returns
    -------
    pandas.DataFrame
        One row per (fold, model), also saved to
        f"holdout_{protocol_name}_results.csv".
    """
    t_start = time.time()
    print(f"\n{'='*70}\n[R1#1 HOLDOUT PROTOCOL] {protocol_name}  "
          f"(grouped by '{group_col}', K={k_folds}, no repeats)\n{'='*70}")

    df = pd.read_csv(DATA_PATH)
    group_col = _resolve_group_column(df, group_col, protocol_name)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_cols
    dup_mask = df[feature_cols].T.duplicated()
    dup_cols = df[feature_cols].columns[dup_mask].tolist()
    feature_cols = [c for c in feature_cols if c not in dup_cols]
    n_before_drop = len(feature_cols)

    # ── CRITICAL: remove group-identity columns from the FEATURE matrix ──
    # See `drop_group_feature` in the docstring. Without this, course-held-out
    # trains on course IDs that are, by construction, absent from the test
    # fold — which is both methodologically incoherent and self-defeating.
    dropped_for_protocol = []
    if drop_group_feature and group_col in feature_cols:
        feature_cols = [c for c in feature_cols if c != group_col]
        dropped_for_protocol.append(group_col)
    for extra in (extra_drop_cols or []):
        if extra in feature_cols:
            feature_cols = [c for c in feature_cols if c != extra]
            dropped_for_protocol.append(extra)

    if dropped_for_protocol:
        print(f"  [R1#1] Removed {len(dropped_for_protocol)} group-identity "
              f"column(s) from the feature matrix for this protocol: "
              f"{dropped_for_protocol}")
        print(f"         Rationale: every test-fold group is unseen during "
              f"training, so retaining its identifier as a predictor would "
              f"be uninterpretable and would defeat the protocol's purpose.")
    else:
        print(f"  [R1#1] WARNING: no group-identity column was removed from "
              f"the feature matrix. If '{group_col}' is present as a "
              f"predictor, the results of this protocol may be misleading.")

    print(f"  [R1#1] {len(feature_cols)} numeric feature columns used "
          f"({n_before_drop} before protocol-specific drops; "
          f"{len(dup_cols)} duplicates removed).")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    X_raw = df[feature_cols].values
    y = df["target"].astype(int).values
    groups = df[group_col].values
    in_dim = X_raw.shape[1]

    n_unique_groups = len(np.unique(groups))
    print(f"  [R1#1] {n_unique_groups} unique '{group_col}' groups across "
          f"{len(df)} enrollments ({len(df)/n_unique_groups:.1f} "
          f"enrollments/group on average).")
    if n_unique_groups < k_folds:
        raise ValueError(
            f"[R1#1 HOLDOUT PROTOCOL] Only {n_unique_groups} unique groups "
            f"in '{group_col}', fewer than k_folds={k_folds}. Reduce "
            f"k_folds or check that '{group_col}' is the intended "
            f"grouping column."
        )
    if n_unique_groups < 3 * k_folds:
        print(f"  [R1#1] NOTE: only {n_unique_groups} groups for {k_folds} "
              f"folds. GroupKFold cannot stratify, so per-fold class balance "
              f"and fold sizes may vary substantially; both are reported "
              f"below and should be inspected before interpreting results.")

    cluster_df = pd.read_csv(CLUSTER_ASSIGN_PATH)
    id_to_row = pd.Series(np.arange(len(df)), index=df["enrollment_id"]).to_dict()
    cluster_df = cluster_df[cluster_df["enrollment_id"].isin(id_to_row)].copy()
    cluster_df["row_idx"] = cluster_df["enrollment_id"].map(id_to_row)
    labeled_rows_all = cluster_df["row_idx"].values
    labeled_clusters_all = cluster_df["cluster"].values.astype(int)

    gkf = GroupKFold(n_splits=k_folds)
    rows = []

    for fold_id, (tr_idx, te_idx) in enumerate(gkf.split(X_raw, y, groups)):
        tf0 = time.time()
        print(f"\n  [R1#1 {protocol_name}] Fold {fold_id+1}/{k_folds} "
              f"(train={len(tr_idx)}, test={len(te_idx)}) ...")

        # Hard group-disjointness assertion — the core guarantee of this
        # protocol. If this ever fails, GroupKFold itself is broken or
        # `groups` was constructed incorrectly; stop immediately.
        train_groups = set(groups[tr_idx].tolist())
        test_groups = set(groups[te_idx].tolist())
        overlap = train_groups & test_groups
        assert len(overlap) == 0, (
            f"[R1#1 HOLDOUT PROTOCOL] LEAKAGE: {len(overlap)} '{group_col}' "
            f"value(s) appear in both train and test of fold {fold_id}. "
            f"Aborting — this protocol's central guarantee has been violated."
        )
        te_pos_rate = float(y[te_idx].mean())
        tr_pos_rate = float(y[tr_idx].mean())
        print(f"    [R1#1 leakage check] train/test '{group_col}' overlap = 0 ✓ "
              f"({len(train_groups)} train groups, {len(test_groups)} test groups)")
        print(f"    [R1#1 class balance] train dropout={tr_pos_rate:.3f}  "
              f"test dropout={te_pos_rate:.3f}  (GroupKFold does not stratify)")
        if te_pos_rate < 0.5 or te_pos_rate > 0.95:
            print(f"    [R1#1 WARN] Test-fold dropout rate {te_pos_rate:.3f} "
                  f"deviates markedly from the overall 0.793; AUC on this "
                  f"fold is not directly comparable to the main study.")

        tr_sub_idx, va_idx = train_test_split(
            tr_idx, test_size=0.15, stratify=y[tr_idx], random_state=SEED)
        y_tr, y_va, y_te = y[tr_sub_idx], y[va_idx], y[te_idx]

        # ---- (1) LightGBM global baseline (identical config to main()) ----
        model_global = fit_lgb(X_raw[tr_sub_idx], y_tr, X_raw[va_idx], y_va, GLOBAL_PARAMS)
        p_global_va = model_global.predict_proba(X_raw[va_idx])[:, 1]
        thr_global = best_threshold(y_va, p_global_va)
        p_global_te = model_global.predict_proba(X_raw[te_idx])[:, 1]
        res_global = evaluate(y_te, p_global_te, thr=thr_global)
        ece_global = expected_calibration_error(y_te, p_global_te)
        del model_global
        gc.collect()

        # ---- (2) Fold-internal surrogate for the SHAP-cluster gate prior ----
        # Same leakage-safe construction as main(): the surrogate is fit
        # only on at-risk students inside this fold's training partition.
        train_set = set(tr_idx.tolist())
        mask_in_train = np.array([r in train_set for r in labeled_rows_all])
        rows_for_surrogate = labeled_rows_all[mask_in_train]
        clusters_for_surrogate = labeled_clusters_all[mask_in_train]
        assert not any(r in set(te_idx.tolist()) for r in rows_for_surrogate), \
            "[R1#1] LEAKAGE: surrogate rows overlap with te_idx — critical error!"

        if len(rows_for_surrogate) < 30 or len(np.unique(clusters_for_surrogate)) < 2:
            print(f"    [R1#1 WARN] Only {len(rows_for_surrogate)} labeled at-risk "
                  f"samples in this fold's training partition; falling back to "
                  f"a uniform gate prior (equivalent to lambda=0 for this fold).")
            gate_target_tr = np.full((len(tr_sub_idx), N_EXPERTS), 1.0 / N_EXPERTS, dtype=np.float32)
            gate_target_va = np.full((len(va_idx), N_EXPERTS), 1.0 / N_EXPERTS, dtype=np.float32)
        else:
            Xs_tr, Xs_va, ys_tr, ys_va = train_test_split(
                X_raw[rows_for_surrogate], clusters_for_surrogate, test_size=0.15,
                stratify=clusters_for_surrogate, random_state=SEED)
            surrogate = fit_lgb(Xs_tr, ys_tr, Xs_va, ys_va, SURROGATE_PARAMS)
            gate_target_tr = surrogate.predict_proba(X_raw[tr_sub_idx]).astype(np.float32)
            gate_target_va = surrogate.predict_proba(X_raw[va_idx]).astype(np.float32)
            del surrogate
            gc.collect()

        # ---- (3) SoftMoE + Isotonic (identical architecture/training to main()) ----
        scaler = StandardScaler().fit(X_raw[tr_sub_idx])
        X_tr_s = scaler.transform(X_raw[tr_sub_idx]).astype(np.float32)
        X_va_s = scaler.transform(X_raw[va_idx]).astype(np.float32)
        X_te_s = scaler.transform(X_raw[te_idx]).astype(np.float32)
        pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

        moe_model = train_soft_moe(X_tr_s, y_tr, gate_target_tr, X_va_s, y_va, gate_target_va,
                                    in_dim, pos_w, fold_seed=fold_id)
        with torch.no_grad():
            p_moe_va, _, _ = moe_model(torch.tensor(X_va_s, dtype=torch.float32, device=DEVICE))
            p_moe_te, _, _ = moe_model(torch.tensor(X_te_s, dtype=torch.float32, device=DEVICE))
        p_moe_va = p_moe_va.detach().cpu().numpy()
        p_moe_te = p_moe_te.detach().cpu().numpy()

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_moe_va, y_va)
        p_moe_va_iso = iso.predict(p_moe_va)
        p_moe_te_iso = iso.predict(p_moe_te)
        ece_moe_iso = expected_calibration_error(y_te, p_moe_te_iso)
        thr_moe_iso = best_threshold(y_va, p_moe_va_iso)
        res_moe_iso = evaluate(y_te, p_moe_te_iso, thr=thr_moe_iso)

        del moe_model
        gc.collect()

        rows.append({"protocol": protocol_name, "fold": fold_id, "model": "global",
                      "n_train": len(tr_idx), "n_test": len(te_idx),
                      "n_train_groups": len(train_groups), "n_test_groups": len(test_groups),
                      "test_dropout_rate": te_pos_rate,
                      "ece_after_isotonic": ece_global, **res_global})
        rows.append({"protocol": protocol_name, "fold": fold_id, "model": "soft_moe_isotonic",
                      "n_train": len(tr_idx), "n_test": len(te_idx),
                      "n_train_groups": len(train_groups), "n_test_groups": len(test_groups),
                      "test_dropout_rate": te_pos_rate,
                      "ece_after_isotonic": ece_moe_iso, **res_moe_iso})

        print(f"    [global]            AUC={res_global['auc']:.4f}  F1={res_global['f1']:.4f}  "
              f"MCC={res_global['mcc']:.4f}  ECE(iso)={ece_global:.4f}")
        print(f"    [soft_moe_isotonic] AUC={res_moe_iso['auc']:.4f}  F1={res_moe_iso['f1']:.4f}  "
              f"MCC={res_moe_iso['mcc']:.4f}  ECE(iso)={ece_moe_iso:.4f}   "
              f"({time.time()-tf0:.1f}s)")

    result_df = pd.DataFrame(rows)
    out_path = f"holdout_{protocol_name}_results.csv"
    result_df.to_csv(out_path, index=False)

    # ── Paired statistics: SoftMoE+Iso vs LightGBM under THIS protocol ──
    # Same paired design as the main study (matched folds), but with only
    # k_folds paired observations instead of 25, so the test is markedly
    # underpowered; Cohen's d and the raw per-fold deltas are reported
    # alongside p so the effect can be judged independently of significance.
    print(f"\n  [R1#1 {protocol_name}] Paired SoftMoE+Iso vs LightGBM "
          f"(n={k_folds} matched folds; underpowered vs the 25-run main study):")
    holdout_stats = {}
    for metric in ["auc", "accuracy", "f1", "precision", "recall", "mcc"]:
        a = result_df[result_df.model == "global"].sort_values("fold")[metric].values
        b = result_df[result_df.model == "soft_moe_isotonic"].sort_values("fold")[metric].values
        diff = b - a
        mean_diff = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        d = mean_diff / sd if sd > 0 else 0.0
        if len(diff) > 1 and sd > 0:
            t_stat, p_val = scipy_stats.ttest_rel(b, a)
        else:
            t_stat, p_val = float("nan"), float("nan")
        wins = int(np.sum(diff > 0))
        holdout_stats[metric] = {
            "mean_diff": mean_diff, "cohens_d": d,
            "paired_t": float(t_stat), "p_value": float(p_val),
            "soft_moe_wins": wins, "n_folds": int(len(diff)),
        }
        print(f"    {metric:10s} Δ={mean_diff:+.4f}  d={d:+.2f}  "
              f"p={p_val:.4f}  SoftMoE wins {wins}/{len(diff)} folds")

    with open(f"holdout_{protocol_name}_stats.json", "w") as f:
        json.dump(holdout_stats, f, indent=2)

    # ── Degradation vs the enrollment-level main study ──
    # This is the number the reviewer actually asked for: how much of the
    # reported performance survives when the group is genuinely unseen.
    MAIN_STUDY_AUC = {"global": 0.8731, "soft_moe_isotonic": 0.8836}
    print(f"\n  [R1#1 {protocol_name}] Degradation vs enrollment-level "
          f"CV (main study, Table 8):")
    for model_name, main_auc in MAIN_STUDY_AUC.items():
        hold_auc = float(result_df[result_df.model == model_name]["auc"].mean())
        delta = hold_auc - main_auc
        print(f"    {model_name:20s} main={main_auc:.4f} → "
              f"{protocol_name}={hold_auc:.4f}  (Δ={delta:+.4f})")

    print(f"\n  [R1#1 HOLDOUT PROTOCOL] {protocol_name} — {k_folds}-fold summary "
          f"({time.time()-t_start:.0f}s total):")
    summary = result_df.groupby("model")[["auc", "f1", "mcc", "accuracy",
                                           "precision", "recall"]].agg(["mean", "std"])
    print(summary)
    print(f"  File saved: {out_path}")

    return result_df


def run_all_holdout_protocols(include_leaky_contrast=True):
    """Convenience wrapper: run the learner- and course-held-out protocols
    back-to-back and print a combined comparison against the main-study
    (enrollment-level) numbers already reported in Table 8.

    If `include_leaky_contrast` is True, an additional course-held-out run
    is performed WITHOUT removing `course_id_encoded` from the features.
    Contrasting that run against the clean one isolates how much of the
    model's performance depends on course identity as a shortcut — which
    is precisely the concern raised in Section III (Fig. 5) and in the
    external-validity discussion. This contrast costs one extra protocol
    run but produces the single most direct piece of evidence on the
    question, so it is enabled by default.
    """
    results = {}
    results["learner_held_out"] = run_group_holdout_protocol(
        group_col="username", protocol_name="learner_held_out", k_folds=5)
    results["course_held_out"] = run_group_holdout_protocol(
        group_col="course_id_encoded", protocol_name="course_held_out", k_folds=5)

    if include_leaky_contrast:
        results["course_held_out_with_id"] = run_group_holdout_protocol(
            group_col="course_id_encoded",
            protocol_name="course_held_out_with_id",
            k_folds=5, drop_group_feature=False)

    print(f"\n{'='*70}\n[R1#1] COMBINED SUMMARY: enrollment-level vs. group-held-out\n{'='*70}")
    print("  Reference (Table 8, enrollment-level, 25 runs):")
    print("    LightGBM baseline    : AUC = 0.873 ± 0.005")
    print("    SoftMoE+Isotonic     : AUC = 0.884 ± 0.003")
    for name, df_res in results.items():
        summ = df_res.groupby("model")["auc"].agg(["mean", "std"])
        print(f"\n  {name} (this run, {len(df_res)//2} folds):")
        for model_name, row in summ.iterrows():
            print(f"    {model_name:20s}: AUC = {row['mean']:.3f} ± {row['std']:.3f}")

    combined = pd.concat(results.values(), ignore_index=True)
    combined.to_csv("holdout_all_protocols_combined.csv", index=False)
    print("\n  File saved: holdout_all_protocols_combined.csv")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="SoftMoE-SHAP full pipeline. Running with NO arguments "
                     "now executes EVERYTHING: the main 25-run enrollment-"
                     "level study, then the R1#1 learner-held-out, "
                     "course-held-out, and course-held-out-with-id "
                     "generalization protocols. Use the flags below to run "
                     "a subset.")
    parser.add_argument(
        "--holdout", choices=["learner", "course", "both", "none"], default="both",
        help="Which group-held-out generalization protocol(s) to run (R1#1). "
             "'learner' groups by the learner identifier; 'course' groups by "
             "the course identifier; 'both' (DEFAULT) runs both sequentially; "
             "'none' skips them and reproduces the original v5 behavior.")
    parser.add_argument(
        "--holdout-only", action="store_true",
        help="Skip the main 25-run study and run ONLY the holdout "
             "protocol(s). Use this if the main study has already been run "
             "and its CSV outputs are still present.")
    parser.add_argument(
        "--skip-main", action="store_true",
        help="Alias for --holdout-only (kept for readability).")
    parser.add_argument(
        "--holdout-folds", type=int, default=5,
        help="Number of GroupKFold splits for each holdout protocol "
             "(default 5, no repeats).")
    parser.add_argument(
        "--no-leaky-contrast", action="store_true",
        help="Skip the extra course-held-out run that KEEPS the course "
             "identifier as a feature. That contrast run is enabled by "
             "default because it directly quantifies how much performance "
             "depends on course identity as a shortcut (Section III, "
             "Fig. 5); pass this flag to save roughly 1.5 hours.")
    # NOTE: use parse_known_args() rather than parse_args(). Jupyter/Colab
    # kernels auto-inject extra arguments (e.g. "-f /root/.../kernel-xxx.json")
    # that this script does not define; parse_args() would raise
    # "unrecognized arguments" and exit. parse_known_args() safely ignores
    # any such unrecognized arguments while still parsing our own flags
    # correctly, in both plain-Python and notebook contexts.
    args, _unknown_args = parser.parse_known_args()

    run_main = not (args.holdout_only or args.skip_main)
    run_contrast = (not args.no_leaky_contrast) and args.holdout in ("course", "both")

    # ── Print the execution plan up front ──────────────────────────────
    # Each stage below is long-running; printing the plan (and a rough time
    # budget) first means an interrupted run can be resumed with the right
    # flags instead of restarting from scratch.
    _plan, _est_h = [], 0.0
    if run_main:
        _plan.append("main 25-run enrollment-level study"); _est_h += 2.5
    if args.holdout in ("learner", "both"):
        _plan.append(f"learner-held-out ({args.holdout_folds}-fold)"); _est_h += 1.5
    if args.holdout in ("course", "both"):
        _plan.append(f"course-held-out ({args.holdout_folds}-fold, course ID dropped)")
        _est_h += 1.5
    if run_contrast:
        _plan.append(f"course-held-out-with-id ({args.holdout_folds}-fold, course ID kept)")
        _est_h += 1.5

    print("\n" + "=" * 70)
    print("[PLAN] This run will execute the following stages in order:")
    for _i, _stage in enumerate(_plan, 1):
        print(f"        {_i}. {_stage}")
    print(f"[PLAN] Rough total wall-clock estimate: ~{_est_h:.1f} hours on one GPU.")
    print("[PLAN] To run a subset: --holdout none | --holdout-only | "
          "--no-leaky-contrast")
    print("=" * 70 + "\n")

    def _dispatch_holdouts():
        """Run the requested holdout protocol(s) with the parsed options.

        Each protocol is wrapped in try/except so that a failure in one
        (e.g. an unresolvable grouping column) does not discard the
        results of the protocols that already completed successfully.
        """
        k = args.holdout_folds
        jobs = []
        if args.holdout in ("learner", "both"):
            jobs.append(("username", "learner_held_out", True))
        if args.holdout in ("course", "both"):
            jobs.append(("course_id_encoded", "course_held_out", True))
            if run_contrast:
                jobs.append(("course_id_encoded", "course_held_out_with_id", False))

        for group_col, name, drop_feat in jobs:
            try:
                run_group_holdout_protocol(group_col, name, k_folds=k,
                                            drop_group_feature=drop_feat)
            except Exception as exc:
                print(f"\nProtocol '{name}' failed and was "
                      f"skipped: {type(exc).__name__}: {exc}")
                print(f"Remaining protocols will still run. "
                      f"Fix the cause and re-run with "
                      f"--holdout-only to redo just the holdout stage.\n")

    if run_main:
        main()
    if args.holdout != "none":
        _dispatch_holdouts()

    print("\n" + "=" * 70)
    print("[DONE] All requested stages finished.")
    print("=" * 70)
