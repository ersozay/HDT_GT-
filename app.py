"""
Dijital Twin — Heyelan Duyarlılık Demo (Streamlit)
==================================================

Eğitilmiş PINN modelini interaktif olarak sunar:
  1) Tek Nokta Tahmini — eğim/yükseklik/litoloji/PGA gir → P(heyelan) + Ky + Dn
  2) Kritik PGA         — seçili yamaç için PGA taraması, tetiklenme eşiği
  3) Bölgesel Harita    — önceden üretilmiş duyarlılık haritası + sonuç figürleri

Çalıştırma:
    cd "digital twin"
    python3 -m streamlit run app.py
"""
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import MODELS_DIR, FIGURES_DIR, DATA_PROCESSED, Z_REF_M, DN_THRESHOLD_M
from src.geomechanics.lithology import LITHOLOGY_TABLE
from src.data_engineering.feature_matrix import build_feature_matrix, ALL_FEATURE_COLUMNS

st.set_page_config(page_title="Heyelan Dijital Twin", page_icon="⛰️", layout="wide")


# ----------------------------------------------------------------------------
# Model + scaler (önbelleğe alınır)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    import torch
    from src.pinn.model import LandslidePINN

    scaler = pickle.loads((MODELS_DIR / "scaler.pkl").read_bytes())
    ckpt = torch.load(MODELS_DIR / "best_model.pt", map_location="cpu")
    # Öznitelik sayısı checkpoint'ten çıkarılır (ilk katmanın giriş boyutu)
    n_feat = ckpt["model_state"]["layer1.0.weight"].shape[1]
    model = LandslidePINN(in_features=n_feat)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    thr_path = MODELS_DIR / "threshold.json"
    thr = 0.5
    if thr_path.exists():
        thr = json.loads(thr_path.read_text()).get("decision_threshold", 0.5)
    return model, scaler, ckpt, float(thr)


def predict_point(model, scaler, *, slope, elevation, aspect, litho_id, pga, z_ref=Z_REF_M):
    """Tek bir nokta için P(heyelan), Ky ve Dn döndürür."""
    import torch

    a = lambda v: np.array([[v]], dtype=float)
    df, feat_cols = build_feature_matrix(
        elevation_m=a(elevation), slope_deg=a(slope), aspect_deg=a(aspect),
        litho_array=np.array([[litho_id]]), pga_array=a(pga),
        u_array=None, z_ref=z_ref, include_onehot=True,
    )
    cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    X = df[cols].values.astype(np.float32)
    Xn = (X - scaler["mean"]) / scaler["std"]
    with torch.no_grad():
        p = float(model(torch.tensor(Xn, dtype=torch.float32)).numpy().ravel()[0])
    return p, float(df["ky_g"].iloc[0]), float(df["dn_cm"].iloc[0])


LITHO_OPTIONS = {f"{lid} — {v['description']}": lid for lid, v in LITHOLOGY_TABLE.items()}


# ----------------------------------------------------------------------------
# Arayüz
# ----------------------------------------------------------------------------
model, scaler, ckpt, DEC_THR = load_model()

st.title("⛰️ Fizik Tabanlı Dijital Twin — Heyelan Duyarlılık Demosu")
st.caption(
    "Kahramanmaraş 6 Şubat 2023 depremi · Hibrit FEM + Newmark + PINN · "
    f"Model: epoch {ckpt.get('epoch','?')}, val ROC-AUC {ckpt.get('val_auc',0):.3f} · "
    f"Karar eşiği {DEC_THR:.2f}"
)

tab1, tab2, tab3 = st.tabs(
    ["🎯 Tek Nokta Tahmini", "📈 Kritik PGA", "🗺️ Bölgesel Harita"]
)

# ============================ TAB 1: Tek Nokta ==============================
with tab1:
    st.subheader("Tek Yamaç için Heyelan Olasılığı")
    c1, c2 = st.columns([1, 1.3])

    with c1:
        slope = st.slider("Eğim (°)", 0.0, 60.0, 30.0, 0.5)
        elevation = st.slider("Yükseklik (m)", 0, 3000, 1200, 50)
        aspect = st.slider("Bakı (° · 0=K, 90=D, 180=G, 270=B)", 0, 359, 180, 5)
        litho_label = st.selectbox("Litoloji sınıfı", list(LITHO_OPTIONS.keys()),
                                   index=list(LITHO_OPTIONS.values()).index(11)
                                   if 11 in LITHO_OPTIONS.values() else 0)
        litho_id = LITHO_OPTIONS[litho_label]
        pga = st.slider("PGA — Yüzey İvmesi (g)", 0.05, 1.50, 0.55, 0.01)

    p, ky, dn = predict_point(model, scaler, slope=slope, elevation=elevation,
                              aspect=aspect, litho_id=litho_id, pga=pga)

    with c2:
        risk = "YÜKSEK" if p >= DEC_THR else "DÜŞÜK"
        color = "#c0392b" if p >= DEC_THR else "#27ae60"
        st.markdown(
            f"<div style='text-align:center;padding:1.2rem;border-radius:12px;"
            f"background:{color}20;border:2px solid {color}'>"
            f"<div style='font-size:0.9rem;color:gray'>Heyelan Olasılığı</div>"
            f"<div style='font-size:3rem;font-weight:800;color:{color}'>{p*100:.0f}%</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:{color}'>{risk} RİSK</div>"
            f"</div>", unsafe_allow_html=True,
        )
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Kritik ivme Ky (g)", f"{ky:.3f}",
                  help="Newmark kritik ivmesi — bu değerin altındaki PGA blok kaymasını başlatmaz.")
        m2.metric("Newmark Dn (cm)", f"{dn:.1f}",
                  delta="≥10 cm → tetiklenme" if dn >= DN_THRESHOLD_M*100 else "eşik altı",
                  delta_color="inverse" if dn >= DN_THRESHOLD_M*100 else "normal",
                  help="Kalıcı yer değiştirme (Jibson 2007). 10 cm üstü heyelan tetiklenmesi.")

        if ky <= 0.02:
            st.warning("Bu yamaç **statik olarak marjinal** (Ky≈0): deprem olmadan da "
                       "duraysız sınırında. Tetiklenme için ek ivme gerekmez.")

    st.info("Bu, modelin **tek bir yamaç** için verdiği olasılıktır. "
            "Litolojiyi/eğimi değiştirip risk nasıl değişiyor gözlemleyin.")

# ============================ TAB 2: Kritik PGA =============================
with tab2:
    st.subheader("Kritik PGA — Yamaç Hangi İvmede Kaymaya Başlar?")
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        slope2 = st.slider("Eğim (°)", 0.0, 60.0, 35.0, 0.5, key="s2")
        elev2 = st.slider("Yükseklik (m)", 0, 3000, 1200, 50, key="e2")
        litho2_label = st.selectbox("Litoloji sınıfı", list(LITHO_OPTIONS.keys()),
                                    index=list(LITHO_OPTIONS.values()).index(11)
                                    if 11 in LITHO_OPTIONS.values() else 0, key="l2")
        litho2 = LITHO_OPTIONS[litho2_label]

    pga_grid = np.round(np.arange(0.05, 1.51, 0.05), 2)
    probs, dns = [], []
    for pg in pga_grid:
        p, ky2, dn2 = predict_point(model, scaler, slope=slope2, elevation=elev2,
                                    aspect=180, litho_id=litho2, pga=pg)
        probs.append(p); dns.append(dn2)
    probs = np.array(probs); dns = np.array(dns)

    over = np.where(probs >= DEC_THR)[0]
    crit = pga_grid[over[0]] if len(over) else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhspan(DEC_THR, 1.02, color="#c0392b", alpha=0.05)
    ax.plot(pga_grid, probs, "o-", color="#c0392b", lw=2.2, label="ML: P(heyelan)")
    ax.axhline(DEC_THR, ls=":", color="gray", label=f"Karar eşiği = {DEC_THR:.2f}")
    if crit is not None:
        ax.axvline(crit, color="#c0392b", alpha=0.5)
        ax.plot([crit], [DEC_THR], "*", color="#c0392b", ms=18)
        ax.annotate(f"Kritik PGA ≈ {crit:.2f} g", xy=(crit, DEC_THR),
                    xytext=(crit + 0.1, 0.25), color="#c0392b", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.set_xlabel("PGA (g)"); ax.set_ylabel("Heyelan olasılığı")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    ax.set_title(f"Eğim {slope2:.0f}° · {litho2_label.split('—')[1].strip()}")
    fig.tight_layout()
    with cc2:
        st.pyplot(fig)
    plt.close(fig)

    if crit is not None:
        st.success(f"**Kritik PGA ≈ {crit:.2f} g** — bu yamaç yaklaşık bu ivmede "
                   f"heyelan olasılığı %{DEC_THR*100:.0f} eşiğini aşıyor.")
    else:
        st.info("Bu yamaç, incelenen PGA aralığında (≤1.5 g) karar eşiğini aşmıyor "
                "— görece stabil.")

# ============================ TAB 3: Bölgesel ==============================
with tab3:
    st.subheader("Bölgesel Duyarlılık Haritası ve Sonuçlar")
    figs = {
        "Duyarlılık Haritası (envanter bindirmeli)": "susceptibility_map_real.png",
        "Kritik PGA — ML vs Newmark": "critical_pga_curve_real.png",
        "SHAP — Öznitelik Önemi": "shap_summary.png",
        "Eğitim Geçmişi": "training_history.png",
        "Karışıklık Matrisi (test)": "confusion_matrix.png",
    }
    choice = st.selectbox("Figür seç", list(figs.keys()))
    fpath = FIGURES_DIR / figs[choice]
    if fpath.exists():
        st.image(str(fpath), use_container_width=True)
    else:
        st.warning(f"Figür bulunamadı: {fpath.name} — önce "
                   "`python3 scripts/05_susceptibility_map.py` çalıştırın.")

    st.divider()
    st.markdown(
        "**Özet sonuçlar:** Test ROC-AUC ≈ 0.96 · Heyelan recall ≈ 0.86 · "
        f"Kritik PGA (ML) ≈ 0.55 g. Litoloji katmanı Macrostrat **proxy**'sidir "
        "(resmi MTA değil). FEM konturları analitik yedek (SfePy kurulu değil)."
    )
