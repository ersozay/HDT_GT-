"""
Script 7: Fizik-Transfer Doğrulaması — 2008 Wenchuan (Çin) Depremi

Danışman (Aytuk) isteği: "yöntemi başka bir depremle de test edelim (örn. Çin
Wenchuan)". Bu betik, Kahramanmaraş için kurulan Newmark rijit-blok fiziğinin
(Jibson 2007 + Ambraseys & Menu 1988) BAŞKA bir büyük depreme transfer
edildiğinde, yayınlanmış saha gözlemleriyle NİTELİKSEL olarak tutarlı sonuç
verdiğini gösterir.

ÖNEMLİ — Bu bir FİZİK-TRANSFER gösterimidir:
  · PINN YENİDEN EĞİTİLMEZ; yalnızca global (evrensel) Newmark bağıntıları
    Wenchuan tipik yamaç parametrelerine uygulanır.
  · Kullanılan yamaç/PGA aralıkları LİTERATÜRDEN alınmış temsili değerlerdir
    (kaynaklar aşağıda). Yerel saha kalibrasyonu DEĞİLDİR.
  · Karşılaştırma niteldir: "yüksek PGA + dik yamaç → Dₙ > 10 cm → yaygın
    tetiklenme" beklentisinin gözlemle uyumu.

Yayınlanmış referanslar (Wenchuan 2008, Mw7.9):
  · Dai, F.C. et al. (2011). Spatial distribution of landslides triggered by the
    2008 Ms 8.0 Wenchuan earthquake. J. Asian Earth Sciences 40(4), 883–895.
  · Xu, C. et al. (2014). Three (nearly) complete inventories of landslides
    triggered by the May 12, 2008 Wenchuan Mw 7.9 earthquake. Landslides 11.
  · Jibson, R.W. (2007). Engineering Geology 91(2–4). (Dₙ regresyonları)

Kullanım:
  python3 scripts/07_wenchuan_validation.py
"""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.geomechanics.newmark import NewmarkDisplacement

DN_LIMIT_CM = config.DN_THRESHOLD_M * 100.0  # 10 cm

# Wenchuan için temsili yamaç sınıfları (literatür tabanlı; saha kalibrasyonu değil).
# aᵧ (kritik ivme) değerleri, yaygın gözlenen tetiklenme geometrilerini temsil eder:
#   dik/zayıf yamaç  → düşük aᵧ (kolay tetiklenir)
#   orta yamaç       → orta aᵧ
#   nazik/sağlam     → yüksek aᵧ (zor tetiklenir)
WENCHUAN_CLASSES = [
    {"name": "Dik / zayıf yamaç (aᵧ≈0.05g)",  "ky": 0.05, "color": "#dc2626"},
    {"name": "Orta yamaç (aᵧ≈0.15g)",         "ky": 0.15, "color": "#d97706"},
    {"name": "Nazik / sağlam yamaç (aᵧ≈0.30g)", "ky": 0.30, "color": "#16a34a"},
]

# Wenchuan yakın-fay PGA aralığı (USGS ShakeMap / kayıt istasyonları): ~0.1–1.0g+,
# fay yakınında lokal olarak >1g raporlanmıştır.
WENCHUAN_PGA_TYPICAL = (0.2, 0.6, 1.0)  # temsili düşük / orta / yüksek

# Yayınlanmış gözlem özeti (nitel karşılaştırma için)
PUBLISHED = {
    "event": "2008 Wenchuan (Çin) — Mw 7.9",
    "landslide_count": "~197.000 (Xu ve ark. 2014; Dai ve ark. 2011)",
    "peak_pga_g": "yakın-fayda ~1.0g (lokal >1g)",
    "note": "Yaygın koseismik heyelan; dik yamaç + yüksek PGA bölgelerinde yoğun.",
}


def build_comparison():
    """Her PGA senaryosu × yamaç sınıfı için Dₙ ve tetiklenme durumu."""
    rows = []
    for pga in WENCHUAN_PGA_TYPICAL:
        for cls in WENCHUAN_CLASSES:
            dn_j = float(NewmarkDisplacement(cls["ky"], pga, "jibson2007") * 100)
            dn_a = float(NewmarkDisplacement(cls["ky"], pga, "ambraseys1988") * 100)
            rows.append({
                "pga_g": pga,
                "class": cls["name"],
                "ky_g": cls["ky"],
                "dn_jibson_cm": round(dn_j, 2),
                "dn_ambraseys_cm": round(dn_a, 2),
                "triggered_jibson": bool(dn_j >= DN_LIMIT_CM),
                "triggered_ambraseys": bool(dn_a >= DN_LIMIT_CM),
            })
    return rows


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle(
        "Fizik-Transfer Doğrulaması — 2008 Wenchuan (Mw 7.9) Depremi",
        fontsize=15, fontweight="bold", y=0.99,
    )

    # Sol: Dn vs PGA eğrileri (yamaç sınıfları)
    pga = np.linspace(0.05, 1.30, 200)
    for cls in WENCHUAN_CLASSES:
        dn_j = NewmarkDisplacement(cls["ky"], pga, "jibson2007") * 100
        dn_a = NewmarkDisplacement(cls["ky"], pga, "ambraseys1988") * 100
        axL.plot(pga, dn_j, color=cls["color"], lw=2.4, label=f"{cls['name']} · Jibson")
        axL.plot(pga, dn_a, color=cls["color"], lw=1.8, ls="--", label=f"{cls['name']} · Ambraseys")

    axL.axhline(DN_LIMIT_CM, color="#111827", lw=1.8, ls="-.")
    axL.text(1.28, DN_LIMIT_CM * 1.2, "10 cm — tetiklenme eşiği",
             ha="right", va="bottom", fontsize=9.5, fontweight="bold")
    # yakın-fay PGA bandı
    axL.axvspan(0.6, 1.0, color="#fca5a5", alpha=0.18)
    axL.text(0.8, 500, "yakın-fay\nPGA bandı", ha="center", va="center",
             fontsize=8.5, color="#b91c1c")
    axL.set_xlabel("Tepe Yer İvmesi  PGA  (g)", fontsize=11)
    axL.set_ylabel("Kalıcı Yer Değiştirme  Dₙ  (cm)", fontsize=11)
    axL.set_title("Newmark Dₙ – PGA (Wenchuan temsili yamaç sınıfları)", fontsize=12)
    axL.set_yscale("log")
    axL.set_ylim(0.1, 1100)
    axL.grid(True, which="both", alpha=0.25)
    axL.legend(fontsize=7.8, loc="lower right", framealpha=0.92)

    # Sağ: nitel karşılaştırma tablosu / özet
    axR.axis("off")
    txt = [
        "GÖZLEM (yayınlanmış):",
        f"  Olay: {PUBLISHED['event']}",
        f"  Heyelan sayısı: {PUBLISHED['landslide_count']}",
        f"  Tepe PGA: {PUBLISHED['peak_pga_g']}",
        "",
        "MODEL (fizik-transfer, global Newmark):",
    ]
    for cls in WENCHUAN_CLASSES:
        dn_hi = float(NewmarkDisplacement(cls["ky"], 1.0, "jibson2007") * 100)
        trig = "TETIKLENIR (Dn>10cm)" if dn_hi >= DN_LIMIT_CM else "tetiklenmez"
        name_ascii = cls["name"].replace("aᵧ", "a_y")
        axR.plot([], [])
        txt.append(f"  {name_ascii}:")
        txt.append(f"     PGA=1.0g -> Dn~{dn_hi:.0f} cm -> {trig}")
    txt += [
        "",
        "SONUC (nitel uyum):",
        "  Yuksek PGA (yakin-fay) + dik/zayif yamac ->",
        "  Dn >> 10 cm -> yaygin tetiklenme beklentisi,",
        "  gozlemle (yaklasik 197.000 heyelan) tutarli.",
    ]
    axR.text(0.02, 0.98, "\n".join(txt), va="top", ha="left", fontsize=9.6,
             family="monospace", transform=axR.transAxes)

    fig.text(0.99, 0.012,
             "Fizik-transfer gösterimi: PINN yeniden eğitilmedi; global (evrensel) "
             "Newmark bağıntıları\nWenchuan temsili parametrelerine uygulandı. "
             "Yerel saha kalibrasyonu DEĞİLDİR.",
             fontsize=7.8, va="bottom", ha="right", color="#64748b", style="italic")

    fig.tight_layout(rect=[0, 0.06, 1, 0.965])
    out = config.FIGURES_DIR / "wenchuan_validation.png"
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def write_validation_json(rows):
    payload = {
        "title": "Fizik-Transfer Doğrulaması — 2008 Wenchuan (Mw 7.9)",
        "method": (
            "Kahramanmaraş için kurulan Newmark rijit-blok fiziği (Jibson 2007 + "
            "Ambraseys & Menu 1988) Wenchuan temsili yamaç parametrelerine uygulandı. "
            "PINN yeniden eğitilmedi; bu bir fizik-transfer gösterimidir."
        ),
        "published": PUBLISHED,
        "dn_threshold_cm": DN_LIMIT_CM,
        "comparison": rows,
        "sources": [
            "Dai, F.C. et al. (2011). J. Asian Earth Sciences 40(4), 883–895.",
            "Xu, C. et al. (2014). Landslides 11, 1461–1481.",
            "Jibson, R.W. (2007). Engineering Geology 91(2–4), 209–218.",
            "Ambraseys, N.N. & Menu, J.M. (1988). EESD 16(7), 985–1006.",
        ],
        "caveats": [
            "Fizik-transfer gösterimidir; yerel saha kalibrasyonu / PINN yeniden eğitim DEĞİLDİR.",
            "Yamaç sınıfları ve PGA aralıkları literatür-tabanlı temsili değerlerdir.",
            "Karşılaştırma niteldir (tetiklenme eğilimi), nicel heyelan sayımı değildir.",
            "Newmark rijit-blok varsayar; ampirik regresyonların tahmin bandı vardır.",
        ],
    }
    out = Path(__file__).parent.parent / "backend" / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    rows = build_comparison()
    print("Karşılaştırma (PGA × yamaç sınıfı):")
    for r in rows:
        print(f"  PGA={r['pga_g']}g · {r['class']}: "
              f"Dn_J={r['dn_jibson_cm']}cm (trig={r['triggered_jibson']}) · "
              f"Dn_A={r['dn_ambraseys_cm']}cm")
    fig = make_figure(rows)
    js = write_validation_json(rows)
    print(f"\nFigür: {fig}\nJSON : {js}")


if __name__ == "__main__":
    main()
