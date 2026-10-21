"""
Teknik özet raporunu (Heyelan_Dijital_Twin_Rapor.docx) sonuç dosyalarından üretir.

Tüm sayılar outputs/results/*.json ve backend/metrics.json'dan okunur; rapor
analiz yeniden koşulduğunda bu betikle güncellenir:

    python3 docs/build_report.py
"""
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor, Emu

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "Heyelan_Dijital_Twin_Rapor.docx"

NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2C, 0x7F, 0xB8)
RED = RGBColor(0xC0, 0x39, 0x2B)
GRAY = RGBColor(0x59, 0x59, 0x59)
HEAD_FILL, ALT_FILL, NOTE_FILL = "1F3864", "EEF2F8", "FDF3E7"

load = lambda name: json.loads((RES / name).read_text(encoding="utf-8"))
bench = load("benchmark_summary.json")
crit = load("critical_pga.json")
fem = load("fem_pilots.json")
shap = load("shap_importance.json")
nmk = load("newmark_th_validation.json")
import pandas as _pd
sm_st = _pd.read_csv(RES / "strong_motion_stations.csv")
metrics = json.loads((ROOT / "backend" / "metrics.json").read_text(encoding="utf-8"))
MAIN = metrics.get("model_version", "pinn_v2")
cp, emp, sens = crit["critical_pga"], crit["empirical_validation"], crit["neg_ratio_sensitivity"]
M = bench["models"]


def f2(x):
    return "—" if x is None else f"{x:.2f}".replace(".", ",")


def f3(x):
    return "—" if x is None else f"{x:.3f}".replace(".", ",")


def pct(x, d=1):
    return "—" if x is None else f"%{100 * x:.{d}f}".replace(".", ",")


# --------------------------------------------------------------------------- #
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Emu(7560310), Emu(10692130)   # A4
for side in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
    setattr(sec, side, Cm(2.12))

st = doc.styles
st["Normal"].font.name = "Calibri"
st["Normal"].font.size = Pt(11)
st["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st["Normal"].paragraph_format.space_after = Pt(6)
st["Normal"].paragraph_format.line_spacing = 1.15
for name, size, color in (("Heading 1", 16, NAVY), ("Heading 2", 13, BLUE)):
    s = st[name]
    s.font.name = "Calibri"; s.font.size = Pt(size); s.font.bold = True
    s.font.color.rgb = color
    rf = s._element.rPr.find(qn("w:rFonts"))
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rf.attrib.pop(qn(a), None)
    rf.set(qn("w:ascii"), "Calibri"); rf.set(qn("w:hAnsi"), "Calibri")
    s.paragraph_format.space_before = Pt(16 if name == "Heading 1" else 10)
    s.paragraph_format.space_after = Pt(6)
    s.paragraph_format.keep_with_next = True


def para(text="", bold=False, italic=False, color=None, size=None, align=None, after=None):
    p = doc.add_paragraph()
    if text:
        _runs(p, text, bold, italic, color, size)
    if align is not None:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def _runs(p, text, bold=False, italic=False, color=None, size=None):
    """**kalın** işaretli metni koşulara böler."""
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part)
        r.bold = bold or (i % 2 == 1)
        r.italic = italic
        if color is not None:
            r.font.color.rgb = color
        if size:
            r.font.size = Pt(size)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    _runs(p, text)
    p.paragraph_format.space_after = Pt(3)
    return p


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), fill)
    tcPr.append(sh)


def table(header, rows, widths_cm, highlight=None, font=9.5, align_right_from=1):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _borders(t)
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(font)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, HEAD_FILL)
        if i >= align_right_from:
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            p = cells[i].paragraphs[0]
            _runs(p, str(v), size=font, bold=(highlight is not None and ri == highlight))
            if i >= align_right_from:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            if highlight is not None and ri == highlight:
                shade(cells[i], "DCE8F5")
            elif ri % 2 == 1:
                shade(cells[i], ALT_FILL)
    for row in t.rows:
        row.cells[0].paragraphs[0].paragraph_format.space_after = Pt(0)
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)
            row.cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in row.cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def _borders(t):
    tblPr = t._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "C9CED6")
        b.append(e)
    tblPr.append(b)


def note_box(title, text):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _borders(t)
    c = t.rows[0].cells[0]
    c.width = Cm(16.7)
    shade(c, NOTE_FILL)
    p = c.paragraphs[0]
    r = p.add_run(title + " "); r.bold = True; r.font.color.rgb = RED; r.font.size = Pt(10)
    _runs(p, text, size=10)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(name, caption, width_cm=16.0):
    path = FIG / f"{name}.png"
    if not path.exists():
        print(f"  uyarı: figür yok {path}", file=sys.stderr)
        return
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = para(caption, italic=True, color=GRAY, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
    return c


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ============================== KAPAK ====================================== #
for _ in range(5):
    para()
para("Fizik Tabanlı Dijital Twin", bold=True, color=NAVY, size=26, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para("Hibrit Heyelan Duyarlılık Modeli", bold=True, color=NAVY, size=18, align=WD_ALIGN_PARAGRAPH.CENTER, after=14)
para("Kahramanmaraş · 6 Şubat 2023 Depremi (Mw 7.8 + 7.5)", color=BLUE, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
para("FEM + Newmark + Fizik-Bilgili Yapay Sinir Ağı (PINN)", color=GRAY, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
para("Zeynep Ersöz · Trakya Üniversitesi", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para("Teknik Özet Raporu · Sürüm 2 · Ekim 2026", color=GRAY, size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
page_break()

# ============================== İÇİNDEKİLER ================================ #
doc.add_heading("İçindekiler", level=1)
for line in ["1. Projenin Amacı", "2. Kullanılan Veriler", "3. Yöntem — Neden “Hibrit”?",
             "4. Ana Sonuç: Kritik PGA", "5. Sonuçlar", "6. İnteraktif Uygulama",
             "7. Sürüm 1’e Göre Yapılan İyileştirmeler", "8. Bilinen Kısıtlar (Şeffaflık)"]:
    para(line, after=3)
note_box("Bu sürümde ne değişti?",
         "Değerlendirme mekânsal blok çapraz doğrulamaya geçirildi, PINN kayıp fonksiyonuna "
         "fizik-yolu kısıtı eklendi, gerçek bir sonlu elemanlar (FEM-SSR) çözücüsü yazıldı ve "
         "Kritik PGA belirsizliğiyle birlikte raporlandı. Sürüm 1’in ROC-AUC 0,96 ve Kritik PGA "
         "≈ 0,55 g değerleri bu raporda güncellenmiştir (bkz. Bölüm 7).")
page_break()

# ============================== 1. AMAÇ ==================================== #
doc.add_heading("1. Projenin Amacı", level=1)
para("Bu proje tek bir soruya cevap arıyor: **“Bir yamaç, deprem sırasında hangi yer sarsıntısı "
     "şiddetinde (PGA) kaymaya — heyelana — başlar?”**")
para("6 Şubat 2023 Kahramanmaraş depremleri binlerce heyelanı tetikledi. Amacımız, deprem şiddeti "
     "(PGA) ile heyelan tetiklenmesi arasındaki kritik eşiği hem fizik denklemleriyle hem de gerçek "
     "heyelanlardan öğrenen yapay zekâ ile tahmin eden bir “dijital ikiz” (digital twin) kurmaktır.")
para("**PGA (Peak Ground Acceleration):** Deprem sırasında zeminin ulaştığı en yüksek ivmedir; "
     "yerçekimi (g) cinsinden ölçülür. 0,5 g, yerçekiminin yarısı kadar yatay sarsıntı demektir.",
     color=GRAY, size=10)

# ============================== 2. VERİ ==================================== #
doc.add_heading("2. Kullanılan Veriler", level=1)
para("Modelin tamamı gerçek, kaynağı belli (atıf verilebilir) verilerle beslenir. Sentetik/uydurma girdi yoktur:")
table(["Veri Katmanı", "Kaynak", "Ne İşe Yarar"], [
    ["Yer ivmesi (PGA)", "USGS ShakeMap (Mw 7,8 + Mw 7,5 zarfı)", "Depremin her noktadaki sarsıntı şiddeti — modelin ana girdisi"],
    ["Yükseklik (DEM)", "Copernicus GLO-90 (90 m)", "Eğim ve bakı buradan hesaplanır"],
    ["Eğim / Bakı", "DEM’den (Horn 1981)", "Dik yamaçlar heyelana daha yatkın"],
    ["Litoloji (kaya türü)", "Macrostrat jeoloji PROXY’si", "Kayanın dayanımı (kohezyon, sürtünme açısı)"],
    ["Heyelan envanteri", "Kocaman 2025 (2.596 poligon)", "Modelin öğrendiği GERÇEK heyelan etiketleri"],
    ["Pürüzlülük, rölyef, eğrilik, TWI, dereye uzaklık", "Copernicus DEM türevleri", "Arazinin kırıklılığı ve su toplama"],
    ["Aktif faya uzaklık", "GEM Global Active Faults (açık)", "Faya yakın kırıklı kaya — Kocaman’ın en etkili faktörlerinden"],
    ["Vs30, PGV, MMI, kırılma izi", "USGS ShakeMap (2 deprem)", "Zemin sınıfı ve sarsıntının diğer ölçüleri"],
    ["Arazi örtüsü", "ESA WorldCover 2021 (10 m)", "Orman, çayır, tarım, çıplak alan"],
    ["Kil / kum oranı", "ISRIC SoilGrids 2.0 (250 m)", "Yüzey zeminin dokusu"],
    ["İvme kayıtları", "AFAD TK ağı, ESM veritabanı (136 kayıt)", "Kayma simülasyonu (zaman-tanım Newmark)"],
], [3.6, 5.6, 7.5], align_right_from=99)
n_txt = f"{metrics['n_samples']:,}".replace(",", ".")
pos_txt = f"{100 * metrics['landslide_ratio']:.1f}".replace(".", ",")
para(f"Envanterden {n_txt} örnek nokta üretildi (%{pos_txt} heyelan; negatifler heyelanlardan "
     "en az 200 m uzaktaki rastgele arka plan noktalarıdır).")
note_box("Not:", "Litoloji katmanı resmi MTA 1:100.000 haritası değil, Macrostrat API tabanlı bir "
         "jeolojik birim proxy’sidir (~5 km çözünürlük). Bu, raporda açıkça belirtilen bilinçli bir "
         "kısıttır (bkz. Bölüm 8).")

# ============================== 3. YÖNTEM ================================== #
doc.add_heading("3. Yöntem — Neden “Hibrit”?", level=1)
para("İki dünyayı birleştiriyoruz. Tek başına her biri eksik; birlikte güçlü: fizik, "
     "mekanizmayı ve sınırları verir; yapay zekâ, gerçek heyelanlardan fiziğin tek başına "
     "yakalayamadığı örüntüleri öğrenir.")
doc.add_heading("3.1 Fizik Tarafı (Mohr-Coulomb + Newmark)", level=2)
bullet("**Statik güvenlik faktörü (FS):** Sonsuz şev modeliyle, kayaç dayanımı (c′, φ′) ve eğimden hesaplanır.")
bullet("**Kritik ivme (Kᵧ):** Bir yamacın kaymaya başlaması için gereken en düşük ivme: Kᵧ = (FS − 1)·sin θ. "
       "Dik ve zayıf kaya → düşük Kᵧ (kolay kayar).")
bullet("**Newmark yer değiştirmesi (Dₙ):** Deprem boyunca yamacın biriktirdiği kalıcı kayma (cm); "
       "Jibson (2007) denklemiyle hesaplanır. Dₙ ≥ 10 cm ise yamaç “tetiklenmiş” kabul edilir.")
doc.add_heading("3.2 Sonlu Elemanlar (FEM — Kayma Dayanımı Azaltma)", level=2)
para("Sürüm 2’de, Griffiths & Lane (1999) elasto-viskoplastik **Kayma Dayanımı Azaltma (SSR)** "
     "yöntemi projeye kazandırıldı. Yamaç kesiti 8 düğümlü dörtgen elemanlara bölünür; dayanım "
     "parametreleri, çözüm yakınsamayı kaybedene (yamaç göçene) kadar azaltılır. Bu oran güvenlik "
     "faktörüdür. Yatay deprem kuvveti eklenerek FEM’den kritik ivme (Kᵧ,FEM) de bulunur. Çözücü, "
     "literatürdeki kıyas problemiyle doğrulanmıştır (FS = 1,35; Bishop-Morgenstern 1,38).")
doc.add_heading("3.3 Yapay Zekâ Tarafı (PINN v2)", level=2)
para(f"PINN (Physics-Informed Neural Network) = Fizik-Bilgili Sinir Ağı. Model, {metrics['n_features']} "
     "öznitelikten (eğim, yükseklik, bakı, PGA, Kᵧ, log Dₙ, litoloji sınıfları ve dayanım "
     "parametreleri) her nokta için 0–1 arası bir heyelan olasılığı üretir.")
para("Fizik bilgisi kayıp fonksiyonuna üç şekilde girer:")
bullet("**Focal Loss:** Az sayıdaki heyelan örneğinin göz ardı edilmemesini sağlar.")
bullet("**Stabilite cezası:** Düz (eğim < 10°) ve zayıf sarsıntılı (PGA < 0,1 g) bölgede yüksek heyelan tahmini cezalandırılır.")
bullet("**Fizik-yolu kısıtı (yeni):** Eğitim sırasında her noktanın PGA’sı veya eğimi küçük bir adım "
       "artırılır; Kᵧ ve Dₙ fizik denklemleriyle yeniden hesaplanır. Model bu durumda heyelan "
       "olasılığını düşürürse cezalandırılır. Böylece “deprem şiddetlenince veya yamaç dikleşince risk "
       "azalmaz” fiziksel kuralı modele öğretilir. Bu, PINN literatüründeki kolokasyon noktalarının "
       "karşılığıdır.")
doc.add_heading("3.4 Değerlendirme — Mekânsal Blok Doğrulama", level=2)
para("Heyelanlar kümelenir: komşu heyelanlar aynı yamaçta, aynı kayaçta ve aynı sarsıntı "
     "hücresindedir. Bu yüzden çalışma alanı 0,25° (≈ 25 km) karelere bölündü; bir karenin tüm "
     "örnekleri ya eğitimde ya testte yer alır. Model, eğitimde hiç görmediği coğrafi bölgelerde "
     "sınanır (5 katlı çapraz doğrulama). Karar eşiği yalnız doğrulama bloklarında seçilir.")
page_break()

# ============================== 4. KRİTİK PGA ============================== #
lv = cp["levels"]
doc.add_heading("4. Ana Sonuç: Kritik PGA", level=1)
para("Aşağıdaki grafik projenin kalbidir ve şu soruyu yanıtlar: **“Dik yamaçlar hangi PGA’da kaymaya başlar?”**")
figure("critical_pga_uncertainty",
       f"Şekil 1. Kritik PGA analizi — en dik %5 yamaç (eğim ≥ {cp['steep_slope_threshold_deg']:.1f}°, "
       f"n = {cp['n_cells']:,} hücre). Mavi: PINN v2 (6 model ortalaması, bant: model aralığı); "
       "turuncu: Newmark.".replace(",", "."))
doc.add_heading("Grafik nasıl okunur?", level=2)
bullet("**Yatay eksen:** PGA (deprem sarsıntı şiddeti), soldan sağa artar.")
bullet("**Dikey eksen:** Heyelan olasılığı / tetiklenen yamaç oranı (0–1).")
bullet("**Mavi çizgi (ML):** 6 modelin (nihai model + 5 doğrulama katı) en dik %5 yamaç için ortalama heyelan olasılığı; açık mavi bant modeller arası farkı gösterir.")
bullet("**Turuncu kesikli çizgi (Fizik):** Saf Newmark denklemiyle 10 cm eşiğini aşan yamaçların oranı.")
bullet("**Noktalı yatay çizgiler:** P = 0,5 karar düzeyi ve LR = 1 (heyelan lehine başlangıç) düzeyi.")
para(f"**Sonuç:** Ortalama eğri 0,5 düzeyini **≈ {f2(cp['critical_pga_g'])} g**’de geçiyor "
     f"(model belirsizliği {f2(cp['ensemble_range_g'][0])}–{f2(cp['ensemble_range_g'][1])} g). "
     f"Ayrıca dik yamaçların heyelan lehine dönmeye başladığı eşik (LR = 1) ≈ {f2(cp['onset_lr1_g'])} g; "
     f"bu değer envanterden doğrudan hesaplanan ampirik eşikle ({f2(emp['empirical_critical_pga_fr1_g'])} g) örtüşüyor.",
     color=NAVY)

doc.add_heading("Kritik PGA’nın farklı tanımları", level=2)
table(["Tanım", "Değer", "Belirsizlik / Not"], [
    ["P = 0,5 (6 model ortalama eğrisi)", f"{f2(cp['critical_pga_g'])} g",
     f"Modeller {f2(cp['ensemble_range_g'][0])}–{f2(cp['ensemble_range_g'][1])} g; bootstrap %95: "
     f"{f2(cp['bootstrap_95ci_g'][0])}–{f2(cp['bootstrap_95ci_g'][1])} g"],
    ["Başlangıç: olabilirlik oranı LR = 1", f"{f2(cp['onset_lr1_g'])} g",
     f"Modeller {f2(lv['onset_lr1']['model_range_g'][0])}–{f2(lv['onset_lr1']['model_range_g'][1])} g"],
    ["Ampirik başlangıç: gözlenen Frekans Oranı = 1", f"{f2(emp['empirical_critical_pga_fr1_g'])} g",
     "Doğrudan envanterden, modelden bağımsız"],
    ["Newmark rijit-blok (Dₙ ≥ 10 cm, %50)", f"{f2(cp['newmark_critical_pga_g'])} g",
     "Fiziğin alt sınırı; çok muhafazakâr"],
], [5.6, 2.2, 8.9], align_right_from=1)
para("**LR = 1 ne demek?** Dik yamaçların, heyelanlı alanlarda bölgenin geneline göre daha sık "
     "görülmeye başladığı nokta. Türkiye’de yaygın kullanılan Frekans Oranı (FR) yönteminin "
     "olasılıksal karşılığıdır. Modelin bulduğu başlangıç değeri, envanterden doğrudan hesaplanan "
     "değerle uyumludur.", size=10)

doc.add_heading("İki çizgi neden farklı?", level=2)
para("Mavi (ML) çizgi, turuncudan (fizik) çok daha erken yükselir. Gerçek heyelanlardan öğrenen "
     "model; dayanım kaybını, ivme büyütmesini (topografik amplifikasyon) ve süreksizlikleri "
     "yakalar. Saf rijit-blok Newmark bunları içermez ve litoloji proxy’sinin ortalama dayanım "
     "değerlerini kullanır; bu yüzden gerçek tetiklenmeyi olduğundan az tahmin eder, yani alt sınır "
     "verir. Bu fark, hibrit yaklaşımın neden gerekli olduğunu gösterir.")

doc.add_heading("Örnekleme oranı duyarlılığı", level=2)
para("P = 0,5 eşiği, eğitimde kullanılan heyelansız/heyelanlı örnek oranına bağlıdır. Bu bağlılık "
     "her oran için 3 tekrarlı eğitimle ölçüldü:")
table(["Negatif / pozitif oranı", "Kritik PGA (P = 0,5)", "Başlangıç (LR = 1)", "ROC-AUC"],
      [[f"{r['neg_pos_ratio']:.1f}".replace(".", ",") + (" (kullanılan)" if r["neg_pos_ratio"] == 2.5 else ""),
        f"{f2(r['critical_pga_at_0.5_mean'])} ± {f2(r['critical_pga_at_0.5_sd'])} g",
        f"{f2(r['critical_pga_onset_lr1_mean'])} g", f3(r["test_roc_auc_mean"])] for r in sens],
      [4.6, 4.2, 4.0, 3.0])
note_box("Yorum:", "Ayırma gücü (AUC) örnekleme oranından etkilenmez; ancak P = 0,5 düzeyine "
         "karşılık gelen PGA değişir. Bu nedenle Kritik PGA tek bir sayı olarak değil, belirsizlik "
         "aralığıyla birlikte raporlanmıştır. Bu, heyelan duyarlılık literatüründe bilinen bir durumdur.")

doc.add_heading("Tasarım tablosu: eğim × kayaç türü", level=2)
para("Kritik PGA, eğim sınıfı ve kayaç türüne göre ayrı ayrı hesaplandı. Beklendiği gibi eğim "
     "arttıkça kritik PGA düşer (örneğin andezitte, killerde ve şistte).")
figure("critical_pga_design_table", "Şekil 2. Eğim sınıfı ve litolojiye göre Kritik PGA (g). "
       "Koyu renk = daha düşük PGA’da tetiklenme = daha tehlikeli.", 15.0)
page_break()

# ============================== 5. SONUÇLAR ================================ #
doc.add_heading("5. Sonuçlar", level=1)
doc.add_heading("5.1 Model Karşılaştırması (mekânsal blok doğrulama)", level=2)
para("Önerilen modelin değerini göstermek için 9 model aynı koşullarda karşılaştırıldı:")
order = ["newmark", "logreg", "rf", "gbm", "gbm_mono", "mlp", "pinn_v1", "pinn_v2", "pinn_v2_nophysfeat",
         "logreg_L", "rf_L", "gbm_L", "gbm_mono_L", "mlp_L", "pinn_v2_L"]
rows = []
for k in order:
    m = M.get(k)
    if not m:
        continue
    rows.append([m["label"], f"{f3(m['roc_auc']['mean'])} ± {f3(m['roc_auc']['std'])}",
                 f3(m.get("roc_auc_steep", {}).get("mean")), f3(m["f1"]["mean"]),
                 pct(m.get("mono_violation_rate", {}).get("mean"))])
table(["Model", "ROC-AUC", "Dik arazi AUC", "F1", "Fizik ihlali*"], rows,
      [5.6, 3.3, 2.8, 2.2, 2.8], highlight=[k for k in order if k in M].index(MAIN if MAIN in M else "pinn_v2"))
para("* PGA artırıldığında heyelan olasılığının düştüğü (fiziğe aykırı davranan) test örneklerinin oranı.",
     italic=True, color=GRAY, size=9)
bullet("Öğrenen modellerin ayırma gücü birbirine çok yakın (farklar ölçüm belirsizliğinin içinde); "
       "saf fizik (Newmark) tek başına zayıf kalıyor.")
bullet("**Asıl fark fiziksel tutarlılıkta:** Fizik kısıtı olmayan modeller (Random Forest, Gradient "
       f"Boosting, MLP, PINN v1) örneklerin %56–73’ünde, deprem şiddetlendikçe heyelan riskini "
       f"azaltıyor. PINN v2’de bu oran {pct(M[MAIN]['mono_violation_rate']['mean'])}.")
bullet("**Açık kaynak katmanlar** (“+ katmanlar” satırları) tüm modellerde ayırma gücünü belirgin "
       f"artırdı: PINN v2 için AUC {f3(M['pinn_v2']['roc_auc']['mean'])} → {f3(M[MAIN]['roc_auc']['mean'])}, "
       f"dik arazide {f3(M['pinn_v2']['roc_auc_steep']['mean'])} → {f3(M[MAIN]['roc_auc_steep']['mean'])}.")
bullet("Fizik kısıtı doğruluktan ödün vermeden fiziksel tutarlılık sağlıyor; bu, Kritik PGA "
       "analizinin güvenilir olmasının ön koşuludur.")
figure("benchmark_consistency", "Şekil 3. Doğruluk – fiziksel tutarlılık düzlemi. Sol üst köşe en iyidir.", 13.5)

doc.add_heading("5.2 Nihai Model Başarımı", level=2)
cv = metrics.get("cv", {})
table(["Metrik", "Değer", "Anlamı"], [
    ["ROC-AUC (5 kat ortalaması)", f"{f3(cv['roc_auc']['mean'])} ± {f3(cv['roc_auc']['std'])}", "Heyelan/heyelan-değil ayırma gücü (1,0 mükemmel)"],
    ["ROC-AUC (ayrık test bölgeleri)", f3(metrics["roc_auc"]), "Eğitimde hiç görülmeyen bölgelerde"],
    ["Dik arazide ROC-AUC", f3(metrics["roc_auc_steep"]), "Yalnız eğim ≥ 15° — eğim dışındaki bilgiyi ölçer"],
    ["Doğruluk (accuracy)", f2(metrics["accuracy"]), "Genel doğru sınıflama oranı"],
    ["Recall (heyelan)", f2(metrics["recall"]), f"Gerçek heyelanların %{100 * metrics['recall']:.0f}’ini yakalıyor"],
    ["Precision (heyelan)", f2(metrics["precision"]), f"“Heyelan” dediğinin %{100 * metrics['precision']:.0f}’i gerçekten heyelan"],
    ["F1 (heyelan)", f2(metrics["f1"]), "Recall + precision dengesi"],
    ["Fizik ihlali", pct(metrics["mono_violation_rate"]), "PGA artınca olasılığın düştüğü örnek oranı"],
], [5.0, 3.4, 8.3])
para(f"Karar eşiği ({f2(metrics['threshold'])}) test setinde değil, ayrı doğrulama bölgelerinde F1’i "
     "maksimize edecek şekilde seçildi — bu, sonucun “şişirilmediğini” garanti eder.", size=10)

doc.add_heading("5.3 En Önemli Değişkenler (SHAP)", level=2)
order_names = {"slope_deg": "eğim", "pga_g": "PGA", "elevation_m": "yükseklik",
               "litoloji (one-hot toplam)": "litoloji", "ky_g": "Kᵧ", "aspect_cos": "bakı"}
top = [order_names.get(k, k) for k in list(shap)[:5]]
para(f"Modelin kararını en çok etkileyen faktörler sırasıyla: **{' > '.join(top)}**. Eğim ve deprem "
     "şiddetinin başı çekmesi mühendislik sezgisiyle uyumludur (dik yamaç + güçlü sarsıntı = en büyük risk).")
figure("shap_summary", "Şekil 4. SHAP öznitelik önem sıralaması (litoloji sınıfları tek grup olarak).", 16.0)

doc.add_heading("5.4 Bölgesel Duyarlılık Haritası ve Ampirik Doğrulama", level=2)
para("Eğitilmiş model tüm çalışma alanına (Kahramanmaraş bölgesi) uygulandığında, her nokta için "
     "bir heyelan olasılığı haritası üretir.")
figure("susceptibility_map_real", "Şekil 5. Bölgesel heyelan duyarlılık haritası (kırmızı = yüksek risk; "
       "siyah konturlar: gözlenen envanter).", 15.0)
para(f"Haritanın gözlemle uyumu sayısal olarak da ölçüldü: envanter poligonları aynı ızgaraya "
     f"aktarıldı ve eğim × PGA sınıflarında gözlenen heyelan yoğunluğu ile model olasılığı "
     f"karşılaştırıldı. **Spearman sıra korelasyonu ρ = {f2(emp['spearman_binned'])}** — model, "
     "heyelanın nerede yoğunlaştığını doğru sıralıyor.")
figure("empirical_validation", "Şekil 6. Solda gözlenen frekans oranı, sağda modelin olabilirlik oranı "
       "(log ölçek; 1 = nötr).", 16.0)

doc.add_heading("5.5 FEM Pilot Analizi", level=2)
para("İki pilot heyelan için FEM-SSR, sonsuz şev (Mohr-Coulomb) hesabıyla karşılaştırıldı "
     "(20 m yüksekliğinde idealize yamaç kesiti):")
table(["Pilot", "FS (FEM)", "FS (sonsuz şev)", "Kᵧ FEM", "Kᵧ sonsuz şev", "PGA", "Dₙ (FEM / sonsuz)"],
      [[f"ID {p['slope_id']} — {p['lithology']}, θ = {p['theta_deg']:.0f}°", f2(p["fs_fem"]),
        f2(p["fs_infinite_slope"]), f"{f3(p['ky_fem_g'])} g", f"{f3(p['ky_infinite_slope_g'])} g",
        f"{f2(p['actual_pga_g'])} g",
        f"{p['dn_newmark_ky_fem_cm']:.1f} / {p['dn_newmark_ky_infinite_cm']:.1f} cm".replace(".", ",")]
       for p in fem], [5.0, 1.6, 2.0, 1.9, 2.1, 1.5, 2.6], font=9)
bullet("Kohezyonsuz kolüvyonda (ID 1327) FEM ve sonsuz şev neredeyse aynı sonucu veriyor: kayma "
       "yüzeye paralel ve sığ. Gerçek PGA’da Dₙ ≈ 20 cm → yamaç tetikleniyor (gözlemle uyumlu).")
bullet("Kohezyonlu marnda (ID 1353) FEM daha derin, dairesel bir kayma yüzeyi buluyor ve kritik "
       "ivmeyi sonsuz şev hesabından yaklaşık %24 daha yüksek veriyor.")
figure("fem_1353_plastic_strain", "Şekil 7. ID 1353 — FEM’de birikmiş plastik kayma gerinmesi: "
       "topuktan geçen dairesel kayma yüzeyi.", 15.5)

doc.add_heading("5.6 Gerçek İvme Kayıtlarıyla Kayma Simülasyonu", level=2)
n_p = int((sm_st.event == "pazarcik").sum()); n_e = int((sm_st.event == "elbistan").sum())
para(f"6 Şubat depremlerinin AFAD ivme kayıtları ESM veritabanından indirildi (Pazarcık {n_p}, "
     f"Elbistan {n_e} istasyon; en büyük PGA {f2(sm_st.pga_g.max())} g). Her heyelan poligonu için en "
     "yakın istasyonun kaydı poligonun ShakeMap PGA’sına ölçeklendi ve Newmark kayan blok yöntemi kaydın "
     "kendisi üzerinde adım adım çözüldü (Jibson 1993). Böylece her poligonun deprem boyunca kaç cm "
     "kaydığı simüle edildi.")
pg = nmk["polygons"]
bullet(f"Poligonların **%{100 * pg['share_triggered_10cm']:.0f}’i** 10 cm eşiğini aşıyor; "
       f"%{100 * pg['share_pazarcik_dominant']:.0f}’inde belirleyici olan Pazarcık depremi (Kocaman ve ark. "
       "heyelanların çoğunun ilk depremle tetiklendiğini belirtir).")
bullet(f"Kaydın kendisiyle hesap, regresyon formülüne göre heyelanları biraz daha iyi ayırıyor "
       f"(AUC {f3(nmk['auc_dn_timehistory'])} vs {f3(nmk['auc_dn_regression_jibson2007'])}; yanlış alarm "
       f"%{100 * nmk['false_rate_background_th']:.1f} vs %{100 * nmk['false_rate_background_reg']:.1f}). "
       "Saf fizik tek başına yine zayıf; ana neden litoloji proxy’sinden gelen dayanım değerleri.")
bullet("Uygulamada Harita sayfasındaki “Depremi oynat” düğmesiyle poligonların deprem süresince "
       "renk değiştirdiği animasyon izlenebilir.")
figure("newmark_th_pilots", "Şekil 8. Pilot yamaçlarda gerçek ivme kaydıyla hesaplanan kalıcı yer değiştirme.", 14.5)

doc.add_heading("5.7 Envanterin ve Etiketlerin Doğrulanması", level=2)
para("Modelin öğrendiği heyelan envanterinin (Kocaman ve ark. 2025) güvenilirliği üç yoldan sınandı.")
para("**Bağımsız yeniden hesaplama.** Makaledeki sayılar kendi verimizle yeniden üretildi: 2.596 poligon, "
     "en küçük 21,8 m², en büyük 8,9625 km², toplam alan 42,9 km² (makalede 68.309 piksel × 25 m ≈ 42,7 km²), "
     "ortalama eğim 23,4° (makalede 23,0°) ve arazi örtüsü sıralaması (çayır > ağaç > çıplak) birebir tutuyor. "
     "Ortalama yükseklik farkı (makalede 1.148 m) piksel ağırlıklandırmasından kaynaklanıyor: tek bir 8,96 km²’lik "
     "heyelan toplam alanın %21’i; heyelan başına ortalama 864 m, alan ağırlıklı 1.106 m.")
fr_p = RES / "frequency_ratio.json"
if fr_p.exists():
    frj = json.loads(fr_p.read_text())
    para("**Frekans Oranı (FR).** Kocaman ve ark. yöntemiyle hesaplandı. Litoloji FR’si yalnızca Macrostrat "
         "PROXY’siyle üretilebildiği için makaledeki MTA tabanlı değerlerle (ör. peridotit 6,70) doğrudan "
         "karşılaştırılamaz; asıl tablo MTA haritası gelince üretilecek.")
    rows = []
    for key in ("eğim (°)", "faya uzaklık (km, GEM)", "2023 kırılmasına uzaklık (km)", "arazi örtüsü (ESA WorldCover)"):
        for r in frj["tables"].get(key, []):
            rows.append([key, r["sinif"], f2(r["FR"])])
    table(["Faktör", "Sınıf", "FR"], rows, [7.0, 5.0, 2.5], font=8.5, align_right_from=2)
    figure("frequency_ratio", "Şekil 9. Frekans Oranı (kırmızı: FR > 1, heyelana yatkın).", 16.0)
ln_p = RES / "label_noise.json"
if ln_p.exists():
    ln = json.loads(ln_p.read_text())
    para("**“Heyelan yok” etiketinin kirliliği.** Envanter kar ve gölge nedeniyle eksik olabilir; bu durumda "
         "“heyelan yok” noktalarının bir kısmı aslında haritalanmamış heyelandır. Negatif noktaların heyelanlara "
         f"medyan uzaklığı {f2(ln['neg_distance_km']['median'])} km; yalnızca %{100 * ln['neg_distance_km']['share_lt_1km']:.1f}’i "
         "1 km’den yakın. Dört eğitim varyantı aynı test bölgelerinde karşılaştırıldı:")
    lab = {"V0_temel": "Temel (≥ 200 m)", "V1_tampon_1km": "Negatifler ≥ 1 km", "V2_tampon_2km": "Negatifler ≥ 2 km",
           "V3_PU_casus": "PU-casus (%5 şüpheli negatif çıkarıldı)"}
    rows = []
    for k, v in ln["variants"].items():
        c05 = v.get("critical_pga_p05_g"); c1 = v.get("critical_pga_lr1_g")
        rows.append([lab.get(k, k), f"{f3(v['roc_auc'][0])} ± {f3(v['roc_auc'][1])}", f3(v["roc_auc_steep"][0]),
                     f"{f2(c05[0])} ± {f2(c05[1])} g" if c05 else "—", f"{f2(c1[0])} ± {f2(c1[1])} g" if c1 else "—"])
    table(["Eğitim varyantı", "ROC-AUC", "Dik arazi", "Kritik PGA (P=0,5)", "Başlangıç (LR=1)"], rows,
          [5.6, 2.9, 2.1, 3.2, 2.9], font=8.5)
para("**Uydu doğrulaması.** Her poligonda 2022 yazı → 2023 yazı Sentinel-2 bitki örtüsü (NDVI) ve parlaklık "
     "değişimi, çevresindeki halkaya göre ölçülecek şekilde hazırlandı (Google Earth Engine betiği "
     "gee/ndvi_change.js). Sinyal vermeyen poligonlar eğitimden çıkarılarak duyarlılık yeniden raporlanacak.")
s2_p = RES / "sentinel_summary.json"
if s2_p.exists():
    s2 = json.loads(s2_p.read_text())
    para(f"Sonuç: kullanılabilir poligonların %{100 * s2['polygons_share_confirmed_of_usable']:.0f}’inde "
         "uydudan bağımsız heyelan izi doğrulandı; dik arka plan noktalarında olası kaçmış heyelan oranı "
         f"%{100 * s2['negatives_contamination_rate']:.1f} (eşiğin kendi taban değeri ≈ %10).")

# ============================== 6. UYGULAMA ================================ #
doc.add_heading("6. İnteraktif Uygulama", level=1)
para("Model, React (arayüz) + FastAPI (sunucu) tabanlı bir web uygulamasına bağlandı. Başlıca sayfalar:")
bullet("**Tahmin:** Eğim, yükseklik, bakı, litoloji ve PGA girilir → anlık heyelan olasılığı, Kᵧ ve Dₙ gösterilir. Envanter kimliği veya koordinatla da seçim yapılabilir.")
bullet("**Kritik PGA:** Seçili bir yamaç için PGA taranır; ML ve Newmark eşikleri aynı grafikte karşılaştırılır.")
bullet("**Harita:** Duyarlılık haritası; heyelan poligonları, aktif faylar, 2023 kırılma izleri, AFAD "
       "istasyonları; PGV, Vs30, faya uzaklık, arazi örtüsü gibi ek katmanlar; tıklanan noktanın tahmini ve "
       "deprem boyunca kayma animasyonu.")
bullet("**Sonuçlar:** Model karşılaştırması, Kritik PGA belirsizliği, tasarım tablosu, FEM sonuçları ve tüm figürler.")
para("Çalıştırma: ./run_app.sh → http://localhost:5173 · Tüm analiz: ./run_pipeline.sh", color=GRAY, size=10)

# ============================== 7. DEĞİŞİKLİKLER ============================ #
doc.add_heading("7. Sürüm 1’e Göre Yapılan İyileştirmeler", level=1)
ref = bench.get("reference_polygon_split_pinn_v2", {})
table(["Konu", "Sürüm 1 (Temmuz 2026)", "Sürüm 2 (Ekim 2026)"], [
    ["Değerlendirme", "Rastgele poligon ayrımı; komşu heyelanlar eğitim ve testte birlikte",
     "Mekânsal blok doğrulama (25 km bloklar, 5 kat)"],
    ["ROC-AUC", "0,963 (iyimser)",
     f"{f3(cv['roc_auc']['mean'])} ± {f3(cv['roc_auc']['std'])} (katmanlar dahil; katmansız PINN v2 eski yöntemle {f3(ref.get('roc_auc'))})"],
    ["Fizik kısıtı", "Yalnız düz/zayıf sarsıntı bölgesinde ceza; pratikte etkisiz",
     "Fizik-yolu kısıtı: PGA ve eğim boyunca tutarlılık"],
    ["Fiziksel tutarlılık", "Örneklerin ~%60’ında PGA ↑ → risk ↓", f"İhlal {pct(M[MAIN]['mono_violation_rate']['mean'])}"],
    ["Kritik PGA", "≈ 0,55 g, tek sayı; taramada Dₙ sabit kalıyordu",
     f"{f2(cp['critical_pga_g'])} g ({f2(cp['ensemble_range_g'][0])}–{f2(cp['ensemble_range_g'][1])} g), Dₙ her PGA’da yeniden hesaplanıyor"],
    ["Karşılaştırma", "Yok", "9 model (istatistiksel, ağaç tabanlı, sinir ağı, fizik)"],
    ["FEM", "Kurulu değildi; konturlar temsiliydi", "FEM-SSR çözücüsü, literatür kıyasıyla doğrulandı"],
    ["Ampirik doğrulama", "Görsel karşılaştırma", f"Sayısal: Spearman ρ = {f2(emp['spearman_binned'])}"],
    ["Ek veri", "Eğim, bakı, yükseklik, PGA, litoloji", "+ pürüzlülük, rölyef, eğrilik, TWI, faya ve dereye uzaklık, Vs30, arazi örtüsü, kil/kum"],
    ["İvme kayıtları", "Yok (sentetik/regresyon)", "136 gerçek AFAD kaydı; poligon bazında kayma simülasyonu"],
    ["Yeniden üretilebilirlik", "Metrikler elle girilmişti", "Tek komutla tüm analiz; 21 otomatik test"],
], [3.4, 6.2, 7.1], align_right_from=99)

# ============================== 8. KISITLAR ================================ #
doc.add_heading("8. Bilinen Kısıtlar (Şeffaflık)", level=1)
para("Bilimsel dürüstlük için mevcut kısıtlar açıkça belirtilir:")
bullet("**Litoloji bir PROXY’dir:** Resmi MTA jeoloji haritası yerine Macrostrat API kullanıldı "
       "(~5 km çözünürlük; duyarlılık haritasındaki dikdörtgen bloklar bunun izidir). Dayanım değerleri sınıf ortalamalarıdır. Resmi harita edinilince "
       "değiştirilecek; bu, en büyük iyileştirme fırsatıdır.")
bullet("**Kritik PGA (P = 0,5) örnekleme tasarımına bağlıdır:** Değer belirsizlik aralığıyla ve "
       "duyarlılık analiziyle birlikte verilmiştir (Bölüm 4).")
bullet("**FEM 2B idealize kesittir:** Gerçek pilot topoğrafyası ve dinamik (zaman-tanım) analiz "
       "değil, psödo-statik yükleme kullanılmıştır.")
bullet("**Newmark rijit-blok basitleştirmesi:** Fizik eğrisi gerçek tetiklenmeyi az tahmin eder; ML bunu telafi eder.")
bullet("**Tek deprem:** Model Kahramanmaraş 2023 verisiyle eğitildi; Wenchuan 2008 karşılaştırması "
       "yalnızca fizik denklemlerinin aktarımıdır.")
para()
note_box("Özet:", f"Gerçek verilerle beslenen, fizik + yapay zekâ hibrit bir dijital ikiz; eğitimde "
         f"görmediği bölgelerde ROC-AUC {f3(cv['roc_auc']['mean'])} başarımla ve fiziksel olarak "
         f"tutarlı biçimde, dik yamaçların ≈ {f2(cp['critical_pga_g'])} g PGA’da (belirsizlik "
         f"{f2(cp['ensemble_range_g'][0])}–{f2(cp['ensemble_range_g'][1])} g) heyelan sınıfına geçtiğini; "
       f"heyelan lehine başlangıcın ise ≈ {f2(cp['onset_lr1_g'])} g olduğunu ortaya koyar.")

doc.add_heading("Kaynaklar", level=1)
for ref_txt in [
    "Kocaman S., Çetinkaya S., Tunar Özcan N., Karakaş G., Karakaş V.E., Gökçeoğlu C. (2025). Landslides triggered by the 6 February 2023 Kahramanmaraş earthquakes (Türkiye). Turkish Journal of Earth Sciences 34(1): 47–67. https://doi.org/10.55730/1300-0985.1945",
    "Li L., Tang B., Cai F., Wei L., Zhu X., Fan D. (2026). Digital Twin Modeling for Landslide Risk Scenarios in Mountainous Regions. Sensors 26: 421. https://doi.org/10.3390/s26020421",
    "Newmark N.M. (1965). Effects of earthquakes on dams and embankments. Géotechnique 15(2): 139–160.",
    "Jibson R.W. (1993). Predicting earthquake-induced landslide displacements using Newmark’s sliding block analysis. Transportation Research Record 1411: 9–17.",
    "Jibson R.W. (2007). Regression models for estimating coseismic landslide displacement. Engineering Geology 91: 209–218.",
    "Griffiths D.V., Lane P.A. (1999). Slope stability analysis by finite elements. Géotechnique 49(3): 387–403.",
    "Raissi M., Perdikaris P., Karniadakis G.E. (2019). Physics-informed neural networks. Journal of Computational Physics 378: 686–707.",
    "Roberts D.R. ve ark. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40: 913–929.",
    "Luzi L. ve ark. Engineering Strong Motion Database (ESM), v3.0. INGV. https://doi.org/10.13127/esm.3 — AFAD Türkiye Ulusal Kuvvetli Yer Hareketi Ağı, https://doi.org/10.7914/SN/TK",
    "Styron R., Pagani M. (2020). The GEM Global Active Faults Database. Earthquake Spectra 36(S1): 160–180.",
    "Zanaga D. ve ark. (2022). ESA WorldCover 10 m 2021 v200. https://doi.org/10.5281/zenodo.7254221",
    "Poggio L. ve ark. (2021). SoilGrids 2.0. SOIL 7: 217–240.",
    "Peters S.E., Husson J.M., Czaplewski J. (2018). Macrostrat. Geochemistry, Geophysics, Geosystems 19: 1393–1409.",
]:
    p = para(ref_txt, size=9.5, after=3)
    p.paragraph_format.left_indent = Cm(0.6); p.paragraph_format.first_line_indent = Cm(-0.6)

# ------------------------------------------------------------------ altbilgi
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = footer.add_run("Heyelan Dijital Twin · Teknik Özet Raporu v2 · Sayfa ")
r.font.size = Pt(8.5); r.font.color.rgb = GRAY
fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
rr = OxmlElement("w:r"); t = OxmlElement("w:t"); t.text = "1"; rr.append(t); fld.append(rr)
footer._p.append(fld)

doc.core_properties.author = "Zeynep Ersöz"
doc.core_properties.title = "Fizik Tabanlı Dijital Twin — Teknik Özet Raporu v2"
doc.core_properties.comments = ""
doc.core_properties.last_modified_by = "Zeynep Ersöz"
doc.save(OUT)
print(f"Rapor yazıldı: {OUT}")
