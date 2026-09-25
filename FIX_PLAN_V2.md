# FIX_PLAN_V2 — `fix/elimination-hardening` branch'ini (5063f14) mustaqil tekshirish natijasida

> Bu reja `FIX_PLAN.md` (V1) ning davomi. V1 dagi bo'lim raqamlariga havola beriladi (masalan "V1 §4.3").
> V1 dan farqi: **avval ishga tushmaydigan narsalar**, keyin V1 dan bajarilmay qolgan bandlar, keyin yangi paydo bo'lgan regressiyalar.

---

## 0. Nega V2 kerak: da'vo va haqiqat

Oldingi agent "hammasi ajoyib, 85/100" degan. Uni mustaqil tekshirdim: 8 soniyalik klip bilan rasmiy `run_submission.py` ni ishga tushirdim, har bir modulni import qildim, svetoforni haqiqiy kadrlarda sinadim.

| Tekshiruv | Natija (HEAD = 5063f14) | Isbot |
|---|---|---|
| `detect_events` (Part A) | ❌ **Har bir videoda birinchi kadrdayoq yiqiladi** | `TypeError: ByteTrack.__init__() got an unexpected keyword argument 'track_thresh'` (solution.py:332) |
| `RiskEstimator` (Part B) | ❌ **`reset()` da yiqiladi** | Xuddi shu xato (solution.py:788) |
| Birinchi crash tuzatilsa | ❌ **Keyingi crash** | `active_events = {"_emitted": set()}` → `for (t_id, ev_label), ev_info in active_events.items()` → `ValueError: too many values to unpack` (solution.py:731, 754) |
| Veb-sayt `app.py` | ❌ **Import paytida yiqiladi** | `ImportError: cannot import name 'SCENE_CONFIG' from 'solution'` (app.py:16) |
| `src/tools/visualizer.py` | ❌ Yiqiladi | Endi mavjud bo'lmagan `SCENE_CONFIG`, `get_traffic_light_state`, `shift_scene_config` ni import qiladi |
| `predictions_samples.json` | ❌ Eski koddan qolgan | Oxirgi marta `6bc8ad6 last_test` da o'zgargan (agent o'zgarishlaridan oldin). Joriy kod uni qayta yarata olmaydi (crash) |
| Yangi svetofor (`src/traffic_light.py`) | ❌ Ishlamaydi | C3896/C3897/C3902 da ~100% `UNKNOWN`; qizil chiroq yonib turgan kadrlar ham `U`. Sababi: yonib turgan lampa ~5 qator piksel, modul esa ROI ning ≥5% (~160 px) yonishini talab qiladi. Yashil lampa bbox'ning "o'rta" uchdan biriga tushadi |
| `congestion` `inf` xatosi (V1 §4.4) | ❌ Tuzatilmagan | 8 s klipda `[0.0, 0.801, "congestion"]` — aynan birinchi 0.8 s |
| `stop_line` teskari mantiq (V1 §4.2) | ❌ Tuzatilmagan | solution.py:534–541 hanuz 1- va 2-chiziq *orasida* to'xtaganlarni belgilaydi |
| V1 §4 dagi 13 ta qoida tuzatishi | ❌ ~0 tasi bajarilgan | red_light, stopped_vehicle, jaywalking, failure_to_yield, wrong_way, illegal_turn, u_turn, solid_line, accident, road_obstacle — kod o'zgarmagan |
| `near_miss` | ⚠️ **Regressiya** | Endi "markazlar orasi < 30 px **absolyut** va nisbiy tezlik > 10 px/s". 4K da bu faqat bir obyektning ikki marta aniqlanishi (car+truck ikki track) bo'lganda sodir bo'ladi, ya'ni FP generatori. Tormoz sharti olib tashlangan |
| `jaywalking` | ⚠️ **Regressiya** | 18-zona (piyoda oroli) `sidewalks` dan `ped_refuge` ga ko'chirilgan, lekin jaywalking faqat crosswalk/sidewalk ni istisno qiladi. Endi orolda turgan piyoda jaywalker hisoblanadi |
| Part B TTC | ⚠️ Birliklar mos emas | `dist` 4K pikselda, tezlik esa 640-masshtabda, shuning uchun TTC ~6× katta chiqadi va risk deyarli yonmaydi |
| Dev set (V1 §2) | ❌ Yo'q | `devset/labels.json` yo'q; `label_tool.py` — faqat `print(...)`. `make_candidates.py` JSON'ning yuqori darajasini aylanadi (`"team"`, `"log"` ham video deb olinadi) va `annotate.py` da yo'q `--pred` argumentini beradi |
| Testlar, `docs/`, `ruff`, `half=True`, stride (V1 §6, §8) | ❌ Yo'q | `tests/`, `docs/` papkalari mavjud emas; `src/models.py` da "half later" izohi qolgan |
| Veb-sayt (V1 §10) | ❌ Bajarilmagan | umumiy `temp_uploaded.mp4`, `maxUploadSize = 10240`, noto'g'ri qoidalar jadvali, placeholder LinkedIn — hammasi joyida; deploy URL yo'q |
| Repo gigienasi | ⚠️ Yomonlashgan | `.agents/skills/ui-ux-pro-max/` (75 fayl, ~100k qator) commit qilingan; `reference.png` (15 MB), PDF'lar, `benchmark_runner.py` joyida |
| ✅ Bajarilgan | | `requirements.txt` numpy<2.3 / pandas<3 (Py3.10 bilan mos, lekin toza 3.10 da sinalmagan), `src/paths.py` + `_load_yolo` (CWD mustaqil), `SHA256SUMS` + `download.py`, auto-alignment o'chirilgan, `build_scene` masshtablash, `Dockerfile`, harness fayllari o'zgarmagan |

**Ikkita crash vaqtincha tuzatilgan nusxa bilan to'liq C3905 o'lchovi** (RTX 3050): jami 237 s / 127.6 s = **1.86×** (oldin 2.47×). 50 ta hodisa, 10 ta sinf: illegal_turn 11, failure_to_yield 8 (oldin 1), near_miss 8, congestion 5, jaywalking 4, stopped_vehicle 3, accident 3, illegal_u_turn 3, solid_line_crossing 3, red_light 2. Hodisalar profili oldingisiga deyarli teng, ya'ni Part A sifati amalda o'zgarmagan. Part B: risk ≥ 0.5 bo'lgan kadrlar 21% dan **3%** ga tushgan. Bu haqiqiy yaxshilanish, lekin maqsad < 1%.

**Xulosa:** hozirgi branch'ni topshirsangiz, Model = 0 (har bir video bo'sh), sayt ochilmaydi, "Runs as submitted" = 0. **Bu `master` dagi holatdan ham yomonroq.**

---

## 1. Agent uchun "isbotsiz bajarildi deyish taqiqlanadi" protokoli

V1 §0 qoidalari amal qiladi. Qo'shimcha:

1. **Har bir vazifa `DONE` deb belgilanishi uchun** agent `docs/PROGRESS.md` ga shularni yozadi: vazifa ID, commit hash, ishga tushirilgan **aniq buyruq** va uning **chiqishidan nusxa** (oxirgi 20 qator). Chiqish bo'lmasa — `DONE` emas.
2. Har bir commit'dan oldin **majburiy smoke-test** (§2.4) o'tishi shart. O'tmasa commit qilinmaydi.
3. "Yaxshilandi" degan har qanday da'vo **raqam bilan** bo'lsin: oldin/keyin (dev F1, hodisalar soni, vaqt). Raqamsiz da'vo qabul qilinmaydi.
4. Ball berish agentning ishi emas. Agent faqat o'lchovlarni yozadi.
5. Kutubxona API'sini ishlatishdan oldin o'rnatilgan versiyada tekshiring: `python -c "import inspect, supervision as sv; print(inspect.signature(sv.ByteTrack.__init__))"`. (`track_thresh` crash'i shu tekshiruv qilinmagani uchun paydo bo'lgan.)

---

## 2. P0 — Ishga tushirish (1–2 soat). Boshqa hamma narsadan OLDIN

### 2.1 ByteTrack argumentlari `[AGENT]`
- `solution.py:332` va `:788`:
  ```python
  sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=int(fps * 4), frame_rate=int(round(fps)))
  ```
  (supervision 0.30.5 imzosi: `track_activation_threshold, lost_track_buffer, minimum_matching_threshold, frame_rate, minimum_consecutive_frames`.) Keyinchalik stride qo'shilsa (§5), `frame_rate = fps / stride`.
- `src/annotate.py:110` va `src/tools/visualizer.py:249` dagi `sv.ByteTrack()` ham bir xil factory'dan olinsin: `src/tracking.py::make_tracker(fps, stride)`.

### 2.2 `_emitted` sentinel'ni olib tashlash `[AGENT]`
- `active_events` faqat `{(track_id, label): info}` bo'lsin. "Allaqachon chiqarilgan" to'plami alohida o'zgaruvchi `emitted: set[tuple[int, str]]` bo'lsin va `_open_event` / `_close_event` ga argument sifatida berilsin.
- Diqqat, semantika: "bir track bir sinfni faqat bir marta chiqaradi" qoidasi jaywalking/wrong_way uchun noto'g'ri bo'lishi mumkin (bir piyoda ikki marta yo'lga chiqadi). Buni faqat `red_light`, `illegal_turn`, `illegal_u_turn` uchun qo'llang; qolganlarida `merge_same_class_segments` yetarli.

### 2.3 Import zanjiri `[AGENT]`
- `app.py:16` → `from solution import CLASSES, RiskEstimator, detect_events` + `from src.scene import SCENE_CONFIG`.
- `src/tools/visualizer.py`: yo `src.scene.build_scene` + `src.traffic_light` ga o'tkazing, yoki butunlay o'chiring (`src/annotate.py` bir xil ishni qiladi). **Tavsiya: o'chirish** (V1 §8.2 "no dead code").
- Butun repo bo'yicha tekshiring: `grep -rn "from solution import" --include=*.py . | grep -v venv` — har bir nom haqiqatan mavjud bo'lsin.

### 2.4 Majburiy smoke-test `[AGENT]`
`tests/test_smoke.py` (pytest) + `scripts/smoke.sh`:
1. `samples/C3905.MP4` ning birinchi 8 soniyasidan `tests/data/clip8s.mp4` yaratilsin (skript bilan, fayl commit qilinmaydi; yoki 1280px, ~2 MB versiya commit qilinadi — CI uchun qulay).
2. `python run_submission.py --videos tests/data/clip8s.mp4 --out /tmp/smoke.json` → JSON dagi `log[...]["errors"] == []`, `risk` uzunligi = kadrlar soni.
3. `python evaluate.py --pred /tmp/smoke.json --validate-only` → exit 0.
4. `python -c "import app"` (streamlit o'rnatilgan holda) → exit 0. Streamlit'ni "bare mode" da import qilish ogohlantirish beradi, lekin xato bermaydi.
5. `python -m src.annotate --help`, `python -m src.deep_eda --help` → exit 0.

**Qabul mezoni:** `bash scripts/smoke.sh` → `SMOKE OK`. Chiqishi `docs/PROGRESS.md` ga.

### 2.5 `predictions_samples.json` ni qayta yaratish `[AGENT]`
- Faqat §2.1–2.4 o'tgandan keyin: `python run_submission.py --videos samples --out predictions_samples.json --team dewew`.
- `log` dagi har bir videoda `errors == []` va `total_sec / duration` qiymati `docs/PROGRESS.md` ga yozilsin.
- **Bu fayl har safar kod o'zgarganda (§3–5 dan keyin) yana qayta yaratiladi.** Yakuniy tag'da u aynan tag'dagi kodga mos bo'lishi shart (rubric: Reproducibility).

---

## 3. P0/P1 — Dev set: haqiqiy, ishlaydigan jarayon

V1 §2 bajarilmagan. GUI belgilash vositasi yozish shart emas, u vaqt yeydi. **Pragmatik yo'l:**

### 3.1 CSV orqali belgilash `[HUMAN]` (eng muhim inson ishi)
- Har bir a'zo o'z videosini **VLC** da ko'radi (`E` tugmasi = kadrma-kadr, pastki panelda vaqt) va `devset/raw/<video>.csv` ga yozadi:
  ```
  start,end,label,note
  12.4,18.9,jaywalking,ped crosses left of zebra 1
  ```
- Konvensiyalar PDF jadvalidagi Start/End ustunlari bo'yicha. Bir vaqtdagi bir xil sinf → bitta segment.
- Taqsimot: Ollabergan — C3896; Seymonbek — C3897 + C3905; Siroj — C3902. ~1.5–2 soat/kishi.
- Minimal talab: **barcha 14 sinf bo'yicha videoni oxirigacha ko'rish**. Hodisa bo'lmasa — CSV da faqat header qoladi. Bu ham ma'lumot: o'sha sinf bo'yicha har qanday bashorat FP ekanini bildiradi.

### 3.2 Konvertor va baholash `[AGENT]`
- `src/devset/csv_to_gt.py`: `devset/raw/*.csv` → `devset/labels.json` (evaluate.py GT formati, `duration` va `fps` videodan olinadi). Tekshiruvlar: label ∈ CLASSES, start<end≤duration, bir sinf ichida ustma-ust kelmaslik (kelsa, birlashtirib ogohlantiradi).
- `src/devset/make_candidates.py` ni tuzating: `preds["videos"].items()` ni aylansin; `annotate.py` ga `--events` (`--pred` emas) berilsin; `duration` ni `preds["log"][v]["duration"]` dan olsin. U FP larni tez ko'rish uchun kerak, belgilashning o'rnini bosmaydi.
- `src/devset/summarize.py`: xom JSON dump emas, **jadval** chiqarsin: sinf × (TP/FP/FN, P, R, F1@0.3/0.5/0.7), hamda Score A, Score B.
- `label_tool.py` stub'ini o'chiring (ishlamaydigan kod = o'lik kod).

### 3.3 Baseline `[AGENT]`
§2 dan keyingi kod bilan birinchi o'lchov → `devset/BASELINE.md`. Keyingi har bir o'zgarish shu bilan solishtiriladi.

**Qabul mezoni:** `python evaluate.py --pred predictions_samples.json --gt devset/labels.json` Score A / Score B chiqaradi; `devset/REPORT.md` sinfma-sinf jadval.

---

## 4. P1 — Part A to'g'riligi

Umumiy tamoyillar V1 §4 dagidek. Bu yerda faqat **aniq, joriy kodga bog'langan** vazifalar berilgan.

### 4.1 Svetofor — uchinchi urinish, lekin o'lchov bilan `[AGENT + HUMAN]`
Hozirgi `_analyze_crop` (bbox'ni 3 ga teng bo'lish + ROI ning 5% yonishi) tamoyilan noto'g'ri. O'lchangan faktlar:
- Asosiy svetofor bbox'i `(2290, 720, 2360, 860)` 4K da. Yonib turgan lampa kichik: yorqin-to'yingan piksellar ~5 qator (masalan C3896 t=30s: yashil lampa y≈808–812, x taxminan bbox markazi).
- bbox'ning pastki qismi korpusdan tashqarida (yo'l foni). U yerdagi V>170 piksellar lampa emas.

Qanday qilish kerak:
1. `scripts/tl_calibrate.py`: 3 ta video × bir necha vaqtda bbox crop'ni 8× kattalashtirib saqlaydi. `[HUMAN]` qizil va yashil lampa **markazlarini** (x, y, r≈6–8 px) yozadi. Chap tomondagi ikkinchi svetofor boshqa yo'nalish uchun, uni aralashtirmang.
2. Lampa ball'i = disk ichidagi **eng yorqin 10 piksel**ning o'rtacha V qiymati × (hue mos kelsa 1, aks holda 0). Hue: qizil [0,12]∪[160,180], S>70; yashil [55,100], S>50. Maydon ulushi (%) ishlatilmasin.
3. Qaror: `red_score - green_score > margin` → RED; teskarisi → GREEN; ikkalasi past → sariq lampa tekshiruvi → YELLOW/UNKNOWN.
4. Hysteresis **soniyada** (`t_sec` bo'yicha, 0.4 s), frame counter bo'yicha emas. UNKNOWN hech qachon GREEN deb qaytarilmasin: qoidalar uchun `UNKNOWN` = "qaror yo'q", ya'ni `red_light`/`stop_line` chiqarilmaydi.
5. Kun yorug'ligi/soya o'zgaradi (EDA). Chegaralarni 4 video bo'yicha tekshiring.

**Qabul mezoni:** `scripts/tl_contact_sheet.py` → har 2 s dan namuna. `[HUMAN]` kamida 200 ta namunani belgilaydi (`devset/tl_truth.csv`). Moslik ≥98% va C3896 da qizil ulushi 25–70% oralig'ida bo'lsin (hozir ~0%, eski versiyada ~99%).

### 4.2 `stop_line` — mantiqni to'g'rilash (V1 §4.2, hanuz teskari) `[AGENT]`
solution.py:534–541 ni almashtiring:
- `past_tolerance = crossed_jam_line_map[track]` (2-chiziqdan **o'tgan**),
- `not in_intersection_core`,
- nisbiy tezlik < chegara ≥ 1 s,
- svetofor `RED` (UNKNOWN emas).
Start = to'xtagan vaqt; End = GREEN ga o'tgan yoki qo'zg'algan vaqt. LineZone yo'nalishi: faqat LTR oqimi yo'nalishidagi kesishish (`crossed_in` yoki `crossed_out` dan qaysi biri to'g'ri ekanini bitta klipda tekshirib, konstantaga yozing).

### 4.3 `congestion` `inf` xatosi (isbotlangan) `[AGENT]`
solution.py:657–659 va 679–681: `valid_speeds` < 4 bo'lsa `is_cong = False`. `else 0.0` fallback olib tashlansin. Keyin V1 §4.4 (≥5 s hysteresis, lane-tasmalar, qizil navbat istisnosi).
**Qabul mezoni:** `clip8s.mp4` da `congestion [0.0, 0.8]` yo'qoladi (unit test sifatida).

### 4.4 `near_miss` — regressiyani qaytarish va to'g'ri qilish `[AGENT]`
- Hozirgi "markazlar < 30 px absolyut" qoidasini **o'chiring**.
- Avval **dublikat track'larni bostirish** (barcha qoidalarga ta'sir qiladi): bir kadrda IoU > 0.7 bo'lgan ikki transport detektsiyasidan (masalan car + truck) faqat eng yuqori conf'lisi qoladi. `sv.Detections.with_nms(threshold=0.7, class_agnostic=True)` ni tracker'dan **oldin** qo'llang.
- Keyin V1 §4.10: nisbiy (diag bo'yicha normallashtirilgan) masofa, TTC < 1.5 s, keskin tormoz yoki burilish, kontakt yo'q, ≥ 0.8 s.
- Dev'da precision < 0.4 bo'lsa → §4.10 (sinfni o'chirish).

### 4.5 `jaywalking` — regressiya `[AGENT]`
- `ped_refuge` zonalari (11, 12, 13, 18) piyodalar uchun **xavfsiz**: `is_jaywalking = in_road and not (crosswalk or sidewalk or ped_refuge)`.
- Chavandozlarni bostirish (V1 §4 umumiy): `person` ning bottom-center nuqtasi motorcycle/bicycle bbox ichida (kengaytirilgan 10%) bo'lsa → piyoda emas. Bu bir xil funksiya orqali `failure_to_yield` da ham ishlatilsin.
- Minimal davomiylik 1.0 s (`MIN_EVENT_DURATION`).

### 4.6 V1 dan hali bajarilmagan qoidalar `[AGENT]`
Har biri V1 dagi bo'yicha, alohida commit bilan, har biridan keyin dev jadval:
| Sinf | V1 bo'limi | Joriy koddagi joy | Asosiy tuzatish |
|---|---|---|---|
| red_light | §4.1 | :528–531 | faqat LTR yo'nalishi, RED ≥1 s, fiksal `+2.0` o'rniga track core'dan chiqquncha |
| stopped_vehicle | §4.3 | :592–598 | signal navbatini istisno qilish, nisbiy tezlik, median |
| failure_to_yield | §4.6 | :442–446, :552–559 | bir xil zebra, masofa, harakatlanayotgan mashina, rider emas |
| wrong_way | §4.7 | :576–586 | oqim vektori bilan cos < −0.5, ≥1.5 s, core'dan tashqarida |
| illegal_turn / u_turn | §4.8 | :617–646 | kirish/chiqish darvozalari + `ALLOWED_MOVES` (`[HUMAN]` jadvali). Jadval bo'lmasa → o'chirish |
| solid_line_crossing | §4.9 | :565–569 | Hozir `ped_refuge + barriers` poligoniga kirish hisoblanadi. `[HUMAN]` yaxlit chiziqlarni polyline qilib bermaguncha **o'chirilsin** |
| accident / fire_smoke | §4.11 | :388–404 | conf ≥0.6, yo'lda ≥2 ishtirokchi + kontakt + to'xtash, ≥1 s. Fiksal `+2.0` o'rniga haqiqiy tugash vaqti |
| road_obstacle | §4.12 | :604–611 | `inf → 0.0` olib tashlansin; odam ko'tarib yurgan buyum istisno |

### 4.7 Piksel chegaralarini nisbiy qilish `[AGENT]`
Hamma joyda absolyut `10.0`, `30.0`, `60.0`, `5.0`, `15.0` px qiymatlari bor, `build_scene` esa geometriyani masshtablaydi. Natijada 1080p upload'da tezlik chegaralari 2× noto'g'ri ishlaydi. Barcha tezlik/masofa chegaralari `bbox diagonal` ga nisbiy bo'lsin. Barchasi `src/config.py::RULES` ga ko'chirilsin.

### 4.8 Sinf tanlash siyosati (V1 §5) `[AGENT]`
§3 dev jadvali asosida `ENABLED_CLASSES`. O'chirilgan sinflar `CLASSES` dan ham olib tashlanadi. Qaror jadvali `docs/class_policy.md` da.

---

## 5. P1 — Tezlik (V1 §6, bajarilmagan) `[AGENT]`
Hozirgi (tuzatilgan nusxa bilan o'lchangan) holat `docs/PROGRESS.md` ga yozilsin. Keyin:
1. `half=True` (CUDA bo'lsa) barcha YOLO chaqiruvlarida. `src/models.py` dagi "half later" izohi o'rniga amalga oshirilsin.
2. Part A stride 2: toq kadrlarda `cap.grab()`, tracker `frame_rate = fps/2`.
3. Anomaliya modeli har 10-kadrda.
4. `detect_events` ichida vaqt nazorati: `elapsed > 1.8 × duration` bo'lsa stride 4 ga oshiriladi va log'ga yoziladi.

**Qabul mezoni:** 4 ta sample'da `total_sec / duration ≤ 1.3` (RTX 3050), dev F1 stride 1 ga nisbatan ≤ 0.02 ga tushgan (ablation jadvali saytga).

---

## 6. P2 — Part B `[AGENT]`
1. **Birlik xatosi:** solution.py:935–951 da `dist` 4K pikselda, 961–976 da tezliklar `scale_to_640` bilan. Hammasini bitta birlikka o'tkazing, yaxshisi bbox diag'ga nisbiy.
2. Tezlikni `frame_count` emas, `t_sec` farqi bilan hisoblang (V1 §7.6). Shunda demo va harness bir xil egri chiziq beradi.
3. `_ensure_scene_polys` izohida hali "AI-aligned… alignment probe" deyilgan, u endi noto'g'ri. Tuzatilsin.
4. Kalibrlash va kauzallik testi — V1 §7.1–7.8.

**Qabul mezoni:** sample'larda risk ≥ 0.5 ulushi < 1% (dev'da accident yo'q bo'lsa); `tests/test_risk_causal.py` o'tadi.

---

## 7. P1 — Veb-sayt (V1 §10 — deyarli butunlay bajarilmagan)

Tartib muhim — birinchi uchtasi bo'lmasa, qolgani befoyda:
1. §2.3 import tuzatilishi → sayt ochiladi.
2. **Deploy** (V1 §10.1) `[HUMAN + AGENT]`: HF Spaces (Streamlit). Weights'ni birinchi ishga tushishda `weights/download.py` yuklaydi. URL README ga va saytning Links bo'limiga yoziladi. Hakamlar davrida doimiy onlayn.
3. **Demo ishonchliligi** (V1 §10.2): `app.py:2081` dagi umumiy `temp_uploaded.mp4` → `tempfile.mkdtemp()` + session UUID; `.streamlit/config.toml` `maxUploadSize = 300`; CPU demo rejimi (1280px, stride 3, yengil detektor) ochiq aytilgan holda. Part B'ni har kadrda chaqirish (yoki §6.2 dan keyin istalgan stride).
4. **Qoidalar jadvali** (app.py:1494–1508) `RULES` dan avtomatik generatsiya qilinsin. Hozir u kodga zid (red ≥15 px, failure_to_yield <80 px, wrong_way dot<−0.3, stopped <3 px/s, congestion ≥3, near_miss 0.85×diag — hech biri kodda yo'q).
5. **Results sahifasi:** yangi kod bilan qayta render qilingan 4 ta preview (`samples/previews/*` hozir eski kod bilan). Har bir video uchun timeline va risk egri chizig'i. Dev jadvali (§3), halol FP/FN misollari.
6. **Team:** app.py:1348 dagi `https://linkedin.com` placeholder. Siroj uchun haqiqiy GitHub, LinkedIn va portfolio `[HUMAN]`. Havola bo'lmasa tugma ko'rsatilmasin.
7. **Report:** "nima ishlamadi" bo'limiga shu tarixni halol yozing: svetofor v1 (99% RED) → v2 (≈100% UNKNOWN) → v3 (o'lchangan lampa disklari) va o'chirilgan sinflar.
8. V1 §10.3–10.8 dagi qolgan bandlar (EDA "topilma → qaror", ablation, telefon, bosilganda videoga o'tadigan timeline).

**Qabul mezoni:** inkognito brauzerda URL ochiladi. `tests/test_web_smoke.py` (Streamlit `AppTest`) 8 s klipni yuklaydi va natija oladi. 4K, 1080p va buzilgan fayl bilan qo'lda sinov `[HUMAN]`.

---

## 8. P2 — Repo gigienasi va tuzilma `[AGENT]`
1. `git rm -r --cached .agents skills-lock.json` va `.gitignore` ga `.agents/`, `skills-lock.json`. Bu branch 75 fayl / ~100k qator begona kod qo'shgan, hakam "structure" bahosida buni ko'radi.
2. `reference.png` (15 MB), `reference.jpg`, `Videos.pdf`, `WIUT Hackathon _ CV Track Elimination Task.pdf` → repodan olib tashlash; sahna rasmi `docs/img/scene_zones.jpg` (≤1 MB) sifatida, `reference resource.txt` → `docs/scene.md`.
3. `benchmark_runner.py` → `scripts/bench.py` (yoki o'chirish, `scripts/smoke.sh` + `scripts/eval_dev.sh` yetarli).
4. `solution.py` hanuz 1014 qator. V1 §8.1 bo'yicha qoidalarni `src/rules/*.py` ga ajrating. **Faqat §2–§4 tuzatishlari dev o'lchovi bilan qotirilgandan keyin**, aks holda refactor yangi xato qo'shadi. Refactor'dan oldin va keyin `predictions_samples.json` bir xil bo'lishi kerak (`scripts/compare_preds.py`).
5. O'lik kod: `get_ai_offset` (endi faqat `return 0, 0` + ikkita chalg'ituvchi print — "Shifted … using YOLO Traffic Light Detection"), `_compute_iou` wrapper, ishlatilmagan `copy`, `Path`, `YOLO` importlari, "BRUTAL" izohlari.
6. `pyproject.toml` + `ruff check .` toza; `pytest -q` (smoke, postprocess, traffic_light fixture crop'lari, risk causal, csv_to_gt).
7. Ixtiyoriy: GitHub Actions — ruff + pytest (GPU'siz qismlar).

---

## 9. P2 — README
V1 §11 checklist'i. Qo'shimcha:
- Hozirgi README "Dev set", natijalar va sayt URL'ini o'z ichiga olmaydi. Qo'shilsin.
- "Tested on" jadvali: Python 3.10 va 3.12, GPU, vaqt nisbati.
- `accident_model.pt` training dataset va litsenziyasi hanuz "AGPL-3.0 (Ultralytics)" deb taxmin qilingan. `[HUMAN]` HF model kartasini tekshirsin. Litsenziya/dataset noma'lum bo'lsa, model olib tashlanadi va accident faqat qoidaga asoslangan bo'ladi (§4.6).

---

## 10. Yakuniy qabul (V1 §12 + qo'shimchalar) `[AGENT]` — natija `docs/FINAL_CHECK.md` ga, chiqishlar bilan

```bash
# Docker o'rnatilgan (29.x) — toza Python 3.10 da haqiqiy sinov:
docker run --rm -it --gpus all -v "$PWD/samples:/data/test:ro" python:3.10-slim bash -lc '
  apt-get update && apt-get install -y git curl && \
  git clone <repo> /r && cd /r && git checkout <tag> && \
  pip install -r requirements.txt && bash weights/download.sh'
# keyin tarmoqsiz:
docker run --rm --network none --gpus all ... python run_submission.py --videos /data/test --out /tmp/p.json
python evaluate.py --pred /tmp/p.json --validate-only
python scripts/compare_preds.py /tmp/p.json predictions_samples.json
bash scripts/smoke.sh && pytest -q && ruff check .
```
- [ ] Har bir video `errors: []`, `total/duration ≤ 1.5`.
- [ ] Toza klondan `streamlit run app.py` ishlaydi; deploy URL ishlaydi.
- [ ] `predictions_samples.json` tag'dagi kodga mos.
- [ ] Dev Score A/B `docs/FINAL_CHECK.md` da, baseline bilan.
- [ ] `run_submission.py`, `evaluate.py` o'zgarmagan.
- [ ] Tag + push + tashkilotchilarga link `[HUMAN]`.

---

## 11. Bajarish tartibi (kritik yo'l)

```
§2 (P0, 1–2 soat)  ──►  §2.5 predictions  ──►  §7.1–7.2 sayt ochiladi va deploy
        │
        ├──► §3.1 [HUMAN] CSV belgilash (parallel, birinchi kuni)
        │
        ▼
§4.1 svetofor ─► §4.2–4.5 regressiyalar/buglar ─► §3.3 baseline ─► §4.6 qoidalar ─► §4.8 sinf siyosati
        ▼
§5 tezlik ─► §6 Part B ─► §2.5 predictions (qayta) ─► §7.3–7.8 sayt ─► §8 gigiena/refactor ─► §9 ─► §10
```

**Muhim:** agar vaqt juda kam bo'lsa, faqat §2 + §2.5 + §7.1–7.3 + §4.3 + §4.5 + §8.1 ni bajaring. Bu "ishlaydigan, ochiladigan" holatni tiklaydi (≈ master darajasi yoki biroz yuqori).
