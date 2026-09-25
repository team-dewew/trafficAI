# FIX_PLAN — WIUT CV Track: barcha kamchiliklarni yopish rejasi

> Bu hujjat AI agent (Antigravity) va jamoa uchun bosqichma-bosqich ish rejasi.
> Har bir vazifada: **nima**, **qayerda**, **qanday**, **qabul mezoni** (acceptance) bor.
> `[HUMAN]` belgisi — faqat inson bajaradigan ish (video ko'rish, belgilash, deploy akkauntlari).
> `[AGENT]` — agent mustaqil bajaradi.

---

## 0. Agent uchun QAT'IY qoidalar (buzish = diskvalifikatsiya yoki 0 ball)

1. `run_submission.py` va `evaluate.py` fayllariga **tegmang** (bir bayt ham o'zgarmasin). Tekshiruv: `git diff --exit-code a76904e -- run_submission.py evaluate.py`.
2. `solution.py` interfeysi o'zgarmaydi: `CLASSES`, `detect_events(video_path) -> list[list]`, `RiskEstimator.reset(meta)`, `RiskEstimator.step(frame, t_sec) -> float`. `detect_events` ga qo'shimcha **ixtiyoriy** kwarg'lar (`progress_callback=None`, `config=None`) qo'shish mumkin.
3. Inference vaqtida hech qanday tarmoq so'rovi yo'q (OpenAI/Gemini/Anthropic/HF API — taqiqlangan). Ultralytics avtomatik yuklab olishiga ham yo'l qo'ymang (4.2-bandga qarang).
4. Sample videolar uchun javoblarni hard-code qilmang (vaqt, video nomi bo'yicha if-lar va h.k.). Faqat sahna geometriyasi hard-code qilinishi mumkin.
5. `RiskEstimator` videoni o'zi ochmaydi va Part A natijalaridan foydalanmaydi (kauzallik).
6. `camera.md` yo'q — tashkilotchilar uni e'tiborsiz qoldirishni aytgan. Sahna bilimi manbai: `reference resource.txt` + `SCENE_CONFIG`.
7. Har bir bosqich oxirida alohida commit qiling (Conventional Commits). `master` ga emas, `fix/elimination-hardening` branch'ida ishlang.
8. Har bosqichdan keyin **regressiya** (9-bosqichdagi `scripts/regress.sh`) ishga tushirilsin.

---

## Bosqichlar tartibi va ustuvorligi

| # | Bosqich | Ball ta'siri | Ustuvorlik |
|---|---|---|---|
| 1 | Ishga tushmay qolish xavflarini yopish (install, yo'llar, git) | Model=0 xavfi, Code 40% | 🔴 P0 |
| 2 | Dev to'plam (o'z belgilarimiz) + baholash quvuri | Barcha keyingi tuning shunga bog'liq | 🔴 P0 |
| 3 | Sahna va svetofor to'g'riligi | red_light, stop_line, butun qoidalar | 🔴 P0 |
| 4 | Part A qoidalarini sinfma-sinf tuzatish | Score A (Model 70%) | 🟠 P1 |
| 5 | Sinf tanlash siyosati (faqat ishonchli sinflarni chiqarish) | Macro-F1 ni keskin oshiradi | 🟠 P1 |
| 6 | Tezlik va vaqt zaxirasi | Budjetdan oshsa video = 0 | 🟠 P1 |
| 7 | Part B (RiskEstimator) sifati | Score B (Model 30%) | 🟡 P2 |
| 8 | Kod tuzilmasi va toza repo | Code 20% + 15% | 🟡 P2 |
| 9 | Determinizm, reproduktivlik, regressiya | Code 25% | 🟡 P2 |
| 10 | Veb-sayt: deploy, demo, to'g'rilik, qo'shimchalar | Website 25% | 🟠 P1 |
| 11 | README va hisobot | Code + Website | 🟡 P2 |
| 12 | Yakuniy qabul tekshiruvi va tag | Hammasi | 🔴 P0 |

---

## 1-bosqich — Ishga tushmay qolish xavflarini yopish (P0)

### 1.1 `requirements.txt` ni Python 3.10 bilan moslash `[AGENT]`
**Muammo:** `numpy==2.5.2` Python ≥3.12, `pandas==3.0.6` ≥3.11 talab qiladi. PDF: "Python 3.10 or newer". 3.10 da `pip install` yiqiladi → Model score 0.

**Qanday:**
- Inference (hakamlar mashinasi) va veb-sayt bog'liqliklarini ajrating:
  - `requirements.txt` (hakamlar ishlatadi, minimal):
    ```
    torch==2.6.0
    torchvision==0.21.0
    ultralytics==8.4.161
    supervision==0.30.5
    numpy>=1.26,<2.3
    opencv-python-headless==4.11.0.86   # yoki 3.10 wheel'i borligi tekshirilgan 5.x
    lap>=0.5.12                           # ultralytics tracker bog'liqligi, oflayn o'rnatilsin
    ```
  - `requirements-web.txt`: `-r requirements.txt` + `streamlit`, `pandas>=2.2,<3`, `imageio-ffmpeg`, `plotly`.
- `opencv-python-headless` serverda GUI kutubxonasisiz ishlaydi (`libGL` xatosi bo'lmaydi). Agar `visualizer.py` GUI kerak bo'lsa — u faqat dev vositasi, `requirements-dev.txt` ga.
- Har bir versiya uchun PyPI da `cp310` manylinux wheel borligini tekshiring.

**Qabul mezoni:**
- Toza `python3.10 -m venv /tmp/v310 && /tmp/v310/bin/pip install -r requirements.txt` xatosiz o'tadi (Docker `python:3.10-slim` ichida sinab ko'ring).
- Xuddi shunisi `python:3.12-slim` da ham.
- Ixtiyoriy, lekin tavsiya qilinadi: ildizga `Dockerfile` qo'shing (`FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`), shunda hakamlarda 2 ta variant bo'ladi.

### 1.2 Weights yo'llarini `__file__` ga nisbatan qilish `[AGENT]`
**Muammo:** `Path("weights/yolo11l.pt")` CWD ga bog'liq; topilmasa `"yolo11l.pt"` ga tushadi → Ultralytics oflayn mashinada yuklab olishga urinadi → exception → video bo'sh.

**Qanday:**
- `src/paths.py`: `REPO_ROOT = Path(__file__).resolve().parent.parent`, `WEIGHTS_DIR = REPO_ROOT / "weights"`.
- `_load_yolo(name)` faqat `WEIGHTS_DIR / name` dan yuklasin; fayl yo'q bo'lsa aniq xabarli `FileNotFoundError("run bash weights/download.sh")` ko'tarsin. Bare nomga fallback **olib tashlansin**.
- Ildizdagi takroriy `yolo11l.pt`, `yolov8n.pt` o'chirilsin (gitignore qilingan, lekin chalkashtiradi).

**Qabul mezoni:** `cd /tmp && python /repo/run_submission.py --solution /repo/solution.py --videos /repo/samples/C3905.MP4 --out /tmp/p.json` ishlaydi (tarmoq o'chirilgan holda: `HF_HUB_OFFLINE=1 YOLO_OFFLINE=1`, yoki `unshare -n`).

### 1.3 `weights/download.sh` mustahkamligi `[AGENT]`
- `curl -fL --retry 3` (xatoda to'xtasin), yuklangandan keyin `sha256sum -c weights/SHA256SUMS`.
- `weights/SHA256SUMS` faylini joriy weights'dan yarating.
- Windows foydalanuvchilari uchun `weights/download.py` (faqat stdlib `urllib`) ham qo'shing, README da ikkalasi ko'rsatilsin.

**Qabul mezoni:** Weights'siz toza klonda `bash weights/download.sh` → 3 ta fayl, checksum OK.

### 1.4 Git holatini to'g'rilash `[AGENT]`
**Muammo:** `src/annotate.py`, `src/deep_eda.py`, `samples/previews/`, `eda_results/`, `assets/team/*` commit qilinmagan; `app.py` `src.annotate` ni import qiladi → toza klonda sayt yiqiladi.
- Ularni commit qiling. `samples/previews/*.mp4` hajmini tekshiring: bitta fayl >50 MB bo'lsa, Git LFS ga o'tkazing yoki 720p/CRF 28 ga qayta kodlang.
- `.gitignore` dagi `samples/*` + `!samples/previews/` qoidasi haqiqatda ishlashini `git check-ignore -v` bilan tekshiring.

**Qabul mezoni:** `git clone <repo> /tmp/c && cd /tmp/c && python -c "import app"` import xatosiz (streamlit o'rnatilgan holda); `git status` toza.

---

## 2-bosqich — Dev to'plam va baholash quvuri (P0)

> PDF: *"Without a dev set you are guessing."* Hozir hech bir chegara (threshold) o'lchanmagan. Bu bosqichsiz 4–7-bosqichlarni "ko'r-ko'rona" qilish befoyda.

### 2.1 Nomzod kliplar generatori `[AGENT]`
`src/devset/make_candidates.py`:
- Kirish: `predictions_samples.json` + videolar.
- Har bir bashorat qilingan hodisa uchun `[s-3, e+3]` oralig'ida 960px klip kesadi (`src/annotate.py` orqali zonalar, tracklar va hodisa banneri bilan).
- `devset/review.csv` yaratadi: `video,idx,label,start,end,clip_path,verdict(TP/FP/?),true_start,true_end,true_label,notes`.

### 2.2 To'liq belgilash `[HUMAN]` (eng muhim inson ishi, ~4–6 soat, 3 kishiga bo'linadi)
- 4 ta sample videoni (jami ~18 daqiqa) **boshidan oxirigacha** ko'ring. PDF dagi start/end konvensiyalariga amal qiling.
- Belgilash vositasi: `src/devset/label_tool.py` `[AGENT yozadi]` — OpenCV oynasi: `space` pauza, `←/→` ±1 s, `,/.` ±1 kadr, `1..9,0,q,w,e,r` sinf tanlash, `s` start, `e` end, `u` bekor qilish; natija `devset/labels.json` ga yoziladi.
- Format (evaluate.py ground-truth formati bilan bir xil):
  ```json
  {"C3896.MP4": {"duration": 340.34, "fps": 29.97, "events": [[12.0, 19.0, "jaywalking"], ...]}}
  ```
- Qoidalar: bir vaqtda bir xil sinfdagi ikki hodisa → bitta segment (FAQ). Noaniq hollar `devset/ambiguous.md` ga yozilsin.
- Belgilashda 2.1 dagi kliplar yordam beradi (FP larni tez ko'rish), lekin **o'tkazib yuborilgan hodisalar (FN)** uchun videoni to'liq ko'rish shart.

### 2.3 Baholash skripti `[AGENT]`
`scripts/eval_dev.sh`:
```bash
python run_submission.py --videos samples --out predictions_samples.json --team dewew
python evaluate.py --pred predictions_samples.json --gt devset/labels.json --per-video --json devset/report.json
```
- `src/devset/summarize.py`: `report.json` dan sinfma-sinf P/R/F1 (τ=0.3/0.5/0.7) jadvali, FP/FN ro'yxati (vaqtlari bilan) → `devset/REPORT.md`.
- **Tezkor iteratsiya:** `src/cache.py` — har bir video uchun detektor + tracker natijalarini (kadr, track_id, cls, xyxy, conf) `cache/<video>.npz` ga saqlash. Qoidalar keshdan qayta ishlanadi (`scripts/replay_rules.py`) → chegaralarni sozlash soniyalarda, 10 daqiqada emas. **Kesh faqat dev uchun**; `detect_events` hakamlarda keshsiz ishlaydi.

**Qabul mezoni:** `devset/labels.json` 4 video uchun bor; `evaluate.py --gt devset/labels.json` ishlaydi; joriy (tuzatishlardan oldingi) baseline ball `devset/BASELINE.md` ga yozilgan.

---

## 3-bosqich — Sahna va svetofor to'g'riligi (P0)

### 3.1 Svetofor holatini aniqlash — qayta yozish `[AGENT]`
**Muammo (tasdiqlangan):** C3896 t=30s da yashil chiroq yonib turibdi, lekin funksiya `RED` qaytaradi; C3896 ning ~99% kadrlari RED. Sabab: bbox ichidagi >5 ta "qizilroq" piksel (S,V≥40) — korpus/soya piksellari ham o'tadi.

**Qanday (`src/traffic_light.py`):**
1. Bbox ichida **3 ta lampa ROI** ni alohida belgilang (yuqori = qizil, o'rta = sariq, past = yashil); koordinatalarni 4K kadrdan qo'lda oling (chiroq ~70×140 px).
2. Har bir ROI uchun "yonganlik" = ROI dagi yorqin piksellar ulushi (`V > 170`), rang tekshiruvi bilan: qizil ROI da hue ∈ [0,12]∪[165,180] & S>90; yashil ROI da hue ∈ [60,100] & S>60; sariq ROI da hue ∈ [12,35].
3. Holat = eng yuqori ball bergan lampa, agar u `min_on` dan katta bo'lsa; aks holda `UNKNOWN`.
4. **Vaqtli silliqlash:** holat faqat ≥ 0.4 s (≈12 kadr) barqaror bo'lsagina o'zgaradi (hysteresis); `UNKNOWN` oxirgi ma'lum holatni saqlaydi.
5. Holatni har kadrda emas, har 3-kadrda hisoblash yetarli.
6. Sariq (`YELLOW`) qizilga tenglashtirilmaydi.

**Tekshiruv vositasi `[AGENT]`:** `scripts/tl_contact_sheet.py` — har bir videodan har 2 s da ROI crop + bashorat yorlig'i bilan kontakt varag'i (PNG) yaratadi.
**Qabul mezoni `[HUMAN tasdiqlaydi]`:** 4 video × ~60 namunada qo'lda tekshirilgan holat bilan ≥ 98% moslik; C3896 da RED ulushi real tsiklga mos (odatda 30–60%). Natijalar `devset/tl_check.md` ga yozilsin.

### 3.2 Avto-alignment ni xavfsiz qilish yoki o'chirish `[AGENT]`
**Muammo:** `get_ai_offset` conf≥0.10 va ≤400 px masofadagi istalgan svetoforni oladi (yonidagi ikkinchi svetofor ~100 px chapda!) → barcha 21 zona siljiydi.
- Kamera qat'iy (PDF). Standart: `dx=dy=0`.
- Agar saqlansa: conf ≥ 0.5, masofa ≤ 40 px, **va** ikkala svetofor (asosiy + yon) juftligi mos kelishi kerak; aks holda 0. Yoki ORB/ECC bilan birinchi kadrni `eda_results/frames/*_first_frame.jpg` ref kadrga moslash (butun sahna bo'yicha, bitta obyekt emas), siljish > 60 px bo'lsa rad etish.
- Siljish `log` da chiqarilsin; Part B da alignment uchun yolo11l **yuklanmasin** (6-bosqich).

**Qabul mezoni:** 4 sample videoda offset (0,0) yoki ≤ ±5 px.

### 3.3 Resolution-mustaqil geometriya `[AGENT]`
- `SCENE_CONFIG` 4K koordinatada qolsin; `detect_events` boshida `sx = W/3840, sy = H/2160` bilan masshtablansin (Part B da allaqachon shunday — Part A bilan bir xil funksiyadan foydalanilsin: `src/scene.py::build_scene(W, H, dx, dy)`).
- Barcha piksel chegaralari (`10 px/s`, `30 px`, `60 px/s` va h.k.) **nisbiy** bo'lsin: obyekt bbox diagonaliga yoki kadr eniga bo'lingan (masalan, "tezlik < 0.05 × diag / s"). Bu perspektivani ham hisobga oladi (uzoqdagi mashina kichik).
- `fps` ga bog'liq barcha narsa soniyada ifodalansin.

**Qabul mezoni:** 1080p ga qisqartirilgan sample (`ffmpeg -vf scale=1920:-2`) da hodisalar soni 4K dagidan ±20% farq qiladi (demo va boshqa o'lchamdagi upload uchun muhim).

### 3.4 Sahna semantikasini kodda aniq ifodalash `[AGENT]`
`reference resource.txt` asosida `src/scene.py` da har bir zonaga **nom va ma'no** bering (raqamlar emas):
- `stop_line_strict` (1-chiziq), `stop_line_tolerance` (2-chiziq: tirbandlikda shu chiziqqacha to'xtash qoidabuzarlik EMAS).
- `lane_ltr` + oqim vektori, `lane_rtl` + oqim vektori (vektorlarni tracklardan o'rtacha yo'nalish sifatida EDA dan chiqaring: `eda_results/*_trajectories.png`).
- `ped_refuge` (11–13 va 18: piyoda uchun ruxsat, mashina uchun man).
- `barriers` (16–17: to'siqlar — mashina tegsa → `accident` nomzodi, **solid_line_crossing emas**).
- `main_signal` faqat `lane_ltr` yondashuviga taalluqli; `ped_signal` zebra uchun.
- Hujjat: `docs/scene.md` — rasm + har bir zona tavsifi (saytda ham ishlatiladi).

---

## 4-bosqich — Part A qoidalarini sinfma-sinf tuzatish (P1)

> Umumiy tamoyillar (hamma sinflarga):
> - **Chavandozni (rider) bostirish:** `person` bbox'i `motorcycle`/`bicycle` bbox'i bilan kesishsa (person bottom-center velosiped bbox ichida yoki IoU>0.2 ≥ 0.5 s) — u piyoda emas. `src/rules/common.py::is_rider()`.
> - **Anchor:** yerdagi joylashuv uchun `BOTTOM_CENTER` (PolygonZone default) — hamma joyda bir xil.
> - **Hysteresis:** har bir holatli qoida `enter_after` va `exit_after` (soniya) bilan; bir kadrlik o'chib-yonish segmentni bo'lmasin.
> - **Chegaralar** (boundaries) — PDF dagi start/end konvensiyasiga aniq mos; fiksal `+2.0 s` oynalar olib tashlansin.
> - Har bir sinf uchun barcha raqamlar `src/config.py::RULES[label]` dict'ida (sayt ham shu yerdan o'qiydi — 10.5).
> - Har bir o'zgarishdan keyin `scripts/replay_rules.py` + `evaluate.py` bilan o'lchang; natija `devset/REPORT.md` ga.

### 4.1 `red_light` `[AGENT]`
- Faqat `lane_ltr` yondashuvidan kelayotgan (oqim vektori bilan dot > 0.5) transport.
- Start = `stop_line_strict` ni **to'g'ri yo'nalishda** kesib o'tgan kadr (supervision `LineZone` `crossed_in` / `crossed_out` dan faqat bittasi — yo'nalishni aniqlang), svetofor holati shu paytda ≥ 1.0 s dan beri `RED` (sariqdan qizilga o'tish lahzasi emas).
- **Tirbandlik istisnosi:** agar mashina `stop_line_tolerance` ni kesmasdan to'xtasa — `red_light` emas (bu `stop_line` ham emas, 4.2).
- End = mashina `intersection_core` dan chiqqan yoki kadrdan yo'qolgan vaqt (track bo'yicha), `+2.0` emas.
- Tasdiqlash: mashina start dan keyin ≤ 5 s ichida `intersection_core` ga kirishi kerak.

### 4.2 `stop_line` — **mantiq teskari, qayta yozish** `[AGENT]`
**Muammo:** hozirgi kod 1- va 2-chiziq orasida to'xtagan mashinani belgilaydi — bu `reference resource.txt` ga ko'ra ruxsat etilgan joy.
- To'g'ri: RED paytida mashina **`stop_line_tolerance` (2-chiziq) dan o'tib**, lekin `intersection_core` ga kirmasdan, ≥ 1 s to'xtab turadi.
- Start = to'xtagan vaqt; End = svetofor `GREEN` ga o'tgan vaqt (PDF: "Signal turns green") yoki mashina qo'zg'algan vaqt, qaysi biri oldin bo'lsa.

### 4.3 `stopped_vehicle` `[AGENT]`
**Muammo:** svetofor navbatidagi mashinalar istisno qilinmagan.
- Istisno: (a) mashina `lane_ltr` yondashuvida stop-chiziq oldidagi `queue_zone` da (yangi polygon, stop chiziqdan ~60 m orqaga) **va** svetofor RED yoki GREEN ga o'tganiga < 8 s; (b) oldida (oqim yo'nalishida, 1.5×diag ichida) boshqa to'xtagan mashina bor (navbat zanjiri); (c) `congestion` faol bo'lgan yo'nalishda.
- Tezlik: bbox diagonaliga nisbiy, 2 s oynada median (jitterga chidamli).
- Track uzilishi: ByteTrack ID almashsa ham ushlab qolish uchun — to'xtagan joyda (IoU>0.6) yangi track paydo bo'lsa, oldingi segmentni davom ettirish (`src/rules/stopped.py` da "stationary slot" xotirasi).
- ≥ 10 s talab saqlanadi; start = to'xtagan vaqt (10 s emas).

### 4.4 `congestion` `[AGENT]`
**Muammo (bug):** ≥4 mashinaning barchasi yangi (tezlik `inf`) bo'lsa, `valid_speeds` bo'sh → o'rtacha 0 → congestion yolg'on yonadi.
- Faqat haqiqiy tezligi o'lchangan (≥ 1 s tarix) mashinalar hisobga olinadi; ular ≥ 4 ta bo'lishi shart.
- "All lanes of a direction": yo'nalish zonasini 2–3 lane-tasmaga bo'ling; har bir tasmada kamida bitta sekin mashina bo'lsin.
- Median nisbiy tezlik < chegara **≥ 5 s** davomida → start (start = sekinlashuv boshlangan vaqt, orqaga surilgan); ≥ 5 s tezlashgach → end.
- Oddiy qizil chiroq navbati (≤ bir tsikl) congestion emas: agar RED dan GREEN ga o'tgach 10 s ichida tarqalsa — segmentni tashlab yuboring.

### 4.5 `jaywalking` `[AGENT]`
- Chavandozlar istisno (umumiy qoida).
- Yo'l polygonlarini 0.5 m (~15–25 px, perspektivaga qarab) ichkariga `erode` qiling — bordyur chetida turganlar hisoblanmasin.
- `ped_refuge` (11–13, 18) — ruxsat etilgan.
- Minimal davomiylik 1.0 s; hysteresis 0.5 s.
- Bir vaqtdagi bir necha piyoda → bitta segment (FAQ), buni `merge` bajaradi.

### 4.6 `failure_to_yield` `[AGENT]`
**Muammo:** piyoda istalgan zebrada, mashina boshqa zebrada bo'lsa ham ishga tushadi.
- Juftlik darajasida: **bir xil zebra** poligonida piyoda (chavandoz emas, ≥ 0.5 s zebrada) va harakatlanayotgan mashina (nisbiy tezlik > chegara).
- Masofa sharti: mashina bbox'i piyodadan ≤ 1.5 × mashina diagonali, va mashina piyodaning **harakat yo'lagini** kesib o'tmoqda.
- Piyoda svetofori (`ped_signal`) GREEN bo'lsa kuchliroq signal; RED bo'lsa (piyoda qoidabuzar) → hodisa chiqarilmaydi.
- `yield_ped_line` (8-chiziq) va `right_turn_zone` (10) mantiqlari `reference resource.txt` ga mos.
- Start = mashina zebraga kirgan vaqt; End = mashina zebradan chiqqan vaqt.

### 4.7 `wrong_way` `[AGENT]`
- `mv_dx` o'rniga: 1.5 s oynadagi siljish vektori va lane oqim vektori orasidagi `cos < -0.5`, siljish ≥ 0.3 × diag.
- ≥ 1.5 s barqaror bo'lsin (jitter va burilishdagi qisqa teskari harakat hisoblanmasin); `intersection_core` ichida baholanmasin (u yerda yo'nalish erkin).
- Start = qarama-qarshi tasmaga kirgan vaqt; End = to'g'ri tasmaga qaytgan yoki kadrdan chiqqan vaqt.

### 4.8 `illegal_turn` va `illegal_u_turn` — manevr matritsasi `[AGENT + HUMAN]`
**Muammo:** hozir har qanday 90° burilish "illegal".
- `[HUMAN]`: kadrdan yo'l belgilarini (yo'nalish strelkalari, "burilish taqiqlangan" belgilari) aniqlab, **ruxsat etilgan manevrlar jadvalini** yozing: `docs/scene.md` → `ALLOWED_MOVES = {("W_in","E_out"), ("W_in","S_out"), ...}`.
- `[AGENT]`: sahnaga **kirish/chiqish darvozalari** (gate polygons) qo'shing: har bir yo'l tarmog'ining chegarasida `*_in`/`*_out`.
- Track butun hayoti bo'yicha klassifikatsiya: birinchi kirgan darvoza → oxirgi chiqqan darvoza.
  - `illegal_u_turn`: kirish va chiqish bir xil tarmoqda (qarama-qarshi yo'nalishda) va bu U-burilish ruxsat etilmagan joyda.
  - `illegal_turn`: (kirish, chiqish) juftligi `ALLOWED_MOVES` da yo'q, **yoki** burilish noto'g'ri tasmadan boshlangan (kirish vaqtida lateral pozitsiya).
  - Start = heading o'zgarishi boshlangan kadr (egrilik > chegara); End = heading barqarorlashgan kadr.
- Bu sinflar kauzal bo'lishi shart emas (Part A offlayn) — track tugagach aniqlash mumkin.
- Agar `[HUMAN]` qaysi burilishlar ruxsatini aniqlay olmasa → 5-bosqichda sinf **o'chiriladi**.

### 4.9 `solid_line_crossing` `[AGENT + HUMAN]`
**Muammo:** hozir "orol/to'siq poligoniga kirish" deb hisoblanadi; `reference resource.txt` bo'yicha 16–17 to'siqlar — bu yerga tegish `accident` nomzodi.
- `[HUMAN]`: kadrdagi **haqiqiy yaxlit chiziqlarni** polyline sifatida belgilang (`solid_lines: list[np.ndarray]`).
- `[AGENT]`: track bottom-center traektoriyasi polyline'ni kesib o'tsa (segment-kesishish testi, har 0.2 s), va mashina ≥ 1 s ichida yangi tasmada to'liq bo'lsa → hodisa. Start = kesishgan kadr; End = bbox butunlay yangi tasmada.
- Orol/to'siq poligoniga kirish → alohida signal: `accident` nomzodi (4.11) yoki umuman chiqarilmaydi.

### 4.10 `near_miss` `[AGENT]`
- Tezlik va tormozlash nisbiy (diag/s) va 3 nuqtali silliqlangan traektoriyadan.
- TTC: nisbiy tezlik vektori bo'yicha `TTC = d / closing_speed`; `TTC < 1.5 s` **va** keyin bir tomonda keskin tormoz (tezlik < 45%) yoki keskin burilish (heading o'zgarishi > 25°/0.5 s), **va** kontakt yo'q (IoU < 0.03 va keyin ham tegmaydi).
- Svetofor navbatidagi odatiy sekinlashuv (stop chiziq oldida, RED) — istisno.
- Minimal davomiylik 0.8 s. Bir juftlik uchun bitta segment.
- Dev natijasida precision < 0.4 bo'lsa → 5-bosqichda o'chiriladi.

### 4.11 `accident` va `fire_smoke` (o'rganilgan model) `[AGENT]`
- `accident_model.pt` sinflari: `detected-injury, fire, high, low, medium, smoke` (severity klassifikatori). Bu CCTV ko'rinishiga moslashmagan → sample videolarda 3–13 ta "accident" chiqyapti, ular deyarli aniq FP.
- Gating (hammasi bajarilishi shart):
  1. conf ≥ 0.6 (dev da sozlang);
  2. anomaliya bbox'i yo'l poligonida va ichida ≥ 2 ta yo'l foydalanuvchisi (yoki 1 ta + `barriers`) bor;
  3. shu yo'l foydalanuvchilari tracklari: kontakt (IoU > 0.05) + keyin ikkalasining ham keskin to'xtashi (4.10 dagi tormoz mezonlari);
  4. ≥ 1.0 s (4 ta ketma-ket tekshiruvdan 3 tasi) barqaror.
- Qoidaga asoslangan ikkinchi yo'l: **model signalisiz ham** — ikki track bbox IoU > 0.1 ≥ 0.5 s, keyin ikkalasi ham ≥ 3 s harakatsiz, svetofor navbati emas → `accident`.
- Start = birinchi kontakt kadri; End = barcha ishtirokchilar to'xtagan yoki kadrdan chiqqan vaqt (PDF konvensiyasi).
- `fire_smoke`: conf ≥ 0.6, ≥ 2 s barqaror, bbox yo'lda yoki mashina ustida. Bulut/chang/quyosh chaqnashlariga qarshi: HSV da tutun kulrang + past to'yinganlik tekshiruvi (ixtiyoriy).
- Tezlik uchun: anomaliya modelini har 10-kadrda, `half=True`, `imgsz=640` da ishlating.

### 4.12 `road_obstacle` `[AGENT]`
- `backpack/umbrella/suitcase`: faqat ≥ 1.5 s hech qanday `person` bbox'i bilan kesishmasa (odamda ko'tarilgan narsa emas).
- Hayvonlar: yo'lda ≥ 1 s.
- Tezlik `inf` → `0` qilib olish **olib tashlansin** (yangi detektsiya darhol "harakatsiz" hisoblanmasin); ≥ 1 s tarix talab qilinsin.
- Detektsiya conf ≥ 0.35; kichik obyektlar uchun ixtiyoriy: 1280px tile'lar bilan har 15-kadrda qo'shimcha o'tish.

### 4.13 Post-processing `[AGENT]`
`src/postprocess.py`:
- Sinfga xos `merge_gap` (masalan jaywalking 1.0 s, congestion 3.0 s, red_light 0.5 s) va `min_duration` (`RULES` dan).
- Clip: `0 ≤ start < end ≤ duration`.
- Bir sinfda ustma-ust kelish yo'q (harness ham tashlaydi, lekin biz o'zimiz birlashtiramiz).
- Unit testlar: `tests/test_postprocess.py` (bo'sh ro'yxat, ustma-ust, chegara holatlari, blip).

**4-bosqich qabul mezoni:** `devset/REPORT.md` da har bir sinf uchun baseline → yangi F1 jadvali; dev Score A baseline'dan sezilarli yuqori; hech bir sinfning precision'i o'zgarishsiz pasaymagan.

---

## 5-bosqich — Sinf tanlash siyosati (P1)

> Macro-F1 da: biz bashorat qilgan, lekin testda yo'q sinf → 0 bilan C ga qo'shiladi. Ya'ni ishonchsiz sinf **zarar**.

- `src/config.py::ENABLED_CLASSES` — dev natijasiga ko'ra:
  - dev da precision ≥ 0.5 va F1(avg τ) ≥ 0.3 → yoqilgan;
  - dev da namunasi yo'q sinflar (masalan `fire_smoke`) → faqat juda yuqori chegarali, "deyarli hech qachon yonmaydigan" rejimda (testda bor bo'lsa ham FN = 0 ball, lekin FP ham 0 → macro'ga zarar yo'q, chunki sinf testda bo'lsa baribir C da);
  - qolganlari → o'chirilgan va `CLASSES` dan olib tashlangan (FAQ ruxsat beradi).
- Qaror jadvali `docs/class_policy.md` + saytdagi Report sahifasida (ochiq, halol).

**Qabul mezoni:** `ENABLED_CLASSES` bilan dev Score A ≥ barcha sinflar yoqilgandagidan.

---

## 6-bosqich — Tezlik va vaqt zaxirasi (P1)

**Muammo:** RTX 3050 da 1.86×–2.47× (C3905: 315 s / 383 s budjet). T4 da oshib ketishi mumkin → video butunlay bo'sh.
**Maqsad:** RTX 3050 da **≤ 1.3×** jami (A+B), T4 da ≤ 2× (katta zaxira).

`[AGENT]`:
1. `half=True` (CUDA bo'lsa) barcha YOLO chaqiruvlarida.
2. Part A: detektor har **2-kadrda** (`cap.grab()` qolganlarida — dekodlashni tejaydi), `sv.ByteTrack(frame_rate=fps/2)`. Dev da stride 1 va 2 ni solishtiring (ablation, saytga).
3. Detektor tanlovi: `yolo11l@640` va `yolo11m@960` ni dev da solishtiring (uzoqdagi piyodalar uchun 960 muhim bo'lishi mumkin). Eng yaxshi F1/vaqt nisbatini oling.
4. Dekodlash: 4K H.264 CPU dekodlash ham qimmat. `cv2.CAP_PROP_HW_ACCELERATION` (`cv2.VIDEO_ACCELERATION_ANY`) ni sinab ko'ring; yoki PyAV + `thread_type="AUTO"`.
5. Kadrni YOLO ga berishdan oldin bir marta `cv2.resize` (1280 kenglik) → preprocessingni tezlashtiradi; koordinatalarni qayta masshtablang.
6. Svetofor holati har 3-kadrda; zona triggerlari vektorlashtirilgan.
7. Part B: yolo11l ni alignment uchun **yuklamang** (3.2); `yolov8n` `half=True`, har 3-kadrda — saqlansin.
8. Warm-up: model yuklash vaqti birinchi videoga tushadi — `_load_yolo` da bir marta bo'sh tensor bilan warm-up.
9. **O'z-o'zini himoya:** `detect_events` ichida vaqtni kuzating; agar `elapsed > 1.8 × duration` bo'lsa, qolgan qismda stride'ni 4 ga oshiring (degradatsiya, lekin 0 emas). Log'ga yozing.

**Qabul mezoni:** `scripts/bench.sh` — har video uchun `part_a_sec`, `part_b_sec`, `total/duration` jadvali (`devset/timing.md`); barchasi ≤ 1.3× (3050). GPU yo'q holatda qancha bo'lishi ham yozilsin (ma'lumot uchun).

---

## 7-bosqich — Part B (RiskEstimator) sifati (P2)

**Muammo:** kadrlarning 15–32% ida risk ≥ 0.5 → ko'p alarm, precision past; AP past.

`[AGENT]`:
1. Kalibrlash maqsadi: accident bo'lmagan sample videolarda risk ≥ 0.5 bo'lgan kadrlar ulushi **< 1%**, alarmlar soni ≤ 1 / 5 daqiqa.
2. Signal: tracklardan **haqiqiy TTC** (nisbiy pozitsiya va nisbiy tezlik vektori, 1 s oynada silliqlangan). `risk_pair = sigmoid(a·(T0 − TTC))`, T0 ≈ 1.5–2 s; faqat yaqinlashayotgan (closing speed > 0) va trajektoriyalari kesishadigan juftliklar.
3. Qo'shimcha kuchaytiruvchilar (ko'paytma, har biri ≤ 1.3×): wrong-way track, RED da stop chiziqni kesib o'tayotgan track (3.1 dagi to'g'ri svetofor bilan), yo'ldagi piyoda (chavandoz emas) oldida tez mashina, keskin tormoz.
4. Svetofor navbatidagi zich, sekin oqim → risk yo'q (normalizatsiya: nisbiy tezlik < chegara bo'lsa juftlik e'tiborsiz).
5. Silliqlash: EMA + **persistence** (0.5 dan oshishi uchun ≥ 0.5 s barqaror signal).
6. `step` har chaqiriqda frame_count emas, **`t_sec`** ga tayansin (tezlik px/s) — shunda harness (stride=1) va demo (istalgan stride) bir xil egri chiziq beradi.
7. Part B vaqti Part A bilan jami budjetga kiradi — `yolov8n@640 half` har 3-kadrda.
8. **Kauzallik testi** `tests/test_risk_causal.py`: bir videoning birinchi N kadrini ikki xil davomida (qolgan kadrlar boshqa) berganda birinchi N ta natija bir xil.

**Tekshiruv:** dev labels'da accident yo'q bo'lsa — `[HUMAN]` jamoa ochiq datasetlardan (masalan CCD/DoTA ning litsenziyasi mos qismi) 5–10 ta qisqa klip bilan sinov qilsin (faqat tekshiruv uchun; README da litsenziya bilan). Egri chiziqlar saytga (8.x).

**Qabul mezoni:** sample videolarda risk ≥ 0.5 ulushi < 1%; kauzallik testi o'tadi; egri chiziq qiymati ∈ [0,1].

---

## 8-bosqich — Kod tuzilmasi va toza repo (P2)

Rubric: "Clear modules, no dead code, no notebooks as the only source".

### 8.1 Modullarga ajratish `[AGENT]`
```
solution.py              # yupqa: CLASSES, detect_events, RiskEstimator — src/ dan chaqiradi
src/
  paths.py               # REPO_ROOT, WEIGHTS_DIR
  config.py              # RULES (barcha chegaralar), ENABLED_CLASSES, SEED
  scene.py               # SCENE_CONFIG (4K), build_scene(W,H,dx,dy), nomlangan zonalar, oqim vektorlari
  traffic_light.py       # 3.1
  models.py              # _load_yolo (kesh, half, warm-up)
  tracking.py            # detektor+ByteTrack, TrackState (tarix, tezlik, heading, is_rider)
  rules/                 # har bir sinf alohida modul: red_light.py, stop_line.py, stopped.py, ...
  learned.py             # accident_model gating
  postprocess.py         # 4.13
  risk.py                # RiskEstimator ichki mantig'i
  annotate.py            # render (sayt/preview)
  eda/                   # deep_eda.py, eda_extractor.py
  devset/                # label_tool.py, make_candidates.py, summarize.py
scripts/                 # eval_dev.sh, bench.sh, regress.sh, tl_contact_sheet.py, replay_rules.py
tests/                   # pytest: postprocess, geometry, traffic_light (fixture crop'lar), risk causal
web/ yoki app.py         # sayt (10-bosqich)
```
- `detect_events` ichidagi 500 qatorli sikl → `tracking` + `rules` registri (`for rule in ENABLED_RULES: rule.update(frame_ctx)`).

### 8.2 O'lik kod va keraksiz fayllarni tozalash `[AGENT]`
O'chirish yoki ko'chirish:
- `visualizer.py` → `src/tools/visualizer.py` (yoki `annotate.py` bilan birlashtirish; `build_scene_zones` duplikatini `scene.py` ga).
- `benchmark_runner.py` → `scripts/bench.py`.
- `scratch/` (gitignored, lokal tozalash), `reference.png` (15 MB) va `reference.jpg` → `docs/img/scene_zones.jpg` (siqilgan, ≤ 1 MB), `reference resource.txt` → `docs/scene.md` ga integratsiya.
- `Videos.pdf`, `WIUT Hackathon _ CV Track Elimination Task.pdf` → repodan olib tashlash (tashkilotchi materiali; kerak bo'lsa `docs/` ga havola).
- `.agents/`, `skills-lock.json` — repodan olib tashlash (gitignore).
- `_compute_iou` wrapper, ishlatilmagan `copy`/argumentlar, `get_ai_offset` dagi ishlatilmagan `e`.
- Kod ichidagi marketing iboralari ("BRUTAL PART A FILTERING", "High-Precision") → oddiy texnik izohlar.

### 8.3 Sifat vositalari `[AGENT]`
- `ruff` (lint + format) konfiguratsiyasi `pyproject.toml` da; `ruff check .` toza.
- `pytest -q` ≥ 15 ta test, barchasi o'tadi.
- Ixtiyoriy: GitHub Actions — `ruff` + `pytest` (GPU'siz testlar) + `evaluate.py --validate-only predictions_samples.json`.

### 8.4 Training/tuning skriptlari `[AGENT]`
Rubric "training scripts present": biz model o'qitmadik. README da aniq yozing: "Biz hech qanday modelni fine-tune qilmadik; barcha chegaralar `scripts/replay_rules.py` + `devset/labels.json` bilan grid-search orqali tanlandi" va `scripts/tune_thresholds.py` (grid-search) repoda bo'lsin.

---

## 9-bosqich — Determinizm, reproduktivlik, regressiya (P2)

`[AGENT]`:
1. `src/config.py::seed_everything(42)`: `random`, `numpy`, `torch`, `torch.cuda`, `torch.backends.cudnn.deterministic=True`, `benchmark=False`, `os.environ["PYTHONHASHSEED"]` (README da eslatma).
2. Bir xil mashinada ikki marta ishga tushirib `scripts/compare_preds.py a.json b.json` — hodisalar bir xil, risk farqi ≤ 1e-3.
3. `predictions_samples.json` **yakuniy kod bilan** qayta generatsiya qilinsin va commit qilinsin (rubric: "results on the samples match predictions_samples.json"). Sample videolar `samples/` da bo'lishi README da ko'rsatilsin (ular gitignored).
4. `scripts/regress.sh`: `pytest` → sample'larda `run_submission.py` → `evaluate.py --validate-only` → `evaluate.py --gt devset/labels.json` → timing jadvali → oldingi natija bilan diff.

**Qabul mezoni:** ikki ketma-ket ishga tushirish `compare_preds.py` da farqsiz.

---

## 10-bosqich — Veb-sayt (P1, Website = 25%)

### 10.1 Deploy `[HUMAN + AGENT]`
- **Muammo:** ommaviy URL topilmadi. Sayt hakamlar davrida doim onlayn bo'lishi shart.
- Tavsiya: **Hugging Face Spaces (Streamlit SDK)**; CPU Basic bepul, lekin 4K yolo11l CPU da juda sekin — `[HUMAN]` imkon bo'lsa T4 small (pullik) yoki ZeroGPU. Muqobil: o'z serveringiz (GPU) + Cloudflare Tunnel.
- `[AGENT]`: `Dockerfile.web` yoki Spaces uchun `README.md` header (`sdk: streamlit`, `app_file: app.py`), `requirements-web.txt`, weights'ni birinchi ishga tushishda `weights/download.py` orqali yuklash (Spaces'da internet bor), yoki HF model repo'dan.
- Spaces fayl limitlari: preview videolar ≤ 50 MB/fayl (1.4 bosqichga qarang).
- **Qabul mezoni:** inkognito brauzerda URL ochiladi, 5 ta sahifa ham xatosiz; README va sayt bir-biriga havola qiladi.

### 10.2 Live demo ishonchliligi `[AGENT]`
- **Umumiy fayl muammosi:** `temp_uploaded.mp4` barcha tashrif buyuruvchilar uchun bitta → poyga holati. `tempfile.mkdtemp()` + `uuid` har bir sessiyaga, ish tugagach (va 1 soatdan eski) tozalash.
- **Demo rejimi (CPU uchun):** `detect_events(path, config=DEMO_CONFIG)` — kadrni 1280 ga kichraytirish, stride 3, `yolo11s` yoki `yolo11m` (weights repo'da). Saytda ochiq yozing: "Demo tezlik uchun yengil sozlamada ishlaydi; to'liq sozlama natijalari Results sahifasida." Maqsad: 2 daqiqalik 4K klip CPU da ≤ 3–4 daqiqada.
- **Part B pariteti:** demo `step` ni har kadrda chaqirsin (yoki 7.6 dagi `t_sec`-asosli tezlik bilan istalgan stride bir xil natija bersin). Hozir har 5-kadr × ichki 3-kadr → boshqa egri chiziq.
- **Cheklovlar ochiq ko'rsatilsin:** "≤ 2 daqiqa, ≤ 300 MB, .mp4 (H.264)". `.streamlit/config.toml` da `maxUploadSize = 300` (hozir 10240 MB — xavfli).
- **Boshqa kamera/o'lcham:** 3.3 dagi masshtablash bilan 1080p/720p upload ham ishlashi; kamera boshqa bo'lsa — ogohlantirish ("zonalar bu kamera uchun kalibrlangan") lekin crash yo'q.
- Buzilgan/audio-only/0 kadrli fayl → aniq xato xabari, crash emas.
- Progress: Part A va B uchun ETA bilan (bor — saqlansin), bekor qilish tugmasi.
- Natija: hodisalar jadvali + interaktiv timeline (**hodisani bosganda video shu vaqtga o'tadi** — `st.video(start_time=...)` yoki custom HTML5 player komponenti) + risk egri chizig'i (Plotly, 0.5 chizig'i bilan) + annotatsiyalangan kliplar + `events.json` yuklab olish tugmasi.
- **Qabul mezoni:** `tests/test_web_smoke.py` (Streamlit `AppTest`) — sample 20 s klipni yuklab, natija qaytishini tekshiradi; 3 xil fayl (4K, 1080p, buzilgan) bilan qo'lda sinov `[HUMAN]`.

### 10.3 Sample natijalari sahifasi `[AGENT]`
- 4 ta sample'ning **yakuniy kod** bilan qayta render qilingan annotatsiyalangan videolari (`src/annotate.py`), har biri uchun timeline + risk egri chizig'i (interaktiv).
- Har bir aniqlangan sinfdan **misol klip** (2–5 s GIF/MP4).
- **Halol xatolar bo'limi:** dev set'dagi eng xarakterli FP va FN lar (klip + tushuntirish).
- Dev set natijalari: sinfma-sinf P/R/F1 jadvali (τ=0.3/0.5/0.7), confusion (sinflar orasida vaqt bo'yicha kesishish matritsasi).

### 10.4 EDA sahifasi `[AGENT]`
PDF talabi: resolution, fps, duration, yorug'lik; vaqt bo'yicha sinf sonlari; motion heatmap; trajektoriyalar va yo'nalishlar; zichlik vaqt bo'yicha. Mavjudlarini saqlab, qo'shing:
- Yorug'lik (kadr yorqinligi histogrami vaqt bo'yicha), svetofor tsikli taqsimoti (3.1 dan: qizil/yashil davomiyligi).
- Tezlik taqsimoti har bir lane bo'yicha; piyoda oqimi zebralar bo'yicha.
- **"Bu topilma yechimga qanday ta'sir qildi"** bloki har grafik ostida (rubric: "findings that shaped the solution") — masalan "svetofor tsikli ~X s → congestion uchun 10 s qoidasi", "trajektoriyalardan oqim vektorlari olindi".

### 10.5 Approach sahifasi to'g'riligi `[AGENT]`
- **Muammo:** qoidalar jadvali kodga zid (red ≥15 px vs 5; failure_to_yield "<80 px" vs global flag; wrong_way dot-product vs dx; stopped <3 px/s vs 10; congestion ≥3 vs ≥4).
- Jadvalni **`src/config.py::RULES` dan avtomatik** generatsiya qiling — endi hech qachon ajralmaydi.
- Pipeline diagrammasi (Mermaid yoki SVG) haqiqiy modullarga mos; "nima o'rganilgan, nima qoidaga asoslangan" aniq ro'yxat.
- Ablation jadvali (6-bosqichdan): stride 1/2/3, yolo11l vs yolo11m, tracking bilan/siz, svetofor eski/yangi — dev F1 va vaqt bilan.

### 10.6 Team sahifasi `[HUMAN + AGENT]`
- `[HUMAN]`: har bir a'zo uchun: haqiqiy GitHub profili, LinkedIn, portfolio sayti, 1–2 ta faxrlanadigan oldingi loyiha (havola bilan), **aniq kim nima qildi** (commit'larga mos).
- `[AGENT]`: Siroj uchun placeholder `https://linkedin.com` va repo-havola "GitHub" ni haqiqiylari bilan almashtirish; havolasi yo'q tugmani ko'rsatmaslik.
- Rol nomlaridagi mubolag'alarni ("Lead ... Architect") real ishga moslash.

### 10.7 Report sahifasi `[AGENT]`
Bir sahifa: nima qurildi; **nima ishladi** (raqamlar bilan); **nima ishlamadi** (masalan: "anomaliya modeli CCTV da ko'p FP berdi → gating"; "svetofor HSV v1 99% RED → lampa-ROI v2"; o'chirilgan sinflar va sababi); keyingi qadamlar. Mubolag'asiz, "no marketing" ohangida.

### 10.8 Links va UX `[AGENT]`
- Links: repo (tag bilan), weights (download.sh + HF havolalari), `predictions_samples.json` (raw GitHub havola), dev labels.
- Telefon: 375px kenglikda barcha sahifalar gorizontal scroll'siz (`[HUMAN]` real telefonda tekshiradi).
- Tezlik: og'ir videolar lazy-load, `st.cache_data` bilan EDA CSV'lar.
- Qo'shimcha kredit (vaqt qolsa): operator dashboard (soat/lane/sinf bo'yicha hodisalar), webcam/stream demo (`streamlit-webrtc`), confusion matrix.

---

## 11-bosqich — README va hisobot (P2)

PDF bo'yicha README da **majburiy**:
- [ ] O'rnatish (Python 3.10–3.12), `bash weights/download.sh` (internet bilan bir marta), ikki buyruq bilan ishga tushirish, Docker varianti.
- [ ] Yondashuv: arxitektura diagrammasi, modellar, **nima o'rganilgan / nima qoidaga asoslangan**.
- [ ] **Datasetlar va litsenziyalar:** COCO (YOLO pre-training, CC BY 4.0); `accident_model.pt` — HF model kartasidan **uning o'qitilgan dataseti va litsenziyasini** aniqlang va yozing (`[HUMAN]` model kartasini tekshiradi; litsenziya noaniq bo'lsa — modelni olib tashlash yoki muallifdan so'rash; qoida buzilishi diskvalifikatsiya xavfi).
- [ ] Ochiq kodni qayta ishlatish atributsiyasi (Ultralytics AGPL-3.0, supervision MIT, ByteTrack MIT).
- [ ] Seedlar va determinizm (9-bosqich); nodeterministik qismlar (cuDNN) haqida eslatma.
- [ ] Jamoa a'zolari va **kim nima qildi**.
- [ ] Dev set: qanday belgilandi, natijalar jadvali, vaqt jadvali.
- [ ] Veb-sayt URL'i, predictions_samples.json haqida.
- [ ] Ma'lum cheklovlar (halol).

---

## 12-bosqich — Yakuniy qabul tekshiruvi va tag (P0)

`[AGENT]` quyidagini **toza muhitda** bajaradi va natijani `docs/FINAL_CHECK.md` ga yozadi:

```bash
# 1. Toza klon, Python 3.10
docker run --gpus all -it --rm -v /path/samples:/data/test python:3.10 bash
git clone https://github.com/DeWeWO/wiut && cd wiut && git checkout <tag>
pip install -r requirements.txt
bash weights/download.sh
# 2. Internetni o'chirib (docker --network none bilan qayta ishga tushirish)
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
# 3. Sample natijasi commit qilingan fayl bilan mos
python scripts/compare_preds.py predictions.json predictions_samples.json
# 4. Dev ball
python evaluate.py --pred predictions.json --gt devset/labels.json
```

Checklist:
- [ ] `run_submission.py`, `evaluate.py` o'zgarmagan (`git diff` bo'sh).
- [ ] Python 3.10 va 3.12 da install OK.
- [ ] Oflayn ishlaydi, hech qanday yuklab olish urinishi yo'q.
- [ ] Har bir video `total_sec / duration ≤ 1.5` (lokal GPU), log'da xato yo'q.
- [ ] `predictions_samples.json` yakuniy kod bilan mos.
- [ ] Dev Score A va Score B `docs/FINAL_CHECK.md` da, baseline bilan solishtirilgan.
- [ ] Ikki ketma-ket ishga tushirish bir xil natija.
- [ ] `ruff check .` va `pytest` toza.
- [ ] Sayt ommaviy URL'da, 5 sahifa + demo ishlaydi, telefonda ko'rinadi.
- [ ] README barcha majburiy bandlarni o'z ichiga oladi.
- [ ] `git tag -a v1.0-elimination -m "Elimination submission"` va push; tashkilotchilarga **tag + sayt URL** yuboriladi `[HUMAN]`.

---

## Ishni taqsimlash tavsiyasi (3 kishi + agent)

| Kim | Vazifalar |
|---|---|
| Agent | 1, 2.1, 2.3, 3.1–3.4 kodi, 4 (barcha kod), 5, 6, 7, 8, 9, 10 (kod), 11 (qoralama), 12 |
| Ollabergan | 2.2 (C3896), 3.1 tasdiqlash, 4.8/4.9 manevr jadvali va yaxlit chiziqlar, 10.1 deploy |
| Seymonbek | 2.2 (C3897 + C3905), 7 uchun ochiq dataset kliplari, Part B tekshiruvi |
| Siroj | 2.2 (C3902), 10.6 profillar, 11 litsenziya tekshiruvi, 12 yakuniy toza-muhit sinovi |

**Kritik yo'l:** 1 → 2 (belgilash) → 3.1 → 4 → 5 → 6 → 9.3 (predictions qayta generatsiya) → 10.3 (natijalar sahifasi) → 12.
Belgilash (2.2) bo'lmasa 4–5 bosqichlar o'lchovsiz qoladi — **birinchi kuniyoq boshlang**.

## Kutilayotgan natija

| Komponent | Hozir (taxmin) | Reja bajarilgach (maqsad) |
|---|---|---|
| Model (0.7A + 0.3B) | ~0.10 (yoki 0 — install xatosi) | 0.30–0.45 |
| Website | ~0.50 (deploy qilinmagan) | 0.80–0.90 |
| Code | ~0.50 | 0.85–0.95 |
| **Elimination** | **~0.28–0.35** | **~0.52–0.64** |
