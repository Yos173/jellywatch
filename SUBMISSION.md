# JellyWatch AI — submission kit

**Team:** Blue Sentinel   **Member:** Yosef (solo)
**Challenge:** ATRC ChallengeOn · ENEC — sustainable management of seasonal jellyfish blooms
**Cover image:** `assets/cover.png` (1024×576)   **Logo:** `assets/logo.png` / `assets/logo.svg`

---

## Short description (EN, ~100 words, for the form)

JellyWatch AI is an early-warning system for jellyfish blooms on the UAE coast. A YOLO11 computer-vision
model counts jellyfish in real time on underwater or cooling-water-intake cameras. A machine-learning risk
model reads live sea temperature, waves, currents and onshore wind from open ocean data and forecasts bloom
risk (0–100) for the next 3 days at Barakah's intake, Abu Dhabi beaches and Ruwais fishing grounds.
When risk rises, targeted alerts tell plant operators to prepare screens, beach managers to warn swimmers and
fishermen which zone to avoid — hours to days before damage occurs.

## One-liner (EN)

Detect, predict and alert: AI that warns the Barakah plant, beaches and fishermen about jellyfish blooms
up to 3 days before they arrive.

## Краткое описание (RU)

JellyWatch AI — система раннего предупреждения о нашествии медуз у побережья ОАЭ. Модель компьютерного
зрения YOLO11 в реальном времени считает медуз на подводных камерах и камерах водозабора. ML-модель риска
анализирует реальные данные о температуре моря, волнах, течениях и нагонном ветре и прогнозирует риск
(0–100) на 3 дня для водозабора АЭС «Барака», пляжей Абу-Даби и рыболовных зон Рувайса. При росте риска
система отправляет адресные алерты: операторам станции — подготовить сетки и усилить очистку, пляжам —
предупредить купающихся, рыбакам — избегать опасной зоны.

---

## Problem
Jellyfish blooms clog the seawater cooling intakes of coastal power and desalination plants — forcing power
reductions or shutdowns (documented worldwide, including nuclear plants). They also close beaches and destroy
fishing nets. Today operators usually find out when the screens are already blocked.

## Solution — 3 layers
1. **Detect** — fine-tuned YOLO11s + ByteTrack: jellyfish count, unique individuals, smoothed density per camera.
2. **Predict** — Gradient-Boosting model on real Open-Meteo marine & weather data → 0–100 risk + level, 3 days ahead,
   with human-readable drivers ("warm water 31 °C; calm sea 5 days; onshore wind 14 km/h").
3. **Alert** — playbook per audience (plant operators / beach managers / fishermen) with concrete actions;
   live camera evidence instantly raises the forecast.

## Sustainable management angle
Early warning turns an emergency into a planned operation: fewer shutdowns, less chemical/mechanical cleaning,
and jellyfish can be **collected before they reach the intake** and valorised (collagen, fertiliser, feed, food)
instead of being landfilled as waste.

## Honesty statement
Ocean/weather data and training images are real; bloom labels for the risk model are **synthetic** (expert rule),
because no public daily Gulf bloom records exist. Demo footage is not from Barakah. Coordinates are approximate.

## 60-second demo script
1. Open dashboard → map shows today's risk at Barakah, Corniche, Ruwais (real weather, live).
2. Alerts feed: Medium alert for plant operators with drivers and action.
3. Live detection → run `demo_bloom.mp4` → boxes + count + density climb to *Critical*.
4. Camera evidence feeds the forecast → Barakah jumps to High/Critical → new alert "prepare screens / increase cleaning".
5. Forecast panel: 3-day risk + "why" (feature importance) + last-12-months pattern.
