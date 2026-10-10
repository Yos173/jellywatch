// Builds docs/JellyWatch_AI_Proposal.docx — the written proposal for ATRC ChallengeOn (ENEC challenge).
// Run:  node docs/build_proposal.js     (metrics are read live from models/*.json)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, Header, PageNumber,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const readJson = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : {});
const det = readJson(path.join(ROOT, "models", "detector_metrics.json"));
const risk = readJson(path.join(ROOT, "models", "risk_metrics.json"));
const pct = (x) => (x === undefined ? "n/a" : x.toFixed(2));

const NAVY = "0B2236", CYAN = "0E7490", GREY = "5B6B7B", ORANGE = "C2410C";

// ---------------------------------------------------------------- helpers
// "**bold** *italic* plain" -> TextRuns
function runs(text, base = {}) {
  return text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g).filter(Boolean).map((s) => {
    if (s.startsWith("**")) return new TextRun({ ...base, text: s.slice(2, -2), bold: true });
    if (s.startsWith("*") && s.endsWith("*") && s.length > 2) return new TextRun({ ...base, text: s.slice(1, -1), italics: true });
    return new TextRun({ ...base, text: s });
  });
}
const P = (text, opts = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, ...opts, children: runs(text, opts.run || {}) });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 60, line: 264 }, children: runs(text) });
const N = (text) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60, line: 264 }, children: runs(text) });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const BR = () => new Paragraph({ children: [new PageBreak()] });

// Slide banner: "SLIDE 3 · key message" — helps turn this document into the 10-slide PDF deck
const slide = (n, msg) => new Paragraph({
  spacing: { before: 0, after: 160 },
  shading: { type: ShadingType.CLEAR, fill: "E8F7FB", color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 24, color: "22D3EE", space: 8 } },
  children: [new TextRun({ text: `SLIDE ${n}  ·  `, bold: true, color: CYAN, size: 18 }),
             new TextRun({ text: msg, italics: true, color: NAVY, size: 20 })],
});

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function img(rel, widthPx, caption) {
  const file = path.join(ROOT, rel);
  if (!fs.existsSync(file)) return [P(`[image missing: ${rel}]`, { run: { color: ORANGE } })];
  const { w, h } = pngSize(file);
  const out = [new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
      transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
      altText: { title: caption || rel, description: caption || rel, name: path.basename(rel) } })],
  })];
  if (caption) out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: caption, italics: true, size: 18, color: GREY })] }));
  return out;
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "C9D6E2" };
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, w, head) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: head ? { type: ShadingType.CLEAR, fill: NAVY, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({ spacing: { after: 0, line: 252 },
      children: runs(text, head ? { bold: true, color: "FFFFFF", size: 19 } : { size: 19 }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], true)) }),
           ...rows.map((r) => new TableRow({ children: r.map((t, i) => cell(t, widths[i], false)) }))],
  });
}
const gap = () => new Paragraph({ spacing: { after: 120 }, children: [] });

// ---------------------------------------------------------------- content
const detTest = det.test || {}, detVal = det.val || {};
const children = [];

// Title page
children.push(
  ...img("assets/logo.png", 150),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new TextRun({ text: "JellyWatch AI", bold: true, size: 64, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
    children: [new TextRun({ text: "Detect · Predict · Alert — early warning for jellyfish blooms on the UAE coast", size: 26, color: CYAN })] }),
  ...img("assets/cover.png", 560),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 300, after: 60 },
    children: [new TextRun({ text: "Proposal — ATRC ChallengeOn · ENEC Challenge", bold: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "“How might we advance the sustainable management of seasonal jellyfish blooms in coastal environments?”", italics: true, size: 22, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 },
    children: [new TextRun({ text: "Team Blue Sentinel  ·  Yosef", bold: true, size: 24, color: NAVY })] }),
  BR(),
);

// 1. Executive summary
children.push(
  H1("1. Executive summary"),
  slide(1, "AI that warns Barakah, beaches and fishermen about jellyfish blooms up to 3 days before they arrive."),
  P("Seasonal jellyfish blooms can block the seawater cooling intakes of coastal power plants, forcing power reductions or emergency shutdowns, and they also close beaches and damage fishing gear. Today, operators typically discover a bloom when the intake screens are already clogged."),
  P("**JellyWatch AI** turns this emergency into a planned operation with three layers:"),
  B("**Detect** — a computer-vision model (YOLO11, fine-tuned on jellyfish) counts and tracks jellyfish in real time on intake and underwater cameras."),
  B("**Predict** — a machine-learning model reads real ocean and weather data (sea temperature, waves, currents, onshore wind, season) and forecasts bloom risk 0–100 for today and the next 3 days at each monitored site."),
  B("**Alert** — targeted, actionable alerts for plant operators (prepare screens, increase cleaning), beach managers (swimming caution) and fishermen (avoid zone), with live camera evidence instantly raising the risk."),
  P(`A **working prototype** already exists: detector (test mAP@50 = ${pct(detTest.mAP50)}), live 3-day risk forecast on real Open-Meteo data for Barakah, Abu Dhabi Corniche and Ruwais, an alert engine and an interactive dashboard. The next step is a pilot with a real camera at the Barakah intake and ENEC's historical screen-cleaning logs to replace our clearly-labelled synthetic training labels.`),
  table(["What", "Status today"], [
    ["Jellyfish detector (video, image, webcam)", `Working — test mAP@50 ${pct(detTest.mAP50)}, precision ${pct(detTest.precision)}`],
    ["3-day bloom risk per site", "Working — real ocean & weather inputs; synthetic labels (declared)"],
    ["Alerts for 3 audiences", "Working — in dashboard; Telegram integration ready"],
    ["Dashboard (map, live detection, forecast, alerts)", "Working — Streamlit web app"],
  ], [4200, 5438]),
  BR(),
);

// 2. Team
children.push(
  H1("2. Introduction & team overview"),
  slide(2, "Team Blue Sentinel — a solo builder with hands-on real-time computer-vision experience."),
  P("**Team name: Blue Sentinel.** A sentinel that watches the sea so that people and infrastructure do not have to react too late."),
  table(["Member", "Role", "Relevant experience"], [
    ["**Yosef**", "Founder · AI & software lead (end-to-end: data, models, dashboard, pitch)",
     "Built **AI Fit** — a real-time pose-tracking web app with MediaPipe (live camera → model → feedback loop). JellyWatch reuses the same real-time vision pattern for the ocean."],
  ], [1800, 3300, 4538]),
  gap(),
  H2("Why we are well placed"),
  B("Experience shipping **real-time computer vision** on ordinary hardware (laptop / web)."),
  B("Built and trained the full prototype end-to-end: data pipelines, model training, forecasting, alerting and UI."),
  B("Lean, low-cost approach: free open data, open-source models, runs on a laptop GPU or CPU."),
  H2("Declared use of AI tools"),
  P("As permitted by the challenge rules, we used an AI coding assistant (**Claude Code by Anthropic**) to speed up code scaffolding, debugging, documentation and visual design. All design decisions — choice of datasets, bloom drivers, risk thresholds, alert actions, honest labelling of synthetic data — were reviewed and owned by the team, and every component can be explained and demonstrated by the team in its own words. All trained models and results were produced by running our own code on our own machine."),
  BR(),
);

// 3. Problem
children.push(
  H1("3. The problem and why it matters"),
  slide(3, "Jellyfish blooms are a known, recurring threat to coastal energy security — and today they are managed reactively."),
  H2("What happens during a bloom"),
  B("**Power plants:** dense swarms are sucked into seawater cooling intakes, clog the travelling screens and filters, reduce cooling-water flow and can force power reduction or a precautionary shutdown. Well-documented international cases include nuclear plants in Sweden (Oskarshamn, 2013), Scotland (Torness, 2011) and the USA (Diablo Canyon, 2008)."),
  B("**Beaches & tourism:** stings close swimming areas and harm visitor safety and the UAE's tourism image."),
  B("**Fisheries:** heavy jellyfish catch tears nets, spoils the haul and costs fishermen time and income."),
  H2("Why it matters for the UAE and ENEC"),
  B("Barakah supplies up to a quarter of the UAE's electricity; its availability depends on reliable seawater cooling from the Arabian Gulf."),
  B("The Gulf is warm, shallow and semi-enclosed; rising sea temperatures and coastal development are expected to favour jellyfish."),
  B("Operators currently lack an **early, site-specific warning** — the response starts when screens are already blocked, which is the most expensive moment to act."),
  H2("The gap we address"),
  table(["Today", "With JellyWatch AI"], [
    ["Bloom noticed when screens clog", "Risk forecast up to 3 days ahead + live camera confirmation"],
    ["Generic, manual observation", "Automatic counting, tracking and density per camera"],
    ["One-size-fits-all reaction", "Targeted actions for operators, beaches and fishermen"],
    ["Jellyfish removed as waste", "Planned collection and valorisation of biomass"],
  ], [4819, 4819]),
  BR(),
);

// 4. Solution
children.push(
  H1("4. Proposed solution and approach"),
  slide(4, "Three layers — Detect, Predict, Alert — that work today on open data and scale to ENEC's own sensors."),
  H2("4.1 Detect — computer vision on intake & underwater cameras"),
  B("**Model:** YOLO11s (Ultralytics), transfer-learned from COCO and fine-tuned on a single class “jellyfish”."),
  B("**Training data (all open, no account needed):** Roboflow *Aquarium* (CC BY 4.0), Aalborg University *Brackish* — frames from a **permanently mounted underwater camera**, the closest public analogue to an intake camera (CC BY 4.0), and Google *Open Images V7* “Jellyfish” close-ups. Images of fish, crabs and sharks are kept as negatives so the model does not fire on other animals."),
  B("**Outputs per frame:** jellyfish count, unique individuals (ByteTrack multi-object tracking), and a smoothed **density score** (rolling mean of count). Density thresholds map to camera levels Low / Medium / High / Critical."),
  B(`**Accuracy:** held-out test set mAP@50 = ${pct(detTest.mAP50)}, mAP@50-95 = ${pct(detTest.mAP50_95)}, precision = ${pct(detTest.precision)}, recall = ${pct(detTest.recall)} (validation mAP@50 = ${pct(detVal.mAP50)}).`),
  H2("4.2 Predict — 3-day bloom risk per site"),
  B("**Inputs (real):** Open-Meteo Marine & Weather APIs — sea-surface temperature, wave height, ocean current speed/direction, wind speed/direction, solar radiation; history since 2022 and a 4-day forecast."),
  B("**Features:** onshore wind component calculated for each coastline's orientation, consecutive calm-sea days, 7-day SST warming trend, SST anomaly vs monthly climatology, current strength, seasonality."),
  B("**Model:** Gradient Boosting classifier → probability × 100 = risk score; levels Low (<30), Medium (30–55), High (55–80), Critical (≥80). Every forecast comes with plain-language **drivers** (e.g. “calm sea 5 days; onshore wind 14 km/h”)."),
  B("**Labels:** no public daily bloom record exists for the Gulf, so labels are generated from a transparent expert rule (see Assumptions). This is clearly declared and designed to be replaced by ENEC records."),
  H2("4.3 Alert — the right action for the right people"),
  table(["Audience", "Example alert", "Recommended action"], [
    ["Plant operators", "High jellyfish risk at intake in ~24h", "Prepare screens, increase cleaning frequency, deploy booms/bubble curtain, brief control room on possible load reduction"],
    ["Beach managers", "Swimming caution advised", "Purple flag, lifeguard patrol, first-aid kits, public advisory"],
    ["Fishermen", "Avoid zone X", "Avoid nearshore zone, shorten soak times, report sightings"],
  ], [1900, 3000, 4738]),
  P("Each alert carries level, location, validity time, drivers and the recommended action. **Fusion:** when a camera sees a high density, today's forecast risk is immediately raised, so a real observation always overrides a calm forecast."),
  BR(),
);

// 5. System design
children.push(
  H1("5. System design"),
  slide(5, "From camera pixels and ocean data to a concrete action — in one pipeline."),
  ...img("docs/img/architecture.png", 620, "Figure 1 — JellyWatch AI architecture"),
  table(["Component", "Technology", "Runs on"], [
    ["Detector + tracker", "YOLO11s, ByteTrack, OpenCV (Python)", "Edge GPU / laptop; CPU fallback"],
    ["Data ingestion", "Open-Meteo Marine/Weather APIs (no key); synthetic fallback if offline", "Any server"],
    ["Risk model", "scikit-learn Gradient Boosting, pandas", "CPU, milliseconds per forecast"],
    ["Alert engine", "Rule playbook per audience; Telegram/SMS connector", "Any server"],
    ["Dashboard", "Streamlit, folium (Esri basemap), plotly", "Web browser"],
  ], [2400, 4400, 2838]),
  gap(),
  P("The design is **modular**: any camera (intake, underwater, drone), any additional data source (sonar, satellite chlorophyll, ENEC screen-pressure sensors) or any new recipient channel can be added without changing the rest of the pipeline."),
  BR(),
);

// 6. Impact
children.push(
  H1("6. Intended impact"),
  slide(6, "Fewer forced shutdowns, safer beaches, protected livelihoods — and jellyfish turned from waste into a resource."),
  table(["Stakeholder", "Impact"], [
    ["ENEC / Barakah operators", "Hours-to-days of lead time to prepare screens and crews; fewer emergency power reductions; better-planned maintenance; protection of grid stability and energy security"],
    ["Beach managers & public", "Earlier, targeted swimming advisories; fewer stings; protected tourism experience"],
    ["Fishermen", "Avoid high-risk zones, fewer torn nets and spoiled catches"],
    ["Environment & science", "A continuous, open record of jellyfish presence and ocean conditions — data that does not exist today for the Gulf"],
  ], [2800, 6838]),
  gap(),
  H2("Sustainable management — the circular angle"),
  P("Early warning makes it possible to **collect jellyfish offshore before they reach the intake** instead of disposing of them as clogging waste. Collected biomass can be valorised — collagen for cosmetics and biomedicine, fertiliser and soil conditioner, animal feed and, for edible species, food — creating value from a problem. JellyWatch provides the “where and when” that such a collection programme needs."),
  H2("How we will measure impact"),
  B("Lead time between first alert and screen-pressure rise at the intake (target: ≥ 24 h)."),
  B("Share of real bloom events preceded by a High/Critical alert (hit rate) and false-alarm rate."),
  B("Avoided power-reduction hours and reduced emergency cleaning interventions vs previous seasons."),
  BR(),
);

// 7. Feasibility
children.push(
  H1("7. Feasibility and real-world implementation"),
  slide(7, "Low-cost, open-source and buildable now — the prototype already runs on a laptop."),
  H2("Implementation roadmap"),
  table(["Phase", "Duration", "What happens"], [
    ["0 · Prototype", "Done", "Detector, risk model, alerts, dashboard on open data"],
    ["1 · Data partnership", "1–2 months", "Access ENEC intake-screen and cleaning logs, EAD jellyfish surveys; retrain risk model on real events"],
    ["2 · Pilot camera", "2–3 months", "1–2 underwater cameras at the Barakah intake; fine-tune detector on local species and water turbidity"],
    ["3 · Validation season", "One bloom season", "Run in shadow mode next to current procedures; measure lead time, hit rate, false alarms"],
    ["4 · Operational roll-out", "After validation", "Integrate alerts with control-room systems; extend to beaches, desalination plants and fishing communities"],
  ], [2300, 1800, 5538]),
  gap(),
  H2("Practical considerations"),
  B("**Hardware:** standard IP underwater cameras with lighting and wipers; an edge GPU box or existing plant server. The detector already runs in real time on a laptop GPU (RTX 2060)."),
  B("**Cost:** open-source software and free ocean data; main costs are cameras, installation and maintenance."),
  B("**Integration & security:** read-only, one-way data flow from cameras to the analytics network; no connection to plant control systems is required — alerts are **advisory** to human operators."),
  B("**Data governance:** no personal data is processed; camera views face the water."),
  H2("Risks and mitigations"),
  table(["Risk", "Mitigation"], [
    ["Turbid water / bio-fouling reduces camera visibility", "Lighting, wipers, cleaning schedule; add sonar as a second sensor"],
    ["Few real bloom records for training", "Start with expert rule (declared), retrain as ENEC logs arrive; active learning from operator feedback"],
    ["Local species differ from training data", "Fine-tune on footage from the pilot camera (a few hundred labelled frames)"],
    ["Alert fatigue", "Tunable thresholds per site; one alert per site per horizon; operator feedback loop"],
  ], [4000, 5638]),
  BR(),
);

// 8. Testing
children.push(
  H1("8. How we would test and demonstrate the solution"),
  slide(8, "A live demo today, and a measurable pilot tomorrow."),
  H2("Live demonstration (if shortlisted)"),
  N("Open the dashboard: the map shows today's risk at Barakah, Corniche and Ruwais from **live** ocean data."),
  N("Show the alerts feed with drivers and recommended actions per audience."),
  N("Run the detector on bloom footage: boxes, count, tracked individuals and density rise in real time."),
  N("Camera evidence is fused into the forecast: Barakah's risk jumps and a new alert for plant operators appears instantly."),
  N("Show the 3-day forecast, the “why” (feature importance) and the last-12-months risk pattern on real weather."),
  H2("Technical validation"),
  B("**Detector:** precision, recall and mAP on a held-out test set (already reported); next on labelled frames from the Barakah intake."),
  B("**Risk model:** time-based back-testing (train on past seasons, test on the most recent) — today against synthetic labels; in the pilot against ENEC bloom/cleaning events, reporting hit rate, false-alarm rate and average lead time."),
  B("**End-to-end:** shadow-mode operation for one season alongside current procedures; operators rate alert usefulness."),
  H2("Success criteria for the pilot"),
  table(["Metric", "Target"], [
    ["Detector precision on intake footage", "≥ 0.85"],
    ["Bloom events preceded by a High/Critical alert", "≥ 70 %"],
    ["Median warning lead time", "≥ 24 hours"],
    ["False alarms per month per site", "≤ 2"],
  ], [6000, 3638]),
  BR(),
);

// 9. Visuals
children.push(
  H1("9. Visuals, mock-ups and prototype"),
  slide(9, "It already works — map, live detection, forecast and alerts in one dashboard."),
  ...img("docs/img/dashboard.png", 620, "Figure 2 — JellyWatch AI dashboard: KPIs, coastal risk map and alerts feed (live data)"),
  ...img("docs/img/detection.png", 560, "Figure 3 — Live detection: jellyfish boxes, tracking IDs, count, density and level"),
  ...img("docs/img/forecast.png", 620, "Figure 4 — 3-day bloom risk forecast per site and model explainability"),
  H2("Prototype facts"),
  table(["Item", "Value"], [
    ["Detector", `YOLO11s, test mAP@50 ${pct(detTest.mAP50)} · precision ${pct(detTest.precision)} · recall ${pct(detTest.recall)}`],
    ["Risk model", `Gradient Boosting · ROC-AUC ${risk.roc_auc ?? "n/a"} on the most recent 12 months (vs synthetic labels)`],
    ["Ocean/weather history", `${(risk.train_days || 0) + (risk.test_days || 0)} site-days of real data since 2022 (3 sites)`],
    ["Code", "Python · open-source stack · runs on a laptop"],
  ], [2800, 6838]),
  BR(),
);

// 10. Assumptions
children.push(
  H1("10. Assumptions, data sources and next steps"),
  slide(10, "No dataset was provided — here is every assumption we made, stated openly."),
  H2("Assumptions"),
  table(["#", "Assumption", "Why / how it will be replaced"], [
    ["A1", "Blooms are favoured by warm water (≈24–33 °C, optimum ≈30 °C), a warming trend, several calm-sea days, onshore wind, weak currents and the spring–summer season", "Encoded as a transparent expert rule to create **synthetic training labels**; replaced by ENEC/EAD bloom records"],
    ["A2", "Camera density thresholds: Medium ≥ 2, High ≥ 5, Critical ≥ 10 jellyfish per frame (smoothed)", "Calibrate against intake screen-pressure data during the pilot"],
    ["A3", "Site coordinates for Barakah intake, Abu Dhabi Corniche and Ruwais are **approximate**", "Replace with exact intake / beach / fishing-zone coordinates"],
    ["A4", "Open-Meteo model data is representative of near-shore conditions at each site", "Add ENEC / NCM in-situ sensors where available"],
    ["A5", "Demo videos (public moon-jellyfish footage) stand in for an intake camera", "Pilot camera footage from Barakah"],
    ["A6", "Alerts are advisory; humans decide on operational actions", "Integrate with existing ENEC procedures"],
  ], [600, 4700, 4338]),
  gap(),
  H2("What is real vs synthetic"),
  table(["Component", "Status"], [
    ["Sea temperature, waves, currents, wind (history + forecast)", "**REAL** — Open-Meteo (CC BY 4.0)"],
    ["Detector training images", "**REAL** — Aquarium (Roboflow), Brackish (Aalborg University), Open Images V7 (Google)"],
    ["Risk-model training labels", "**SYNTHETIC** — expert rule + noise (declared)"],
    ["Demo footage", "**DEMO** — Wikimedia Commons, not filmed at Barakah"],
  ], [5200, 4438]),
  gap(),
  H2("Next steps"),
  B("Real camera at the Barakah intake; real ENEC/EAD bloom and screen-cleaning data."),
  B("Sonar and drone integration; satellite chlorophyll and sea-surface temperature as additional features."),
  B("Jellyfish collection and valorisation programme (collagen, fertiliser, feed, food) as the sustainable-management loop."),
  B("Arabic/English mobile alerts for fishermen; API for the plant control room."),
);

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "Team Blue Sentinel", title: "JellyWatch AI — Proposal", description: "ATRC ChallengeOn — ENEC challenge proposal",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21, color: "1F2937" } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 34, bold: true, color: NAVY, font: "Calibri" },
        paragraph: { spacing: { before: 120, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: CYAN, font: "Calibri" },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 270 } } } } ] },
    { reference: "numbers", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } } } ] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "JellyWatch AI · Team Blue Sentinel · ATRC ChallengeOn — ENEC", size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: GREY })] })] }) },
    children,
  }],
});

const out = path.join(__dirname, "JellyWatch_AI_Proposal.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("Wrote", out); });
