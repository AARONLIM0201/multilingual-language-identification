import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const artifactToolPath =
  "C:\\Users\\Aaron Lim\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules\\@oai\\artifact-tool\\dist\\artifact_tool.mjs";
const { Presentation, PresentationFile } = await import(pathToFileURL(artifactToolPath).href);

const ROOT = "C:\\Users\\Aaron Lim\\Dropbox\\PC\\Documents\\1School\\FYP";
const OUT_DIR = path.join(ROOT, "Inventex");
const OUT = path.join(OUT_DIR, "FYP_Demo_Video_Presentation.pptx");
const QA_DIR = path.join(OUT_DIR, "demo_deck_preview");

const W = 1280;
const H = 720;
const margin = 64;
const bg = "#F7FAFC";
const navy = "#102A43";
const teal = "#0E7490";
const cyan = "#06B6D4";
const red = "#EF4444";
const green = "#16A34A";
const orange = "#F59E0B";
const slate = "#475569";
const light = "#E2E8F0";

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: opts.size ?? 22,
    bold: opts.bold ?? false,
    color: opts.color ?? navy,
    alignment: opts.align ?? "left",
  };
  return shape;
}

function addTitle(slide, title, subtitle = "") {
  addText(slide, title, margin, 38, 880, 54, { size: 34, bold: true, color: navy });
  if (subtitle) {
    addText(slide, subtitle, margin, 94, 820, 30, { size: 15, color: slate });
  }
  slide.shapes.add({
    geometry: "rect",
    position: { left: margin, top: 132, width: 1152, height: 3 },
    fill: teal,
    line: { style: "solid", fill: teal, width: 0 },
  });
}

function addFooter(slide, index) {
  addText(slide, "Multilingual Low-Resource Language Identification", margin, 684, 650, 20, {
    size: 11,
    color: "#64748B",
  });
  addText(slide, String(index), 1190, 684, 40, 20, {
    size: 11,
    color: "#64748B",
    align: "right",
  });
}

function addCard(slide, x, y, w, h, title, body, accent = teal) {
  slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill: "white",
    line: { style: "solid", fill: light, width: 1 },
    borderRadius: "rounded-xl",
    shadow: "shadow-sm",
  });
  slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: 8, height: h },
    fill: accent,
    line: { style: "solid", fill: accent, width: 0 },
  });
  addText(slide, title, x + 22, y + 18, w - 44, 26, { size: 18, bold: true, color: navy });
  addText(slide, body, x + 22, y + 54, w - 44, h - 70, { size: 15, color: slate });
}

function addPlaceholder(slide, x, y, w, h, label) {
  slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill: "#F1F5F9",
    line: { style: "dash", fill: "#94A3B8", width: 2 },
    borderRadius: "rounded-xl",
  });
  addText(slide, label, x + 20, y + h / 2 - 20, w - 40, 40, {
    size: 20,
    bold: true,
    color: "#64748B",
    align: "center",
  });
}

function addPipelineNode(slide, x, y, w, label, detail, color) {
  slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: 78 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
    borderRadius: "rounded-xl",
  });
  addText(slide, label, x + 12, y + 14, w - 24, 24, { size: 18, bold: true, color: "white", align: "center" });
  addText(slide, detail, x + 12, y + 42, w - 24, 22, { size: 12, color: "white", align: "center" });
}

function addArrow(slide, x, y, w) {
  addText(slide, "->", x, y, w, 34, { size: 28, bold: true, color: slate, align: "center" });
}

function addMiniTable(slide, x, y, columns, rows, widths, rowH = 34) {
  const totalW = widths.reduce((a, b) => a + b, 0);
  slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: totalW, height: rowH * (rows.length + 1) },
    fill: "white",
    line: { style: "solid", fill: light, width: 1 },
    borderRadius: "rounded-lg",
  });
  let cx = x;
  for (let i = 0; i < columns.length; i++) {
    slide.shapes.add({
      geometry: "rect",
      position: { left: cx, top: y, width: widths[i], height: rowH },
      fill: teal,
      line: { style: "solid", fill: teal, width: 0 },
    });
    addText(slide, columns[i], cx + 6, y + 8, widths[i] - 12, rowH - 12, {
      size: 12,
      bold: true,
      color: "white",
      align: i === 0 ? "left" : "center",
    });
    cx += widths[i];
  }
  for (let r = 0; r < rows.length; r++) {
    cx = x;
    for (let c = 0; c < rows[r].length; c++) {
      addText(slide, rows[r][c], cx + 6, y + rowH * (r + 1) + 8, widths[c] - 12, rowH - 12, {
        size: 12,
        color: navy,
        align: c === 0 ? "left" : "center",
      });
      cx += widths[c];
    }
  }
}

function addBarChart(slide, x, y, w, h, title, labels, values, colors) {
  addText(slide, title, x, y, w, 28, { size: 18, bold: true, color: navy });
  const plotY = y + 44;
  const plotH = h - 70;
  const max = 100;
  const barGap = 22;
  const barW = (w - barGap * (values.length - 1)) / values.length;
  values.forEach((v, i) => {
    const bh = (v / max) * plotH;
    const bx = x + i * (barW + barGap);
    const by = plotY + plotH - bh;
    slide.shapes.add({
      geometry: "rect",
      position: { left: bx, top: by, width: barW, height: bh },
      fill: colors[i],
      line: { style: "solid", fill: colors[i], width: 0 },
    });
    addText(slide, `${v.toFixed(2)}%`, bx, by - 24, barW, 20, { size: 12, bold: true, color: navy, align: "center" });
    addText(slide, labels[i], bx, plotY + plotH + 8, barW, 34, { size: 11, color: slate, align: "center" });
  });
}

const p = Presentation.create({ slideSize: { width: W, height: H } });

// 1
{
  const s = p.slides.add();
  s.background.fill = bg;
  addText(s, "Multilingual Low-Resource\nLanguage Identification", margin, 84, 720, 115, {
    size: 42,
    bold: true,
    color: navy,
  });
  addText(s, "Using audio, OCR text, and multimodal late fusion", margin, 220, 700, 36, {
    size: 23,
    color: teal,
  });
  addText(s, "FYP demo video deck | English, Malay, Mandarin", margin, 278, 520, 28, { size: 17, color: slate });
  addPlaceholder(s, 760, 90, 430, 300, "Insert poster or Streamlit hero screenshot");
  addCard(
    s,
    margin,
    430,
    1080,
    130,
    "Demo focus",
    "Explain the problem briefly, then demonstrate the working Streamlit app and summarize the key evaluation results.",
    cyan,
  );
  addText(s, "Aaron Lim Cjun Shien | Faculty of Computing & Informatics, MMU", margin, 610, 780, 26, {
    size: 16,
    color: slate,
  });
  addFooter(s, 1);
}

// 2
{
  const s = p.slides.add();
  s.background.fill = bg;
  addTitle(s, "Problem and Objectives", "Why this project matters in Malaysia's multilingual setting");
  addCard(
    s,
    72,
    170,
    520,
    170,
    "Problem",
    "Many language identification systems rely on one modality and are stronger for high-resource languages. Malay is relatively under-resourced in speech, OCR, and multimodal datasets.",
    red,
  );
  addCard(
    s,
    72,
    374,
    520,
    188,
    "Research focus",
    "Build a language identification system for English, Malay, and Mandarin using speech audio and image-based text.",
    teal,
  );
  addCard(
    s,
    650,
    170,
    500,
    392,
    "Objectives",
    "1. Prepare audio and image-based text datasets.\n2. Develop audio-only and text-only models.\n3. Evaluate OCR engines.\n4. Compare multimodal late fusion strategies.\n5. Analyse Malay recognition performance.",
    cyan,
  );
  addFooter(s, 2);
}

// 3
{
  const s = p.slides.add();
  s.background.fill = bg;
  addTitle(s, "Dataset and Preprocessing", "Audio from VoxLingua-based sources; text images from Wikipedia samples");
  addCard(
    s,
    72,
    170,
    350,
    178,
    "Audio modality",
    "Approximately 7.5 hours per language for English, Malay, and Mandarin. Audio was segmented for CNN and wav2vec2 experiments.",
    teal,
  );
  addCard(
    s,
    465,
    170,
    350,
    178,
    "Text image modality",
    "Wikipedia text samples were rendered into image format. Each language contains 5,000 clean text images.",
    cyan,
  );
  addCard(
    s,
    858,
    170,
    350,
    178,
    "Noisy OCR images",
    "Noise augmentation simulates blur, compression artifacts, pixel noise, and lighting changes.",
    orange,
  );
  addPlaceholder(s, 92, 400, 330, 165, "Insert clean text image examples");
  addPlaceholder(s, 475, 400, 330, 165, "Insert noisy image examples");
  addPlaceholder(s, 858, 400, 330, 165, "Insert dataset summary chart");
  addFooter(s, 3);
}

// 4
{
  const s = p.slides.add();
  s.background.fill = bg;
  addTitle(s, "System Architecture", "Decision-level late fusion combines separate audio and text predictions");
  addPipelineNode(s, 70, 190, 170, "Audio Input", "speech waveform", teal);
  addArrow(s, 245, 210, 48);
  addPipelineNode(s, 300, 190, 190, "wav2vec2", "audio classifier", cyan);
  addArrow(s, 495, 210, 48);
  addPipelineNode(s, 550, 190, 190, "Audio Prob.", "EN / MS / ZH", teal);

  addPipelineNode(s, 70, 330, 170, "Image Input", "text image", orange);
  addArrow(s, 245, 350, 48);
  addPipelineNode(s, 300, 330, 190, "EasyOCR", "text extraction", green);
  addArrow(s, 495, 350, 48);
  addPipelineNode(s, 550, 330, 190, "DistilBERT", "text classifier", cyan);

  addArrow(s, 760, 278, 58);
  addPipelineNode(s, 830, 260, 190, "Late Fusion", "probability level", red);
  addArrow(s, 1025, 278, 48);
  addPipelineNode(s, 1080, 260, 140, "Final", "language", navy);
  addCard(
    s,
    110,
    520,
    1040,
    106,
    "Key idea",
    "Audio and text models make separate predictions first. Fusion then combines their probability outputs, allowing one modality to compensate when the other is uncertain.",
    teal,
  );
  addFooter(s, 4);
}

// 5
{
  const s = p.slides.add();
  s.background.fill = bg;
  addTitle(s, "OCR Evaluation", "TrOCR and EasyOCR were compared using character accuracy and Levenshtein distance");
  addMiniTable(
    s,
    80,
    175,
    ["Language", "TrOCR", "EasyOCR", "Selected"],
    [
      ["English", "62.24%", "87.90%", "EasyOCR"],
      ["Malay", "46.65%", "87.62%", "EasyOCR"],
      ["Mandarin", "2.56%", "76.82%", "EasyOCR"],
    ],
    [150, 110, 120, 130],
    42,
  );
  addCard(
    s,
    650,
    165,
    470,
    168,
    "Demo OCR strategies",
    "The Streamlit app includes TrOCR, EasyOCR, and Hybrid OCR. Hybrid OCR runs multiple OCR candidates and chooses the strongest extraction using confidence and text quality.",
    cyan,
  );
  addPlaceholder(s, 80, 420, 510, 150, "Insert OCR comparison chart");
  addPlaceholder(s, 650, 420, 470, 150, "Insert OCR output example");
  addFooter(s, 5);
}

// 6
{
  const s = p.slides.add();
  s.background.fill = bg;
  addTitle(s, "Streamlit App Demo Flow", "The app demonstrates the same pipeline used in the experiment");
  addCard(
    s,
    72,
    170,
    345,
    140,
    "1. Project Results",
    "Show saved accuracies, fusion comparison, and generated figures.",
    teal,
  );
  addCard(
    s,
    468,
    170,
    345,
    140,
    "2. Image / OCR",
    "Upload an image, run OCR, review extracted text, then classify with DistilBERT.",
    cyan,
  );
  addCard(
    s,
    864,
    170,
    345,
    140,
    "3. Audio",
    "Upload or record speech and classify it using wav2vec2.",
    orange,
  );
  addCard(
    s,
    270,
    360,
    345,
    140,
    "4. Fusion",
    "Provide both audio and text inputs, then compare audio-only, text-only, and fused predictions.",
    red,
  );
  addCard(
    s,
    665,
    360,
    345,
    140,
    "5. Explain outcome",
    "Relate the live prediction to the saved evaluation results and report findings.",
    green,
  );
  addPlaceholder(s, 392, 550, 496, 70, "Insert Streamlit app screenshot");
  addFooter(s, 6);
}

// 7
{
  const s = p.slides.add();
  s.background.fill = bg;
  addTitle(s, "Key Evaluation Results", "Transformer models and late fusion achieved the strongest performance");
  addMiniTable(
    s,
    80,
    168,
    ["Model", "Modality", "Accuracy"],
    [
      ["CNN", "Audio", "73.56%"],
      ["wav2vec2", "Audio", "93.91%"],
      ["FastText", "Text", "98.44%"],
      ["DistilBERT", "Text", "99.73%"],
      ["Audio Only", "Fusion set", "94.16%"],
      ["OCR Text Only", "Fusion set", "97.28%"],
      ["Equal Fusion", "Audio + Text", "99.18%"],
    ],
    [190, 145, 105],
    34,
  );
  addBarChart(
    s,
    600,
    168,
    520,
    310,
    "Fusion Strategy Accuracy",
    ["Equal\nFusion", "Confidence\nAdaptive", "Accuracy\nWeighted"],
    [99.18, 99.01, 97.86],
    [red, cyan, "#3B82F6"],
  );
  addCard(
    s,
    600,
    510,
    520,
    104,
    "Result to say in video",
    "Equal Weighting Fusion achieved the best result by combining complementary audio and OCR-text predictions.",
    teal,
  );
  addFooter(s, 7);
}

// 8
{
  const s = p.slides.add();
  s.background.fill = bg;
  addTitle(s, "Malay Low-Resource Analysis", "Fusion improved Malay recognition in the paired fusion test set");
  addBarChart(
    s,
    120,
    178,
    500,
    340,
    "Malay Accuracy by Method",
    ["Audio\nOnly", "OCR Text\nOnly", "Equal\nFusion"],
    [94.32, 92.84, 98.52],
    [orange, cyan, green],
  );
  addCard(
    s,
    690,
    178,
    430,
    160,
    "Interpretation",
    "Malay recognition improved when audio and OCR text were combined. This supports the objective of using multimodal fusion to reduce low-resource weaknesses.",
    green,
  );
  addPlaceholder(s, 690, 380, 430, 160, "Insert per-language comparison figure");
  addFooter(s, 8);
}

// 9
{
  const s = p.slides.add();
  s.background.fill = bg;
  addTitle(s, "Contributions, Limitations, and Future Work", "Final closing slide for the demo video");
  addCard(
    s,
    72,
    170,
    360,
    320,
    "Contributions",
    "- Multimodal dataset pipeline\n- Baseline vs transformer comparison\n- OCR engine evaluation\n- Late fusion with wav2vec2 + DistilBERT\n- Streamlit demo interface",
    teal,
  );
  addCard(
    s,
    460,
    170,
    360,
    320,
    "Limitations",
    "- Three-language scope\n- Wikipedia-based text images\n- OCR errors on noisy/handwritten inputs\n- Audio robustness depends on speakers and noise\n- Late fusion only",
    orange,
  );
  addCard(
    s,
    848,
    170,
    360,
    320,
    "Future work",
    "- Add more Malaysian languages\n- Collect real-world image/audio samples\n- Improve OCR for handwriting\n- Explore feature-level fusion\n- Enhance Streamlit demo",
    cyan,
  );
  addText(s, "Closing line: multimodal late fusion improves robustness by combining speech and OCR-based text.", 118, 555, 1040, 40, {
    size: 21,
    bold: true,
    color: navy,
    align: "center",
  });
  addFooter(s, 9);
}

await fs.mkdir(OUT_DIR, { recursive: true });
await fs.mkdir(QA_DIR, { recursive: true });

for (const [index, slide] of p.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await p.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(QA_DIR, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
}

const montage = await p.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(QA_DIR, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const pptx = await PresentationFile.exportPptx(p);
await pptx.save(OUT);

console.log(OUT);
