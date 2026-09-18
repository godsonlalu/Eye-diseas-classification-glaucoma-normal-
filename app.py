from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# ── Load .env so HF_TOKEN / HF_MODEL are available via os.getenv ──────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

ROOT            = Path(__file__).resolve().parent
IMAGE_SIZE_MNV2  = (224, 224)   # MobileNetV2 input
IMAGE_SIZE_XCEP  = (299, 299)   # Xception input
THRESHOLD        = 0.40
MOBILENET_PATH   = ROOT / "saved_models"       / "mobilenetv2_best.h5"
XCEPTION_PATH    = ROOT / "backbone_benchmark" / "xception_best.pt"
HF_MODEL         = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# ── Backbone benchmark (from backbone_benchmark/) ─────────────────────────────
BENCHMARK = [
    {"model": "Xception",    "test_acc": 0.9663, "test_auc": 0.9944, "glaucoma_f1": 0.9651, "normal_f1": 0.9674},
    {"model": "ResNet50",    "test_acc": 0.9607, "test_auc": 0.9929, "glaucoma_f1": 0.9605, "normal_f1": 0.9609},
    {"model": "MobileNetV2", "test_acc": 0.9607, "test_auc": 0.9929, "glaucoma_f1": 0.9500, "normal_f1": 0.9550},
    {"model": "InceptionV3", "test_acc": 0.9438, "test_auc": 0.9879, "glaucoma_f1": 0.9195, "normal_f1": 0.9231},
    {"model": "VGG19",       "test_acc": 0.8989, "test_auc": 0.9714, "glaucoma_f1": 0.8875, "normal_f1": 0.9082},
    {"model": "VGG16",       "test_acc": 0.8202, "test_auc": 0.9543, "glaucoma_f1": 0.7746, "normal_f1": 0.8505},
]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OcuLens | Glaucoma Screening",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Styles ─────────────────────────────────────────────────────────────────────
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --ink:      #e2e8e4;
            --muted:    #8a9e93;
            --paper:    #0f1410;
            --surface:  #161d18;
            --surface2: #1c2620;
            --line:     #263029;
            --leaf:     #4ade80;
            --leaf-dk:  #22c55e;
            --mint:     #1a2e20;
            --coral:    #f87171;
            --coral-bg: #2a1515;
            --amber:    #fbbf24;
            --amber-bg: #2a2010;
        }

        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }

        /* Main background */
        .stApp { background: var(--paper) !important; }

        /* Hide sidebar + collapse toggle completely */
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] { display: none !important; }

        /* Top header bar */
        header[data-testid="stHeader"] {
            background: #0a0f0b !important;
            border-bottom: 1px solid var(--line);
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--paper); }
        ::-webkit-scrollbar-thumb { background: var(--line); border-radius: 3px; }

        /* Typography */
        .eyebrow {
            font-family: 'DM Mono', monospace;
            text-transform: uppercase;
            font-size: .72rem;
            color: var(--leaf);
            letter-spacing: .12em;
        }
        h1 {
            font-size: clamp(2.2rem, 4vw, 4.5rem) !important;
            line-height: .96 !important;
            letter-spacing: -.04em !important;
            margin: .25rem 0 1rem !important;
            color: var(--ink) !important;
        }
        h2, h3 { letter-spacing: -.025em; color: var(--ink); }
        .lead { max-width: 650px; font-size: 1.05rem; line-height: 1.7; color: var(--muted); }

        /* Section labels */
        .panel-title {
            font-size: .73rem;
            text-transform: uppercase;
            letter-spacing: .1em;
            color: var(--muted);
            font-family: 'DM Mono', monospace;
            margin-bottom: .8rem;
        }

        /* Result metrics */
        .metric       { font-size: 2.6rem; font-weight: 800; letter-spacing: -.06em; color: var(--leaf); }
        .metric-coral { font-size: 2.6rem; font-weight: 800; letter-spacing: -.06em; color: var(--coral); }
        .caption      { color: var(--muted); font-size: .82rem; line-height: 1.5; }

        /* Disclaimer / info banners */
        .disclaimer {
            border-left: 3px solid var(--coral);
            background: var(--coral-bg);
            padding: .8rem 1rem;
            color: #fca5a5;
            font-size: .82rem;
            line-height: 1.5;
            margin-bottom: 1rem;
        }
        .info-banner {
            border-left: 3px solid var(--leaf-dk);
            background: var(--mint);
            padding: .8rem 1rem;
            color: #86efac;
            font-size: .82rem;
            line-height: 1.5;
        }

        /* Image preview card */
        .img-card {
            display: inline-flex;
            flex-direction: column;
            align-items: flex-start;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 4px;
            padding: 8px 8px 6px;
            gap: 6px;
            max-width: 356px;
        }
        .img-card img { border-radius: 2px; display: block; }
        .img-meta {
            font-family: 'DM Mono', monospace;
            font-size: .70rem;
            color: var(--muted);
            letter-spacing: .03em;
            padding: 0 2px;
        }

        /* File uploader */
        div[data-testid="stFileUploader"] {
            background: var(--surface);
            border: 1px dashed var(--line);
            padding: .5rem;
        }

        /* Buttons */
        .stButton > button {
            background: var(--leaf-dk); color: #0a0f0b;
            border: 0; border-radius: 2px; font-weight: 700; padding: .7rem 1.1rem;
        }
        .stButton > button:hover { background: var(--leaf); }

        /* Progress bar */
        [data-testid="stProgress"] > div > div { background: var(--leaf-dk) !important; }
        [data-testid="stProgress"]             { background: var(--line) !important; }

        /* Metric widgets */
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            padding: .7rem 1rem;
            border-radius: 2px;
        }
        [data-testid="stMetricValue"] { color: var(--ink) !important; }
        [data-testid="stMetricLabel"] { color: var(--muted) !important; }

        /* Notification boxes */
        [data-testid="stInfo"]    { background: var(--mint);     border-color: var(--leaf-dk); color: var(--ink); }
        [data-testid="stWarning"] { background: var(--amber-bg); border-color: var(--amber);   color: #fde68a; }
        [data-testid="stError"]   { background: var(--coral-bg); border-color: var(--coral);   color: #fca5a5; }

        /* Divider */
        hr { border-color: var(--line) !important; }

        code { font-family: 'DM Mono', monospace; background: var(--surface); color: var(--leaf); }

        /* DataFrame / table */
        [data-testid="stDataFrame"] { background: var(--surface) !important; }
        thead tr th {
            background: var(--surface2) !important;
            color: var(--muted) !important;
            font-family: 'DM Mono', monospace !important;
            font-size: .73rem !important;
            text-transform: uppercase !important;
            letter-spacing: .06em !important;
        }
        tbody tr:nth-child(even) td { background: var(--surface2) !important; }
        tbody tr:hover td { background: var(--mint) !important; }

        /* Tabs */
        [data-baseweb="tab-list"] { background: var(--surface) !important; border-bottom: 1px solid var(--line); }
        [data-baseweb="tab"]      { color: var(--muted) !important; }
        [aria-selected="true"]    { color: var(--leaf) !important; border-bottom: 2px solid var(--leaf-dk) !important; }

        /* Expander */
        [data-testid="stExpander"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_mobilenetv2() -> Any:
    import tf_keras  # Keras 2 shim for TF2/Keras 2.x saved models
    return tf_keras.models.load_model(MOBILENET_PATH, compile=False)


@st.cache_resource(show_spinner=False)
def load_xception() -> Any:
    import torch
    import torch.nn as nn
    import timm

    # Rebuild exact architecture: Sequential([xception_backbone, classifier_head])
    # Classifier indices from saved state dict: 0=Flatten,1=Linear(2048→128),2=ReLU,3=Dropout,4=Linear(128→2)
    backbone   = timm.create_model("xception", pretrained=False, num_classes=0)
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2),
    )
    net = nn.Sequential(backbone, classifier)
    sd  = torch.load(XCEPTION_PATH, map_location="cpu", weights_only=False)["model_state"]
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(image: Image.Image) -> tuple[str, float, float, float]:
    import torch
    import torch.nn.functional as F

    model  = load_xception()
    img    = image.convert("RGB").resize(IMAGE_SIZE_XCEP)
    pixels = np.asarray(img, dtype=np.float32) / 255.0
    # Xception trained with [-1, 1] normalisation (mean=0.5, std=0.5)
    pixels = (pixels - 0.5) / 0.5
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,299,299)
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    # class_names: ['glaucoma', 'normal'] → idx 0 = glaucoma, idx 1 = normal
    glaucoma_prob = float(probs[0])
    normal_prob   = float(probs[1])
    label         = "glaucoma" if glaucoma_prob > normal_prob else "normal"
    confidence    = max(glaucoma_prob, normal_prob)
    return label, glaucoma_prob, normal_prob, confidence


# ── Local rule-based explanation (zero dependencies, always works) ────────────
def _local_explanation(label: str, glaucoma_prob: float, normal_prob: float) -> str:
    if label == "glaucoma":
        return (
            f"The screening model assigned a glaucoma probability of {glaucoma_prob:.1%} "
            f"(normal {normal_prob:.1%}), which exceeds the normal probability and triggered a "
            f"glaucoma flag. Fundus classifiers learn visual patterns associated with optic-disc "
            f"changes such as increased cup-to-disc ratio or retinal nerve fibre layer thinning. "
            f"However, this result is not a diagnosis — only a qualified eye-care professional "
            f"examining your eye can confirm or rule out glaucoma."
        )
    else:
        return (
            f"The screening model assigned a normal probability of {normal_prob:.1%} "
            f"(glaucoma {glaucoma_prob:.1%}), suggesting no strong visual patterns associated "
            f"with glaucomatous optic-disc changes were detected. Fundus classifiers learn "
            f"population-level patterns and can miss early or atypical disease. "
            f"This result is not a clinical clearance — regular eye examinations with an "
            f"eye-care professional remain essential."
        )


# ── Groq explanation (free tier — no credits needed) ─────────────────────────
def _groq_explanation(label: str, glaucoma_prob: float, normal_prob: float) -> str | None:
    import requests as req
    groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not groq_key:
        return None
    prompt = (
        f"In one short paragraph, explain this fundus image screening result: "
        f"label='{label}', glaucoma probability={glaucoma_prob:.1%}, "
        f"normal probability={normal_prob:.1%}. "
        "Mention that the model learned patterns associated with optic-disc or retinal changes, "
        "but only an eye-care professional can make a diagnosis. "
        "Do not claim to see specific image findings."
    )
    try:
        resp = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "You are a cautious ophthalmology education assistant. Never diagnose from a classifier output alone."},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 150,
                "temperature": 0.2,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


# ── Explanation dispatcher: Groq → local fallback ────────────────────────────
def get_explanation(label: str, glaucoma_prob: float, normal_prob: float) -> str:
    groq = _groq_explanation(label, glaucoma_prob, normal_prob)
    return groq if groq else _local_explanation(label, glaucoma_prob, normal_prob)


# ── Benchmark table ───────────────────────────────────────────────────────────
def render_benchmark() -> None:
    import pandas as pd

    df = pd.DataFrame(BENCHMARK).rename(columns={
        "model":       "Model",
        "test_acc":    "Test Accuracy",
        "test_auc":    "Test AUC",
        "glaucoma_f1": "Glaucoma F1",
        "normal_f1":   "Normal F1",
    })
    df = df.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    df["Test Accuracy"] = df["Test Accuracy"].map("{:.1%}".format)
    df["Test AUC"]      = df["Test AUC"].map("{:.4f}".format)
    df["Glaucoma F1"]   = df["Glaucoma F1"].map("{:.3f}".format)
    df["Normal F1"]     = df["Normal F1"].map("{:.3f}".format)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_styles()

    # Header
    st.markdown('<div class="eyebrow">Fundus image intelligence</div>', unsafe_allow_html=True)
    st.title("A clearer view of\nretinal screening.")
    st.markdown(
        '<p class="lead">Upload a fundus image and choose a model to run glaucoma screening, '
        "then receive a brief, cautious AI explanation of the result.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer">⚠ This is a research screening aid, not a medical diagnosis. '
        "Do not use it to make treatment decisions.</div>",
        unsafe_allow_html=True,
    )

    # Tabs
    tab_screen, tab_bench = st.tabs(["🔬  Screening", "📊  Model Benchmark"])

    # ── Screening tab ─────────────────────────────────────────────────────────
    with tab_screen:
        st.write("")
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.markdown('<div class="panel-title">01 / Add an image</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Choose a JPG or PNG fundus image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.markdown('<div class="img-card">', unsafe_allow_html=True)
                st.image(image, width=340)
                st.markdown(
                    f'<div class="img-meta">{uploaded.name}&nbsp;&nbsp;·&nbsp;&nbsp;'
                    f'{image.width} × {image.height} px</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="info-banner">📂 Upload a fundus image to begin screening.</div>',
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown('<div class="panel-title">02 / Screening result</div>', unsafe_allow_html=True)

            if not uploaded:
                st.markdown('<div class="metric">Waiting&hellip;</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="caption">Upload an image on the left to begin.</div>',
                    unsafe_allow_html=True,
                )
            else:
                try:
                    with st.spinner("Loading model…"):
                        load_xception()
                    with st.spinner("Running inference…"):
                        label, glaucoma_prob, normal_prob, confidence = predict(image)

                    # Label — colour-coded by class
                    if label == "glaucoma":
                        st.markdown(
                            f'<div class="metric-coral">{label.title()}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="metric">{label.title()}</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f'<div class="caption">Confidence {confidence:.1%}</div>',
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.progress(confidence, text=f"Confidence  {confidence:.1%}")
                    st.write("")

                    c1, c2 = st.columns(2)
                    c1.metric("Glaucoma probability", f"{glaucoma_prob:.1%}")
                    c2.metric("Normal probability",   f"{normal_prob:.1%}")

                    st.divider()
                    st.markdown(
                        '<div class="panel-title">03 / AI explanation</div>',
                        unsafe_allow_html=True,
                    )
                    with st.spinner("Generating explanation…"):
                        explanation = get_explanation(label, glaucoma_prob, normal_prob)
                    st.write(explanation)

                except Exception as error:
                    st.error(f"Could not run the model: {error}")

    # ── Benchmark tab ─────────────────────────────────────────────────────────
    with tab_bench:
        st.write("")
        st.markdown(
            '<div class="panel-title">Backbone benchmark — glaucoma classification</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="lead" style="font-size:.9rem">Results on the held-out test set (178 images). '
            "All models fine-tuned from ImageNet weights.</p>",
            unsafe_allow_html=True,
        )
        render_benchmark()

        st.write("")
        st.markdown('<div class="panel-title">Accuracy comparison</div>', unsafe_allow_html=True)

        import pandas as pd
        chart_df = (
            pd.DataFrame(BENCHMARK)
            .set_index("model")[["test_acc", "test_auc"]]
            .rename(columns={"test_acc": "Test Accuracy", "test_auc": "Test AUC"})
            .sort_values("Test Accuracy", ascending=True)
        )
        st.bar_chart(chart_df, use_container_width=True)

        st.write("")
        with st.expander("ℹ  About the models"):
            st.markdown(
                """
| Model | Parameters | Notes |
|---|---|---|
| **Xception** | 22 M | Best overall accuracy (96.6 %) |
| **ResNet50** | 25 M | Strong AUC (0.993), fast training |
| **MobileNetV2** | 3.4 M | Deployed in this app — best accuracy/size trade-off |
| **InceptionV3** | 23 M | Good AUC, moderate accuracy |
| **VGG19** | 143 M | Slow training (4 946 s), lower accuracy |
| **VGG16** | 138 M | Lowest accuracy of the benchmark set |

All models trained for up to 50 epochs with early stopping, batch size 32, input 224 × 224 RGB.
MobileNetV2 is selected for deployment due to its low parameter count and competitive test accuracy.
                """
            )


if __name__ == "__main__":
    main()
