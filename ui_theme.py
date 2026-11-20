"""Shared visual styling for the two existing Streamlit pages."""

from __future__ import annotations

import streamlit as st
from html import escape


def apply_app_theme() -> None:
    st.html(
        """
        <style>
        :root {
            --city-green: #075b32;
            --city-green-dark: #064426;
            --city-green-bright: #16864d;
            --city-green-soft: #e8f3eb;
            --city-ink: #18211b;
            --city-muted: #69756d;
            --city-line: #e3e9e4;
            --city-paper: #ffffff;
            --city-page: #f5f8f6;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(ellipse at 82% 2%,rgba(212,232,218,.52),transparent 30%),var(--city-page);
        }
        [data-testid="stHeader"] { background: rgba(245,248,246,.88); }
        [data-testid="stToolbar"] { visibility: visible; height: 2.5rem; background: transparent; }
        [data-testid="stBaseButton-header"] { display: none; }
        [data-testid="stDecoration"] { background: linear-gradient(90deg,#075b32,#28a05e,#b5d8be); }
        footer { visibility: hidden; }
        [data-testid="stSidebar"] {
            background: var(--city-paper);
            border-right: 1px solid var(--city-line);
            box-shadow: 5px 0 24px rgba(22,54,33,.035);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.15rem; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--city-muted); }
        [data-testid="stSidebar"] [role="radiogroup"] { gap: .3rem; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 11px;
            padding: .45rem .65rem;
            transition: background .15s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display: none; }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #f1f6f2; }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: var(--city-green);
            color: white;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: white; }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) svg { fill: white; }
        [data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] { accent-color: var(--city-green); }
        .block-container { max-width: 1220px; padding-top: 1.65rem; padding-bottom: 3rem; }
        h1, h2, h3, h4 { color: var(--city-ink); letter-spacing: -.025em; }
        h1 { font-size: clamp(1.8rem, 3vw, 2.55rem) !important; font-weight: 750 !important; }
        h2 { font-size: 1.4rem !important; font-weight: 720 !important; }
        [data-testid="stMarkdownContainer"] p { color: var(--city-ink); }
        [data-testid="stCaptionContainer"] { color: var(--city-muted); }
        .sidebar-brand {
            display: flex; align-items: center; gap: .7rem; padding: .35rem .25rem 1.2rem;
            color: var(--city-ink); font-size: 1.15rem; font-weight: 800; line-height: 1.1;
        }
        .sidebar-brand-icon {
            display: grid; place-items: center; width: 48px; height: 48px; flex: 0 0 48px;
            border-radius: 14px; color: white; background: linear-gradient(145deg,#0c7740,#034323);
            box-shadow: 0 5px 12px rgba(7,91,50,.2); font-size: 1.5rem;
        }
        .sidebar-brand small { display: block; margin-top: .28rem; color: var(--city-muted); font-size: .7rem; font-weight: 500; letter-spacing: .04em; }
        .page-hero {
            position: relative; display: flex; align-items: center; gap: 1rem; overflow: hidden;
            margin: .2rem 0 1.5rem; padding: 1.35rem 1.45rem;
            border: 0; border-radius: 19px; color: white;
            background: linear-gradient(112deg,#063d24 0%,#075b32 58%,#147342 100%);
            box-shadow: 0 10px 25px rgba(5,65,35,.17);
        }
        .page-hero::after { content:""; position:absolute; right:-42px; top:-90px; width:270px; height:270px; border:1px solid rgba(255,255,255,.13); border-radius:50%; box-shadow:0 0 0 30px rgba(255,255,255,.035),0 0 0 60px rgba(255,255,255,.025); }
        .page-hero-icon {
            position:relative; z-index:1; display: grid; place-items: center; width: 58px; height: 58px; flex: 0 0 58px;
            border:1px solid rgba(255,255,255,.25); border-radius: 16px;
            background: rgba(255,255,255,.13); box-shadow:inset 0 1px rgba(255,255,255,.15); font-size: 1.75rem;
        }
        .page-hero > div:last-child { position:relative; z-index:1; }
        .page-hero-eyebrow { margin: 0 0 .25rem; color: #c5e7cf; font-size: .69rem; font-weight: 850; letter-spacing: .15em; }
        .page-hero h1 { margin: 0; color: white; line-height: 1.1; text-shadow:0 1px 1px rgba(0,0,0,.12); }
        .page-hero p { margin: .4rem 0 0; color: rgba(255,255,255,.84); }
        .landing-hero {
            position: relative; display: flex; align-items: flex-end; min-height: 250px;
            margin: .2rem 0 1.2rem; overflow: hidden; border-radius: 20px;
            background: var(--city-green-dark); color: white;
            box-shadow: 0 8px 24px rgba(20,55,33,.12);
        }
        .landing-hero > img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
        .landing-hero::after { content: ""; position: absolute; inset: 0; background: linear-gradient(90deg,rgba(4,43,25,.86),rgba(4,43,25,.3)); }
        .landing-hero-copy { position: relative; z-index: 1; max-width: 720px; padding: 1.5rem 1.7rem; color: white; }
        .landing-hero-copy .eyebrow { color: #cde7d4; font-size: .72rem; font-weight: 800; letter-spacing: .14em; }
        .landing-hero-copy h1 { margin: .35rem 0; color: white; font-size: clamp(1.7rem,3.5vw,2.6rem) !important; line-height: 1.1; }
        .landing-hero-copy p { margin: 0; color: rgba(255,255,255,.92); font-size: 1.05rem; }
        .landing-hero-empty { min-height: 215px; align-items: center; }
        .landing-hero-empty::after { background: radial-gradient(circle at 85% 15%,rgba(137,190,151,.35),transparent 32%),linear-gradient(120deg,#064426,#0b6740); }
        .comparison-heading { margin: 0 0 .25rem; font-size: 1.18rem; font-weight: 750; }
        .comparison-label { display: inline-block; margin-bottom: .45rem; padding: .18rem .65rem; border-radius: 999px; background: #e8f3eb; color: #075b32; font-size: .72rem; font-weight: 750; }
        .comparison-arrow { display: grid; place-items: center; width: 42px; height: 42px; margin: auto; border-radius: 50%; background: white; color: var(--city-green); box-shadow: 0 2px 10px rgba(20,55,33,.15); font-size: 1.5rem; font-weight: 750; }
        .comparison-card-note { color: var(--city-muted); text-align: center; font-size: .83rem; }
        .project-description-strip { display: flex; align-items: center; gap: .75rem; margin: .7rem 0 1.2rem; padding: .8rem 1rem; border-radius: 12px; background: var(--city-green-soft); color: var(--city-green-dark); }
        .project-description-strip .leaf { font-size: 1.5rem; }
        .section-kicker { color: var(--city-green); font-size: .72rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
        [data-testid="stFileUploaderDropzone"] {
            border: 1.5px dashed #83ad91; border-radius: 16px; background: rgba(255,255,255,.94);
            box-shadow: 0 4px 15px rgba(20,55,33,.04);
        }
        [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--city-green); background: #fbfdfb; }
        [data-testid="stImage"] img { border-radius: 12px; }
        [data-testid="stCustomComponentV1"] { border-radius: 15px; }
        [data-testid="stMetric"] {
            padding: .85rem .9rem; border: 1px solid #e0ebe3; border-radius: 14px;
            background: var(--city-green-soft);
        }
        [data-testid="stMetricLabel"] { color: var(--city-green-dark); }
        [data-testid="stMetricValue"] { color: var(--city-ink); }
        .stButton > button, [data-testid="stFormSubmitButton"] > button {
            min-height: 2.7rem; border-radius: 11px; border: 1px solid var(--city-green);
            background: linear-gradient(115deg,#075b32,#0d7540); color: white; font-weight: 700;
            box-shadow: 0 3px 8px rgba(7,91,50,.12);
            transition: background .15s ease, transform .15s ease, box-shadow .15s ease;
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
            border-color: var(--city-green-dark); background: linear-gradient(115deg,#064426,#075b32); color: white;
            box-shadow: 0 6px 14px rgba(7,91,50,.2); transform: translateY(-1px);
        }
        .stButton > button p, [data-testid="stFormSubmitButton"] > button p { color: white !important; }
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            border-radius: 11px;
        }
        [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus { border-color:#419264; box-shadow:0 0 0 2px rgba(22,134,77,.13); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--city-line); border-radius: 17px; background: white;
            box-shadow: 0 5px 18px rgba(20,55,33,.045);
        }
        .editor-step-heading { display:flex; align-items:center; gap:.8rem; margin:0 0 .85rem; }
        .editor-step-number { display:grid; place-items:center; width:42px; height:42px; flex:0 0 42px; border-radius:13px; background:linear-gradient(145deg,#e6f3e9,#d3ead9); color:var(--city-green); font-size:.82rem; font-weight:850; box-shadow:inset 0 0 0 1px rgba(7,91,50,.08); }
        .editor-step-heading h2 { margin:0; font-size:1.25rem!important; line-height:1.15; }
        .editor-step-heading p { margin:.23rem 0 0; color:var(--city-muted); font-size:.88rem; }
        .st-key-upload_panel, .st-key-drawing_panel, .st-key-comparison_panel,
        .st-key-results_panel, .st-key-proposal_panel { margin:0 0 1.25rem; }
        .st-key-image_source [role="radiogroup"] { gap:.55rem; }
        .st-key-image_source [role="radiogroup"] label { padding:.48rem .8rem; border:1px solid var(--city-line); border-radius:11px; background:white; }
        .st-key-image_source [role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display:none; }
        .st-key-image_source [role="radiogroup"] label:has(input:checked) { border-color:#93b9a0; background:var(--city-green-soft); color:var(--city-green-dark); font-weight:750; }
        .st-key-image_source [role="radiogroup"] input[type="radio"] { accent-color:var(--city-green); }
        .st-key-drawing_panel [data-testid="stCustomComponentV1"] { overflow:hidden; border:1px solid #e4ebe6; border-radius:14px; box-shadow:0 3px 12px rgba(20,55,33,.05); }
        .st-key-comparison_panel [data-testid="stImage"] img { border:1px solid #e7ece8; }
        .editor-result-caption { margin:.15rem 0 .8rem; color:var(--city-muted); font-size:.9rem; }
        .city-project-card {
            display: block; height: 100%; overflow:hidden; margin: 0 0 8px;
            border: 1px solid #e0e9e2; border-radius: 18px; color: inherit;
            background: white; text-decoration: none !important;
            box-shadow: 0 6px 20px rgba(20,55,33,.065);
            transition: border-color .18s, box-shadow .18s, transform .18s;
        }
        .city-project-card:hover {
            border-color: #75a786; box-shadow: 0 13px 28px rgba(20,55,33,.14);
            transform: translateY(-3px);
        }
        .project-card-visual { position:relative; height:245px; overflow:hidden; background:#e9f0eb; }
        .project-card-visual::after { content:""; position:absolute; inset:35% 0 0; background:linear-gradient(transparent,rgba(3,35,20,.82)); pointer-events:none; }
        .project-card-visual img {
            display: block; width: 100%; height: 100%; object-fit: cover;
            transition: transform .35s ease;
        }
        .city-project-card:hover .project-card-visual img { transform:scale(1.035); }
        .project-card-overlay { position:absolute; z-index:1; right:1rem; bottom:.9rem; left:1rem; color:white; }
        .city-project-card .project-location { display:inline-block; margin:0 0 .45rem; padding:.25rem .6rem; border:1px solid rgba(255,255,255,.23); border-radius:999px; background:rgba(5,50,28,.7); color:white; font-size:.73rem; font-weight:700; backdrop-filter:blur(5px); }
        .city-project-card h3 { margin:0; color:white; font-size:1.17rem; line-height:1.2; text-shadow:0 1px 3px rgba(0,0,0,.25); }
        .project-card-open { position:absolute; z-index:2; top:.8rem; right:.8rem; display:grid; place-items:center; width:36px; height:36px; border:1px solid rgba(255,255,255,.7); border-radius:50%; background:rgba(255,255,255,.93); color:var(--city-green); font-size:1.15rem; box-shadow:0 3px 9px rgba(0,0,0,.12); }
        .project-card-copy { padding:.9rem 1rem .95rem; }
        .city-project-card p { min-height:2.75em; margin:0 0 .7rem; color:var(--city-muted); line-height:1.42; }
        .project-card-link { display:flex; align-items:center; justify-content:space-between; padding-top:.65rem; border-top:1px solid #edf1ee; color:var(--city-green); font-size:.83rem; font-weight:800; }
        .project-card-link b { display:grid; place-items:center; width:27px; height:27px; border-radius:50%; background:var(--city-green-soft); font-size:1rem; }
        [data-testid="stImage"] figcaption { color: var(--city-muted); }
        @media (max-width: 700px) {
            .block-container { padding: 1rem 1rem 2rem; }
            .page-hero { align-items: flex-start; padding: .95rem; gap: .75rem; }
            .page-hero-icon { width: 48px; height: 48px; flex-basis: 48px; font-size: 1.4rem; }
            .page-hero h1 { font-size:1.6rem!important; }
            .page-hero p { font-size:.88rem; line-height:1.45; }
            .editor-step-heading { align-items:flex-start; }
            .editor-step-number { width:36px; height:36px; flex-basis:36px; border-radius:11px; }
            .editor-step-heading h2 { font-size:1.1rem!important; }
            .project-card-visual { height:215px; }
            .city-project-card p { min-height:0; }
            .landing-hero { min-height: 200px; }
            .landing-hero-copy { padding: 1rem; }
        }
        </style>
        """
    )


def render_page_header(eyebrow: str, title: str, description: str, icon: str) -> None:
    st.html(
        f"""
        <section class="page-hero">
          <div class="page-hero-icon" aria-hidden="true">{escape(icon)}</div>
          <div>
            <div class="page-hero-eyebrow">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
          </div>
        </section>
        """
    )


def render_editor_step(number: str, title: str, description: str) -> None:
    st.html(
        f"""
        <div class="editor-step-heading">
          <span class="editor-step-number">{escape(number)}</span>
          <div><h2>{escape(title)}</h2><p>{escape(description)}</p></div>
        </div>
        """
    )
