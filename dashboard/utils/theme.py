"""
Theme: dark, AI-quality polish.

Inspired by Linear, Vercel, and the Anthropic console: generous spacing,
mono-typography for numbers, accent-color for the hero metric, subtle
borders, no gradients/clip-art.
"""

import streamlit as st


PRIMARY      = "#8b7cff"   # softer violet accent
PRIMARY_DIM  = "#5d4ed4"
SUCCESS      = "#34d399"   # softer green for beats-benchmark
WARNING      = "#fbbf24"   # warmer amber
DANGER       = "#f87171"   # softer red
NEUTRAL      = "#9ca3af"
BG           = "#0E1B3A"   # SLIDE navy — matches the deck for projector
SURFACE      = "#16264F"   # slide CARD_FACE — lighter navy for cards
SURFACE_ALT  = "#1F3567"   # mid-navy for hover states
BORDER       = "#3D5187"   # subtle blue-tinted border
TEXT         = "#f5f6fa"   # near-white with cool tint
TEXT_MUTED   = "#c5cad8"   # cool muted gray (brightened for recording)
TEXT_DIM     = "#a8aec0"   # even more muted (brightened for recording)


_CUSTOM_CSS = f"""
<style>
    /* ---------- Global ---------- */
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    section.main > div {{
        max-width: 1280px;
        padding-top: 1.5rem;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT};
        font-weight: 600;
        letter-spacing: -0.02em;
    }}
    h1 {{ font-size: 2.0rem !important; }}
    h2 {{ font-size: 1.5rem !important; }}
    h3 {{ font-size: 1.15rem !important; color: {TEXT_MUTED}; font-weight: 500; }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] > div {{
        padding-top: 2rem;
    }}
    /* Force sidebar text BRIGHT for projector / recording visibility */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: #e8eaf2 !important;
        font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #d8dce8 !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="radio"] label,
    section[data-testid="stSidebar"] [data-baseweb="checkbox"] label {{
        color: #ecedf5 !important;
        font-weight: 500 !important;
    }}

    /* ---------- Metric cards (st.metric) ---------- */
    [data-testid="stMetric"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED};
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT};
        font-size: 1.8rem !important;
        font-weight: 600;
        font-feature-settings: "tnum";
    }}
    [data-testid="stMetricDelta"] {{
        font-feature-settings: "tnum";
    }}

    /* ---------- Tables ---------- */
    [data-testid="stTable"], [data-testid="stDataFrame"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}
    /* Zebra stripes for data frames */
    [data-testid="stDataFrame"] [role="row"]:nth-child(even) {{
        background-color: rgba(255, 255, 255, 0.015);
    }}
    [data-testid="stDataFrame"] [role="row"]:hover {{
        background-color: rgba(139, 124, 255, 0.04);
    }}

    /* ---------- Variant card (custom) — explicit BG everywhere ---------- */
    div.variant-card,
    [data-testid="stExpander"] div.variant-card,
    [data-testid="stExpander"] div div.variant-card {{
        background: {SURFACE} !important;
        background-image: linear-gradient(180deg, {SURFACE} 0%, {SURFACE_ALT} 100%);
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: border-color 0.15s ease;
        color: {TEXT};
    }}
    div.variant-card:hover {{
        border-color: {PRIMARY_DIM};
    }}
    div.variant-card .v-name {{
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: {TEXT} !important;
        margin-bottom: 0.2rem;
    }}
    div.variant-card .v-folder {{
        font-family: 'SF Mono', 'JetBrains Mono', Menlo, monospace;
        font-size: 0.7rem;
        color: {TEXT_DIM} !important;
        margin-bottom: 0.6rem;
    }}
    div.variant-card .v-sub {{
        font-size: 0.83rem;
        color: {TEXT_MUTED} !important;
        line-height: 1.4;
        margin-bottom: 0.8rem;
    }}
    div.variant-card .v-row {{
        font-family: 'SF Mono', 'JetBrains Mono', Menlo, monospace;
        font-size: 0.85rem;
        font-feature-settings: "tnum";
        margin-top: 0.3rem;
        color: {TEXT} !important;
    }}
    div.variant-card .v-rule {{
        font-size: 0.7rem;
        color: {TEXT_DIM} !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 0.6rem;
    }}

    /* Force expander body background dark (was light by default) */
    [data-testid="stExpander"] details {{
        background-color: {BG} !important;
    }}
    [data-testid="stExpander"] details summary {{
        background-color: {SURFACE} !important;
        color: {TEXT} !important;
    }}
    [data-testid="stExpander"] details[open] > div {{
        background-color: transparent !important;
    }}

    /* ---------- Expander ---------- */
    [data-testid="stExpander"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}

    /* ---------- Tabs ---------- */
    [data-baseweb="tab-list"] {{
        background-color: transparent;
        border-bottom: 1px solid {BORDER};
        gap: 0.25rem;
    }}
    [data-baseweb="tab"] {{
        color: {TEXT_MUTED};
        font-weight: 500;
    }}
    [data-baseweb="tab"][aria-selected="true"] {{
        color: {PRIMARY};
        border-bottom-color: {PRIMARY};
    }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        background-color: {PRIMARY};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }}
    .stButton > button:hover {{
        background-color: #6b4def;
    }}

    /* ---------- Selectbox ---------- */
    [data-baseweb="select"] > div {{
        background-color: {SURFACE} !important;
        border-color: {BORDER} !important;
    }}

    /* ---------- Custom hero card ---------- */
    .hero-card {{
        background: linear-gradient(135deg, {SURFACE} 0%, {SURFACE_ALT} 100%);
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
    }}
    .hero-card h2 {{
        margin: 0 0 0.5rem 0;
        font-size: 1.4rem !important;
    }}
    .hero-card .hero-sub {{
        color: {TEXT_MUTED};
        font-size: 0.95rem;
    }}

    /* ---------- Custom badges ---------- */
    .badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        margin-right: 0.3rem;
    }}
    .badge-success {{ background-color: rgba(34, 197, 94, 0.15);  color: {SUCCESS}; }}
    .badge-warning {{ background-color: rgba(245, 158, 11, 0.15); color: {WARNING}; }}
    .badge-danger  {{ background-color: rgba(239, 68, 68, 0.15);  color: {DANGER}; }}
    .badge-neutral {{ background-color: rgba(161, 161, 170, 0.15);color: {NEUTRAL}; }}
    .badge-accent  {{ background-color: rgba(124, 92, 255, 0.15); color: {PRIMARY}; }}

    /* ---------- Mono numbers ---------- */
    .mono-num {{
        font-family: 'SF Mono', 'JetBrains Mono', Menlo, monospace;
        font-feature-settings: "tnum";
    }}

    /* ---------- Footer ---------- */
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
</style>
"""


def apply_theme():
    """Inject custom CSS. Call once per page after st.set_page_config()."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    """Return HTML for a colored badge. Use with st.markdown(..., unsafe_allow_html=True)."""
    return f'<span class="badge badge-{kind}">{text}</span>'
