"""
OIRseg Theme — Dark Gold Glassmorphism
Inspired by aged stone / weathered metal with champagne gold accents.
"""

import streamlit as st
import streamlit.components.v1 as components


# ── Palette ───────────────────────────────────────────────────────────────────
GOLD = "#C5A55A"
GOLD_BRIGHT = "#D4B96A"
GOLD_LIGHT = "#E8D5A3"
GOLD_DARK = "#8B7635"
GOLD_MUTED = "#9A8B6F"
BG_DEEP = "#0c0a08"
TEXT_BODY = "#BFB39A"


def inject_theme():
    """Call once after st.set_page_config to apply the full custom theme."""
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:'
        'ital,wght@0,400..900;1,400..900&family=Inter:wght@300;400;500;600'
        '&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    components.html(_JS, height=0)


# ── Reusable UI helpers ───────────────────────────────────────────────────────

def page_header(title: str, subtitle: str, caption: str):
    st.markdown(f"""
    <div style="padding:0.5rem 0 0.8rem 0;position:relative;">
        <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.35rem;">
            <span style="
                display:inline-flex;align-items:center;justify-content:center;
                width:30px;height:30px;border-radius:50%;
                border:1px solid rgba(197,165,90,0.45);
                color:{GOLD};font-size:0.85rem;
            ">✦</span>
            <h1 style="
                margin:0;padding:0;
                font-family:'Playfair Display',serif;
                font-size:2.6rem;font-weight:700;font-style:italic;
                background:linear-gradient(135deg,{GOLD_LIGHT},{GOLD},{GOLD_DARK});
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;
            ">{title}</h1>
        </div>
        <span id="oirseg-mouse-toggle"
            data-running="0"
            style="position:absolute;top:calc(0.35rem + 15px);left:50%;transform:translateX(-50%);font-size:2.4rem;line-height:1;cursor:pointer;display:inline-flex;align-items:center;">🐭</span>
        <p style="
            color:{GOLD_MUTED};font-family:'Inter',sans-serif;
            font-size:0.85rem;letter-spacing:0.12em;text-transform:uppercase;
            margin:0 0 1rem 2.9rem;font-weight:400;
        ">{subtitle}</p>
        <div style="height:1px;background:linear-gradient(90deg,{GOLD},rgba(197,165,90,0.12),transparent);margin-bottom:0.55rem;"></div>
        <p style="
            color:#5A5345;font-family:'Inter',sans-serif;
            font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;margin:0;
        ">{caption}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f"""
    <div style="margin:2rem 0 1rem 0;">
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.45rem;">
            <span style="color:{GOLD};font-size:0.8rem;">✦</span>
            <h3 style="
                margin:0;padding:0;
                font-family:'Playfair Display',serif;
                color:{GOLD_LIGHT};font-weight:600;font-size:1.35rem;
            ">{title}</h3>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,rgba(197,165,90,0.5),rgba(197,165,90,0.08),transparent);max-width:400px;"></div>
    </div>
    """, unsafe_allow_html=True)


def gold_divider():
    st.markdown(
        f'<div style="height:1px;margin:1.5rem 0;background:linear-gradient(90deg,transparent,{GOLD_DARK},{GOLD},{GOLD_DARK},transparent);"></div>',
        unsafe_allow_html=True,
    )


def sidebar_legend(items: list):
    """items: [(label, description, (r_float, g_float, b_float)), ...]"""
    for label, desc, rgb in items:
        r, g, b = (int(c * 255) for c in rgb)
        st.sidebar.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin:7px 0;">
            <div style="
                width:13px;height:13px;border-radius:50%;flex-shrink:0;
                background:rgb({r},{g},{b});
                border:1px solid rgba(197,165,90,0.3);
                box-shadow:0 0 8px rgba({r},{g},{b},0.35);
            "></div>
            <span style="color:{GOLD_MUTED};font-size:0.82rem;font-family:'Inter',sans-serif;">
                <strong style="color:{GOLD_LIGHT};">{label}</strong> — {desc}
            </span>
        </div>
        """, unsafe_allow_html=True)


def sidebar_status(text: str):
    st.sidebar.markdown(f"""
    <div style="
        background:rgba(197,165,90,0.08);
        border:1px solid rgba(197,165,90,0.25);
        border-radius:8px;padding:0.55rem 0.8rem;margin-bottom:0.5rem;
        color:{GOLD_LIGHT};font-family:'Inter',sans-serif;font-size:0.82rem;
        display:flex;align-items:center;gap:8px;
    ">
        <span style="color:{GOLD};font-size:0.7rem;">✦</span> {text}
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CSS
# ═════════════════════════════════════════════════════════════════════════════

_CSS = """<style>
/* ── Palette variables ── */
:root{
    --bg-deep:#0c0a08;--bg-primary:#141210;--bg-secondary:#1c1917;
    --bg-card:rgba(22,19,16,0.75);--bg-card-hover:rgba(30,26,22,0.88);
    --gold:#C5A55A;--gold-bright:#D4B96A;--gold-light:#E8D5A3;
    --gold-dark:#8B7635;--gold-muted:#9A8B6F;
    --border-gold:rgba(197,165,90,0.22);--border-gold-s:rgba(197,165,90,0.48);
    --text-primary:#E8D5A3;--text-body:#BFB39A;--text-sec:#9A8B6F;
    --glass-bg:rgba(20,18,14,0.62);--glass-shadow:rgba(0,0,0,0.4);
}

/* ── Background with grunge texture ── */
.stApp{
    background-color:var(--bg-deep)!important;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='600'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.7' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='600' height='600' filter='url(%23n)' opacity='0.055'/%3E%3C/svg%3E"),
        radial-gradient(ellipse at center,transparent 30%,rgba(0,0,0,0.55) 100%),
        radial-gradient(ellipse at 15% 25%,rgba(75,50,22,0.09) 0%,transparent 50%),
        radial-gradient(ellipse at 70% 65%,rgba(55,40,18,0.07) 0%,transparent 45%),
        radial-gradient(ellipse at 45% 80%,rgba(65,45,20,0.08) 0%,transparent 55%),
        radial-gradient(ellipse at 88% 12%,rgba(50,35,15,0.05) 0%,transparent 40%),
        linear-gradient(170deg,#12100e 0%,#0c0a08 50%,#100e0c 100%);
    background-attachment:fixed;
}

/* ── Hide Streamlit chrome ── */
[data-testid="stHeader"]{background:transparent!important;border:none!important}
#MainMenu,footer{visibility:hidden}

/* ── Typography ── */
.stApp{font-family:'Inter',sans-serif;color:var(--text-body)}
p,span,label,div,li,td,th{font-family:'Inter',sans-serif}
h1,h2,h3,.stApp h1,.stApp h2,.stApp h3{
    color:var(--gold-light)!important;font-weight:600!important;letter-spacing:.02em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,rgba(18,16,13,0.96),rgba(10,9,7,0.98))!important;
    border-right:1px solid var(--border-gold)!important;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] p{color:var(--text-sec)!important}
[data-testid="stSidebar"] hr{border-color:var(--border-gold)!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{
    font-family:'Playfair Display',serif!important;color:var(--gold)!important;
}

/* ── Sidebar collapse/expand button ── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"]{
    display:flex!important;visibility:visible!important;opacity:1!important;
}
button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button{
    background:rgba(197,165,90,0.14)!important;
    border:1px solid var(--border-gold-s)!important;
    border-radius:10px!important;
    color:var(--gold)!important;
    width:36px!important;height:36px!important;
    min-width:36px!important;min-height:36px!important;
    padding:0!important;
    position:relative!important;
    transition:all .25s ease!important;
}
button[data-testid="baseButton-headerNoPadding"]:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover{
    background:rgba(197,165,90,0.28)!important;
    border-color:var(--gold)!important;
    box-shadow:0 0 18px rgba(197,165,90,0.25)!important;
}
/* Colour the existing Streamlit sidebar toggle SVG — no DOM surgery needed */
button[data-testid="baseButton-headerNoPadding"] svg,
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="collapsedControl"] button svg{
    color:var(--gold)!important;
    stroke:var(--gold)!important;
    fill:none!important;
}

/* ── Glassmorphism metric cards ── */
[data-testid="stMetric"]{
    background:var(--glass-bg)!important;
    border:1px solid var(--border-gold)!important;
    border-radius:12px!important;padding:1rem 1.2rem!important;
    backdrop-filter:blur(16px)!important;-webkit-backdrop-filter:blur(16px)!important;
    box-shadow:0 8px 32px var(--glass-shadow),inset 0 1px 0 rgba(197,165,90,0.07)!important;
    transition:all .3s ease!important;
}
[data-testid="stMetric"]:hover{
    background:var(--bg-card-hover)!important;
    border-color:var(--border-gold-s)!important;
    box-shadow:0 12px 40px var(--glass-shadow),0 0 20px rgba(197,165,90,0.05)!important;
    transform:translateY(-2px);
}
[data-testid="stMetricLabel"]{
    color:var(--gold)!important;font-family:'Playfair Display',serif!important;
    font-size:.95rem!important;font-weight:600!important;letter-spacing:.06em;
}
[data-testid="stMetricValue"]{
    color:var(--gold-light)!important;font-family:'Playfair Display',serif!important;font-weight:700!important;
}
[data-testid="stMetricDelta"]{color:var(--gold-muted)!important}

/* ── Buttons ── */
.stButton>button{
    background:linear-gradient(135deg,rgba(197,165,90,0.14),rgba(197,165,90,0.04))!important;
    border:1px solid var(--border-gold-s)!important;
    color:var(--gold-light)!important;font-family:'Playfair Display',serif!important;
    font-weight:600!important;font-size:1rem!important;letter-spacing:.05em;
    border-radius:10px!important;padding:.6rem 1.5rem!important;
    backdrop-filter:blur(12px)!important;
    transition:all .3s cubic-bezier(.4,0,.2,1)!important;
    box-shadow:0 4px 16px rgba(0,0,0,.3),inset 0 1px 0 rgba(197,165,90,.08)!important;
}
.stButton>button:hover{
    background:linear-gradient(135deg,rgba(197,165,90,.28),rgba(197,165,90,.12))!important;
    border-color:var(--gold)!important;
    box-shadow:0 8px 24px rgba(0,0,0,.4),0 0 25px rgba(197,165,90,.08)!important;
    transform:translateY(-1px);
}

/* Primary button — shimmer */
@keyframes gold-shimmer{
    0%{background-position:-200% center}
    100%{background-position:200% center}
}
button[data-testid="stBaseButton-primary"]{
    background:linear-gradient(90deg,#8B7635 0%,#C5A55A 25%,#D4B96A 50%,#C5A55A 75%,#8B7635 100%)!important;
    background-size:200% 100%!important;
    animation:gold-shimmer 4s ease-in-out infinite!important;
    color:var(--bg-deep)!important;border:none!important;font-weight:700!important;
    box-shadow:0 4px 20px rgba(197,165,90,.3),inset 0 1px 0 rgba(255,255,255,.1)!important;
}
button[data-testid="stBaseButton-primary"]:hover{
    box-shadow:0 8px 30px rgba(197,165,90,.45),inset 0 1px 0 rgba(255,255,255,.15)!important;
}

/* Download buttons */
.stDownloadButton>button{
    background:linear-gradient(135deg,rgba(197,165,90,.1),rgba(197,165,90,.03))!important;
    border:1px solid var(--border-gold)!important;color:var(--gold-light)!important;
    font-family:'Inter',sans-serif!important;font-weight:500!important;
    border-radius:8px!important;backdrop-filter:blur(12px)!important;transition:all .3s ease!important;
}
.stDownloadButton>button:hover{
    background:linear-gradient(135deg,rgba(197,165,90,.22),rgba(197,165,90,.08))!important;
    border-color:var(--gold)!important;
}

/* ── File Uploader — compact, expands on focus/drag ── */
[data-testid="stFileUploader"]{
    background:var(--glass-bg)!important;
    border:1px solid var(--border-gold)!important;
    border-radius:10px!important;padding:0.6rem 1rem!important;
    backdrop-filter:blur(16px)!important;
    transition:all .35s cubic-bezier(.4,0,.2,1)!important;
    max-width:520px;
}
[data-testid="stFileUploader"]:hover,
[data-testid="stFileUploader"]:focus-within{
    max-width:80%!important;
    padding:1.2rem 1.4rem!important;
    border-color:var(--gold)!important;
    background:var(--bg-card-hover)!important;
    box-shadow:0 0 28px rgba(197,165,90,0.12)!important;
}
[data-testid="stFileUploaderDropzone"]{
    background:transparent!important;border:none!important;
    padding:0.4rem 0!important;min-height:0!important;
}
[data-testid="stFileUploaderDropzone"] div[data-testid="stMarkdownContainer"]{
    font-size:0.78rem!important;
}
[data-testid="stFileUploaderDropzone"] button{
    display:none!important;
}
[data-testid="stFileUploaderDropzone"] small{
    font-size:0.7rem!important;
}
[data-testid="stFileUploader"] section{
    padding:0!important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small{color:var(--text-sec)!important}
[data-testid="stFileUploader"] label p{
    font-size:0.82rem!important;margin-bottom:0.2rem!important;
}

/* ── Slider ── */
[data-testid="stSlider"] label{color:var(--text-body)!important}
.stSlider [role="slider"]{
    background-color:var(--gold)!important;border:2px solid var(--gold-light)!important;
    box-shadow:0 0 10px rgba(197,165,90,.3)!important;
}
.stSlider>div>div>div>div{background-color:var(--gold)!important}
[data-testid="stSlider"] [data-testid="stThumbValue"]{color:var(--gold-light)!important}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label span{color:var(--text-body)!important}

/* ── Alerts ── */
.stAlert,[data-testid="stAlert"]{
    background:var(--glass-bg)!important;
    border:1px solid var(--border-gold)!important;
    border-radius:10px!important;backdrop-filter:blur(12px)!important;
}
[data-testid="stAlert"] p{color:var(--text-body)!important}

/* ── Images ── */
[data-testid="stImage"]{
    border-radius:12px!important;overflow:hidden!important;
    border:1px solid var(--border-gold)!important;
    box-shadow:0 8px 32px rgba(0,0,0,.4)!important;transition:all .3s ease!important;
}
[data-testid="stImage"]:hover{
    border-color:var(--border-gold-s)!important;
    box-shadow:0 12px 40px rgba(0,0,0,.5),0 0 20px rgba(197,165,90,.06)!important;
}
[data-testid="stImage"] img{border-radius:11px!important}
[data-testid="stImage"] [data-testid="caption"]{
    color:var(--gold-muted)!important;font-family:'Inter',sans-serif!important;
    font-size:.82rem!important;letter-spacing:.03em;
}

/* ── Spinner ── */
.stSpinner>div{border-top-color:var(--gold)!important}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:7px;height:7px}
::-webkit-scrollbar-track{background:var(--bg-deep)}
::-webkit-scrollbar-thumb{background:var(--gold-dark);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--gold)}

/* ── Layout ── */
[data-testid="stHorizontalBlock"]{gap:1rem}
.block-container{padding-top:2rem!important}

/* ── Toast ── */
.stToast{
    background:var(--bg-card)!important;border:1px solid var(--border-gold)!important;
    color:var(--text-body)!important;
}
[data-testid="stTooltipIcon"]{color:var(--gold-muted)!important}

/* ── Inputs ── */
.stSelectbox>div>div,.stTextInput>div>div>input{
    background:var(--glass-bg)!important;border:1px solid var(--border-gold)!important;
    color:var(--text-primary)!important;border-radius:8px!important;
}
</style>"""


# ═════════════════════════════════════════════════════════════════════════════
#  JavaScript — mouse-tracking golden light
# ═════════════════════════════════════════════════════════════════════════════

_JS = """<script>
(function(){
    var d=window.parent.document;

    // ── Mouse toggle: static emoji ↔ mouse_preview.html SVG animation ──────────
    var MOUSE_SVG = '<svg width="52" height="52" viewBox="48 16 130 128" xmlns="http://www.w3.org/2000/svg" style="display:block;vertical-align:middle;">'
      +'<defs>'
      +'<radialGradient id="mt_wheelGlow" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#C5A55A" stop-opacity="0.18"/><stop offset="100%" stop-color="#C5A55A" stop-opacity="0"/></radialGradient>'
      +'<linearGradient id="mt_goldGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#E8D5A3"/><stop offset="100%" stop-color="#8B7635"/></linearGradient>'
      +'<radialGradient id="mt_bodyGrad" cx="50%" cy="30%" r="60%"><stop offset="0%" stop-color="#CFC4AA"/><stop offset="100%" stop-color="#8A7B62"/></radialGradient>'
      +'<radialGradient id="mt_bellyGrad" cx="50%" cy="60%" r="55%"><stop offset="0%" stop-color="#E0D5BC"/><stop offset="100%" stop-color="#B0A282"/></radialGradient>'
      +'</defs>'
      +'<circle cx="110" cy="78" r="68" fill="url(#mt_wheelGlow)"/>'
      +'<g><animateTransform attributeName="transform" type="rotate" from="0 110 78" to="360 110 78" dur="1.3s" repeatCount="indefinite"/>'
      +'<circle cx="110" cy="78" r="60" fill="none" stroke="#C5A55A" stroke-width="4.5" opacity="0.8"/>'
      +'<circle cx="110" cy="78" r="60" fill="none" stroke="#8B7635" stroke-width="2" stroke-dasharray="5 8" opacity="0.3"/>'
      +'<circle cx="110" cy="78" r="52" fill="none" stroke="#8B7635" stroke-width="1.2" opacity="0.3"/>'
      +'<circle cx="110" cy="78" r="8" fill="#C5A55A" opacity="0.9"/>'
      +'<circle cx="110" cy="78" r="4" fill="#1a1612"/>'
      +'<g stroke="url(#mt_goldGrad)" stroke-width="2.2" opacity="0.6" stroke-linecap="round">'
      +'<line x1="110" y1="70" x2="110" y2="26"/><line x1="110" y1="86" x2="110" y2="130"/>'
      +'<line x1="102" y1="78" x2="58" y2="78"/><line x1="118" y1="78" x2="162" y2="78"/>'
      +'<line x1="104" y1="72" x2="74" y2="42"/><line x1="116" y1="84" x2="146" y2="114"/>'
      +'<line x1="116" y1="72" x2="146" y2="42"/><line x1="104" y1="84" x2="74" y2="114"/>'
      +'</g></g>'
      +'<g><animateTransform attributeName="transform" type="translate" values="0,0; 0,-1.5; 0,0; 0,-1.5; 0,0" dur="0.5s" repeatCount="indefinite"/>'
      +'<g transform="translate(117,117) scale(1.2)">'
      +'<path fill="none" stroke="#9A8B6F" stroke-width="2" stroke-linecap="round"><animate attributeName="d" values="M -15 2 C -25 5 -33 -2 -31 -11 Q -29 -17 -24 -15;M -15 2 C -23 -1 -30 -10 -27 -17 Q -24 -22 -19 -19;M -15 2 C -25 5 -33 -2 -31 -11 Q -29 -17 -24 -15" dur="0.5s" repeatCount="indefinite"/></path>'
      +'<g opacity="0.65"><line stroke="#7A6B52" stroke-width="2.8" stroke-linecap="round" x1="-7" y1="8"><animate attributeName="x2" values="-13;-4;-13" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y2" values="11;9;11" dur="0.5s" repeatCount="indefinite"/></line><line stroke="#7A6B52" stroke-width="2.2" stroke-linecap="round"><animate attributeName="x1" values="-13;-4;-13" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y1" values="8;6;8" dur="0.5s" repeatCount="indefinite"/><animate attributeName="x2" values="-16;-5;-16" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y2" values="13;10;13" dur="0.5s" repeatCount="indefinite"/></line><ellipse rx="3.2" ry="1.6" fill="#7A6B52"><animate attributeName="cx" values="-16;-5;-16" dur="0.5s" repeatCount="indefinite"/><animate attributeName="cy" values="13;10;13" dur="0.5s" repeatCount="indefinite"/></ellipse></g>'
      +'<g opacity="0.65"><line stroke="#7A6B52" stroke-width="2.5" stroke-linecap="round" x1="6" y1="8"><animate attributeName="x2" values="1;10;1" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y2" values="11;9;11" dur="0.5s" repeatCount="indefinite"/></line><line stroke="#7A6B52" stroke-width="2" stroke-linecap="round"><animate attributeName="x1" values="1;10;1" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y1" values="8;6;8" dur="0.5s" repeatCount="indefinite"/><animate attributeName="x2" values="-1;13;-1" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y2" values="13;10;13" dur="0.5s" repeatCount="indefinite"/></line><ellipse rx="2.8" ry="1.5" fill="#7A6B52"><animate attributeName="cx" values="-1;13;-1" dur="0.5s" repeatCount="indefinite"/><animate attributeName="cy" values="13;10;13" dur="0.5s" repeatCount="indefinite"/></ellipse></g>'
      +'<ellipse cx="-2" cy="-1" rx="18" ry="11" fill="url(#mt_bodyGrad)"/>'
      +'<ellipse cx="-1" cy="7" rx="13" ry="4" fill="url(#mt_bellyGrad)" opacity="0.55"/>'
      +'<g><line stroke="#9A8B70" stroke-width="2.8" stroke-linecap="round" x1="-7" y1="8"><animate attributeName="x2" values="-4;-13;-4" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y2" values="9;11;9" dur="0.5s" repeatCount="indefinite"/></line><line stroke="#9A8B70" stroke-width="2.2" stroke-linecap="round"><animate attributeName="x1" values="-4;-13;-4" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y1" values="6;8;6" dur="0.5s" repeatCount="indefinite"/><animate attributeName="x2" values="-5;-16;-5" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y2" values="10;13;10" dur="0.5s" repeatCount="indefinite"/></line><ellipse rx="3.2" ry="1.6" fill="#BFB39A"><animate attributeName="cx" values="-5;-16;-5" dur="0.5s" repeatCount="indefinite"/><animate attributeName="cy" values="10;13;10" dur="0.5s" repeatCount="indefinite"/></ellipse></g>'
      +'<g><line stroke="#9A8B70" stroke-width="2.5" stroke-linecap="round" x1="6" y1="8"><animate attributeName="x2" values="10;1;10" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y2" values="9;11;9" dur="0.5s" repeatCount="indefinite"/></line><line stroke="#9A8B70" stroke-width="2" stroke-linecap="round"><animate attributeName="x1" values="10;1;10" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y1" values="6;8;6" dur="0.5s" repeatCount="indefinite"/><animate attributeName="x2" values="13;-1;13" dur="0.5s" repeatCount="indefinite"/><animate attributeName="y2" values="10;13;10" dur="0.5s" repeatCount="indefinite"/></line><ellipse rx="2.8" ry="1.5" fill="#BFB39A"><animate attributeName="cx" values="13;-1;13" dur="0.5s" repeatCount="indefinite"/><animate attributeName="cy" values="10;13;10" dur="0.5s" repeatCount="indefinite"/></ellipse></g>'
      +'<ellipse cx="12" cy="-5" rx="5.5" ry="4" fill="url(#mt_bodyGrad)"/>'
      +'<circle cx="18" cy="-8" r="8.5" fill="url(#mt_bodyGrad)"/>'
      +'<ellipse cx="14" cy="-16" rx="4.5" ry="5.5" fill="url(#mt_bodyGrad)"/>'
      +'<ellipse cx="14" cy="-16" rx="2.8" ry="3.8" fill="#C5A55A" opacity="0.25"/>'
      +'<circle cx="22" cy="-10" r="2.5" fill="#1a1612"/>'
      +'<circle cx="23" cy="-11" r="0.9" fill="#E8D5A3" opacity="0.9"/>'
      +'<ellipse cx="26" cy="-7" rx="1.8" ry="1.3" fill="#C5A55A" opacity="0.6"/>'
      +'<g stroke="#9A8B6F" stroke-width="0.75" opacity="0.6" stroke-linecap="round"><line x1="26" y1="-8" x2="35" y2="-10"/><line x1="26" y1="-7" x2="35" y2="-7"/><line x1="26" y1="-6" x2="35" y2="-4"/></g>'
      +'</g></g>'
      +'</svg>';

    function bindMouseToggle(){
        var el=d.getElementById('oirseg-mouse-toggle');
        if(!el||el.getAttribute('data-bound'))return;
        el.setAttribute('data-bound','1');
        el.addEventListener('click',function(){
            var running=el.getAttribute('data-running')==='1';
            if(running){
                el.setAttribute('data-running','0');
                el.innerHTML='🐭';
                el.style.fontSize='2rem';
            }else{
                el.setAttribute('data-running','1');
                el.innerHTML=MOUSE_SVG;
                el.style.fontSize='0';
            }
        });
    }
    setInterval(bindMouseToggle, 300);

    // ── Mouse-tracking golden light ──────────────────────────────────────────
    if(!d.getElementById('oirseg-light')){
        var o=d.createElement('div');
        o.id='oirseg-light';
        o.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;'
            +'pointer-events:none;z-index:9999;transition:background .18s ease-out;';
        d.body.appendChild(o);
        d.addEventListener('mousemove',function(e){
            o.style.background='radial-gradient(280px circle at '
                +e.clientX+'px '+e.clientY+'px,rgba(197,165,90,0.13),rgba(197,165,90,0.04) 45%,transparent 70%)';
        });
        d.addEventListener('mouseleave',function(){o.style.background='transparent';});
    }

})();
</script>"""
