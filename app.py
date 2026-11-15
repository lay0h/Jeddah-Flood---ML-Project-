"""Streamlit app: The Saudi Water Paradox - play with the Jeddah flood model.

Run from the project folder:   streamlit run app.py
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))
from predict import load_model, predict_susceptibility  # noqa: E402

BLUE, NAVY, AQUA, SAND = "#0077B6", "#03045E", "#48CAE4", "#E9C46A"
ROUNDS = 10

st.set_page_config(page_title="The Saudi Water Paradox", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&display=swap');
.stApp, .stApp p, .stApp label, .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp li, .stMarkdown div
  {font-family: 'Sora', 'Segoe UI', system-ui, sans-serif;}
.stApp {background:
  radial-gradient(1100px 520px at 85% -8%, rgba(0,119,182,.45), transparent 70%),
  radial-gradient(900px 520px at -10% 35%, rgba(0,180,216,.18), transparent 70%),
  radial-gradient(900px 600px at 60% 110%, rgba(72,202,228,.14), transparent 70%), #030B2E;}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], footer, #MainMenu {display: none !important;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1120px;}

/* brand bar */
.brand {display: flex; align-items: center; gap: .9rem; margin-bottom: .4rem;}
.brand .logo {width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center; font-size: 1.5rem;
              background: linear-gradient(135deg, #48CAE4, #0077B6); box-shadow: 0 8px 26px rgba(0,180,216,.45);}
.brand .name {font-weight: 800; font-size: 1.25rem; letter-spacing: -.01em; line-height: 1.1;}
.brand .tag {color: #9CC7DD; font-size: .82rem;}

/* every radio becomes a row of pills */
div[role="radiogroup"] {gap: .5rem; flex-wrap: wrap;}
div[role="radiogroup"] label {background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.14);
  border-radius: 999px; padding: .42rem 1rem; margin: 0; cursor: pointer; transition: all .18s ease;}
div[role="radiogroup"] label:hover {border-color: #48CAE4; transform: translateY(-1px);}
div[role="radiogroup"] label > div > div:first-child:not([data-testid="stMarkdownContainer"]) {display: none !important;}
div[role="radiogroup"] label[data-selected="true"] {background: linear-gradient(135deg, #00B4D8, #0077B6);
  border-color: transparent; box-shadow: 0 6px 20px rgba(0,180,216,.4);}
div[role="radiogroup"] label[data-selected="true"] p {color: white; font-weight: 600;}
.st-key-nav div[role="radiogroup"] {background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10);
  border-radius: 999px; padding: .35rem; width: fit-content; margin-left: auto; flex-wrap: nowrap;}
.st-key-nav div[role="radiogroup"] label p {white-space: nowrap;}
.st-key-nav div[role="radiogroup"] label {border: none; background: transparent; padding: .5rem 1.15rem;}
.st-key-nav div[role="radiogroup"] label p {font-weight: 600;}
.st-key-nav div[role="radiogroup"] label[data-selected="true"] {background: linear-gradient(135deg, #00B4D8, #0077B6);
  box-shadow: 0 6px 20px rgba(0,180,216,.45);}

/* hero */
.hero {position: relative; overflow: hidden; border-radius: 26px; padding: 2rem 2.2rem 3.2rem 2.2rem; margin: .8rem 0 1.3rem 0;
       background: linear-gradient(120deg, #06206B 0%, #0077B6 62%, #48CAE4 100%); box-shadow: 0 22px 60px rgba(0,60,140,.45);}
.hero h1 {color: white; font-size: 2.35rem; font-weight: 800; letter-spacing: -.02em; margin: 0; padding: 0;}
.hero p {color: #E4F6FF; font-size: 1.02rem; margin: .5rem 0 0 0; max-width: 760px;}
.hero svg {position: absolute; left: 0; bottom: -2px; width: 200%; height: 46px; animation: drift 14s linear infinite;}
.hero svg.back {opacity: .45; animation-duration: 22s; height: 60px;}
@keyframes drift {from {transform: translateX(0);} to {transform: translateX(-50%);}}

/* glass cards */
.glass {background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12); border-radius: 20px;
        backdrop-filter: blur(8px); padding: 1.1rem 1.2rem;}
.clues {display: grid; grid-template-columns: repeat(4, 1fr); gap: .7rem;}
.clue {background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.13); border-radius: 18px;
       padding: 1rem .5rem; text-align: center; animation: rise .5s ease both;}
.clue:nth-child(2) {animation-delay: .08s;} .clue:nth-child(3) {animation-delay: .16s;} .clue:nth-child(4) {animation-delay: .24s;}
.clue .icon {font-size: 1.8rem;}
.clue .val {font-size: 1.25rem; font-weight: 800; color: white; white-space: nowrap;}
.clue .lab {color: #9CC7DD; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;}
@keyframes rise {from {opacity: 0; transform: translateY(14px);} to {opacity: 1; transform: none;}}

/* scoreboard */
.board {display: flex; align-items: center; gap: 1rem; margin-bottom: .2rem;}
.board .side {flex: 1; border-radius: 20px; padding: .9rem 1.2rem; display: flex; align-items: center; justify-content: space-between;
              border: 1px solid rgba(255,255,255,.14);}
.board .you {background: linear-gradient(135deg, rgba(0,180,216,.55), rgba(0,119,182,.35));}
.board .bot {background: linear-gradient(135deg, rgba(123,97,255,.45), rgba(40,30,120,.35));}
.board .who {font-weight: 700; letter-spacing: .08em; font-size: .85rem;}
.board .n {font-size: 2.3rem; font-weight: 800; line-height: 1;}
.board .vs {font-weight: 800; color: #9CC7DD;}
.track {display: flex; gap: .35rem; margin: .5rem 0 0 0;}
.track span {flex: 1; height: 8px; border-radius: 6px; background: rgba(255,255,255,.12);}
.track span.ok {background: #2EE6A6;} .track span.no {background: #FF6B6B;} .track span.now {background: #48CAE4; animation: pulse 1.2s ease infinite;}
@keyframes pulse {50% {opacity: .35;}}
.tracklab {color: #9CC7DD; font-size: .72rem; letter-spacing: .08em; margin-top: .5rem;}
.round {font-size: 1.35rem; font-weight: 800; margin: 1.1rem 0 .7rem 0;}
.round small {color: #9CC7DD; font-weight: 400; font-size: .9rem; margin-left: .5rem;}
.reveal {border-radius: 18px; padding: 1.1rem 1.3rem; margin-top: .9rem; font-size: 1.02rem; animation: pop .35s ease both;
         border: 1px solid rgba(255,255,255,.16);}
.reveal.win {background: linear-gradient(135deg, rgba(46,230,166,.28), rgba(46,230,166,.08));}
.reveal.lose {background: linear-gradient(135deg, rgba(255,107,107,.30), rgba(255,107,107,.08));}
.reveal .big {font-size: 1.3rem; font-weight: 800;}
@keyframes pop {from {opacity: 0; transform: scale(.94);} to {opacity: 1; transform: none;}}
.final {text-align: center; padding: 2.2rem 1rem; animation: pop .4s ease both;}
.final .emoji {font-size: 4rem;} .final h2 {font-size: 2rem; font-weight: 800; margin: .2rem 0;}

/* buttons */
.stButton > button {border-radius: 14px; border: 1px solid rgba(255,255,255,.16); background: rgba(255,255,255,.06);
  color: #EAF6FF; font-weight: 600; padding: .62rem 1rem; transition: all .18s ease;}
.stButton > button:hover {transform: translateY(-2px); border-color: #48CAE4; box-shadow: 0 10px 26px rgba(0,180,216,.28); color: white;}
.stButton > button[kind="primary"] {background: linear-gradient(135deg, #00B4D8, #0077B6); border: none; color: white;}
.st-key-btn_flood button, .st-key-btn_dry button {height: 104px; border: none; border-radius: 20px;}
.st-key-btn_flood button p, .st-key-btn_dry button p {font-size: 1.3rem; font-weight: 800;}
.st-key-btn_flood button {background: linear-gradient(135deg, #00B4D8, #0353A4);}
.st-key-btn_dry button {background: linear-gradient(135deg, #F4D58D, #D9A441);}
.st-key-btn_dry button p {color: #3A2A00;}

/* gauge */
.gauge {width: 236px; height: 236px; border-radius: 50%; position: relative; overflow: hidden; margin: .4rem auto 0 auto;
        background: #06123F; border: 5px solid rgba(255,255,255,.18); box-shadow: 0 0 60px rgba(0,180,216,.35), inset 0 0 30px rgba(0,0,0,.5);}
.gauge .wave {position: absolute; left: -50%; width: 200%; height: 200%; background: linear-gradient(#00B4D8, #0353A4);
              border-radius: 42%; animation: spin 7s linear infinite; transition: top .7s cubic-bezier(.2,.8,.2,1);}
.gauge .wave.two {background: #48CAE4; opacity: .45; border-radius: 38%; animation-duration: 11s;}
.gauge .num {position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
             font-size: 3.6rem; font-weight: 800; color: white; text-shadow: 0 2px 14px rgba(0,0,0,.55);}
.gauge .num small {font-size: .8rem; font-weight: 600; letter-spacing: .08em; opacity: .9;}
@keyframes spin {from {transform: rotate(0deg);} to {transform: rotate(360deg);}}
.verdict {text-align: center; font-size: 1.35rem; font-weight: 800; margin-top: .9rem;}
.best {text-align: center; color: #9CC7DD; font-size: .85rem; margin-top: .2rem;}
.small {color: #9CC7DD; font-size: .84rem;}
.notice {background: rgba(244,213,141,.14); border: 1px solid rgba(244,213,141,.45); border-radius: 16px; padding: .85rem 1.1rem;
         color: #FBE9BD; margin-top: .8rem;}
.legend {display: flex; gap: 1.2rem; flex-wrap: wrap; margin-top: .6rem; font-size: .9rem;}
.legend i {display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: .4rem;}
[data-testid="stDeckGlJsonChart"], .stDeckGlJsonChart {border-radius: 20px; overflow: hidden; border: 1px solid rgba(255,255,255,.12);}
.foot {text-align: center; color: #6F96AD; font-size: .78rem; margin-top: 2.2rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- data
@st.cache_resource
def get_model():
    return load_model(ROOT / "models" / "flood_model.joblib")


@st.cache_data
def get_map_points():
    return pd.read_csv(ROOT / "app_data" / "map_points.csv")


@st.cache_data
def get_game_cells():
    return pd.read_csv(ROOT / "app_data" / "game_cells.csv")


@st.cache_data
def get_examples():
    return json.load(open(ROOT / "app_data" / "examples.json"))


LAND = {"bare_sparse": ("🏜️", "Bare land"), "built_up": ("🏢", "Built-up"),
        "vegetation": ("🌿", "Vegetation"), "water_wetland": ("🌾", "Wetland")}


WAVE = ("M0 30 Q 75 0 150 30 T 300 30 T 450 30 T 600 30 T 750 30 T 900 30 T 1050 30 T 1200 30 "
        "T 1350 30 T 1500 30 T 1650 30 T 1800 30 T 1950 30 T 2100 30 T 2250 30 T 2400 30 V 60 H 0 Z")


def hero(title, text):
    st.markdown(
        f'''<div class="hero"><h1>{title}</h1><p>{text}</p>
<svg class="back" viewBox="0 0 2400 60" preserveAspectRatio="none"><path d="{WAVE}" fill="#030B2E"/></svg>
<svg viewBox="0 0 2400 60" preserveAspectRatio="none"><path d="{WAVE}" fill="#030B2E"/></svg></div>''',
        unsafe_allow_html=True)


# ----------------------------------------------------------------------------- page 1: the game
def new_game(level):
    cells = get_game_cells()
    pool = cells.index[cells.level == level].tolist()
    st.session_state.update(g_level=level, g_order=random.sample(pool, ROUNDS), g_i=0, g_guess=None, g_log=[])


def guess(value):
    s = st.session_state
    cell = get_game_cells().loc[s.g_order[s.g_i]]
    s.g_guess = value
    s.g_log.append((value == cell.label, int(cell.score >= 0.5) == cell.label))


def next_round():
    st.session_state.g_i += 1
    st.session_state.g_guess = None


def page_game():
    hero("Can you beat the model?",
         "A real 30 m square of Jeddah. Four clues. Did the satellite see flood water there after the storm? "
         "You and the model answer the same question, 10 times.")
    s = st.session_state
    level = st.radio("Level", ["easy", "hard"], horizontal=True, key="level_pick", label_visibility="collapsed",
                     format_func=lambda v: "🙂 Easy · anywhere in Jeddah" if v == "easy" else "😈 Hard · only ground near sea level")
    if s.get("g_level") != level:
        new_game(level)

    you, model = sum(a for a, _ in s.g_log), sum(b for _, b in s.g_log)

    def track(col):
        marks = ["ok" if r[col] else "no" for r in s.g_log]
        marks += ["now"] * (len(marks) < ROUNDS and s.g_guess is None) + [""] * ROUNDS
        return "".join(f'<span class="{m}"></span>' for m in marks[:ROUNDS])

    st.markdown(
        f'<div class="board"><div class="side you"><span class="who">YOU</span><span class="n">{you}</span></div>'
        f'<span class="vs">VS</span>'
        f'<div class="side bot"><span class="n">{model}</span><span class="who">THE MODEL</span></div></div>'
        f'<div style="display:flex;gap:1rem"><div style="flex:1"><div class="track">{track(0)}</div></div><span style="width:1.6rem"></span>'
        f'<div style="flex:1"><div class="track">{track(1)}</div></div></div>',
        unsafe_allow_html=True)

    if s.g_i >= ROUNDS:                                   # game over
        if you > model:
            st.balloons()
            emoji, title, line = "🏆", f"You win {you} to {model}!", "You read the land better than a Random Forest."
        elif you == model:
            emoji, title, line = "🤝", f"A tie, {you} to {model}", "You think like a Random Forest."
        else:
            emoji, title, line = "🤖", f"The model wins {model} to {you}", "It has studied 40,000 squares. No shame in that."
        extra = "Now try the hard level: every square is low ground, so the easy trick stops working." if level == "easy" else ""
        st.markdown(f'<div class="glass final"><div class="emoji">{emoji}</div><h2>{title}</h2><p>{line}</p>'
                    f'<p class="small">{extra}</p></div>', unsafe_allow_html=True)
        st.write("")
        st.columns([1, 1, 1])[1].button("Play again", on_click=new_game, args=(level,), type="primary", width="stretch")
        return

    cell = get_game_cells().loc[s.g_order[s.g_i]]
    st.markdown(f'<div class="round">Round {s.g_i + 1} <small>of {ROUNDS} · storm of November {int(cell.year)}</small></div>',
                unsafe_allow_html=True)
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        icon, name = LAND[cell.landcover_group]
        clues = [("⛰️", f"{cell.elevation_m:.1f} m", "height"), ("🌊", f"{cell.dist_to_coast_m / 1000:.1f} km", "to the coast"),
                 ("📐", f"{cell.slope_deg:.1f}°", "slope"), (icon, name, "land")]
        st.markdown('<div class="clues">' + "".join(
            f'<div class="clue"><div class="icon">{ic}</div><div class="val">{val}</div><div class="lab">{lab}</div></div>'
            for ic, val, lab in clues) + "</div>", unsafe_allow_html=True)
        st.write("")
        if s.g_guess is None:
            b1, b2 = st.columns(2)
            b1.button("💧 Flood water", on_click=guess, args=(1,), width="stretch", key="btn_flood")
            b2.button("🏜️ Dry land", on_click=guess, args=(0,), width="stretch", key="btn_dry")
        else:
            truth = "💧 Flood water" if cell.label == 1 else "🏜️ Dry land"
            you_ok, model_ok = s.g_log[-1]
            model_said = "flood water" if cell.score >= 0.5 else "dry land"
            st.markdown(
                f'<div class="reveal {"win" if you_ok else "lose"}"><div class="big">{truth}</div>'
                f'That is what the satellite saw. {"You got it ✅" if you_ok else "Not this time ❌"}<br>'
                f'<span class="small">The model said {model_said} (score {cell.score * 100:.0f}/100) {"✅" if model_ok else "❌"}</span></div>',
                unsafe_allow_html=True)
            st.write("")
            st.button("See the result" if s.g_i == ROUNDS - 1 else "Next square →", on_click=next_round, type="primary")
    with right:
        dot = pd.DataFrame({"lon": [cell.lon], "lat": [cell.lat]})
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", data=dot, get_position="[lon, lat]", get_fill_color=[255, 107, 107, 240],
                              get_radius=350, radius_min_pixels=8, stroked=True, get_line_color=[255, 255, 255],
                              line_width_min_pixels=2)],
            map_style="dark", initial_view_state=pdk.ViewState(latitude=21.55, longitude=39.2, zoom=8.7)), height=300)
    st.markdown('<div class="small">"Flood water" means water the radar satellite still saw 4 to 5 days after the storm.</div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- page 2: build your land
LANDCOVER_CHOICES = {"🏜️ Bare land": "bare_sparse", "🏢 Built-up": "built_up", "🌿 Vegetation": "shrubland", "🌾 Wetland": "herbaceous_wetland"}
DEFAULTS = {"elevation_m": 6.0, "dist_to_coast_km": 4.0, "slope_deg": 0.5, "landcover": "🏜️ Bare land"}
HIDDEN = {"hand_m": 0.0, "dist_to_ref_water_m": 1600.0, "flowacc_km2": 0.005, "twi": 9.5}   # typical values


def load_example(values):
    code_to_name = {v: k for k, v in LANDCOVER_CHOICES.items()}
    st.session_state.update(
        elevation_m=float(np.clip(values["elevation_m"], -2, 150)),
        dist_to_coast_km=float(np.clip(values["dist_to_coast_m"] / 1000, 0, 28)),
        slope_deg=float(np.clip(values["slope_deg"], 0, 15)),
        landcover=code_to_name[values["landcover_name"]],
        hidden={k: values[k] for k in HIDDEN})


def surprise():
    st.session_state.update(elevation_m=round(random.choice([random.uniform(-1, 5), random.uniform(5, 120)]), 1),
                            dist_to_coast_km=round(random.uniform(0, 25), 1), slope_deg=round(random.uniform(0, 6), 1),
                            landcover=random.choice(list(LANDCOVER_CHOICES)), hidden=dict(HIDDEN))


def scene_svg(elev, dist_km, icon, score):
    # side view at dusk: Red Sea on the left, the land rising to the right, a marker where the square is
    px = 130 + 440 * math.sqrt(dist_km / 28)
    py = 150 - 105 * min(1.0, math.log1p(max(elev, 0)) / math.log1p(150)) + (8 if elev < 0 else 0)
    end_y = max(py - 18, 25)
    return f"""
<svg viewBox="0 0 600 205" style="width:100%;border-radius:20px;border:1px solid rgba(255,255,255,.12);margin-top:.6rem">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0B1E5B"/><stop offset="1" stop-color="#1C6FA8"/></linearGradient>
    <linearGradient id="sand" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F4D58D"/><stop offset="1" stop-color="#B9812E"/></linearGradient>
    <radialGradient id="pool"><stop offset="0" stop-color="#48CAE4" stop-opacity="{0.25 + 0.7 * score:.2f}"/><stop offset="1" stop-color="#48CAE4" stop-opacity="0"/></radialGradient>
  </defs>
  <rect width="600" height="205" fill="url(#sky)"/>
  <circle cx="535" cy="40" r="17" fill="#FFF3C4"/><circle cx="535" cy="40" r="30" fill="#FFF3C4" opacity=".15"/>
  <circle cx="90" cy="30" r="1.4" fill="white"/><circle cx="210" cy="52" r="1.2" fill="white"/><circle cx="330" cy="24" r="1.5" fill="white"/><circle cx="430" cy="62" r="1.1" fill="white"/>
  <polygon points="110,150 {px:.0f},{py:.0f} 600,{end_y:.0f} 600,205 110,205" fill="url(#sand)"/>
  <rect x="0" y="150" width="125" height="55" fill="#0353A4"/>
  <path d="M0 150 q 10 -6 20 0 t 20 0 t 20 0 t 20 0 t 20 0 t 20 0 v 8 h -125 z" fill="#48CAE4"/>
  <text x="12" y="188" fill="white" font-size="12" font-weight="700" font-family="Sora, sans-serif">RED SEA</text>
  <ellipse cx="{px:.0f}" cy="{py + 3:.0f}" rx="{20 + 40 * score:.0f}" ry="{6 + 8 * score:.0f}" fill="url(#pool)"/>
  <line x1="{px:.0f}" y1="{py:.0f}" x2="{px:.0f}" y2="{py - 30:.0f}" stroke="white" stroke-width="2"/>
  <circle cx="{px:.0f}" cy="{py:.0f}" r="4" fill="white"/>
  <text x="{px:.0f}" y="{py - 35:.0f}" font-size="26" text-anchor="middle">{icon}</text>
  <text x="{px:.0f}" y="{min(py + 24, 198):.0f}" font-size="11.5" text-anchor="middle" fill="#06123F" font-weight="700" font-family="Sora, sans-serif">{elev:.1f} m · {dist_km:.1f} km</text>
</svg>"""


def page_build():
    hero("Build your land", "Shape a piece of Jeddah and watch the model react. Can you find the most flood-prone spot?")
    s = st.session_state
    for key, value in DEFAULTS.items():
        s.setdefault(key, value)
    s.setdefault("hidden", dict(HIDDEN))
    s.setdefault("best", 0.0)

    cols = st.columns(5)
    for col, (name, values) in zip(cols, get_examples().items()):
        col.button(name, on_click=load_example, args=(values,), width="stretch")
    cols[4].button("🎲 Surprise me", on_click=surprise, width="stretch", type="primary")
    st.write("")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.slider("⛰️  Height above sea level (m)", -2.0, 150.0, step=0.1, key="elevation_m")
        st.slider("🌊  Distance to the coast (km)", 0.0, 28.0, step=0.1, key="dist_to_coast_km")
        st.slider("📐  Slope (degrees)", 0.0, 15.0, step=0.1, key="slope_deg")
        st.radio("Land cover", list(LANDCOVER_CHOICES), key="landcover", horizontal=True)

    row = pd.DataFrame([{**s.hidden, "elevation_m": s.elevation_m, "slope_deg": s.slope_deg,
                         "dist_to_coast_m": s.dist_to_coast_km * 1000, "landcover_name": LANDCOVER_CHOICES[s.landcover]}])
    result = predict_susceptibility(row, get_model()).iloc[0]
    score = float(result["susceptibility_score"])
    s.best = max(s.best, score)

    with right:
        verdict = ("🌊 Water loves this spot" if score >= 0.75 else "💧 Likely to hold water" if score >= 0.5
                   else "🌤️ Probably stays dry" if score >= 0.2 else "☀️ High and dry")
        st.markdown(
            f'<div class="gauge"><div class="wave two" style="top:{103 - score * 100:.0f}%"></div>'
            f'<div class="wave" style="top:{106 - score * 100:.0f}%"></div>'
            f'<div class="num">{score * 100:.0f}<small>OUT OF 100</small></div></div>'
            f'<div class="verdict">{verdict}</div><div class="best">🏅 Your highest so far: <b>{s.best * 100:.0f}</b></div>',
            unsafe_allow_html=True)

    st.markdown(scene_svg(s.elevation_m, s.dist_to_coast_km, s.landcover.split()[0], score), unsafe_allow_html=True)
    if "built-up" in result["warning"]:
        st.markdown('<div class="notice">🏢 <b>City blind spot.</b> The radar satellite cannot see water between buildings, so the '
                    "model never learned what a flooded street looks like. A low score here does <b>not</b> mean the area is safe.</div>",
                    unsafe_allow_html=True)
    st.markdown('<div class="small" style="margin-top:.6rem">The score compares places with each other. It is not a chance of '
                "flooding and not a water depth.</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- page 3: the map
def page_map():
    hero("Explore the storms", "Every column is a real 30 m square. Drag to move, scroll to zoom, right-drag to tilt.")
    points = get_map_points()
    c1, c2, c3 = st.columns([1, 2.6, .7])
    year = c1.radio("Storm", [2022, 2017], horizontal=True, format_func=lambda y: f"Nov {y}", label_visibility="collapsed")
    view = c2.radio("Show", ["🛰️ Satellite saw", "🤖 Model thinks", "🎯 Model mistakes"], horizontal=True, label_visibility="collapsed")
    tilt = c3.toggle("3D", value=True)

    d = points[points.year == year].copy()
    pred = (d.score.values >= 0.5).astype(int)
    flooded = d.label.values == 1
    if view.startswith("🛰️"):
        rgb = np.where(flooded[:, None], [72, 202, 228], [244, 213, 141])
        d["info"] = np.where(flooded, "Flood water seen", "Seen dry")
        d["h"], alpha = np.where(flooded, 1.0, 0.04), np.where(flooded, 235, 120)
        legend = [("#48CAE4", f"flood water seen · {int(flooded.sum()):,}"), ("#F4D58D", f"seen dry · {int((~flooded).sum()):,}")]
    elif view.startswith("🤖"):
        cut = st.slider("Hide squares with a model score below", 0, 100, 0, 5) / 100
        keep = d.score.values >= cut
        d = d[keep].copy()
        t = d.score.values[:, None]
        lo, mid, hi = np.array([244, 213, 141]), np.array([72, 202, 228]), np.array([123, 97, 255])
        rgb = np.where(t < 0.5, lo + (mid - lo) * t / 0.5, mid + (hi - mid) * (t - 0.5) / 0.5)
        d["info"] = "Model score " + (d.score * 100).round().astype(int).astype(str) + " / 100"
        d["h"], alpha = d.score.clip(lower=0.04), np.where(d.score.values >= 0.5, 235, 130)
        legend = [("#F4D58D", "low score"), ("#48CAE4", "middle"), ("#7B61FF", "high score"), ("transparent", f"{len(d):,} squares shown")]
    else:
        missed, false_alarm = flooded & (pred == 0), ~flooded & (pred == 1)
        rgb = np.where(missed[:, None], [123, 97, 255], np.where(false_alarm[:, None], [255, 107, 107], [150, 170, 190]))
        d["info"] = np.where(missed, "Missed flood", np.where(false_alarm, "False alarm", "Model was right"))
        d["h"], alpha = np.where(missed | false_alarm, 1.0, 0.03), np.where(missed | false_alarm, 240, 60)
        right_share = 1 - (missed.sum() + false_alarm.sum()) / len(d)
        legend = [("#96AABE", f"model was right · {right_share:.0%}"), ("#FF6B6B", f"false alarm · {int(false_alarm.sum()):,}"),
                  ("#7B61FF", f"missed flood · {int(missed.sum()):,}")]
    d["r"], d["g"], d["b"], d["a"] = rgb[:, 0].astype(int), rgb[:, 1].astype(int), rgb[:, 2].astype(int), alpha

    cols = ["lon", "lat", "r", "g", "b", "a", "h", "info", "elevation_m"]
    if tilt:
        layer = pdk.Layer("ColumnLayer", data=d[cols], get_position="[lon, lat]", get_elevation="h", elevation_scale=2500,
                          radius=45, get_fill_color="[r, g, b, a]", pickable=True, auto_highlight=True)
        state = pdk.ViewState(latitude=21.52, longitude=39.2, zoom=9.2, pitch=50, bearing=-15)
    else:
        layer = pdk.Layer("ScatterplotLayer", data=d[cols], get_position="[lon, lat]", get_fill_color="[r, g, b, a]",
                          get_radius=45, radius_min_pixels=1.5, pickable=True)
        state = pdk.ViewState(latitude=21.55, longitude=39.2, zoom=9.3)
    st.pydeck_chart(pdk.Deck(layers=[layer], map_style="dark", initial_view_state=state,
                             tooltip={"text": "{info}\nHeight {elevation_m} m"}), height=560)
    st.markdown('<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{t}</span>' for c, t in legend) + "</div>",
                unsafe_allow_html=True)
    st.markdown('<div class="small" style="margin-top:.5rem">The dense city is mostly empty here: radar cannot see water between '
                "buildings, so those squares were left out. Model scores come from a model that never saw the same 5 km area in training.</div>",
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- navigation
PAGES = {"🎮  Beat the model": page_game, "🏗️  Build your land": page_build, "🗺️  Explore the storms": page_map}

top_left, top_right = st.columns([1, 1.6], vertical_alignment="center")
top_left.markdown('<div class="brand"><div class="logo">🌊</div><div><div class="name">The Saudi Water Paradox</div>'
                  '<div class="tag">Where does flood water stay in Jeddah?</div></div></div>', unsafe_allow_html=True)
choice = top_right.radio("Go to", list(PAGES), horizontal=True, key="nav", label_visibility="collapsed")

PAGES[choice]()

st.markdown('<div class="foot">A Random Forest trained on satellite data from the Jeddah storms of November 2017 and November 2022. '
            "It ranks open land by how likely water is to stay there. It is not a flood warning. · Machine Learning final project, Unit 45</div>",
            unsafe_allow_html=True)
