# =============================================================
# DEMOSTRACIÓN INTERACTIVA - TFG PABLO MARTÍN TEJEDOR
# Estimación del precio de vehículos de segunda mano (XGBoost)
# =============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------

st.set_page_config(
    page_title='Estimación de Precio de Vehículos de Segunda Mano',
    page_icon='🚗',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# -------------------------------------------------------------
# CSS PERSONALIZADO (mismo estilo que el prototipo de Jupyter)
# -------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600;700&display=swap');

/* Fondo general de la app */
.stApp {
    background: #0f1117;
    color: #e2e8f0;
}

/* Eliminar el padding del header de Streamlit */
[data-testid="stHeader"] {
    background: #0f1117;
}

.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px;
}

/* HERO superior */
.pred-hero {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0d1f3c 0%, #0f1117 60%, #0d1f3c 100%);
    padding: 130px 40px 80px 40px;
    text-align: center;
    border-bottom: 1px solid #1e293b;
    margin: 0 -2rem 30px -2rem;
    min-height: 320px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
}
.pred-hero > div {
    width: 100%;
    text-align: center;
}
.pred-title {
    font-family: 'Syne', sans-serif;
    font-size: 38px;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 auto 28px auto;
    line-height: 1.1;
    text-align: center;
    white-space: nowrap;
}
.pred-slogan {
    color: #e2e8f0;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-align: center;
    margin: 0 auto;
}

/* Títulos de sección */
.pred-section-title {
    color: #60a5fa;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin: 24px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e293b;
    font-family: 'Inter', sans-serif;
}

/* Labels de los inputs (Streamlit) */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stSelectbox label, .stSlider label,
.stNumberInput label, .stRadio label {
    color: #e2e8f0 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Inputs de tipo selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}

/* Texto que el usuario escribe dentro del selectbox */
.stSelectbox div[data-baseweb="select"] input {
    color: #f1f5f9 !important;
    caret-color: #f1f5f9 !important;
}

/* Inputs numéricos */
.stNumberInput input,
.stTextInput input {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}

/* Botones +/- de los number_input */
.stNumberInput button {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #f1f5f9 !important;
    border-left: 1px solid #f1f5f9 !important;
    border-right: 1px solid #f1f5f9 !important;
}
.stNumberInput button:hover {
    background: #334155 !important;
    color: #ffffff !important;
    border: 1px solid #ffffff !important;
}
.stNumberInput button svg {
    fill: #f1f5f9 !important;
}
.stNumberInput [data-testid="stNumberInputContainer"] {
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

/* === SLIDER - ESTILO COMPLETO === */

/* Barra del slider */
.stSlider [data-baseweb="slider"] {
    background: transparent !important;
}
.stSlider [data-baseweb="slider"] > div {
    background: transparent !important;
}
.stSlider [data-baseweb="slider"] > div > div {
    background: #3b82f6 !important;
}

/* Punto de selección */
.stSlider [role="slider"] {
    background: #60a5fa !important;
    border: 2px solid #60a5fa !important;
}

/* Valor numérico encima del puntito */
.stSlider [data-testid="stThumbValue"] {
    color: #60a5fa !important;
    font-weight: 700 !important;
}

/* === ELIMINAR LOS VALORES 0 Y 30 (NUCLEAR) === */
.stSlider [data-testid="stTickBar"],
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider [data-baseweb="slider"] > div:last-child,
div[data-testid="stTickBar"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}

/* Forzar que cualquier texto debajo del slider desaparezca */
.stSlider > div > div:nth-child(2),
.stSlider > div > div:nth-child(3),
.stSlider > div > div:nth-child(4) {
    display: none !important;
}

/* Quitar selección visual de cualquier elemento del slider */
.stSlider, .stSlider *, .stSlider *::before, .stSlider *::after {
    user-select: none !important;
    -webkit-user-select: none !important;
    -moz-user-select: none !important;
    -ms-user-select: none !important;
}

/* Radio buttons */
.stRadio label {
    color: #e2e8f0 !important;
}

/* Botón principal */
.stButton button {
    background: #2563eb !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 1px !important;
    height: 56px !important;
    border-radius: 8px !important;
    border: none !important;
    width: 100%;
    transition: background 0.2s;
}
.stButton button:hover {
    background: #1d4ed8 !important;
}

/* Botón secundario (Reiniciar) - amarillo suave */
[data-testid="stButton"] button[kind="secondary"] {
    background: #fbbf2422 !important;
    color: #fbbf24 !important;
    border: 1px solid #fbbf2466 !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
    background: #fbbf2433 !important;
    color: #fcd34d !important;
    border: 1px solid #fbbf24 !important;
}

/* Resultado de la predicción */
.pred-result {
    background: linear-gradient(135deg, #0f172a, #0d1525);
    border: 1px solid #2563eb44;
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 20px;
    font-family: 'Inter', sans-serif;
}
.pred-result-top {
    display: flex;
    align-items: center;
    gap: 28px;
    margin-bottom: 20px;
    padding-bottom: 20px;
    border-bottom: 1px solid #1e293b;
    flex-wrap: wrap;
}
.pred-result-price {
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    color: #60a5fa;
    letter-spacing: -1px;
    line-height: 1;
    white-space: nowrap;
}
.pred-result-range {
    color: #e2e8f0;
    font-size: 12px;
    font-weight: 700;
    margin-top: 18px;
}
.pred-result-range span { color: #e2e8f0; font-weight: 700; }
.pred-result-label {
    color: #e2e8f0;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.pred-result-car {
    color: #f1f5f9;
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}
.pred-result-details { color: #e2e8f0; font-size: 13px; line-height: 1.9; }
.pred-result-details span { color: #e2e8f0; font-weight: 700; }

/* Tarjetas comparativas */
.pred-cards {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
}
.pred-card {
    background: #0a0f1e;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 18px 20px;
    font-family: 'Inter', sans-serif;
}
.pred-card-icon { font-size: 22px; margin-bottom: 10px; }
.pred-card-title {
    color: #e2e8f0;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.pred-card-value {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 8px;
}
.pred-card-sub { color: #e2e8f0; font-size: 12px; line-height: 1.6; }
.pred-card-sub span { color: #e2e8f0; font-weight: 700; }
.pred-card-up    { border-top: 2px solid #22c55e66; }
.pred-card-down  { border-top: 2px solid #ef444466; }
.pred-card-market{ border-top: 2px solid #60a5fa66; }

/* Avisos */
.pred-warning {
    background: linear-gradient(135deg, #2a1810, #1a0f08);
    border: 1px solid #f59e0b55;
    border-left: 3px solid #f59e0b;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
    font-family: 'Inter', sans-serif;
}
.pred-warning-title {
    color: #f59e0b;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.pred-warning-msg { color: #fbbf24; font-size: 13px; line-height: 1.6; }

/* Footer */
.pred-footer {
    text-align: center;
    color: #e2e8f0;
    font-size: 13px;
    font-weight: 500;
    padding: 30px 0 10px 0;
    margin-top: 40px;
    border-top: 1px solid #1e293b;
    font-family: 'Inter', sans-serif;
}

/* Frases de explicación de las tarjetas */
.pred-card-explanation {
    color: #e2e8f0 !important;
}
.pred-card-explanation span { 
    color: #94a3b8 !important; 
    font-weight: 600 !important; 
}           

</style>
""", unsafe_allow_html=True)

# -----------------------------
# CARGA DE RECURSOS (cacheado)
# -----------------------------

@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('xgb_opt.pkl')
    media_global = joblib.load('app_artifacts/media_global.pkl')

    manufacturer_cats = joblib.load('app_artifacts/manufacturer_cats.pkl')
    transmission_cats = joblib.load('app_artifacts/transmission_cats.pkl')
    drivetrain_cats   = joblib.load('app_artifacts/drivetrain_cats.pkl')
    fuel_type_cats    = joblib.load('app_artifacts/fuel_type_cats.pkl')
    ext_color_cats    = joblib.load('app_artifacts/ext_color_cats.pkl')
    int_color_cats    = joblib.load('app_artifacts/int_color_cats.pkl')

    manufacturer_to_models = joblib.load('app_artifacts/manufacturer_to_models.pkl')
    encoding_dict          = joblib.load('app_artifacts/encoding_dict.pkl')

    precio_medio  = joblib.load('app_artifacts/precio_medio_por_modelo.pkl')
    mileage_medio = joblib.load('app_artifacts/mileage_medio_por_modelo.pkl')
    age_medio     = joblib.load('app_artifacts/age_medio_por_modelo.pkl')

    return (modelo, media_global, manufacturer_cats, transmission_cats,
            drivetrain_cats, fuel_type_cats, ext_color_cats, int_color_cats,
            manufacturer_to_models, encoding_dict,
            precio_medio, mileage_medio, age_medio)

(modelo, media_global, manufacturer_cats, transmission_cats,
 drivetrain_cats, fuel_type_cats, ext_color_cats, int_color_cats,
 manufacturer_to_models, encoding_dict,
 precio_medio_por_modelo, mileage_medio_por_modelo, age_medio_por_modelo) = cargar_recursos()

# ------------------------
# RESTRICCIONES POR MARCA
# ------------------------

RESTRICCIONES = {
    'Audi':          {'drivetrain': ['AWD', 'FWD', 'RWD']},
    'BMW':           {'drivetrain': ['AWD', 'FWD', 'RWD']},
    'Chrysler':      {'fuel_type': ['Flex Fuel', 'Gasolina', 'Hibrido', 'Plug-in Hybrid']},
    'Dodge':         {'fuel_type': ['Diesel', 'Flex Fuel', 'Gasolina']},
    'Honda':         {'fuel_type': ['Gasolina', 'Hibrido', 'Plug-in Hybrid']},
    'INFINITI':      {'fuel_type': ['Gasolina', 'Hibrido']},
    'Jaguar':        {'drivetrain': ['AWD', 'RWD'], 'transmission': ['Automatica', 'Manual']},
    'Jeep':          {'fuel_type': ['Diesel', 'Flex Fuel', 'Gasolina', 'Hibrido', 'Plug-in Hybrid']},
    'Land Rover':    {'drivetrain': ['4WD', 'AWD'], 'transmission': ['Automatica'],
                     'fuel_type': ['Diesel', 'Flex Fuel', 'Gasolina', 'Hibrido', 'Plug-in Hybrid']},
    'Lexus':         {'fuel_type': ['Gasolina', 'Hibrido']},
    'Lincoln':       {'transmission': ['Automatica', 'CVT']},
    'Mercedes-Benz': {'transmission': ['Automatica', 'Manual', 'Semi-automatica']},
    'Porsche':       {'drivetrain': ['4WD', 'AWD', 'RWD']},
    'RAM':           {'drivetrain': ['4WD', 'FWD', 'RWD']},
    'Subaru':        {'drivetrain': ['AWD', 'RWD']},
    'Tesla':         {'fuel_type': ['Electrico'], 'drivetrain': ['AWD', 'RWD'],
                     'transmission': ['Automatica', 'CVT']},
    'Toyota':        {'fuel_type': ['Electrico', 'Flex Fuel', 'Gasolina', 'Hibrido', 'Plug-in Hybrid']},
    'Volvo':         {'drivetrain': ['AWD', 'FWD', 'RWD']},
}

# -----
# HERO
# -----

st.markdown("""
<div class="pred-hero">
    <div class="pred-title">¿CUÁNTO VALE TU COCHE?</div>
    <div class="pred-slogan">INTRODUCE LAS CARACTERÍSTICAS Y OBTÉN UNA ESTIMACIÓN INSTANTÁNEA</div>
</div>
""", unsafe_allow_html=True)

# -----------
# FORMULARIO
# -----------

# Contador de reset: cuando incrementa, todos los widgets se renuevan completamente
if 'reset_counter' not in st.session_state:
    st.session_state['reset_counter'] = 0
rc = st.session_state['reset_counter']

# === IDENTIDAD DEL VEHÍCULO ===
st.markdown('<div class="pred-section-title">Identidad del vehículo</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    manufacturer = st.selectbox(
        'Marca',
        ['Seleccionar...'] + manufacturer_cats,
        key=f'manufacturer_{rc}'
    )

with col_b:
    if manufacturer == 'Seleccionar...':
        modelos_disponibles = ['Seleccionar...']
    else:
        modelos_disponibles = ['Seleccionar...'] + manufacturer_to_models.get(manufacturer, [])

    model_input = st.selectbox(
        'Modelo',
        modelos_disponibles,
        key=f'model_{rc}'
    )

# Restricciones según marca
restricciones = RESTRICCIONES.get(manufacturer, {})

fuel_options    = restricciones.get('fuel_type',    fuel_type_cats)
drive_options   = restricciones.get('drivetrain',   drivetrain_cats)
trans_options   = restricciones.get('transmission', transmission_cats)

es_tesla = (manufacturer == 'Tesla')

# === CARACTERÍSTICAS DEL VEHÍCULO ===
st.markdown('<div class="pred-section-title">Características del vehículo</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    car_age = st.slider('Antigüedad (Años)', 0, 30, 0, key=f'car_age_{rc}')
    mileage = st.number_input('Millas', min_value=0, max_value=500000, value=0, step=1000, key=f'mileage_{rc}')

    if es_tesla:
        engine_size = 0.0
        st.number_input('Motor (Litros)', value=0.0, disabled=True, help='Tesla es eléctrica.', key=f'engine_size_tesla_{rc}')
    else:
        engine_size = st.number_input('Motor (Litros)', min_value=0.0, max_value=8.4, value=0.0, step=0.1, key=f'engine_size_{rc}')

    if es_tesla:
        mpg_highway = 0
        st.number_input('Eficiencia en Autopista (MPG)', value=0, disabled=True, help='Tesla es eléctrica.', key=f'mpg_tesla_{rc}')
    else:
        mpg_highway = st.number_input('Eficiencia en Autopista (MPG)', min_value=0, max_value=80, value=0, key=f'mpg_highway_{rc}')

    int_color = st.selectbox(
        'Color Interior',
        ['Seleccionar...'] + int_color_cats,
        key=f'int_color_{rc}'
    )

with col2:
    drivetrain = st.selectbox(
        'Tracción',
        ['Seleccionar...'] + drive_options,
        key=f'drivetrain_{rc}'
    )
    fuel_type = st.selectbox(
        'Combustible',
        ['Seleccionar...'] + fuel_options,
        key=f'fuel_type_{rc}'
    )
    transmission = st.selectbox(
        'Transmisión',
        ['Seleccionar...'] + trans_options,
        key=f'transmission_{rc}'
    )
    ext_color = st.selectbox(
        'Color Exterior',
        ['Seleccionar...'] + ext_color_cats,
        key=f'ext_color_{rc}'
    )

# === HISTORIAL DEL VEHÍCULO ===
st.markdown('<div class="pred-section-title">Historial del vehículo</div>', unsafe_allow_html=True)

col_h1, col_h2 = st.columns(2)
with col_h1:
    accidents = st.selectbox('¿Con Accidente o Daño?', ['Seleccionar...', 'No', 'Sí'], key=f'accidents_{rc}')
with col_h2:
    one_owner = st.selectbox('¿Un solo Dueño?', ['Seleccionar...', 'Sí', 'No'], key=f'one_owner_{rc}')

st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

# --------------------
# BOTÓN DE PREDICCIÓN
# --------------------

col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    calcular = st.button('CALCULAR EL PRECIO ESTIMADO', type='primary', use_container_width=True)

with col_btn2:
    reiniciar = st.button('REINICIAR', type='secondary', use_container_width=True)

if reiniciar:
    st.session_state['reset_counter'] = rc + 1
    st.rerun()

if calcular:
    # ---- Validación de campos vacíos ----
    campos_vacios = []
    if manufacturer == 'Seleccionar...': campos_vacios.append('Marca')
    if model_input  == 'Seleccionar...': campos_vacios.append('Modelo')
    if drivetrain   == 'Seleccionar...': campos_vacios.append('Tracción')
    if fuel_type    == 'Seleccionar...': campos_vacios.append('Combustible')
    if transmission == 'Seleccionar...': campos_vacios.append('Transmisión')
    if accidents    == 'Seleccionar...': campos_vacios.append('¿Con Accidente o Daño?')
    if one_owner    == 'Seleccionar...': campos_vacios.append('¿Un solo dueño?')
    if ext_color    == 'Seleccionar...': campos_vacios.append('Color Exterior')
    if int_color    == 'Seleccionar...': campos_vacios.append('Color Interior')

    if campos_vacios:
        msgs = ''.join([f'<div class="pred-warning-msg">· {c}</div>' for c in campos_vacios])
        st.markdown(f'''
        <div class="pred-warning">
            <div class="pred-warning-title">⚠️ Campos pendientes</div>
            {msgs}
        </div>
        ''', unsafe_allow_html=True)
    else:
        # ---- Codificación de inputs ----
        def ec(v, cats):
            return cats.index(v) if v in cats else -1

        model_enc = encoding_dict.get(model_input, media_global)

        entrada = pd.DataFrame([{
            'manufacturer':        ec(manufacturer, manufacturer_cats),
            'model':               model_enc,
            'mileage':             mileage,
            'transmission':        ec(transmission, transmission_cats),
            'drivetrain':          ec(drivetrain, drivetrain_cats),
            'fuel_type':           ec(fuel_type, fuel_type_cats),
            'exterior_color':      ec(ext_color, ext_color_cats),
            'interior_color':      ec(int_color, int_color_cats),
            'accidents_or_damage': 1 if accidents == 'Sí' else 0,
            'one_owner':           1 if one_owner == 'Sí' else 0,
            'car_age':             car_age,
            'engine_size':         engine_size,
            'mpg_highway':         mpg_highway,
        }])

        # ---- Predicción ----
        precio = float(np.exp(modelo.predict(entrada)[0]))
        margen = precio * 0.081

        # ---- Tarjeta 1: Posición en el mercado ----
        media_modelo = precio_medio_por_modelo.get(model_input, None)
        if media_modelo:
            diff_pct = ((precio - media_modelo) / media_modelo) * 100
            mercado_valor = f'+{diff_pct:.1f}%' if diff_pct > 0 else f'{diff_pct:.1f}%'
            mercado_color = '#22c55e' if diff_pct > 0 else '#ef4444'
            mercado_dir = 'Por encima de la media' if diff_pct > 0 else 'Por debajo de la media'
            mercado_ref = f'Media del <span>{model_input}</span>: <span>{media_modelo:,.0f}$</span>'
        else:
            mercado_valor, mercado_color = '—', '#60a5fa'
            mercado_dir, mercado_ref = 'Sin datos de referencia', ''

        # ---- Tarjeta 2: Millas ----
        media_mileage = mileage_medio_por_modelo.get(model_input, None)
        if media_mileage:
            diff_km = mileage - media_mileage
            diff_km_pct = (diff_km / media_mileage) * 100 if media_mileage > 0 else 0
            if diff_km < 0:
                km_valor, km_color = f'{diff_km_pct:.1f}%', '#22c55e'
                km_dir, km_icon = 'Menos millas que la media', '🟢'
            else:
                km_valor, km_color = f'+{diff_km_pct:.1f}%', '#ef4444'
                km_dir, km_icon = 'Más millas que la media', '🔴'
            km_ref = f'Media del <span>{model_input}</span>: <span>{media_mileage:,.0f} mi</span>'
        else:
            km_valor, km_color, km_dir, km_ref, km_icon = '—', '#60a5fa', 'Sin datos de referencia', '', '📏'

        # ---- Tarjeta 3: Antigüedad ----
        media_age = age_medio_por_modelo.get(model_input, None)
        if media_age and media_age > 0:
            diff_age = car_age - media_age
            diff_age_pct = (diff_age / media_age) * 100
            if diff_age < 0:
                age_valor, age_color = f'{diff_age_pct:.1f}%', '#22c55e'
                age_dir, age_icon = 'Más nuevo que la media', '🟢'
            else:
                age_valor, age_color = f'+{diff_age_pct:.1f}%', '#ef4444'
                age_dir, age_icon = 'Más antiguo que la media', '🔴'
            age_ref = f'Media del <span>{model_input}</span>: <span>{media_age:.1f} años</span>'
        else:
            age_valor, age_color, age_dir, age_ref, age_icon = '—', '#60a5fa', 'Sin datos de referencia', '', '📅'

        # ---- Avisos ----
        avisos = []
        if mileage > 350000:
            avisos.append('Millas superiores a 350.000 mi — Fuera del rango del dataset. La predicción puede ser menos fiable.')
        if mpg_highway > 60:
            avisos.append('Eficiencia en autopista superior a 60 MPG — Fuera del rango habitual. La predicción puede ser menos fiable.')

        if avisos:
            msgs = ''.join([f'<div class="pred-warning-msg">{a}</div>' for a in avisos])
            st.markdown(f'''
            <div class="pred-warning">
                <div class="pred-warning-title">⚠️ Aviso de Fiabilidad</div>
                {msgs}
            </div>
            ''', unsafe_allow_html=True)

        # ---- Resultado ----
        st.markdown(f"""
        <div class="pred-result">
            <div class="pred-result-top">
                <div>
                    <div class="pred-result-label">PRECIO ESTIMADO</div>
                    <div class="pred-result-price">{precio:,.0f}$</div>
                    <div class="pred-result-range">Rango Estimado: <span>{precio - margen:,.0f}$ – {precio + margen:,.0f}$</span></div>
                </div>
                <div>
                    <div class="pred-result-car">{manufacturer} {model_input}</div>
                    <div class="pred-result-details">
                        <span>{car_age} años</span> · 
                        <span>{mileage:,} mi</span> · 
                        Motor <span>{engine_size:.1f}L</span> ·
                        <span>{fuel_type}</span><br>
                        <span>{drivetrain}</span> · 
                        <span>{transmission}</span> · 
                        Color Exterior <span>{ext_color}</span> · 
                        Color Interior <span>{int_color}</span>
                    </div>
                </div>
            </div>
            <div class="pred-cards">
                <div class="pred-card pred-card-market">
                    <div class="pred-card-icon">📊</div>
                    <div class="pred-card-title">POSICIÓN EN EL MERCADO</div>
                    <div class="pred-card-value" style="color:{mercado_color}">{mercado_valor}</div>
                    <div class="pred-card-sub">{mercado_dir}<br>{mercado_ref}</div>
                </div>
                <div class="pred-card pred-card-up">
                    <div class="pred-card-icon">{km_icon}</div>
                    <div class="pred-card-title">MILLAS</div>
                    <div class="pred-card-value" style="color:{km_color}">{km_valor}</div>
                    <div class="pred-card-sub">{km_dir}<br>{km_ref}</div>
                </div>
                <div class="pred-card pred-card-down">
                    <div class="pred-card-icon">{age_icon}</div>
                    <div class="pred-card-title">ANTIGÜEDAD</div>
                    <div class="pred-card-value" style="color:{age_color}">{age_valor}</div>
                    <div class="pred-card-sub">{age_dir}<br>{age_ref}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------
# FOOTER
# -------

st.markdown("""
<div class="pred-footer">
    Trabajo de Fin de Grado en Ingeniería Informática · Pablo Martín Tejedor · CUNEF Universidad
</div>
""", unsafe_allow_html=True)