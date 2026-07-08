"""
Sistema de predicción de fútbol con enfoque en apuestas (value bets)
Streamlit + Football-Data.org + Modelo Poisson
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, mean_absolute_error
import requests
from io import StringIO
from datetime import datetime, timedelta
# ------------------------------------------------------------
# 0. AUTENTICACIÓN POR CLAVE (SUSCRIPCIÓN)
# ------------------------------------------------------------
if 'autenticado' not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.title("🔐 Acceso restringido")
    st.markdown("Esta herramienta requiere una clave de acceso. Solicítala a tu administrador.")
    clave = st.text_input("Clave de acceso", type="password")
    
    # Cargar claves desde los secretos de Streamlit
    try:
        claves_validas = st.secrets["claves"]
    except:
        claves_validas = {
            "demo123": "2026-08-07",
        }
    
    if st.button("Acceder"):
        if clave in claves_validas:
            expiracion_str = str(claves_validas[clave]).strip().strip('"').strip("'")
            if expiracion_str.lower() == "perpetua":
                st.session_state.autenticado = True
                st.rerun()
            else:
                # Intentar varios formatos de fecha
                expiracion = None
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
                    try:
                        expiracion = datetime.strptime(expiracion_str, fmt).date()
                        break
                    except ValueError:
                        continue
                if expiracion is None:
                    st.error(f"Formato de fecha no reconocido: '{expiracion_str}'. Contacta al administrador.")
                elif datetime.now().date() <= expiracion:
                    st.session_state.autenticado = True
                    st.rerun()
                else:
                    st.error("❌ Tu suscripción ha caducado. Contacta para renovarla.")
        else:
            st.error("❌ Clave incorrecta.")
    st.stop()


# ------------------------------------------------------------
# 1. CONFIGURACIÓN
# ------------------------------------------------------------
st.set_page_config(page_title="Predicción Fútbol - Apuestas", layout="wide")
st.title("⚽ Sistema de Predicción de Partidos (Value Bets)")
st.markdown("""
Carga un CSV histórico, descarga datos reales y obtén predicciones con **análisis de valor** frente a cuotas de apuestas.
""")

# ------------------------------------------------------------
# 2. FUNCIONES DE DATOS
# ------------------------------------------------------------
def generar_plantilla():
    df = pd.DataFrame(columns=['date','home_team','away_team','home_goals','away_goals'])
    df.loc[0] = ['2024-01-01','Equipo A','Equipo B',2,1]
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

# Datos finalizados (histórico)
@st.cache_data(ttl=600)
def descargar_datos_finalizados(api_key, liga='PD', season='2023'):
    api_key = api_key.strip() if api_key else ''
    url = f"https://api.football-data.org/v4/competitions/{liga}/matches?status=FINISHED&season={season}"
    headers = {'X-Auth-Token': api_key} if api_key else {}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Error HTTP/conexión: {e}")
        return pd.DataFrame()
    partidos = []
    for m in data.get('matches', []):
        try:
            partidos.append({
                'date': m['utcDate'][:10],
                'home_team': m['homeTeam']['name'],
                'away_team': m['awayTeam']['name'],
                'home_goals': m['score']['fullTime']['home'],
                'away_goals': m['score']['fullTime']['away']
            })
        except (KeyError, TypeError):
            continue
    df = pd.DataFrame(partidos)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

# Próximos partidos (SCHEDULED + TIMED)
@st.cache_data(ttl=300)
def descargar_proximos_partidos(api_key, liga='PD', season='2026'):
    api_key = api_key.strip() if api_key else ''
    # Obtenemos todos los partidos programados para la temporada elegida
    url = f"https://api.football-data.org/v4/competitions/{liga}/matches?status=SCHEDULED&season={season}"
    headers = {'X-Auth-Token': api_key} if api_key else {}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Error al obtener próximos partidos: {e}")
        return pd.DataFrame()
    partidos = []
    for m in data.get('matches', []):
        try:
            partidos.append({
                'date': m['utcDate'][:10],
                'home_team': m['homeTeam']['name'],
                'away_team': m['awayTeam']['name'],
                'home_goals': np.nan,
                'away_goals': np.nan
            })
        except:
            continue
    df = pd.DataFrame(partidos)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def cargar_csv(archivo):
    df = pd.read_csv(archivo)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.dropna(subset=['home_team','away_team','home_goals','away_goals'], inplace=True)
    df['home_goals'] = df['home_goals'].astype(int)
    df['away_goals'] = df['away_goals'].astype(int)
    return df

def combinar_datos(df_csv, df_api):
    if df_csv.empty and df_api.empty:
        return pd.DataFrame(columns=['date','home_team','away_team','home_goals','away_goals'])
    comb = pd.concat([df_csv, df_api], ignore_index=True)
    comb.drop_duplicates(subset=['date','home_team','away_team'], inplace=True)
    comb.sort_values('date', inplace=True)
    return comb

# ------------------------------------------------------------
# 3. FEATURES Y MODELO
# ------------------------------------------------------------
def crear_features_rodantes(df, ventana=5):
    if df.empty: return df
    home = df[['date','home_team','away_team','home_goals','away_goals']].copy()
    home['team'] = home['home_team']
    home['scored'] = home['home_goals']
    home['conceded'] = home['away_goals']
    away = df[['date','home_team','away_team','home_goals','away_goals']].copy()
    away['team'] = away['away_team']
    away['scored'] = away['away_goals']
    away['conceded'] = away['home_goals']
    todas = pd.concat([home[['date','team','scored','conceded']],
                       away[['date','team','scored','conceded']]], ignore_index=True)
    todas.sort_values(['team','date'], inplace=True)
    todas['avg_scored_5'] = todas.groupby('team')['scored'].transform(
        lambda x: x.shift(1).rolling(ventana, min_periods=1).mean())
    todas['avg_conceded_5'] = todas.groupby('team')['conceded'].transform(
        lambda x: x.shift(1).rolling(ventana, min_periods=1).mean())
    df = df.merge(todas[['date','team','avg_scored_5','avg_conceded_5']],
                  left_on=['date','home_team'], right_on=['date','team'], how='left')
    df.rename(columns={'avg_scored_5':'home_avg_scored','avg_conceded_5':'home_avg_conceded'}, inplace=True)
    df.drop('team', axis=1, inplace=True)
    df = df.merge(todas[['date','team','avg_scored_5','avg_conceded_5']],
                  left_on=['date','away_team'], right_on=['date','team'], how='left')
    df.rename(columns={'avg_scored_5':'away_avg_scored','avg_conceded_5':'away_avg_conceded'}, inplace=True)
    df.drop('team', axis=1, inplace=True)
    media = (df['home_goals'].mean() + df['away_goals'].mean()) / 2
    for c in ['home_avg_scored','home_avg_conceded','away_avg_scored','away_avg_conceded']:
        df[c].fillna(media, inplace=True)
    return df

def entrenar_modelo_poisson(df):
    if df.empty: return 1.5, 1.2, {}
    mh = df['home_goals'].mean()
    ma = df['away_goals'].mean()
    mg = (mh+ma)/2
    equipos = pd.concat([df['home_team'], df['away_team']]).unique()
    fuerzas = {}
    for eq in equipos:
        loc = df[df['home_team']==eq]
        vis = df[df['away_team']==eq]
        sl = loc['home_avg_scored'].mean() if not loc.empty else mh
        sv = vis['away_avg_scored'].mean() if not vis.empty else ma
        cl = loc['home_avg_conceded'].mean() if not loc.empty else ma
        cv = vis['away_avg_conceded'].mean() if not vis.empty else mh
        nl, nv = len(loc), len(vis)
        total = nl+nv
        if total==0:
            fuerzas[eq]={'ataque':1.0,'defensa':1.0}
        else:
            asc = (nl*sl + nv*sv)/total
            con = (nl*cl + nv*cv)/total
            fuerzas[eq]={
                'ataque': asc/mg if mg>0 else 1.0,
                'defensa': con/mg if mg>0 else 1.0
            }
    return mh, ma, fuerzas

def predecir_partido(home, away, mh, ma, fuerzas, max_goles=10):
    at_h = fuerzas.get(home,{}).get('ataque',1.0)
    df_h = fuerzas.get(home,{}).get('defensa',1.0)
    at_a = fuerzas.get(away,{}).get('ataque',1.0)
    df_a = fuerzas.get(away,{}).get('defensa',1.0)
    lh = mh * at_h * df_a
    la = ma * at_a * df_h
    prob = np.zeros((max_goles+1, max_goles+1))
    for i in range(max_goles+1):
        for j in range(max_goles+1):
            prob[i,j] = poisson.pmf(i,lh)*poisson.pmf(j,la)
    prob /= prob.sum()
    pl = prob[np.triu_indices_from(prob, k=1)].sum()
    pe = np.trace(prob).sum()
    pv = 1 - pl - pe
    tg = np.fromfunction(lambda i,j:i+j, (max_goles+1,max_goles+1), dtype=int)
    under = prob[tg < 2.5].sum()
    btts = prob[np.where((np.arange(max_goles+1)[:,None]>0) & (np.arange(max_goles+1)[None,:]>0))].sum()
    return {
        'lambda_home': lh, 'lambda_away': la,
        'prob_local': pl, 'prob_empate': pe, 'prob_visitante': pv,
        'prob_under25': under, 'prob_over25': 1-under,
        'prob_btts': btts,
    }

# ------------------------------------------------------------
# 4. INTERFAZ PRINCIPAL
# ------------------------------------------------------------
st.sidebar.header("📂 Fuentes de datos")
archivo_csv = st.sidebar.file_uploader("Subir CSV histórico", type=['csv'])

# API
st.sidebar.subheader("🌐 Football-Data.org")
api_key = st.sidebar.text_input("API Key", type="password")
liga = st.sidebar.selectbox("Liga",
    [('LaLiga','PD'),('Premier League','PL'),('Serie A','SA'),('Bundesliga','BL1'),
     ('Champions League','CL'),('Ligue 1','FL1')],
    format_func=lambda x: x[0])
season_hist = st.sidebar.selectbox("Temporada (histórico)",
    ['2024','2023','2022','2021','2020'], index=0)
season_prox = st.sidebar.selectbox("Temporada (próximos partidos)",
    ['2026','2025','2024'], index=0,
    help="Elige la temporada de la que quieres ver los partidos programados. Si no hay nada para 2026/27, prueba con 2025/26.")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Descargar histórico"):
        st.session_state.df_api = descargar_datos_finalizados(api_key, liga[1], season_hist)
        if not st.session_state.df_api.empty:
            st.sidebar.success(f"{len(st.session_state.df_api)} partidos")
        else:
            st.sidebar.warning("Sin datos")
with col2:
    if st.button("Cargar próximos"):
        st.session_state.df_prox = descargar_proximos_partidos(api_key, liga[1], season_prox)
        if not st.session_state.df_prox.empty:
            st.sidebar.success(f"{len(st.session_state.df_prox)} próximos partidos ({season_prox})")
        else:
            st.sidebar.warning(f"No hay próximos partidos en la temporada {season_prox}. Prueba con otra.")

if 'df_api' not in st.session_state: st.session_state.df_api = pd.DataFrame()
if 'df_prox' not in st.session_state: st.session_state.df_prox = pd.DataFrame()
if 'cuotas' not in st.session_state: st.session_state.cuotas = {}

# Plantilla CSV
st.sidebar.download_button("📥 Plantilla CSV", data=generar_plantilla(),
                           file_name="plantilla.csv", mime="text/csv")

# ------------------------------------------------------------
# INICIALIZACIÓN DE VARIABLES DEL MODELO (por defecto)
# ------------------------------------------------------------
media_home = 1.5
media_away = 1.2
fuerzas = {}

# ------------------------------------------------------------
# 5. DATOS COMBINADOS Y ENTRENAMIENTO
# ------------------------------------------------------------
df_csv = cargar_csv(archivo_csv) if archivo_csv else pd.DataFrame()
df_hist = combinar_datos(df_csv, st.session_state.df_api)

if not df_hist.empty:
    st.subheader("📋 Datos históricos")
    st.dataframe(df_hist.head())
    st.write(f"Total partidos: {len(df_hist)}")

    with st.spinner("Calculando estadísticas..."):
        df_hist = crear_features_rodantes(df_hist)
        media_home, media_away, fuerzas = entrenar_modelo_poisson(df_hist)   # aquí se actualizan

    # Evaluación rápida
    if st.sidebar.checkbox("Mostrar métricas"):
        if len(df_hist)>=5:
            corte = int(len(df_hist)*0.8)
            test = df_hist.iloc[corte:]
            yt, yp = [], []
            for _,r in test.iterrows():
                pred = predecir_partido(r['home_team'],r['away_team'],media_home,media_away,fuerzas)
                real = 1 if r['home_goals']>r['away_goals'] else (0 if r['home_goals']==r['away_goals'] else 2)
                pred_res = 1 if pred['prob_local']>max(pred['prob_empate'],pred['prob_visitante']) else (
                           0 if pred['prob_empate']>pred['prob_visitante'] else 2)
                yt.append(real); yp.append(pred_res)
            acc = accuracy_score(yt,yp)
            st.sidebar.metric("Precisión (1X2)", f"{acc:.1%}")
else:
    st.info("Carga datos o descarga el histórico para empezar.")

# ------------------------------------------------------------
# 6. PRÓXIMOS PARTIDOS (JORNADA)
# ------------------------------------------------------------
st.header("📅 Próximos partidos")
if not st.session_state.df_prox.empty:
    df_prox = st.session_state.df_prox.copy()
    
    # Aviso de diagnóstico
    st.info(f"Se han cargado {len(df_prox)} partidos. Calculando predicciones...")
    
    predicciones = []
    equipos_desconocidos = set()
    for _, row in df_prox.iterrows():
        # Si el equipo local o visitante no está en 'fuerzas', usamos fuerza neutral (1.0)
        # Aseguramos que existan claves; si no, se crean temporalmente
        for eq in [row['home_team'], row['away_team']]:
            if eq not in fuerzas:
                equipos_desconocidos.add(eq)
                fuerzas[eq] = {'ataque': 1.0, 'defensa': 1.0}  # fuerza neutra
        try:
            pred = predecir_partido(row['home_team'], row['away_team'], media_home, media_away, fuerzas)
        except Exception as e:
            # Si aun así falla, ponemos valores por defecto
            pred = {
                'prob_local': 0.33, 'prob_empate': 0.33, 'prob_visitante': 0.34,
                'prob_under25': 0.5, 'prob_over25': 0.5, 'prob_btts': 0.5,
                'lambda_home': 1.5, 'lambda_away': 1.2
            }
        predicciones.append(pred)
    
    if equipos_desconocidos:
        st.warning(f"⚠️ Equipos sin histórico reciente: {', '.join(sorted(equipos_desconocidos))}. Se usan parámetros neutros.")
    
    # Construir tabla (sin columnas editables, más sencilla para visualizar)
    tabla = pd.DataFrame({
        'Fecha': df_prox['date'].dt.strftime('%Y-%m-%d'),
        'Local': df_prox['home_team'],
        'Visitante': df_prox['away_team'],
        'Prob. Local': [f"{p.get('prob_local',0):.1%}" for p in predicciones],
        'Prob. Empate': [f"{p.get('prob_empate',0):.1%}" for p in predicciones],
        'Prob. Visit.': [f"{p.get('prob_visitante',0):.1%}" for p in predicciones],
        'Under 2.5': [f"{p.get('prob_under25',0):.1%}" for p in predicciones],
        'BTTS': [f"{p.get('prob_btts',0):.1%}" for p in predicciones],
    })
    
    st.subheader("🎲 Predicciones base (sin cuotas)")
    st.dataframe(tabla, use_container_width=True)
    
    # Sección de cuotas editables (opcional)
    with st.expander("💰 Añadir cuotas para value bets", expanded=False):
        st.markdown("Introduce las cuotas de tu casa de apuestas para cada partido:")
        cuotas_edit = []
        for idx, row in tabla.iterrows():
            partido_id = f"{row['Local']}_{row['Visitante']}"
            if partido_id not in st.session_state.cuotas:
                st.session_state.cuotas[partido_id] = {'cuota_local': 2.0, 'cuota_empate': 3.5, 'cuota_visitante': 4.0}
            c = st.session_state.cuotas[partido_id]
            col1, col2, col3, col4, col5, col6 = st.columns([2,2,1,1,1,1])
            col1.write(row['Fecha'])
            col2.write(f"{row['Local']} - {row['Visitante']}")
            cuota_local = col3.number_input('1', value=c['cuota_local'], step=0.05, key=f"cl_{partido_id}")
            cuota_empate = col4.number_input('X', value=c['cuota_empate'], step=0.05, key=f"ce_{partido_id}")
            cuota_visit = col5.number_input('2', value=c['cuota_visitante'], step=0.05, key=f"cv_{partido_id}")
            # Guardar
            st.session_state.cuotas[partido_id] = {'cuota_local': cuota_local, 'cuota_empate': cuota_empate, 'cuota_visitante': cuota_visit}
            # EV
            prob = predicciones[idx]
            ev_local = prob.get('prob_local',0)*cuota_local - 1
            ev_empate = prob.get('prob_empate',0)*cuota_empate - 1
            ev_visit = prob.get('prob_visitante',0)*cuota_visit - 1
            # Mostrar EV con iconos
            def icono(val):
                return '🟢' if val>0 else ('🔴' if val<-0.1 else '⚪')
            col6.markdown(f"{icono(ev_local)} L: {ev_local:+.1%}  \n{icono(ev_empate)} E: {ev_empate:+.1%}  \n{icono(ev_visit)} V: {ev_visit:+.1%}")
    
    # Exportar CSV
    csv_export = tabla.to_csv(index=False).encode('utf-8')
    st.download_button("📤 Descargar predicciones (CSV)", data=csv_export,
                       file_name="predicciones_jornada.csv", mime="text/csv")
else:
    st.warning("No hay próximos partidos cargados. Usa el botón 'Cargar próximos' en la barra lateral.")

# ------------------------------------------------------------
# 7. PREDICCIÓN INDIVIDUAL CON CUOTAS
# ------------------------------------------------------------
st.header("🔮 Predicción individual")
if not df_hist.empty:
    equipos = sorted(pd.concat([df_hist['home_team'], df_hist['away_team']]).unique())
else:
    if not st.session_state.df_prox.empty:
        equipos = sorted(pd.concat([st.session_state.df_prox['home_team'], st.session_state.df_prox['away_team']]).unique())
    else:
        equipos = []

if equipos:
    col_a, col_b = st.columns(2)
    with col_a:
        eq_local = st.selectbox("Local", equipos)
    with col_b:
        eq_visit = st.selectbox("Visitante", equipos)

    # Cuotas opcionales
    st.subheader("Cuotas de apuesta (opcional)")
    cl = st.number_input('Cuota local', value=2.0, step=0.05)
    ce = st.number_input('Cuota empate', value=3.5, step=0.05)
    cv = st.number_input('Cuota visitante', value=4.0, step=0.05)

    if st.button("Predecir partido", type="primary"):
        if eq_local == eq_visit:
            st.error("Equipos diferentes")
        else:
            pred = predecir_partido(eq_local, eq_visit, media_home, media_away, fuerzas)
            st.subheader(f"{eq_local} vs {eq_visit}")
            col1,col2,col3 = st.columns(3)
            col1.metric("xG Local", f"{pred['lambda_home']:.2f}")
            col2.metric("xG Visitante", f"{pred['lambda_away']:.2f}")
            col3.metric("Total xG", f"{pred['lambda_home']+pred['lambda_away']:.2f}")
            st.markdown("---")
            pcol1,pcol2,pcol3 = st.columns(3)
            pcol1.metric("🏠 Local", f"{pred['prob_local']:.1%}")
            pcol2.metric("🤝 Empate", f"{pred['prob_empate']:.1%}")
            pcol3.metric("🚩 Visitante", f"{pred['prob_visitante']:.1%}")
            st.markdown("---")
            m1,m2,m3 = st.columns(3)
            m1.metric("Under 2.5", f"{pred['prob_under25']:.1%}")
            m2.metric("Over 2.5", f"{pred['prob_over25']:.1%}")
            m3.metric("BTTS", f"{pred['prob_btts']:.1%}")
            
            # Value bet
            ev_local = pred['prob_local'] * cl - 1
            ev_empate = pred['prob_empate'] * ce - 1
            ev_visit = pred['prob_visitante'] * cv - 1
            st.markdown("---")
            st.subheader("💰 Análisis de valor (EV)")
            ev_col1, ev_col2, ev_col3 = st.columns(3)
            def show_ev(label, ev):
                color = "green" if ev > 0 else "red"
                st.markdown(f"**{label}:** :{color}[{ev:+.2%}]")
            with ev_col1:
                show_ev("Local", ev_local)
            with ev_col2:
                show_ev("Empate", ev_empate)
            with ev_col3:
                show_ev("Visitante", ev_visit)
            if max(ev_local, ev_empate, ev_visit) > 0:
                st.success("¡Existe una posible apuesta de valor! 🎉")
            else:
                st.info("No se detecta valor positivo según estas cuotas.")
else:
    st.info("No hay equipos disponibles. Carga datos históricos o próximos partidos.")