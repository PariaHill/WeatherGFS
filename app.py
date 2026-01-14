import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
import xarray as xr
import tempfile
import os
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 페이지 설정
st.set_page_config(page_title="Captain Park's Marine Forecast", layout="wide")

# ============================================================
# 2. 세션 상태 초기화
# ============================================================
if 'lat' not in st.session_state: st.session_state.lat = 31.8700
if 'lon' not in st.session_state: st.session_state.lon = 126.7700
if 'offset' not in st.session_state: st.session_state.offset = 9

MS_TO_KNOTS = 1.94384

# ============================================================
# 3. 유틸리티 함수
# ============================================================
def get_direction_text(deg):
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = int((deg + 11.25) / 22.5) % 16
    return directions[idx]

def get_arrow_html(deg, color="#007BFF"):
    rotate_deg = (deg + 180) % 360 
    return f'<span style="display:inline-block; transform:rotate({rotate_deg}deg); font-size:16px; color:{color}; margin-left:5px;">↑</span>'

def get_available_cycle():
    now_utc = datetime.now(timezone.utc)
    cycles = [18, 12, 6, 0]
    
    for days_ago in range(2):
        check_date = now_utc - timedelta(days=days_ago)
        date_str = check_date.strftime("%Y%m%d")
        
        for cycle in cycles:
            cycle_time = check_date.replace(hour=cycle, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            if cycle_time > now_utc:
                continue
            
            hours_since_cycle = (now_utc - cycle_time).total_seconds() / 3600
            if hours_since_cycle < 4:
                continue
            
            url = (f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
                   f"dir=%2Fgfs.{date_str}%2F{cycle:02d}%2Fatmos&"
                   f"file=gfs.t{cycle:02d}z.pgrb2.0p25.f000&"
                   f"var_PRMSL=on&lev_mean_sea_level=on&"
                   f"subregion=&toplat=32&leftlon=126&rightlon=127&bottomlat=31")
            
            try:
                resp = requests.head(url, timeout=10)
                if resp.status_code == 200:
                    return date_str, cycle, cycle_time
            except:
                continue
    
    return None, None, None

def build_subregion_params(lat, lon, margin=0.25):
    lat_min = math.floor((lat - margin) * 4) / 4
    lat_max = math.ceil((lat + margin) * 4) / 4
    lon_min = math.floor((lon - margin) * 4) / 4
    lon_max = math.ceil((lon + margin) * 4) / 4
    return f"subregion=&toplat={lat_max}&leftlon={lon_min}&rightlon={lon_max}&bottomlat={lat_min}"

def get_forecast_hours():
    return list(range(0, 169, 3))

def fetch_gfs_atmosphere(date_str, cycle, fhour, lat, lon):
    subregion = build_subregion_params(lat, lon)
    url = (f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
           f"dir=%2Fgfs.{date_str}%2F{cycle:02d}%2Fatmos&"
           f"file=gfs.t{cycle:02d}z.pgrb2.0p25.f{fhour:03d}&"
           f"var_PRMSL=on&var_GUST=on&"
           f"lev_mean_sea_level=on&lev_surface=on&"
           f"{subregion}")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 100:
            return resp.content
    except:
        pass
    return None

def fetch_gfswave(date_str, cycle, fhour, lat, lon):
    subregion = build_subregion_params(lat, lon)
    url = (f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl?"
           f"dir=%2Fgfs.{date_str}%2F{cycle:02d}%2Fwave%2Fgridded&"
           f"file=gfswave.t{cycle:02d}z.global.0p25.f{fhour:03d}.grib2&"
           f"var_WIND=on&var_WDIR=on&var_UGRD=on&var_VGRD=on&"
           f"var_HTSGW=on&var_DIRPW=on&var_PERPW=on&"
           f"var_SWELL=on&var_SWDIR=on&var_SWPER=on&"
           f"lev_surface=on&lev_1_in_sequence=on&"
           f"{subregion}")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 100:
            return resp.content
    except:
        pass
    return None

def parse_grib_data(grib_bytes, lat, lon):
    if grib_bytes is None or len(grib_bytes) < 100:
        return {}
    
    result = {}
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as f:
            f.write(grib_bytes)
            temp_path = f.name
        
        try:
            filter_configs = [
                {'typeOfLevel': 'surface'},
                {'typeOfLevel': 'meanSea'},
                {'typeOfLevel': 'orderedSequence'},
                {},
            ]
            
            for filter_keys in filter_configs:
                try:
                    if filter_keys:
                        ds = xr.open_dataset(temp_path, engine='cfgrib',
                                           backend_kwargs={'filter_by_keys': filter_keys, 'errors': 'ignore'})
                    else:
                        ds = xr.open_dataset(temp_path, engine='cfgrib',
                                           backend_kwargs={'errors': 'ignore'})
                except:
                    continue
                
                if ds is None:
                    continue
                    
                lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
                lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
                
                if lat_name not in ds.coords or lon_name not in ds.coords:
                    ds.close()
                    continue
                
                try:
                    point = ds.sel({lat_name: lat, lon_name: lon}, method='nearest')
                except:
                    ds.close()
                    continue
                
                var_mapping = {
                    'prmsl': 'pressure', 'gust': 'gust',
                    'wind': 'wind_speed', 'ws': 'wind_speed',
                    'wdir': 'wind_dir',
                    'u': 'wind_u', 'v': 'wind_v', 'u10': 'wind_u', 'v10': 'wind_v',
                    'htsgw': 'wave_height', 'swh': 'wave_height', 'shww': 'wave_height',
                    'dirpw': 'wave_dir', 'mwd': 'wave_dir',
                    'perpw': 'wave_period', 'mwp': 'wave_period',
                    'swell': 'swell_height', 'shts': 'swell_height',
                    'swdir': 'swell_dir', 'mdts': 'swell_dir',
                    'swper': 'swell_period', 'mpts': 'swell_period',
                }
                
                for var in ds.data_vars:
                    var_lower = var.lower()
                    if var_lower in var_mapping:
                        mapped_key = var_mapping[var_lower]
                        if mapped_key in result:
                            continue
                        try:
                            val = float(point[var].values)
                            if not np.isnan(val):
                                result[mapped_key] = val
                        except:
                            pass
                
                ds.close()
                
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
            
    except:
        pass
    
    return result

def fetch_single_forecast(args):
    date_str, cycle, cycle_time, fhour, lat, lon = args
    valid_time = cycle_time + timedelta(hours=fhour)
    
    row = {'valid_time': valid_time, 'fhour': fhour}
    
    atmos_data = fetch_gfs_atmosphere(date_str, cycle, fhour, lat, lon)
    atmos_parsed = parse_grib_data(atmos_data, lat, lon)
    
    wave_data = fetch_gfswave(date_str, cycle, fhour, lat, lon)
    wave_parsed = parse_grib_data(wave_data, lat, lon)
    
    row.update(atmos_parsed)
    row.update(wave_parsed)
    
    return row

def fetch_all_forecasts_parallel(date_str, cycle, cycle_time, lat, lon, progress_bar, status_text):
    forecast_hours = get_forecast_hours()
    all_data = []
    total = len(forecast_hours)
    completed = 0
    
    args_list = [(date_str, cycle, cycle_time, fhour, lat, lon) for fhour in forecast_hours]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_single_forecast, args): args[3] for args in args_list}
        
        for future in as_completed(futures):
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"데이터 수신 중... ({completed}/{total})")
            
            try:
                row = future.result()
                if len(row) > 2:
                    all_data.append(row)
            except:
                pass
    
    all_data.sort(key=lambda x: x['fhour'])
    return all_data, len(all_data)

# UI 상단
st.title("⚓ 해상 기상 예보 시스템")
st.caption("Data Source: NOAA GFS & GFS-Wave (0.25° Resolution)")

with st.container():
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1: 
        st.session_state.lat = st.number_input("위도 (Lat)", value=st.session_state.lat, format="%.4f")
    with col2: 
        st.session_state.lon = st.number_input("경도 (Lon)", value=st.session_state.lon, format="%.4f")
    with col3:
        opts = list(range(13, -13, -1))
        st.session_state.offset = st.selectbox("시간대 설정 (UTC Offset)", options=opts, index=opts.index(st.session_state.offset))
    with col4:
        st.write(" ")
        fetch_btn = st.button("데이터 수신 시작")

# ============================================================
# 5. 데이터 수집 및 표시
# ============================================================
if fetch_btn or 'data_loaded' in st.session_state:
    
    with st.spinner("최신 GFS Cycle 탐지 중..."):
        date_str, cycle, cycle_time = get_available_cycle()
    
    if date_str is None:
        st.error("❌ 사용 가능한 GFS 데이터를 찾을 수 없습니다.")
    else:
        st.success(f"✅ GFS Cycle: {date_str} {cycle:02d}Z (UTC)")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_data, successful = fetch_all_forecasts_parallel(
            date_str, cycle, cycle_time, 
            st.session_state.lat, st.session_state.lon,
            progress_bar, status_text
        )
        
        progress_bar.empty()
        status_text.empty()
        
        if successful == 0:
            st.error("❌ 데이터를 가져오지 못했습니다.")
        else:
            st.session_state.data_loaded = True
            st.info(f"📊 {successful}개 시간대 데이터 수신 완료")
            
            df = pd.DataFrame(all_data)
            
            time_col = f"Time (UTC{st.session_state.offset:+})"
            df['local_time'] = df['valid_time'].apply(
                lambda x: (x + timedelta(hours=st.session_state.offset)).replace(tzinfo=None)
            )
            df[time_col] = df['local_time'].dt.strftime('%Y-%m-%d %H:%M')
            
            # 데이터 변환
            df['Pressure(hPa)'] = (df['pressure'] / 100).round(1) if 'pressure' in df.columns else np.nan
            
            if 'wind_speed' in df.columns:
                df['Wind Speed(kts)'] = (df['wind_speed'] * MS_TO_KNOTS).round(1)
            elif 'wind_u' in df.columns and 'wind_v' in df.columns:
                df['Wind Speed(kts)'] = (np.sqrt(df['wind_u']**2 + df['wind_v']**2) * MS_TO_KNOTS).round(1)
            else:
                df['Wind Speed(kts)'] = np.nan
            
            if 'wind_dir' in df.columns:
                df['Wind_Deg'] = df['wind_dir']
            elif 'wind_u' in df.columns and 'wind_v' in df.columns:
                df['Wind_Deg'] = (np.degrees(np.arctan2(df['wind_u'], df['wind_v'])) + 180) % 360
            else:
                df['Wind_Deg'] = np.nan
            
            df['Wind Direction'] = df.apply(
                lambda r: f"{r['Wind_Deg']:.1f}° {get_direction_text(r['Wind_Deg'])} {get_arrow_html(r['Wind_Deg'])}" 
                if pd.notna(r['Wind_Deg']) else '-', axis=1)
            
            df['Gust(kts)'] = (df['gust'] * MS_TO_KNOTS).round(1) if 'gust' in df.columns else np.nan
            
            df['Waves(m)'] = df['wave_height'].round(1) if 'wave_height' in df.columns else np.nan
            df['Max Waves(m)'] = (df['wave_height'] * 1.6).round(1) if 'wave_height' in df.columns else np.nan
            
            if 'wave_dir' in df.columns:
                df['Wave_Deg'] = df['wave_dir']
                df['Wave Direction'] = df.apply(
                    lambda r: f"{r['Wave_Deg']:.1f}° {get_direction_text(r['Wave_Deg'])} {get_arrow_html(r['Wave_Deg'], '#28A745')}" 
                    if pd.notna(r['Wave_Deg']) else '-', axis=1)
            else:
                df['Wave_Deg'] = np.nan
                df['Wave Direction'] = '-'
            
            df['Wave Period(s)'] = df['wave_period'].round(1) if 'wave_period' in df.columns else np.nan
            df['Swell(m)'] = df['swell_height'].round(1) if 'swell_height' in df.columns else np.nan
            
            if 'swell_dir' in df.columns:
                df['Swell_Deg'] = df['swell_dir']
                df['Swell Direction'] = df.apply(
                    lambda r: f"{r['Swell_Deg']:.1f}° {get_direction_text(r['Swell_Deg'])} {get_arrow_html(r['Swell_Deg'], '#9932CC')}" 
                    if pd.notna(r['Swell_Deg']) else '-', axis=1)
            else:
                df['Swell_Deg'] = np.nan
                df['Swell Direction'] = '-'
            
            df['Swell Period(s)'] = df['swell_period'].round(1) if 'swell_period' in df.columns else np.nan
            
            tab1, tab2 = st.tabs(["📊 데이터 테이블", "📈 시각화 그래프"])
            
            with tab1:
                st.subheader("데이터 테이블")
                
                display_cols = [time_col, "Pressure(hPa)", "Wind Direction", "Wind Speed(kts)", "Gust(kts)", 
                               "Wave Direction", "Waves(m)", "Max Waves(m)", "Wave Period(s)",
                               "Swell Direction", "Swell(m)", "Swell Period(s)"]
                display_cols = [c for c in display_cols if c in df.columns]
                
                st.write(df[display_cols].to_html(escape=False, index=False, justify='center'), unsafe_allow_html=True)
            
            with tab2:
                st.subheader("그래프 분석")
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15,
                                   subplot_titles=("Wind Speed & Direction (kts)", "Wave Height & Direction (m)"))
                
                graph_time = df['local_time']
                
                # 바람 그래프
                if 'Wind Speed(kts)' in df.columns:
                    fig.add_trace(go.Scatter(x=graph_time, y=df['Wind Speed(kts)'], name="Wind", 
                                            line=dict(color='firebrick')), row=1, col=1)
                if 'Gust(kts)' in df.columns:
                    fig.add_trace(go.Scatter(x=graph_time, y=df['Gust(kts)'], name="Gust", 
                                            line=dict(color='orange', dash='dot'), fill='tonexty'), row=1, col=1)
                
                # 바람 방향 화살표
                if 'Wind_Deg' in df.columns and 'Wind Speed(kts)' in df.columns:
                    wind_max = df['Wind Speed(kts)'].max()
                    if pd.notna(wind_max) and wind_max > 0:
                        for i in range(len(df)):
                            if pd.notna(df['Wind_Deg'].iloc[i]):
                                fig.add_annotation(dict(x=graph_time.iloc[i], y=wind_max * 1.2, text="↑", 
                                                       showarrow=False, font=dict(size=12, color="#007BFF"), 
                                                       textangle=df['Wind_Deg'].iloc[i]+180, xref="x1", yref="y1"))
                
                # 파도 그래프
                if 'Waves(m)' in df.columns:
                    fig.add_trace(go.Scatter(x=graph_time, y=df['Waves(m)'], name="Waves", 
                                            line=dict(color='royalblue', width=3)), row=2, col=1)
                if 'Max Waves(m)' in df.columns:
                    fig.add_trace(go.Scatter(x=graph_time, y=df['Max Waves(m)'], name="Max Waves", 
                                            line=dict(color='navy', width=1, dash='dot')), row=2, col=1)
                if 'Swell(m)' in df.columns:
                    fig.add_trace(go.Scatter(x=graph_time, y=df['Swell(m)'], name="Swell", 
                                            line=dict(color='skyblue', dash='dash')), row=2, col=1)
                
                # 파도 방향 화살표
                if 'Wave_Deg' in df.columns and 'Max Waves(m)' in df.columns:
                    y_max_wave = df['Max Waves(m)'].max()
                    if pd.notna(y_max_wave) and y_max_wave > 0:
                        for i in range(len(df)):
                            if pd.notna(df['Wave_Deg'].iloc[i]):
                                fig.add_annotation(dict(x=graph_time.iloc[i], y=y_max_wave * 1.2, text="↑", 
                                                       showarrow=False, font=dict(size=12, color="#28A745"), 
                                                       textangle=df['Wave_Deg'].iloc[i]+180, xref="x2", yref="y2"))
                
                # 날짜 구분
                for i, day in enumerate(graph_time.dt.date.unique()):
                    if i % 2 == 0:
                        fig.add_vrect(x0=str(day), x1=str(day + timedelta(days=1)), 
                                     fillcolor="gray", opacity=0.07, layer="below", line_width=0)
                
                fig.update_layout(height=700, hovermode="x unified", legend=dict(orientation="h", y=1.05),
                                 paper_bgcolor='white', plot_bgcolor='white')
                fig.update_xaxes(tickformat="%d일\n%H:%M", dtick=21600000, showgrid=True, row=1, col=1)
                fig.update_xaxes(tickformat="%d일\n%H:%M", dtick=21600000, showgrid=True, row=2, col=1)
                
                if 'Wind Speed(kts)' in df.columns:
                    wind_max = df['Wind Speed(kts)'].max()
                    if pd.notna(wind_max) and wind_max > 0:
                        fig.update_yaxes(range=[0, wind_max * 1.4], row=1, col=1)
                
                if 'Max Waves(m)' in df.columns:
                    wave_max = df['Max Waves(m)'].max()
                    if pd.notna(wave_max) and wave_max > 0:
                        fig.update_yaxes(range=[0, wave_max * 1.4], row=2, col=1)
                
                st.plotly_chart(fig, key="main_chart")
