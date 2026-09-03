"""Creates combined glacier-velocity and CARRA climate visualizations.

The script compares velocity at P_CARRA with 2 m air temperature, 2 m relative humidity
and total precipitation for Arnesenbreen, Kvalbreen and Scheelebreen."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
root_dir = Path('D:\\Studia\\praca_magisterska\\predkosci_svalbard_2018-2022')
velocity_dir = root_dir / 'wyniki_2018-2025_path14' / 'CARRA_points_velocity'
climate_dir = root_dir / 'wyniki_CARRA'
output_dir = root_dir / 'wyniki_2018-2025_path14' / 'Figure10_P_CARRA_temp_RH_precip_jeden_wykres'
output_dir.mkdir(parents=True, exist_ok=True)
glaciers = ['Arnesenbreen', 'Kvalbreen', 'Scheelebreen']
analysis_start = pd.Timestamp('2018-01-01')
analysis_end = pd.Timestamp('2025-12-31 23:59:59')
smooth_days = 30
summer_start_month = 6
summer_end_month = 9
BAND_P = (0.0, 16.0)
BAND_RH = (24.0, 40.0)
BAND_T = (48.0, 64.0)
BAND_V = (72.0, 88.0)
TITLE_Y_P = 18.3
TITLE_Y_RH = 42.3
TITLE_Y_T = 66.3
TITLE_Y_V = 90.3
SEP_Y_1 = 20.0
SEP_Y_2 = 44.0
SEP_Y_3 = 68.0
COLOR_V_SCAT = 'black'
COLOR_V_LINE = '#1f77b4'
COLOR_T_SCAT = '#ff7f0e'
COLOR_T_LINE = '#d62728'
COLOR_RH_SCAT = '#2ca02c'
COLOR_RH_LINE = '#1b9e77'
COLOR_P_BAR = '#6a5acd'
COLOR_P_LINE = '#4b3f9f'
COLOR_SUMMER = '#d9d9d9'

def load_velocity(glacier_name: str) -> pd.DataFrame:
    path = velocity_dir / f'{glacier_name}_point_CARRA_velocity_2018-2025.csv'
    if not path.exists():
        raise FileNotFoundError(f'Brak pliku prędkości: {path}')
    df = pd.read_csv(path)
    required = ['date_start', 'date_end', 'date', 'velocity_m_y']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{path.name}: brakuje kolumn {missing}')
    for col in ['date_start', 'date_end', 'date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['velocity_m_y'] = pd.to_numeric(df['velocity_m_y'], errors='coerce')
    df = df.dropna(subset=['date', 'date_start', 'date_end']).sort_values('date').reset_index(drop=True)
    df = df.loc[(df['date_start'] >= analysis_start) & (df['date_end'] <= analysis_end)].copy()
    return df

def load_climate(glacier_name: str) -> pd.DataFrame:
    preferred_path = climate_dir / f'{glacier_name}_CARRA_temp_RH_precip_daily.csv'
    fallback_path = climate_dir / f'{glacier_name}_CARRA_temp_RH_daily.csv'
    if preferred_path.exists():
        path = preferred_path
    elif fallback_path.exists():
        path = fallback_path
    else:
        raise FileNotFoundError(f'Brak pliku klimatycznego.\nSprawdzono:\n{preferred_path}\n{fallback_path}')
    df = pd.read_csv(path)
    required = ['datetime', 'temperature_2m_C', 'relative_humidity_2m_percent', 'total_precipitation_daily_mm']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{path.name}: brakuje kolumn {missing}\nUruchom najpierw nowy skrypt ekstrakcji CARRA z temperaturą, RH i opadem.')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    for col in ['temperature_2m_C', 'relative_humidity_2m_percent', 'total_precipitation_daily_mm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').drop_duplicates(subset=['datetime'], keep='first').reset_index(drop=True)
    df = df.loc[(df['datetime'] >= analysis_start) & (df['datetime'] <= analysis_end)].copy()
    return df

def rolling_mean_time(df: pd.DataFrame, date_col: str, value_col: str, days: int=30) -> pd.Series:
    temp = df[[date_col, value_col]].dropna().sort_values(date_col).set_index(date_col)
    if temp.empty:
        return pd.Series(dtype=float)
    return temp[value_col].rolling(f'{days}D', center=True, min_periods=max(5, days // 4)).mean()

def rolling_sum_time(df: pd.DataFrame, date_col: str, value_col: str, days: int=30) -> pd.Series:
    temp = df[[date_col, value_col]].dropna().sort_values(date_col).set_index(date_col)
    if temp.empty:
        return pd.Series(dtype=float)
    return temp[value_col].rolling(f'{days}D', center=True, min_periods=max(5, days // 4)).sum()

def scale_to_band(values, src_min, src_max, band_min, band_max):
    values = np.asarray(values, dtype=float)
    if np.isclose(src_min, src_max):
        return np.full_like(values, (band_min + band_max) / 2.0)
    return band_min + (values - src_min) * (band_max - band_min) / (src_max - src_min)

def add_summer_shading(ax, start_date, end_date):
    years = range(start_date.year, end_date.year + 1)
    for year in years:
        s = pd.Timestamp(year=year, month=summer_start_month, day=1)
        e = pd.Timestamp(year=year, month=summer_end_month, day=30)
        if e < start_date or s > end_date:
            continue
        ax.axvspan(max(s, start_date), min(e, end_date), color=COLOR_SUMMER, alpha=0.55, zorder=0)

def format_x_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.tick_params(axis='x', which='major', labelsize=9)
    ax.tick_params(axis='x', which='minor', length=3)

def compute_ranges(velocity: pd.DataFrame, climate: pd.DataFrame):
    vel = velocity['velocity_m_y'].dropna()
    temp = climate['temperature_2m_C'].dropna()
    rh = climate['relative_humidity_2m_percent'].dropna()
    precip = climate['total_precipitation_daily_mm'].dropna()
    if vel.empty:
        vmin, vmax = (0.0, 1.0)
    else:
        vmin = float(np.floor(vel.min() / 100.0) * 100.0)
        vmax = float(np.ceil(vel.max() / 100.0) * 100.0)
        if np.isclose(vmin, vmax):
            vmax = vmin + 100.0
    if temp.empty:
        tmin, tmax = (-30.0, 10.0)
    else:
        tmin = float(np.floor(temp.min() / 5.0) * 5.0)
        tmax = float(np.ceil(temp.max() / 5.0) * 5.0)
        if np.isclose(tmin, tmax):
            tmax = tmin + 5.0
    rmin, rmax = (0.0, 100.0)
    pmin = 0.0
    if precip.empty:
        pmax = 1.0
    else:
        pmax = float(np.nanpercentile(precip, 99))
        if not np.isfinite(pmax) or pmax <= 0:
            pmax = float(precip.max())
        if not np.isfinite(pmax) or pmax <= 0:
            pmax = 1.0
        if pmax <= 5:
            step = 1.0
        elif pmax <= 20:
            step = 5.0
        elif pmax <= 50:
            step = 10.0
        else:
            step = 20.0
        pmax = float(np.ceil(pmax / step) * step)
    return ((vmin, vmax), (tmin, tmax), (rmin, rmax), (pmin, pmax))

def add_band_guides(ax, ranges, show_right_axis=False):
    (vmin, vmax), (tmin, tmax), (rmin, rmax), (pmin, pmax) = ranges
    for y in [SEP_Y_1, SEP_Y_2, SEP_Y_3]:
        ax.axhline(y, color='0.75', linewidth=0.8, zorder=0)
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    x_label = 0.01
    ax.text(x_label, TITLE_Y_V, 'Prędkość lodu [m/rok]', transform=trans, fontsize=10, fontweight='normal', va='top', ha='left', clip_on=False, bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.2))
    ax.text(x_label, TITLE_Y_T, 'Temperatura powietrza 2 m [°C]', transform=trans, fontsize=10, fontweight='normal', color=COLOR_T_LINE, va='top', ha='left', clip_on=False, bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.2))
    ax.text(x_label, TITLE_Y_RH, 'Wilgotność względna 2 m [%]', transform=trans, fontsize=10, fontweight='normal', color=COLOR_RH_LINE, va='top', ha='left', clip_on=False, bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.2))
    ax.text(x_label, TITLE_Y_P, 'Opad dobowy [mm]', transform=trans, fontsize=10, fontweight='normal', color=COLOR_P_LINE, va='top', ha='left', clip_on=False, bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.2))
    x_tick_1 = -0.03
    x_tick_2 = -0.004
    x_text = -0.034

    def draw_left_ticks(values, band, color='black', fmt='{:.0f}'):
        for frac, value in zip([0.0, 0.5, 1.0], values):
            y = band[0] + frac * (band[1] - band[0])
            ax.plot([x_tick_1, x_tick_2], [y, y], transform=trans, color=color, linewidth=0.8, clip_on=False)
            ax.text(x_text, y, fmt.format(value), transform=trans, ha='right', va='center', fontsize=8, color=color, clip_on=False)
    draw_left_ticks([vmin, (vmin + vmax) / 2.0, vmax], BAND_V, color='black', fmt='{:.0f}')
    draw_left_ticks([tmin, (tmin + tmax) / 2.0, tmax], BAND_T, color=COLOR_T_LINE, fmt='{:.1f}')
    draw_left_ticks([rmin, (rmin + rmax) / 2.0, rmax], BAND_RH, color=COLOR_RH_LINE, fmt='{:.0f}')
    draw_left_ticks([pmin, (pmin + pmax) / 2.0, pmax], BAND_P, color=COLOR_P_LINE, fmt='{:.1f}')

def annotate_annual_statistics(ax, climate: pd.DataFrame, temp_range, rh_range, precip_range):
    climate = climate.copy()
    climate['year'] = climate['datetime'].dt.year
    grouped = climate.groupby('year').agg(T_mean=('temperature_2m_C', 'mean'), RH_mean=('relative_humidity_2m_percent', 'mean'), P_sum=('total_precipitation_daily_mm', 'sum')).reset_index()
    for _, row in grouped.iterrows():
        year = int(row['year'])
        x = pd.Timestamp(year=year, month=7, day=15)
        if pd.notna(row['T_mean']):
            y_t = scale_to_band([row['T_mean']], temp_range[0], temp_range[1], BAND_T[0], BAND_T[1])[0]
            ax.text(x, y_t + 0.7, f"{row['T_mean']:.1f}", color=COLOR_T_LINE, fontsize=7.5, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor=COLOR_T_LINE, alpha=0.75))
        if pd.notna(row['RH_mean']):
            y_rh = scale_to_band([row['RH_mean']], rh_range[0], rh_range[1], BAND_RH[0], BAND_RH[1])[0]
            ax.text(x, y_rh - 0.7, f"{row['RH_mean']:.0f}", color=COLOR_RH_LINE, fontsize=7.5, ha='center', va='top', bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor=COLOR_RH_LINE, alpha=0.75))
        if pd.notna(row['P_sum']):
            ax.text(x, BAND_P[1] - 0.9, f"Σ {row['P_sum']:.0f}", color=COLOR_P_LINE, fontsize=7.2, ha='center', va='top', bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor=COLOR_P_LINE, alpha=0.75))

def plot_precipitation(ax, climate, pmin, pmax, precip_roll):
    precip_valid = climate.dropna(subset=['total_precipitation_daily_mm']).copy()
    if precip_valid.empty:
        return
    precip_for_plot = np.clip(precip_valid['total_precipitation_daily_mm'].to_numpy(dtype=float), pmin, pmax)
    y_precip = scale_to_band(precip_for_plot, pmin, pmax, BAND_P[0], BAND_P[1])
    bar_height = y_precip - BAND_P[0]
    ax.bar(precip_valid['datetime'], bar_height, bottom=BAND_P[0], width=1.0, color=COLOR_P_BAR, alpha=0.45, linewidth=0, zorder=2, label='Opad dobowy')
    if not precip_roll.empty:
        roll_values = precip_roll.dropna()
        if not roll_values.empty:
            roll_max = float(np.nanpercentile(roll_values.values, 99))
            if not np.isfinite(roll_max) or roll_max <= 0:
                roll_max = float(roll_values.max())
            if np.isfinite(roll_max) and roll_max > 0:
                roll_plot = np.clip(roll_values.values, 0.0, roll_max)
                ax.plot(roll_values.index, scale_to_band(roll_plot, 0.0, roll_max, BAND_P[0], BAND_P[1]), linewidth=1.5, color=COLOR_P_LINE, zorder=5, label=f'Opad — suma krocząca {smooth_days} dni')

def plot_one_glacier(glacier_name: str, velocity: pd.DataFrame, climate: pd.DataFrame):
    ranges = compute_ranges(velocity, climate)
    (vmin, vmax), (tmin, tmax), (rmin, rmax), (pmin, pmax) = ranges
    temp_roll = rolling_mean_time(climate, 'datetime', 'temperature_2m_C', smooth_days)
    rh_roll = rolling_mean_time(climate, 'datetime', 'relative_humidity_2m_percent', smooth_days)
    vel_roll = rolling_mean_time(velocity, 'date', 'velocity_m_y', smooth_days)
    precip_roll = rolling_sum_time(climate, 'datetime', 'total_precipitation_daily_mm', smooth_days)
    fig, ax = plt.subplots(figsize=(15, 7.6))
    start = max(analysis_start, min(velocity['date'].min(), climate['datetime'].min()))
    end = min(analysis_end, max(velocity['date'].max(), climate['datetime'].max()))
    add_summer_shading(ax, start, end)
    vel_valid = velocity.dropna(subset=['velocity_m_y'])
    y_vel = scale_to_band(vel_valid['velocity_m_y'], vmin, vmax, BAND_V[0], BAND_V[1])
    ax.scatter(vel_valid['date'], y_vel, s=14, color=COLOR_V_SCAT, alpha=0.85, zorder=3, label='Prędkość w P_CARRA')
    if not vel_roll.empty:
        ax.plot(vel_roll.index, scale_to_band(vel_roll.values, vmin, vmax, BAND_V[0], BAND_V[1]), linewidth=1.4, color=COLOR_V_LINE, zorder=4, label=f'Prędkość — średnia krocząca {smooth_days} dni')
    temp_valid = climate.dropna(subset=['temperature_2m_C'])
    y_temp = scale_to_band(temp_valid['temperature_2m_C'], tmin, tmax, BAND_T[0], BAND_T[1])
    ax.scatter(temp_valid['datetime'], y_temp, s=6, color=COLOR_T_SCAT, alpha=0.45, zorder=2, label='Temperatura dzienna')
    if not temp_roll.empty:
        ax.plot(temp_roll.index, scale_to_band(temp_roll.values, tmin, tmax, BAND_T[0], BAND_T[1]), linewidth=2.0, color=COLOR_T_LINE, zorder=5, label=f'Temperatura — średnia krocząca {smooth_days} dni')
    rh_valid = climate.dropna(subset=['relative_humidity_2m_percent'])
    y_rh = scale_to_band(rh_valid['relative_humidity_2m_percent'], rmin, rmax, BAND_RH[0], BAND_RH[1])
    ax.scatter(rh_valid['datetime'], y_rh, s=6, color=COLOR_RH_SCAT, alpha=0.35, zorder=1, label='RH dzienna')
    if not rh_roll.empty:
        ax.plot(rh_roll.index, scale_to_band(rh_roll.values, rmin, rmax, BAND_RH[0], BAND_RH[1]), linewidth=2.0, color=COLOR_RH_LINE, zorder=5, label=f'RH — średnia krocząca {smooth_days} dni')
    plot_precipitation(ax, climate, pmin, pmax, precip_roll)
    add_band_guides(ax, ranges, show_right_axis=False)
    annotate_annual_statistics(ax, climate, (tmin, tmax), (rmin, rmax), (pmin, pmax))
    ax.set_xlim(start - pd.Timedelta(days=100), end + pd.Timedelta(days=40))
    ax.set_ylim(-2, 94)
    ax.set_yticks([])
    ax.set_xlabel('Czas', labelpad=24)
    ax.set_title(f'{glacier_name} — prędkość w P_CARRA w relacji do temperatury, wilgotności względnej i opadu', fontsize=13, fontweight='normal')
    format_x_axis(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.145), borderaxespad=0.0, fontsize=8, framealpha=0.9, ncol=4)
    legend_handles = getattr(legend, 'legend_handles', None)
    if legend_handles is None:
        legend_handles = getattr(legend, 'legendHandles', [])
    for lh in legend_handles:
        try:
            lh.set_alpha(1)
        except Exception:
            pass
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.25, top=0.92)
    out_path = output_dir / f'{glacier_name}_P_CARRA_temp_RH_precip_jeden_wykres.jpg'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Zapisano: {out_path}')

def plot_combined(glacier_data: dict):
    fig, axes = plt.subplots(1, len(glaciers), figsize=(19, 7), sharey=True)
    if len(glaciers) == 1:
        axes = [axes]
    for ax, glacier_name in zip(axes, glaciers):
        velocity = glacier_data[glacier_name]['velocity']
        climate = glacier_data[glacier_name]['climate']
        ranges = compute_ranges(velocity, climate)
        (vmin, vmax), (tmin, tmax), (rmin, rmax), (pmin, pmax) = ranges
        temp_roll = rolling_mean_time(climate, 'datetime', 'temperature_2m_C', smooth_days)
        rh_roll = rolling_mean_time(climate, 'datetime', 'relative_humidity_2m_percent', smooth_days)
        vel_roll = rolling_mean_time(velocity, 'date', 'velocity_m_y', smooth_days)
        precip_roll = rolling_sum_time(climate, 'datetime', 'total_precipitation_daily_mm', smooth_days)
        start = max(analysis_start, min(velocity['date'].min(), climate['datetime'].min()))
        end = min(analysis_end, max(velocity['date'].max(), climate['datetime'].max()))
        add_summer_shading(ax, start, end)
        vel_valid = velocity.dropna(subset=['velocity_m_y'])
        temp_valid = climate.dropna(subset=['temperature_2m_C'])
        rh_valid = climate.dropna(subset=['relative_humidity_2m_percent'])
        ax.scatter(vel_valid['date'], scale_to_band(vel_valid['velocity_m_y'], vmin, vmax, BAND_V[0], BAND_V[1]), s=10, color=COLOR_V_SCAT, alpha=0.82)
        ax.scatter(temp_valid['datetime'], scale_to_band(temp_valid['temperature_2m_C'], tmin, tmax, BAND_T[0], BAND_T[1]), s=4, color=COLOR_T_SCAT, alpha=0.4)
        ax.scatter(rh_valid['datetime'], scale_to_band(rh_valid['relative_humidity_2m_percent'], rmin, rmax, BAND_RH[0], BAND_RH[1]), s=4, color=COLOR_RH_SCAT, alpha=0.28)
        plot_precipitation(ax, climate, pmin, pmax, precip_roll)
        if not vel_roll.empty:
            ax.plot(vel_roll.index, scale_to_band(vel_roll.values, vmin, vmax, BAND_V[0], BAND_V[1]), linewidth=1.2, color=COLOR_V_LINE)
        if not temp_roll.empty:
            ax.plot(temp_roll.index, scale_to_band(temp_roll.values, tmin, tmax, BAND_T[0], BAND_T[1]), linewidth=1.6, color=COLOR_T_LINE)
        if not rh_roll.empty:
            ax.plot(rh_roll.index, scale_to_band(rh_roll.values, rmin, rmax, BAND_RH[0], BAND_RH[1]), linewidth=1.6, color=COLOR_RH_LINE)
        ax.set_xlim(start - pd.Timedelta(days=70), end + pd.Timedelta(days=30))
        ax.set_ylim(-2, 94)
        ax.set_yticks([])
        ax.set_title(glacier_name, fontsize=11, fontweight='normal')
        ax.set_xlabel('Czas', labelpad=24)
        format_x_axis(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
    first_glacier = glaciers[0]
    first_ranges = compute_ranges(glacier_data[first_glacier]['velocity'], glacier_data[first_glacier]['climate'])
    add_band_guides(axes[0], first_ranges, show_right_axis=False)
    handles = [plt.Line2D([0], [0], marker='o', linestyle='None', color=COLOR_V_SCAT, markersize=5, label='Prędkość w P_CARRA'), plt.Line2D([0], [0], marker='o', linestyle='None', color=COLOR_T_SCAT, markersize=5, label='Temperatura dzienna'), plt.Line2D([0], [0], marker='o', linestyle='None', color=COLOR_RH_SCAT, markersize=5, label='RH dzienna'), plt.Rectangle((0, 0), 1, 1, color=COLOR_P_BAR, alpha=0.45, label='Opad dobowy'), plt.Line2D([0], [0], color=COLOR_V_LINE, linewidth=1.5, label=f'Prędkość — średnia {smooth_days} dni'), plt.Line2D([0], [0], color=COLOR_T_LINE, linewidth=1.8, label=f'Temperatura — średnia {smooth_days} dni'), plt.Line2D([0], [0], color=COLOR_RH_LINE, linewidth=1.8, label=f'RH — średnia {smooth_days} dni'), plt.Line2D([0], [0], color=COLOR_P_LINE, linewidth=1.5, label=f'Opad — suma krocząca {smooth_days} dni')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, framealpha=0.95, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle('Prędkość w punkcie P_CARRA w relacji do temperatury, wilgotności względnej i opadu CARRA', fontsize=13, fontweight='normal', y=0.98)
    fig.subplots_adjust(left=0.13, right=0.955, bottom=0.22, top=0.9, wspace=0.08)
    out_path = output_dir / 'ALL_GLACIERS_P_CARRA_temp_RH_precip_jeden_wykres.jpg'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Zapisano: {out_path}')
if __name__ == '__main__':
    glacier_data = {}
    for glacier_name in glaciers:
        print('=' * 70)
        print(glacier_name)
        print('=' * 70)
        velocity = load_velocity(glacier_name)
        climate = load_climate(glacier_name)
        print('Zakres opadu dobowego [mm]:', climate['total_precipitation_daily_mm'].min(), '-', climate['total_precipitation_daily_mm'].max())
        glacier_data[glacier_name] = {'velocity': velocity, 'climate': climate}
        plot_one_glacier(glacier_name, velocity, climate)
    plot_combined(glacier_data)
    print('\nGotowe.')
    print(f'Wyniki zapisano w: {output_dir}')
