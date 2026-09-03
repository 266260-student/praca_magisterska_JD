"""Creates dual-axis plots comparing glacier velocity with selected CARRA variables.

For each glacier, the script can plot velocity against 2 m relative humidity and
2 m air temperature, with rolling means used only for visualization."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
root_dir = Path('D:\\Studia\\praca_magisterska\\predkosci_svalbard_2018-2022')
input_dir = root_dir / 'wyniki_2018-2025_path14' / 'CARRA_velocity_temp_RH'
output_dir = input_dir / 'figures_dual_axis'
output_dir.mkdir(parents=True, exist_ok=True)
glaciers = ['Arnesenbreen', 'Kvalbreen', 'Scheelebreen']
make_RH_plot = True
make_TEMP_plot = True
smooth_window = '60D'
min_periods = 3
shade_periods = True
shade_start_month = 6
shade_start_day = 1
shade_end_month = 9
shade_end_day = 30

def rolling_time_mean(df, value_column, window):
    """
    Średnia krocząca po czasie.

    Nie interpoluje brakujących wartości.
    Nie tworzy nowych obserwacji.
    Wygładzenie służy wyłącznie do wizualizacji.
    """
    temp = df[['date', value_column]].dropna().sort_values('date').set_index('date')
    if temp.empty:
        return pd.Series(dtype=float)
    smooth = temp[value_column].rolling(window=window, center=True, min_periods=min_periods).mean()
    return smooth

def add_background_periods(ax, start_date, end_date):
    """
    Dodaje subtelne pionowe pasy dla wskazanego okresu
    każdego roku.

    Jest to wyłącznie element graficzny.
    """
    if not shade_periods:
        return
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    for year in range(start_year, end_year + 1):
        period_start = pd.Timestamp(year=year, month=shade_start_month, day=shade_start_day)
        period_end = pd.Timestamp(year=year, month=shade_end_month, day=shade_end_day)
        ax.axvspan(period_start, period_end, alpha=0.1, zorder=0)

def format_time_axis(ax, start_date, end_date):
    """
    Główne etykiety = lata.
    Drobniejsze podziały = styczeń, maj, wrzesień.
    """
    ax.set_xlim(start_date, end_date)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 5, 9]))
    ax.tick_params(axis='x', which='major', pad=8)
    ax.grid(axis='y', alpha=0.25)

def make_dual_axis_plot(df, glacier_name, climate_column, climate_label, climate_short_name, filename_suffix):
    """
    Lewa oś:
        prędkość w punkcie CARRA

    Prawa oś:
        wybrana zmienna CARRA

    Punkty = wartości rzeczywiste.
    Linie = 60-dniowa średnia krocząca.
    """
    plot_df = df[['date', 'velocity_m_y', climate_column]].copy()
    plot_df['date'] = pd.to_datetime(plot_df['date'], errors='coerce')
    plot_df['velocity_m_y'] = pd.to_numeric(plot_df['velocity_m_y'], errors='coerce')
    plot_df[climate_column] = pd.to_numeric(plot_df[climate_column], errors='coerce')
    plot_df = plot_df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    if plot_df.empty:
        print(glacier_name, '- brak danych do wykresu.')
        return
    start_date = plot_df['date'].min()
    end_date = plot_df['date'].max()
    velocity_smooth = rolling_time_mean(plot_df, 'velocity_m_y', smooth_window)
    climate_smooth = rolling_time_mean(plot_df, climate_column, smooth_window)
    fig, ax_velocity = plt.subplots(figsize=(15, 6))
    ax_climate = ax_velocity.twinx()
    add_background_periods(ax_velocity, start_date, end_date)
    valid_velocity = plot_df['velocity_m_y'].notna()
    velocity_points = ax_velocity.scatter(plot_df.loc[valid_velocity, 'date'], plot_df.loc[valid_velocity, 'velocity_m_y'], s=14, alpha=0.7, label='Prędkość — obserwacje', zorder=3)
    if not velocity_smooth.empty:
        velocity_line = ax_velocity.plot(velocity_smooth.index, velocity_smooth.values, linewidth=2.0, label=f'Prędkość — średnia krocząca ({smooth_window})', zorder=4)[0]
    else:
        velocity_line = None
    valid_climate = plot_df[climate_column].notna()
    climate_points = ax_climate.scatter(plot_df.loc[valid_climate, 'date'], plot_df.loc[valid_climate, climate_column], s=12, alpha=0.6, marker='x', label=f'{climate_short_name} — obserwacje', zorder=3)
    if not climate_smooth.empty:
        climate_line = ax_climate.plot(climate_smooth.index, climate_smooth.values, linewidth=2.0, linestyle='--', label=f'{climate_short_name} — średnia krocząca ({smooth_window})', zorder=4)[0]
    else:
        climate_line = None
    ax_velocity.set_title(f'{glacier_name} — prędkość powierzchniowa i {climate_short_name}', fontsize=12, pad=12)
    ax_velocity.set_xlabel('Czas', fontsize=10)
    ax_velocity.set_ylabel('Prędkość powierzchniowa lodu [m/rok]', fontsize=10)
    ax_climate.set_ylabel(climate_label, fontsize=10)
    format_time_axis(ax_velocity, start_date, end_date)
    valid_velocity_values = plot_df['velocity_m_y'].dropna()
    if len(valid_velocity_values) > 0:
        vmax = float(valid_velocity_values.max())
        if np.isfinite(vmax) and vmax > 0:
            ax_velocity.set_ylim(0, vmax * 1.08)
    if climate_short_name == 'RH':
        ax_climate.set_ylim(0, 100)
    handles = [velocity_points, climate_points]
    if velocity_line is not None:
        handles.append(velocity_line)
    if climate_line is not None:
        handles.append(climate_line)
    labels = [handle.get_label() for handle in handles]
    ax_velocity.legend(handles, labels, loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout()
    output_path = output_dir / f'{glacier_name}_velocity_{filename_suffix}_dual_axis.jpg'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Zapisano:', output_path)
for glacier_name in glaciers:
    print('\n')
    print('=' * 72)
    print(glacier_name)
    print('=' * 72)
    input_csv = input_dir / f'{glacier_name}_point_CARRA_velocity_temp_RH_2018-2025.csv'
    if not input_csv.exists():
        raise FileNotFoundError(f'Nie znaleziono:\n{input_csv}')
    df = pd.read_csv(input_csv)
    required_columns = ['date', 'velocity_m_y', 'temperature_2m_mean_C', 'relative_humidity_2m_mean_percent']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f'{input_csv.name}: brakuje kolumn {missing}')
    if make_RH_plot:
        make_dual_axis_plot(df=df, glacier_name=glacier_name, climate_column='relative_humidity_2m_mean_percent', climate_label='Wilgotność względna 2 m [%]', climate_short_name='RH', filename_suffix='RH')
    if make_TEMP_plot:
        make_dual_axis_plot(df=df, glacier_name=glacier_name, climate_column='temperature_2m_mean_C', climate_label='Temperatura powietrza 2 m [°C]', climate_short_name='temperatura', filename_suffix='temperature')
print('\n')
print('=' * 72)
print('GOTOWE')
print('=' * 72)
print('\nWykresy zapisano w:')
print(output_dir)
print('\nPunkty na wykresie = wartości rzeczywiste.')
print('Linie = średnia krocząca używana wyłącznie do wizualizacji; nie zmienia danych wejściowych.')
