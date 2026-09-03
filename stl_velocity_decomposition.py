"""Regularizes glacier-velocity time series to a 12-day step and performs STL decomposition.

The procedure is applied independently to P1, P2 and P3. Input velocities represent
median values calculated within a 500 m buffer around each profile point."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
root_dir = Path('D:\\Studia\\praca_magisterska\\predkosci_svalbard_2018-2022')
input_dir = root_dir / 'wyniki_2018-2025_path14' / 'csv'
output_dir = root_dir / 'wyniki_2018-2025_path14' / 'STL_12dni_3punkty_buffer500m'
output_dir.mkdir(parents=True, exist_ok=True)
glacier_name = 'Scheelebreen'
point_files = {'P1_upper': input_dir / f'{glacier_name}_point_upper_equal_section_buffer500m_velocity_2018-2025.csv', 'P2_middle': input_dir / f'{glacier_name}_point_middle_equal_section_buffer500m_velocity_2018-2025.csv', 'P3_lower': input_dir / f'{glacier_name}_point_lower_equal_section_buffer500m_velocity_2018-2025.csv'}
point_labels = {'P1_upper': 'P1 — górna część profilu (1/6 L)', 'P2_middle': 'P2 — środkowa część profilu (1/2 L)', 'P3_lower': 'P3 — dolna część profilu (5/6 L)'}
point_colors = {'P1_upper': 'purple', 'P2_middle': 'green', 'P3_lower': 'orange'}
step_days = 12
stl_period = 30
stl_robust = True
max_assignment_distance_days = 6

def run_stl_for_point(point_key, point_label, point_color, point_csv):
    """
    Wykonuje tę samą procedurę, która była wcześniej
    stosowana dla jednego punktu:

    1. wczytanie danych,
    2. oczyszczenie dat i wartości,
    3. budowa regularnej siatki 12-dniowej,
    4. przypisanie obserwacji do najbliższego terminu,
    5. mediana, jeśli kilka obserwacji wpada do jednego terminu,
    6. interpolacja braków wewnętrznych metodą czasową,
    7. STL: observed + trend + seasonal + remainder,
    8. zapis CSV,
    9. zapis wykresu.

    Analiza każdego punktu jest wykonywana niezależnie,
    ale z identycznymi parametrami STL.
    """
    print('\n')
    print('=' * 70)
    print(f'{glacier_name} — {point_label}')
    print('=' * 70)
    if not point_csv.exists():
        raise FileNotFoundError(f'Nie znaleziono pliku wejściowego:\n{point_csv}')
    df = pd.read_csv(point_csv)
    print('\nWczytano plik:')
    print(point_csv)
    print('\nLiczba rekordów:', len(df))
    required_columns = ['date', 'velocity_m_y']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f'{point_label}: brakuje wymaganych kolumn: ' + ', '.join(missing_columns))
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['velocity_m_y'] = pd.to_numeric(df['velocity_m_y'], errors='coerce')
    df = df.dropna(subset=['date', 'velocity_m_y']).sort_values('date').reset_index(drop=True)
    if df.empty:
        raise ValueError(f'{point_label}: brak poprawnych rekordów po oczyszczeniu danych.')
    print('\nZakres rzeczywistych danych:')
    print(df['date'].min(), '→', df['date'].max())
    real_time_diffs = (df['date'].sort_values().diff().dt.total_seconds() / 86400.0).dropna()
    if len(real_time_diffs) > 0:
        print('\nOdstępy między pomiarami [dni]:')
        print('Minimum:', round(real_time_diffs.min(), 2))
        print('Mediana:', round(real_time_diffs.median(), 2))
        print('Średnia:', round(real_time_diffs.mean(), 2))
        print('Maksimum:', round(real_time_diffs.max(), 2))
    grid_start = df['date'].min().normalize()
    grid_end = df['date'].max().normalize()
    regular_index = pd.date_range(start=grid_start, end=grid_end, freq=f'{step_days}D')
    print('\nRegularna siatka:')
    print('Krok:', step_days, 'dni')
    print('Początek:', regular_index.min())
    print('Koniec:', regular_index.max())
    print('Liczba punktów:', len(regular_index))
    assignments = []
    regular_values_ns = regular_index.asi8
    for _, row in df.iterrows():
        obs_date = row['date']
        obs_velocity = row['velocity_m_y']
        obs_ns = pd.Timestamp(obs_date).value
        nearest_idx = int(np.argmin(np.abs(regular_values_ns - obs_ns)))
        nearest_date = regular_index[nearest_idx]
        distance_days = abs((obs_date - nearest_date).total_seconds() / 86400.0)
        if distance_days <= max_assignment_distance_days:
            assignments.append({'grid_date': nearest_date, 'real_date': obs_date, 'velocity_m_y': obs_velocity, 'distance_days': distance_days})
    assignment_df = pd.DataFrame(assignments)
    if assignment_df.empty:
        raise ValueError(f'{point_label}: żaden rzeczywisty pomiar nie został przypisany do siatki 12-dniowej.')
    print('\nLiczba rzeczywistych obserwacji przypisanych do siatki:', len(assignment_df))
    real_on_grid = assignment_df.groupby('grid_date')['velocity_m_y'].median()
    real_count_on_grid = assignment_df.groupby('grid_date')['velocity_m_y'].count()
    series_12d_original = real_on_grid.reindex(regular_index)
    count_12d = real_count_on_grid.reindex(regular_index).fillna(0).astype(int)
    missing_mask = series_12d_original.isna()
    n_total = len(series_12d_original)
    n_real = int((~missing_mask).sum())
    n_missing = int(missing_mask.sum())
    missing_percent = n_missing / n_total * 100
    print('\n====================================')
    print('SIATKA 12-DNIOWA — PODSUMOWANIE')
    print('====================================')
    print('\nLiczba punktów:', n_total)
    print('Punkty z rzeczywistą wartością:', n_real)
    print('Punkty bez rzeczywistej wartości:', n_missing)
    print('Udział punktów wymagających interpolacji [%]:', round(missing_percent, 2))
    is_missing = missing_mask.astype(int)
    group_id = is_missing.ne(is_missing.shift()).cumsum()
    missing_run_lengths = is_missing[is_missing == 1].groupby(group_id[is_missing == 1]).size()
    if len(missing_run_lengths) > 0:
        longest_missing_run = int(missing_run_lengths.max())
    else:
        longest_missing_run = 0
    print('\nNajdłuższa seria brakujących punktów:', longest_missing_run)
    print('Najdłuższa luka w przybliżeniu [dni]:', longest_missing_run * step_days)
    series_12d = series_12d_original.interpolate(method='time', limit_area='inside')
    valid_after_interpolation = series_12d.notna()
    if not valid_after_interpolation.any():
        raise ValueError(f'{point_label}: po interpolacji nie pozostały żadne dane.')
    first_valid = valid_after_interpolation[valid_after_interpolation].index.min()
    last_valid = valid_after_interpolation[valid_after_interpolation].index.max()
    series_12d = series_12d.loc[first_valid:last_valid]
    series_12d_original = series_12d_original.loc[first_valid:last_valid]
    count_12d = count_12d.loc[first_valid:last_valid]
    interpolated_mask = series_12d_original.isna()
    if series_12d.isna().any():
        raise ValueError(f'{point_label}: po interpolacji nadal występują NaN.')
    n_final = len(series_12d)
    n_real_final = int((~interpolated_mask).sum())
    n_interpolated_final = int(interpolated_mask.sum())
    interpolated_percent_final = n_interpolated_final / n_final * 100
    print('\n====================================')
    print('INTERPOLACJA — PODSUMOWANIE')
    print('====================================')
    print('\nPunkty używane przez STL:', n_final)
    print('Rzeczywiste:', n_real_final)
    print('Interpolowane:', n_interpolated_final)
    print('Interpolowane [%]:', round(interpolated_percent_final, 2))
    minimum_length = stl_period * 2
    if len(series_12d) < minimum_length:
        raise ValueError(f'{point_label}: szereg zawiera tylko {len(series_12d)} punktów. Dla period={stl_period} zalecane minimum to {minimum_length} punktów.')
    print('\nUruchamianie STL...')
    print('Period:', stl_period, 'punktów')
    print('Przybliżony okres sezonowy:', stl_period * step_days, 'dni')
    stl = STL(series_12d, period=stl_period, robust=stl_robust)
    result = stl.fit()
    observed = series_12d
    trend = result.trend
    seasonal = result.seasonal
    remainder = result.resid
    decomposition = pd.DataFrame({'observed': observed, 'observed_original': series_12d_original, 'trend': trend, 'seasonal': seasonal, 'remainder': remainder, 'n_observations': count_12d, 'interpolated': interpolated_mask})
    decomposition['reconstructed'] = decomposition['trend'] + decomposition['seasonal'] + decomposition['remainder']
    reconstruction_error = decomposition['observed'] - decomposition['reconstructed']
    print('\nMaksymalny błąd rekonstrukcji:')
    print(np.nanmax(np.abs(reconstruction_error)))
    print('\n====================================')
    print('STL 12 DNI — PODSUMOWANIE')
    print('====================================')
    print('\nLiczba punktów:', len(decomposition))
    print('Liczba punktów interpolowanych:', int(decomposition['interpolated'].sum()))
    print('\nOBSERVED:')
    print('Minimum:', round(decomposition['observed'].min(), 2))
    print('Maksimum:', round(decomposition['observed'].max(), 2))
    print('Średnia:', round(decomposition['observed'].mean(), 2))
    print('\nTREND:')
    print('Minimum:', round(decomposition['trend'].min(), 2))
    print('Maksimum:', round(decomposition['trend'].max(), 2))
    print('\nSEASONAL:')
    print('Minimum:', round(decomposition['seasonal'].min(), 2))
    print('Maksimum:', round(decomposition['seasonal'].max(), 2))
    print('Amplituda peak-to-peak:', round(decomposition['seasonal'].max() - decomposition['seasonal'].min(), 2))
    print('\nREMAINDER:')
    print('Minimum:', round(decomposition['remainder'].min(), 2))
    print('Maksimum:', round(decomposition['remainder'].max(), 2))
    print('Średnia:', round(decomposition['remainder'].mean(), 4))
    print('Odchylenie standardowe:', round(decomposition['remainder'].std(), 2))
    output_csv = output_dir / f'{glacier_name}_{point_key}_STL_12days_buffer500m_decomposition.csv'
    decomposition.to_csv(output_csv, index_label='date')
    assignment_csv = output_dir / f'{glacier_name}_{point_key}_STL_12days_buffer500m_assignments.csv'
    assignment_df.to_csv(assignment_csv, index=False)
    print('\nZapisano:')
    print(output_csv)
    print(assignment_csv)
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 11), sharex=True)
    axes[0].plot(decomposition.index, decomposition['observed'], linewidth=1.0, color=point_color)
    real_data = decomposition[~decomposition['interpolated']]
    axes[0].scatter(real_data.index, real_data['observed'], s=18, color=point_color, zorder=3, label='Obserwacje rzeczywiste')
    interpolated_data = decomposition[decomposition['interpolated']]
    if not interpolated_data.empty:
        axes[0].scatter(interpolated_data.index, interpolated_data['observed'], s=28, marker='x', linewidths=1.1, zorder=4, label='Interpolowane')
    axes[0].legend(fontsize=8, loc='best')
    axes[0].set_ylabel('Prędkość\n[m/rok]')
    axes[0].set_title(f'{glacier_name} — {point_label} — dekompozycja STL, siatka 12-dniowa')
    axes[0].grid(alpha=0.2)
    axes[1].plot(decomposition.index, decomposition['trend'], linewidth=1.8, color=point_color)
    axes[1].set_ylabel('Trend\n[m/rok]')
    axes[1].grid(alpha=0.2)
    axes[2].plot(decomposition.index, decomposition['seasonal'], linewidth=1.2, color=point_color)
    axes[2].axhline(0, linewidth=0.8, linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Sezonowość\n[m/rok]')
    axes[2].grid(alpha=0.2)
    axes[3].scatter(decomposition.index, decomposition['remainder'], s=18, color=point_color)
    axes[3].axhline(0, linewidth=0.8, linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Reszta\n[m/rok]')
    axes[3].set_xlabel('Czas')
    axes[3].grid(alpha=0.2)
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[3].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    for ax in axes:
        ax.tick_params(axis='both', labelsize=9)
    fig.tight_layout()
    figure_png = output_dir / f'{glacier_name}_{point_key}_STL_12days_decomposition.png'
    figure_pdf = output_dir / f'{glacier_name}_{point_key}_STL_12days_decomposition.pdf'
    plt.savefig(figure_png, dpi=300, bbox_inches='tight')
    plt.savefig(figure_pdf, bbox_inches='tight')
    plt.close(fig)
    print('\nZapisano wykres STL:')
    print(figure_png)
    print(figure_pdf)
    return {'point_key': point_key, 'point_label': point_label, 'decomposition': decomposition, 'assignment_df': assignment_df, 'output_csv': output_csv, 'assignment_csv': assignment_csv, 'figure_png': figure_png, 'figure_pdf': figure_pdf, 'n_final': n_final, 'n_real': n_real_final, 'n_interpolated': n_interpolated_final, 'interpolated_percent': interpolated_percent_final, 'longest_missing_run': longest_missing_run}
results = {}
for point_key in ['P1_upper', 'P2_middle', 'P3_lower']:
    results[point_key] = run_stl_for_point(point_key=point_key, point_label=point_labels[point_key], point_color=point_colors[point_key], point_csv=point_files[point_key])
summary_rows = []
for point_key, info in results.items():
    decomposition = info['decomposition']
    summary_rows.append({'point': point_key, 'label': info['point_label'], 'n_stl_points': info['n_final'], 'n_real': info['n_real'], 'n_interpolated': info['n_interpolated'], 'interpolated_percent': info['interpolated_percent'], 'longest_missing_run_bins': info['longest_missing_run'], 'longest_missing_run_days': info['longest_missing_run'] * step_days, 'observed_min': decomposition['observed'].min(), 'observed_max': decomposition['observed'].max(), 'observed_mean': decomposition['observed'].mean(), 'trend_min': decomposition['trend'].min(), 'trend_max': decomposition['trend'].max(), 'seasonal_min': decomposition['seasonal'].min(), 'seasonal_max': decomposition['seasonal'].max(), 'seasonal_peak_to_peak': decomposition['seasonal'].max() - decomposition['seasonal'].min(), 'remainder_min': decomposition['remainder'].min(), 'remainder_max': decomposition['remainder'].max(), 'remainder_mean': decomposition['remainder'].mean(), 'remainder_std': decomposition['remainder'].std()})
summary_df = pd.DataFrame(summary_rows)
summary_csv = output_dir / f'{glacier_name}_3points_STL_12days_buffer500m_summary.csv'
summary_df.to_csv(summary_csv, index=False)
print('\n')
print('=' * 70)
print('PODSUMOWANIE 3 PUNKTÓW')
print('=' * 70)
print(summary_df.to_string(index=False))
print('\nZapisano:')
print(summary_csv)
combined_components = None
for point_key, info in results.items():
    dec = info['decomposition'][['observed', 'observed_original', 'trend', 'seasonal', 'remainder', 'interpolated']].copy()
    dec = dec.rename(columns={'observed': f'{point_key}_observed', 'observed_original': f'{point_key}_observed_original', 'trend': f'{point_key}_trend', 'seasonal': f'{point_key}_seasonal', 'remainder': f'{point_key}_remainder', 'interpolated': f'{point_key}_interpolated'})
    if combined_components is None:
        combined_components = dec
    else:
        combined_components = combined_components.join(dec, how='outer')
combined_csv = output_dir / f'{glacier_name}_3points_STL_12days_buffer500m_components.csv'
combined_components.to_csv(combined_csv, index_label='date')
print('\nZapisano wspólną tabelę komponentów:')
print(combined_csv)
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 12), sharex=True)
for point_key in ['P1_upper', 'P2_middle', 'P3_lower']:
    dec = results[point_key]['decomposition']
    label = point_labels[point_key]
    axes[0].plot(dec.index, dec['observed'], color=point_colors[point_key], linewidth=1.0, label=label)
    axes[1].plot(dec.index, dec['trend'], color=point_colors[point_key], linewidth=1.5, label=label)
    axes[2].plot(dec.index, dec['seasonal'], color=point_colors[point_key], linewidth=1.0, label=label)
    axes[3].plot(dec.index, dec['remainder'], color=point_colors[point_key], linewidth=1.0, label=label)
axes[0].set_ylabel('Prędkość\n[m/rok]')
axes[0].set_title(f'{glacier_name} — Dekompozycja sezonowo-trendowa z wykorzystaniem metody LOESS (STL)')
axes[1].set_ylabel('Trend\n[m/rok]')
axes[2].set_ylabel('Sezonowość\n[m/rok]')
axes[3].set_ylabel('Reszta\n[m/rok]')
axes[3].set_xlabel('Czas')
axes[2].axhline(0, linewidth=0.8, linestyle='--', alpha=0.5)
axes[3].axhline(0, linewidth=0.8, linestyle='--', alpha=0.5)
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc='best')
    ax.tick_params(axis='both', labelsize=9)
axes[3].xaxis.set_major_locator(mdates.YearLocator())
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[3].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
fig.tight_layout()
comparison_png = output_dir / f'{glacier_name}_3points_STL_12days_comparison.png'
comparison_pdf = output_dir / f'{glacier_name}_3points_STL_12days_comparison.pdf'
plt.savefig(comparison_png, dpi=300, bbox_inches='tight')
plt.savefig(comparison_pdf, bbox_inches='tight')
print('\nZapisano wspólny wykres porównawczy:')
print(comparison_png)
print(comparison_pdf)
plt.show()
print('\n')
print('=' * 70)
print('ZAKOŃCZONO STL DLA 3 PUNKTÓW')
print('=' * 70)
print('\nP1 = górna część profilu, bliżej A = L/6')
print('P2 = środkowa część profilu = L/2')
print('P3 = dolna część profilu, bliżej B = 5L/6')
print('Dane wejściowe P1/P2/P3 = mediana prędkości w buforze 500 m')
print('\nKażdy punkt został przetworzony niezależnie z identycznymi ustawieniami:')
print(f'krok = {step_days} dni')
print(f'period = {stl_period}')
print(f'robust = {stl_robust}')
