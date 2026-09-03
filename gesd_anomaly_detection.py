"""Detects GESD anomalies in the STL residual component for three glacier-profile points.

The analysis is performed independently for P1, P2 and P3. Input velocities correspond
to values previously aggregated within a 500 m buffer around each point."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import t
root_dir = Path('D:\\Studia\\praca_magisterska\\predkosci_svalbard_2018-2022')
input_dir = root_dir / 'wyniki_2018-2025_path14' / 'STL_12dni_3punkty_buffer500m'
output_dir = root_dir / 'wyniki_2018-2025_path14' / 'GESD_12dni_3punkty_buffer500m'
output_dir.mkdir(parents=True, exist_ok=True)
glacier_name = 'Scheelebreen'
point_files = {'P1_upper': input_dir / f'{glacier_name}_P1_upper_STL_12days_buffer500m_decomposition.csv', 'P2_middle': input_dir / f'{glacier_name}_P2_middle_STL_12days_buffer500m_decomposition.csv', 'P3_lower': input_dir / f'{glacier_name}_P3_lower_STL_12days_buffer500m_decomposition.csv'}
point_labels = {'P1_upper': 'P1 — górna część profilu (1/6 L)', 'P2_middle': 'P2 — środkowa część profilu (1/2 L)', 'P3_lower': 'P3 — dolna część profilu (5/6 L)'}
point_colors = {'P1_upper': 'purple', 'P2_middle': 'green', 'P3_lower': 'orange'}
point_order = ['P1_upper', 'P2_middle', 'P3_lower']
alpha = 0.05
max_anomaly_fraction = 0.2

def generalized_esd(values, alpha=0.05, max_outliers=None):
    """
    Dwustronny test Generalized ESD Rosnera.
    """
    x = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(x)):
        raise ValueError('Szereg GESD zawiera NaN lub inf.')
    n = len(x)
    if max_outliers is None:
        max_outliers = int(np.floor(0.2 * n))
    max_outliers = max(1, min(int(max_outliers), n - 2))
    working_values = x.copy()
    working_positions = np.arange(n)
    candidate_positions = []
    iteration_rows = []
    for i in range(1, max_outliers + 1):
        n_i = len(working_values)
        if n_i < 3:
            break
        mean_i = np.mean(working_values)
        std_i = np.std(working_values, ddof=1)
        if not np.isfinite(std_i) or std_i == 0:
            break
        deviations = np.abs(working_values - mean_i)
        local_idx = int(np.argmax(deviations))
        candidate_value = working_values[local_idx]
        candidate_position = int(working_positions[local_idx])
        R_i = deviations[local_idx] / std_i
        p = 1 - alpha / (2 * n_i)
        t_critical = t.ppf(p, df=n_i - 2)
        lambda_i = (n_i - 1) * t_critical / np.sqrt(n_i * (n_i - 2 + t_critical ** 2))
        candidate_positions.append(candidate_position)
        iteration_rows.append({'iteration': i, 'n_remaining': n_i, 'mean': mean_i, 'std': std_i, 'candidate_position': candidate_position, 'candidate_value': candidate_value, 'R_i': R_i, 'lambda_i': lambda_i, 'R_gt_lambda': bool(R_i > lambda_i)})
        working_values = np.delete(working_values, local_idx)
        working_positions = np.delete(working_positions, local_idx)
    iterations = pd.DataFrame(iteration_rows)
    if iterations.empty or not iterations['R_gt_lambda'].any():
        n_outliers = 0
    else:
        n_outliers = int(iterations.loc[iterations['R_gt_lambda'], 'iteration'].max())
    return {'n_outliers': n_outliers, 'outlier_positions': candidate_positions[:n_outliers], 'iterations': iterations}

def run_gesd_for_point(point_key, point_label, point_color, input_csv):
    """
    Wykonuje test GESD dla komponentu resztowego STL
    jednego punktu kontrolnego.
    """
    print('\n' + '=' * 70)
    print(f'{glacier_name} — {point_label}')
    print('=' * 70)
    if not input_csv.exists():
        raise FileNotFoundError(f'Nie znaleziono pliku wejściowego:\n{input_csv}')
    df = pd.read_csv(input_csv)
    required_columns = ['date', 'remainder']
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError('Brakuje wymaganych kolumn: ' + ', '.join(missing_columns))
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['remainder'] = pd.to_numeric(df['remainder'], errors='coerce')
    df = df.dropna(subset=['date', 'remainder']).sort_values('date').reset_index(drop=True)
    if len(df) < 3:
        raise ValueError(f'{point_label}: zbyt mało obserwacji do wykonania GESD.')
    n = len(df)
    max_outliers = max(1, int(np.floor(max_anomaly_fraction * n)))
    print('\nPARAMETRY GESD')
    print('Poziom istotności alpha:', alpha)
    print('Maksymalny udział kandydatów na anomalie:', max_anomaly_fraction)
    print('Maksymalna liczba kandydatów na anomalie:', max_outliers)
    result = generalized_esd(df['remainder'].to_numpy(), alpha=alpha, max_outliers=max_outliers)
    iterations = result['iterations']
    outlier_positions = result['outlier_positions']
    df['gesd_anomaly'] = False
    if outlier_positions:
        df.loc[outlier_positions, 'gesd_anomaly'] = True
    df['positive_anomaly'] = df['gesd_anomaly'] & (df['remainder'] > 0)
    df['negative_anomaly'] = df['gesd_anomaly'] & (df['remainder'] < 0)
    df['gesd_rank'] = np.nan
    for rank, position in enumerate(outlier_positions, start=1):
        df.loc[position, 'gesd_rank'] = rank
    if 'interpolated' in df.columns:
        if df['interpolated'].dtype == object:
            df['interpolated'] = df['interpolated'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False).astype(bool)
        else:
            df['interpolated'] = df['interpolated'].astype(bool)
        df['anomaly_on_interpolated_point'] = df['gesd_anomaly'] & df['interpolated']
    n_anomalies = int(df['gesd_anomaly'].sum())
    n_positive = int(df['positive_anomaly'].sum())
    n_negative = int(df['negative_anomaly'].sum())
    if 'anomaly_on_interpolated_point' in df.columns:
        n_interpolated = int(df['anomaly_on_interpolated_point'].sum())
    else:
        n_interpolated = 0
    print('\nGESD — WYNIKI')
    print('Liczba wykrytych anomalii:', n_anomalies)
    print('Dodatnie anomalie:', n_positive)
    print('Ujemne anomalie:', n_negative)
    if 'anomaly_on_interpolated_point' in df.columns:
        print('Anomalie na punktach interpolowanych:', n_interpolated)
    output_csv = output_dir / f'{glacier_name}_{point_key}_GESD_buffer500m_remainder_anomalies.csv'
    anomalies_csv = output_dir / f'{glacier_name}_{point_key}_GESD_buffer500m_anomalies_only.csv'
    iterations_csv = output_dir / f'{glacier_name}_{point_key}_GESD_buffer500m_iterations.csv'
    df.to_csv(output_csv, index=False)
    df[df['gesd_anomaly']].to_csv(anomalies_csv, index=False)
    iterations.to_csv(iterations_csv, index=False)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df['date'], df['remainder'], linewidth=1.0, color=point_color, label='Komponent resztowy')
    ax.scatter(df['date'], df['remainder'], s=16, color=point_color, zorder=3)
    positive = df[df['positive_anomaly']]
    negative = df[df['negative_anomaly']]
    if not positive.empty:
        ax.scatter(positive['date'], positive['remainder'], s=70, marker='o', facecolors='none', edgecolors='red', linewidths=1.7, zorder=5, label='Dodatnia anomalia GESD')
    if not negative.empty:
        ax.scatter(negative['date'], negative['remainder'], s=70, marker='o', facecolors='none', edgecolors='blue', linewidths=1.7, zorder=5, label='Ujemna anomalia GESD')
    ax.axhline(0, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Czas')
    ax.set_ylabel('Komponent resztowy [m/rok]')
    ax.set_title(f'{glacier_name} — {point_label} — test GESD dla komponentu resztowego STL')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    figure_jpg = output_dir / f'{glacier_name}_{point_key}_GESD_buffer500m_remainder.jpg'
    plt.savefig(figure_jpg, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('\nZapisano:')
    print(output_csv)
    print(anomalies_csv)
    print(iterations_csv)
    print(figure_jpg)
    return {'data': df, 'n_anomalies': n_anomalies, 'n_positive': n_positive, 'n_negative': n_negative, 'n_interpolated': n_interpolated}
results = {}
for point_key in point_order:
    results[point_key] = run_gesd_for_point(point_key=point_key, point_label=point_labels[point_key], point_color=point_colors[point_key], input_csv=point_files[point_key])
summary_rows = []
all_anomalies = []
for point_key in point_order:
    info = results[point_key]
    summary_rows.append({'point': point_key, 'label': point_labels[point_key], 'n_anomalies': info['n_anomalies'], 'n_positive': info['n_positive'], 'n_negative': info['n_negative'], 'n_interpolated_anomalies': info['n_interpolated']})
    anomalies = info['data'][info['data']['gesd_anomaly']].copy()
    if not anomalies.empty:
        anomalies.insert(0, 'point', point_key)
        anomalies.insert(1, 'point_label', point_labels[point_key])
        all_anomalies.append(anomalies)
summary_df = pd.DataFrame(summary_rows)
summary_csv = output_dir / f'{glacier_name}_3points_GESD_buffer500m_summary.csv'
summary_df.to_csv(summary_csv, index=False)
if all_anomalies:
    combined_anomalies_df = pd.concat(all_anomalies, ignore_index=True).sort_values(['date', 'point'])
else:
    combined_anomalies_df = pd.DataFrame()
combined_anomalies_csv = output_dir / f'{glacier_name}_3points_GESD_buffer500m_anomalies.csv'
combined_anomalies_df.to_csv(combined_anomalies_csv, index=False)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 11), sharex=True)
for ax, point_key in zip(axes, point_order):
    df_point = results[point_key]['data']
    color = point_colors[point_key]
    ax.plot(df_point['date'], df_point['remainder'], linewidth=1.0, color=color, label='Komponent resztowy')
    ax.scatter(df_point['date'], df_point['remainder'], s=14, color=color, zorder=3)
    positive = df_point[df_point['positive_anomaly']]
    negative = df_point[df_point['negative_anomaly']]
    if not positive.empty:
        ax.scatter(positive['date'], positive['remainder'], s=65, marker='o', facecolors='none', edgecolors='red', linewidths=1.6, zorder=5, label='Dodatnia anomalia GESD')
    if not negative.empty:
        ax.scatter(negative['date'], negative['remainder'], s=65, marker='o', facecolors='none', edgecolors='blue', linewidths=1.6, zorder=5, label='Ujemna anomalia GESD')
    ax.axhline(0, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylabel('Komponent resztowy\n[m/rok]')
    ax.set_title(point_labels[point_key], fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc='best')
axes[-1].set_xlabel('Czas')
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
fig.suptitle(f'{glacier_name} — test GESD dla trzech punktów', fontsize=12, y=0.995)
fig.tight_layout()
comparison_jpg = output_dir / f'{glacier_name}_3points_GESD_buffer500m_comparison.jpg'
plt.savefig(comparison_jpg, dpi=300, bbox_inches='tight')
plt.close(fig)
print('\n' + '=' * 70)
print('ZAKOŃCZONO TEST GESD DLA 3 PUNKTÓW')
print('=' * 70)
print('P1 = górna część profilu, bliżej A = L/6')
print('P2 = środkowa część profilu = L/2')
print('P3 = dolna część profilu, bliżej B = 5L/6')
print('\nParametry GESD:')
print('alpha =', alpha)
print('maksymalny udział kandydatów na anomalie =', max_anomaly_fraction)
print('\nZapisano:')
print(summary_csv)
print(combined_anomalies_csv)
print(comparison_jpg)
