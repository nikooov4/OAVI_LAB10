from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import signal
from scipy.io import wavfile


# ============================================================
# Лабораторная работа №10
# Вариант 1 — Голосовой диапазон, тембр, форманты
# ============================================================

VARIANT = 1
LAB_TITLE = "Голосовой диапазон, тембр, форманты"

BASE_DIR = Path(__file__).resolve().parent

INPUT_AUDIO_DIR = BASE_DIR / "input_audio"

RESULTS_DIR = BASE_DIR / "results_lab10"
SRC_DIR = BASE_DIR / "src_lab10"
REPORT_PATH = BASE_DIR / "report_lab10.md"

AUDIO_DIR = RESULTS_DIR / "audio"
PLOTS_DIR = RESULTS_DIR / "plots"
CSV_DIR = RESULTS_DIR / "csv"

SRC_AUDIO_DIR = SRC_DIR / "audio"
SRC_PLOTS_DIR = SRC_DIR / "plots"
SRC_CSV_DIR = SRC_DIR / "csv"

SUMMARY_CSV = CSV_DIR / "summary.csv"
FORMANTS_CSV = CSV_DIR / "formants.csv"
ENERGY_REGIONS_CSV = CSV_DIR / "energy_regions.csv"

# =========================
# ФАЙЛЫ ДЛЯ ЗАПИСИ
# =========================

REQUIRED_RECORDINGS = {
    "a": {
        "title": "Гласный А",
        "expected_files": ["a.wav", "а.wav"],
        "theoretical_formants": [660.0, 1700.0, 2400.0],
    },
    "i": {
        "title": "Гласный И",
        "expected_files": ["i.wav", "и.wav"],
        "theoretical_formants": [270.0, 2300.0, 3000.0],
    },
    "animal": {
        "title": "Имитация животного / крик",
        "expected_files": ["animal.wav", "bark.wav", "meow.wav", "tarzan.wav"],
        "theoretical_formants": [],
    },
}

# =========================
# ПАРАМЕТРЫ АНАЛИЗА
# =========================

# STFT: окно Ханна.
# N_FFT=4096 при 44100 Гц дает шаг частоты примерно 10.77 Гц,
# то есть форманты считаются не слишком грубо.
N_FFT = 4096
OVERLAP = 0.75

# Для поиска энергетических областей из задания
ENERGY_DT = 0.1
ENERGY_DF = 50.0
FORMANT_FREQ_STEP = 10.0

TOP_ENERGY_REGIONS = 20

# Диапазон поиска основного тона голоса
PITCH_MIN_HZ = 60.0
PITCH_MAX_HZ = 1200.0

# Диапазон поиска формант
FORMANT_MIN_HZ = 100.0
FORMANT_MAX_HZ = 4000.0

# Если нет пользовательских WAV-файлов, код создаст демонстрационные записи.
DEMO_SAMPLE_RATE = 44100


@dataclass
class RecordingResult:
    key: str
    title: str
    input_file: str
    sample_rate: int
    channels_original: int
    duration_sec: float
    samples_count: int

    pitch_min_hz: float | None
    pitch_max_hz: float | None
    richest_f0_hz: float | None
    richest_time_sec: float | None
    harmonic_count: int

    formants_hz: list[float]
    theoretical_formants_hz: list[float]
    formant_errors_hz: list[float]

    waveform_file: str
    spectrogram_file: str
    pitch_file: str
    spectrum_file: str


@dataclass
class EnergyRegion:
    recording_key: str
    rank: int
    time_start: float
    time_end: float
    freq_start: float
    freq_end: float
    energy: float


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def setup_dirs() -> None:
    for path in [
        RESULTS_DIR,
        SRC_DIR,
        AUDIO_DIR,
        PLOTS_DIR,
        CSV_DIR,
        SRC_AUDIO_DIR,
        SRC_PLOTS_DIR,
        SRC_CSV_DIR,
    ]:
        ensure_clean_dir(path)

    INPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def normalize_audio(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        max_abs = max(abs(info.min), abs(info.max))
        audio = data.astype(np.float64) / max_abs
    else:
        audio = data.astype(np.float64)

    return np.clip(audio, -1.0, 1.0)


def load_wav_mono(path: Path) -> tuple[int, np.ndarray, int]:
    sample_rate, data = wavfile.read(path)

    audio = normalize_audio(data)

    if audio.ndim == 1:
        channels = 1
        mono = audio
    else:
        channels = audio.shape[1]
        mono = audio.mean(axis=1)

    mono = mono.astype(np.float64)

    # Убираем постоянную составляющую
    mono = mono - float(mono.mean())

    # Нормализуем, но не до клиппинга
    max_abs = float(np.max(np.abs(mono)) + 1e-12)
    mono = mono / max_abs * 0.95

    return sample_rate, mono, channels


def save_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    audio = np.asarray(audio, dtype=np.float64)
    audio = np.clip(audio, -1.0, 1.0)
    int16_audio = np.round(audio * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, int16_audio)


def find_recording_file(key: str) -> Path | None:
    expected = REQUIRED_RECORDINGS[key]["expected_files"]

    for file_name in expected:
        path = INPUT_AUDIO_DIR / file_name
        if path.exists():
            return path

    return None


def synthesize_vowel(
    formants: list[float],
    duration_sec: float = 8.0,
    sample_rate: int = DEMO_SAMPLE_RATE,
    f0_start: float = 100.0,
    f0_end: float = 700.0,
) -> np.ndarray:
    """
    Демонстрационный синтез гласной.
    Это не замена реальной записи, а тестовый сигнал, чтобы проверить работу кода.
    """

    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    f0 = np.linspace(f0_start, f0_end, t.size)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate

    source = np.zeros_like(t)

    for k in range(1, 18):
        source += (1.0 / k) * np.sin(k * phase)

    # Плавная огибающая
    attack = np.clip(t / 0.3, 0, 1)
    release = np.clip((duration_sec - t) / 0.5, 0, 1)
    envelope = np.minimum(attack, release)

    source *= envelope

    y = np.zeros_like(source)

    for formant in formants:
        q = 12.0
        b, a = signal.iirpeak(w0=formant, Q=q, fs=sample_rate)
        y += signal.lfilter(b, a, source)

    y = y / (np.max(np.abs(y)) + 1e-12) * 0.85

    return y.astype(np.float64)


def synthesize_animal(duration_sec: float = 4.0, sample_rate: int = DEMO_SAMPLE_RATE) -> np.ndarray:
    """
    Демонстрационная имитация коротких звуков.
    Для сдачи лучше записать свой animal.wav.
    """

    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    rng = np.random.default_rng(10)

    y = np.zeros_like(t)

    bursts = [
        (0.4, 0.9, 280.0),
        (1.3, 1.8, 360.0),
        (2.2, 2.8, 520.0),
        (3.1, 3.5, 420.0),
    ]

    for start, end, freq in bursts:
        mask = (t >= start) & (t <= end)
        lt = t[mask] - start

        envelope = np.sin(np.pi * lt / (end - start)) ** 2

        tone = (
            np.sin(2 * np.pi * freq * lt)
            + 0.45 * np.sin(2 * np.pi * 2.2 * freq * lt)
            + 0.25 * np.sin(2 * np.pi * 3.5 * freq * lt)
        )

        noise = 0.5 * rng.normal(size=lt.size)

        y[mask] += envelope * (tone + noise)

    y = y / (np.max(np.abs(y)) + 1e-12) * 0.85

    return y.astype(np.float64)


def get_recording(key: str) -> tuple[str, int, np.ndarray, int]:
    path = find_recording_file(key)

    if path is not None:
        sample_rate, audio, channels = load_wav_mono(path)
        return path.name, sample_rate, audio, channels

    # Демонстрационный режим, если пользователь еще не записал файлы
    if key == "a":
        audio = synthesize_vowel([660.0, 1700.0, 2400.0])
    elif key == "i":
        audio = synthesize_vowel([270.0, 2300.0, 3000.0])
    else:
        audio = synthesize_animal()

    return f"demo_{key}.wav", DEMO_SAMPLE_RATE, audio, 1


def stft_params(sample_rate: int) -> tuple[int, int]:
    nperseg = min(N_FFT, max(256, len_for_min_audio(sample_rate)))

    if nperseg % 2 == 1:
        nperseg += 1

    noverlap = int(round(nperseg * OVERLAP))

    return nperseg, noverlap


def len_for_min_audio(sample_rate: int) -> int:
    # Фиксируем около 4096 для 44100, но оставляем разумное значение
    return int(round(sample_rate * N_FFT / 44100))


def compute_stft(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg, noverlap = stft_params(sample_rate)

    f, t, zxx = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    return f, t, zxx


def active_frame_mask(audio: np.ndarray, sample_rate: int, frame_sec: float = 0.03, hop_sec: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    frame_len = max(1, int(round(frame_sec * sample_rate)))
    hop_len = max(1, int(round(hop_sec * sample_rate)))

    if audio.size < frame_len:
        return np.array([0.0]), np.array([True])

    energies = []
    times = []

    for start in range(0, audio.size - frame_len + 1, hop_len):
        frame = audio[start:start + frame_len]
        energies.append(float(np.sqrt(np.mean(frame * frame) + 1e-12)))
        times.append((start + frame_len / 2) / sample_rate)

    energies_arr = np.array(energies)
    times_arr = np.array(times)

    threshold = max(0.02, 0.15 * float(energies_arr.max()))

    return times_arr, energies_arr >= threshold


def estimate_pitch_track(
    audio: np.ndarray,
    sample_rate: int,
    frame_sec: float = 0.05,
    hop_sec: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Оценка основного тона через автокорреляцию.
    """

    frame_len = int(round(frame_sec * sample_rate))
    hop_len = int(round(hop_sec * sample_rate))

    min_lag = max(1, int(sample_rate / PITCH_MAX_HZ))
    max_lag = min(frame_len - 1, int(sample_rate / PITCH_MIN_HZ))

    times: list[float] = []
    pitches: list[float] = []

    if audio.size < frame_len or max_lag <= min_lag:
        return np.array([]), np.array([])

    global_rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))

    for start in range(0, audio.size - frame_len + 1, hop_len):
        frame = audio[start:start + frame_len].copy()
        frame = frame - float(frame.mean())

        frame_rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))

        if frame_rms < 0.08 * global_rms:
            continue

        frame *= np.hanning(frame_len)

        corr = np.correlate(frame, frame, mode="full")[frame_len - 1:]

        if corr[0] <= 1e-12:
            continue

        corr = corr / corr[0]

        search = corr[min_lag:max_lag]

        if search.size == 0:
            continue

        peak_rel = int(np.argmax(search))
        peak_lag = min_lag + peak_rel
        peak_value = float(corr[peak_lag])

        if peak_value < 0.25:
            continue

        pitch = sample_rate / peak_lag

        times.append((start + frame_len / 2) / sample_rate)
        pitches.append(pitch)

    return np.array(times), np.array(pitches)


def find_richest_tone(
    audio: np.ndarray,
    sample_rate: int,
    pitch_times: np.ndarray,
    pitch_values: np.ndarray,
) -> tuple[float | None, float | None, int]:
    """
    Находит основной тон, для которого прослеживается больше всего обертонов.
    """

    if pitch_values.size == 0:
        return None, None, 0

    frame_len = int(round(0.08 * sample_rate))
    half = frame_len // 2

    best_count = -1
    best_f0 = None
    best_time = None

    for time_sec, f0 in zip(pitch_times, pitch_values):
        center = int(round(time_sec * sample_rate))

        start = max(0, center - half)
        end = min(audio.size, start + frame_len)

        frame = audio[start:end]

        if frame.size < frame_len // 2:
            continue

        frame = frame - float(frame.mean())
        frame *= np.hanning(frame.size)

        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(frame.size, d=1.0 / sample_rate)

        if spectrum.size == 0:
            continue

        fundamental_band = (freqs >= f0 - 25) & (freqs <= f0 + 25)

        if not np.any(fundamental_band):
            continue

        fundamental_amp = float(np.max(spectrum[fundamental_band]) + 1e-12)

        harmonic_count = 0

        for k in range(2, 16):
            target = k * f0

            if target > sample_rate / 2:
                break

            band = (freqs >= target - 35) & (freqs <= target + 35)

            if not np.any(band):
                continue

            amp = float(np.max(spectrum[band]))

            if amp >= 0.12 * fundamental_amp:
                harmonic_count += 1

        if harmonic_count > best_count:
            best_count = harmonic_count
            best_f0 = float(f0)
            best_time = float(time_sec)

    if best_f0 is None:
        return None, None, 0

    return best_f0, best_time, best_count


def estimate_formants(
    f: np.ndarray,
    t: np.ndarray,
    zxx: np.ndarray,
    top_n: int = 3,
) -> list[float]:
    """
    Приближенная оценка формант.

    Используется энергетическая сетка:
    - по времени окно 0.1 с;
    - по частоте полоса 50 Гц;
    - центр полосы двигается с шагом 10 Гц.

    Затем выбираются самые сильные частотные области с разносом,
    чтобы не взять три соседних интервала одной и той же форманты.
    """

    power = np.abs(zxx) ** 2

    if power.size == 0:
        return []

    frame_energy = power.sum(axis=0)

    if frame_energy.size == 0:
        return []

    active = frame_energy >= 0.15 * float(frame_energy.max())

    if not np.any(active):
        active = np.ones_like(frame_energy, dtype=bool)

    centers = np.arange(FORMANT_MIN_HZ, FORMANT_MAX_HZ + 1, FORMANT_FREQ_STEP)

    band_energies: list[tuple[float, float]] = []

    for center in centers:
        f0 = center - ENERGY_DF / 2
        f1 = center + ENERGY_DF / 2

        f_mask = (f >= f0) & (f < f1)

        if not np.any(f_mask):
            continue

        e = float(power[np.ix_(f_mask, active)].sum())

        band_energies.append((center, e))

    band_energies.sort(key=lambda item: item[1], reverse=True)

    selected: list[float] = []

    for center, _energy in band_energies:
        if all(abs(center - existing) >= 180.0 for existing in selected):
            selected.append(float(center))

        if len(selected) >= top_n:
            break

    selected.sort()

    return selected


def find_energy_regions(
    key: str,
    f: np.ndarray,
    t: np.ndarray,
    zxx: np.ndarray,
    top_count: int = TOP_ENERGY_REGIONS,
) -> list[EnergyRegion]:
    power = np.abs(zxx) ** 2

    if t.size == 0 or f.size == 0:
        return []

    time_bins = np.arange(0.0, float(t[-1]) + ENERGY_DT, ENERGY_DT)
    freq_bins = np.arange(0.0, min(float(f[-1]), FORMANT_MAX_HZ) + ENERGY_DF, ENERGY_DF)

    candidates: list[tuple[float, float, float, float, float]] = []

    for ti in range(len(time_bins) - 1):
        t0 = time_bins[ti]
        t1 = time_bins[ti + 1]

        t_mask = (t >= t0) & (t < t1)

        if not np.any(t_mask):
            continue

        for fi in range(len(freq_bins) - 1):
            f0 = freq_bins[fi]
            f1 = freq_bins[fi + 1]

            if f1 < 50:
                continue

            f_mask = (f >= f0) & (f < f1)

            if not np.any(f_mask):
                continue

            local_power = power[np.ix_(f_mask, t_mask)]
            energy = float(local_power.sum())

            candidates.append((energy, t0, t1, f0, f1))

    candidates.sort(key=lambda item: item[0], reverse=True)

    regions: list[EnergyRegion] = []

    for rank, (energy, t0, t1, f0, f1) in enumerate(candidates[:top_count], start=1):
        regions.append(
            EnergyRegion(
                recording_key=key,
                rank=rank,
                time_start=t0,
                time_end=t1,
                freq_start=f0,
                freq_end=f1,
                energy=energy,
            )
        )

    return regions


def save_waveform_plot(audio: np.ndarray, sample_rate: int, title: str, path: Path) -> None:
    duration = audio.size / sample_rate
    time = np.linspace(0.0, duration, audio.size, endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

    ax.plot(time, audio, linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Амплитуда")

    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_spectrogram_plot(
    f: np.ndarray,
    t: np.ndarray,
    zxx: np.ndarray,
    formants: list[float],
    title: str,
    path: Path,
) -> None:
    amp_db = 20.0 * np.log10(np.abs(zxx) + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    mesh = ax.pcolormesh(t, f, amp_db, shading="auto")

    for formant in formants:
        ax.axhline(formant, linestyle="--", linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, Гц")

    ax.set_yscale("log")
    ax.set_ylim(50, min(8000, max(100, f[-1])))

    fig.colorbar(mesh, ax=ax, label="Амплитуда, дБ")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_pitch_plot(
    pitch_times: np.ndarray,
    pitch_values: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

    if pitch_values.size > 0:
        ax.plot(pitch_times, pitch_values, marker="o", markersize=2, linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Основной тон F0, Гц")

    ax.set_ylim(PITCH_MIN_HZ, PITCH_MAX_HZ)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_spectrum_plot(audio: np.ndarray, sample_rate: int, title: str, path: Path) -> None:
    window = np.hanning(audio.size)
    spectrum = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)

    amp_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

    ax.plot(freqs, amp_db, linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Амплитуда, дБ")

    ax.set_xscale("log")
    ax.set_xlim(50, min(12000, sample_rate / 2))
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def copy_to_src(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)


def process_recording(key: str) -> tuple[RecordingResult, list[EnergyRegion]]:
    title = REQUIRED_RECORDINGS[key]["title"]
    theoretical = REQUIRED_RECORDINGS[key]["theoretical_formants"]

    input_file_name, sample_rate, audio, channels = get_recording(key)

    audio_file_name = f"{key}_mono.wav"
    audio_path = AUDIO_DIR / audio_file_name
    save_wav(audio_path, sample_rate, audio)
    copy_to_src(audio_path, SRC_AUDIO_DIR / audio_file_name)

    f, t, zxx = compute_stft(audio, sample_rate)

    pitch_times, pitch_values = estimate_pitch_track(audio, sample_rate)

    if pitch_values.size > 0:
        pitch_min = float(np.min(pitch_values))
        pitch_max = float(np.max(pitch_values))
    else:
        pitch_min = None
        pitch_max = None

    richest_f0, richest_time, harmonic_count = find_richest_tone(
        audio,
        sample_rate,
        pitch_times,
        pitch_values,
    )

    formants = estimate_formants(f, t, zxx, top_n=3)

    errors: list[float] = []

    if theoretical and len(formants) >= 3:
        for actual, expected in zip(formants[:3], theoretical[:3]):
            errors.append(abs(actual - expected))

    waveform_file = f"{key}_waveform.png"
    spectrogram_file = f"{key}_spectrogram.png"
    pitch_file = f"{key}_pitch.png"
    spectrum_file = f"{key}_spectrum.png"

    save_waveform_plot(
        audio,
        sample_rate,
        f"{title}: осциллограмма",
        PLOTS_DIR / waveform_file,
    )

    save_spectrogram_plot(
        f,
        t,
        zxx,
        formants,
        f"{title}: спектрограмма STFT, окно Ханна",
        PLOTS_DIR / spectrogram_file,
    )

    save_pitch_plot(
        pitch_times,
        pitch_values,
        f"{title}: оценка основного тона",
        PLOTS_DIR / pitch_file,
    )

    save_spectrum_plot(
        audio,
        sample_rate,
        f"{title}: усредненный спектр",
        PLOTS_DIR / spectrum_file,
    )

    for file_name in [waveform_file, spectrogram_file, pitch_file, spectrum_file]:
        copy_to_src(PLOTS_DIR / file_name, SRC_PLOTS_DIR / file_name)

    regions = find_energy_regions(key, f, t, zxx, TOP_ENERGY_REGIONS)

    result = RecordingResult(
        key=key,
        title=title,
        input_file=input_file_name,
        sample_rate=sample_rate,
        channels_original=channels,
        duration_sec=audio.size / sample_rate,
        samples_count=audio.size,
        pitch_min_hz=pitch_min,
        pitch_max_hz=pitch_max,
        richest_f0_hz=richest_f0,
        richest_time_sec=richest_time,
        harmonic_count=harmonic_count,
        formants_hz=formants,
        theoretical_formants_hz=theoretical,
        formant_errors_hz=errors,
        waveform_file=waveform_file,
        spectrogram_file=spectrogram_file,
        pitch_file=pitch_file,
        spectrum_file=spectrum_file,
    )

    return result, regions


def save_summary_csv(results: list[RecordingResult]) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow(
            [
                "key",
                "title",
                "input_file",
                "sample_rate",
                "channels_original",
                "duration_sec",
                "pitch_min_hz",
                "pitch_max_hz",
                "richest_f0_hz",
                "richest_time_sec",
                "harmonic_count",
            ]
        )

        for item in results:
            writer.writerow(
                [
                    item.key,
                    item.title,
                    item.input_file,
                    item.sample_rate,
                    item.channels_original,
                    f"{item.duration_sec:.6f}",
                    "" if item.pitch_min_hz is None else f"{item.pitch_min_hz:.3f}",
                    "" if item.pitch_max_hz is None else f"{item.pitch_max_hz:.3f}",
                    "" if item.richest_f0_hz is None else f"{item.richest_f0_hz:.3f}",
                    "" if item.richest_time_sec is None else f"{item.richest_time_sec:.3f}",
                    item.harmonic_count,
                ]
            )


def save_formants_csv(results: list[RecordingResult]) -> None:
    with FORMANTS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow(
            [
                "key",
                "title",
                "formant_index",
                "estimated_hz",
                "theoretical_hz",
                "abs_error_hz",
            ]
        )

        for item in results:
            for idx, estimated in enumerate(item.formants_hz, start=1):
                theoretical = ""
                error = ""

                if idx <= len(item.theoretical_formants_hz):
                    theoretical_value = item.theoretical_formants_hz[idx - 1]
                    theoretical = f"{theoretical_value:.3f}"
                    error = f"{abs(estimated - theoretical_value):.3f}"

                writer.writerow(
                    [
                        item.key,
                        item.title,
                        idx,
                        f"{estimated:.3f}",
                        theoretical,
                        error,
                    ]
                )


def save_energy_regions_csv(regions: list[EnergyRegion]) -> None:
    with ENERGY_REGIONS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow(
            [
                "recording_key",
                "rank",
                "time_start_sec",
                "time_end_sec",
                "freq_start_hz",
                "freq_end_hz",
                "energy",
            ]
        )

        for region in regions:
            writer.writerow(
                [
                    region.recording_key,
                    region.rank,
                    f"{region.time_start:.3f}",
                    f"{region.time_end:.3f}",
                    f"{region.freq_start:.1f}",
                    f"{region.freq_end:.1f}",
                    f"{region.energy:.12f}",
                ]
            )


def format_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "не найдено"
    return f"{value:.{digits}f}"


def write_report(results: list[RecordingResult], regions: list[EnergyRegion]) -> None:
    lines: list[str] = []

    lines.append("# Лабораторная работа №10")
    lines.append("## Обработка голоса")
    lines.append("")
    lines.append(f"### Выбранный вариант: {VARIANT}. {LAB_TITLE}")
    lines.append("")
    lines.append("### Что было записано")
    lines.append("")
    lines.append("- `input_audio/a.wav` — протяжный звук `А` от низкого голоса до высокого.")
    lines.append("- `input_audio/i.wav` — протяжный звук `И` от низкого голоса до высокого.")
    lines.append("- `input_audio/animal.wav` — имитация лая, мяуканья или крик.")
    lines.append("")
    lines.append("Если этих файлов нет, программа создает демонстрационные записи, но для сдачи следует использовать собственный голос.")
    lines.append("")
    lines.append("### Метод анализа")
    lines.append("")
    lines.append("```text")
    lines.append("1. WAV переводится в mono.")
    lines.append("2. Строится STFT с окном Ханна.")
    lines.append("3. Частоты на спектрограмме отображаются в логарифмической шкале.")
    lines.append("4. Основной тон F0 оценивается автокорреляцией по коротким окнам.")
    lines.append("5. Самый тембрально окрашенный тон ищется по числу выраженных обертонов.")
    lines.append("6. Форманты оцениваются как частотные области с максимальной энергией.")
    lines.append("   Используется Δt = 0.1 с, Δf = 50 Гц, шаг частоты = 10 Гц.")
    lines.append("```")
    lines.append("")
    lines.append("### Сводная таблица")
    lines.append("")
    lines.append("| Запись | Файл | Каналы | Длительность, с | F0 min, Гц | F0 max, Гц | Самый окрашенный F0, Гц | Обертонов |")
    lines.append("|:--|:--|--:|--:|--:|--:|--:|--:|")

    for item in results:
        lines.append(
            f"| {item.title} | `{item.input_file}` | "
            f"{item.channels_original} | "
            f"{item.duration_sec:.3f} | "
            f"{format_optional(item.pitch_min_hz)} | "
            f"{format_optional(item.pitch_max_hz)} | "
            f"{format_optional(item.richest_f0_hz)} | "
            f"{item.harmonic_count} |"
        )

    lines.append("")
    lines.append("### Оценка формант")
    lines.append("")
    lines.append("| Запись | F1, Гц | F2, Гц | F3, Гц | Теоретические ориентиры |")
    lines.append("|:--|--:|--:|--:|:--|")

    for item in results:
        fvals = item.formants_hz + [float("nan")] * (3 - len(item.formants_hz))

        if item.theoretical_formants_hz:
            theory = ", ".join(f"{x:.0f}" for x in item.theoretical_formants_hz)
        else:
            theory = "нет"

        lines.append(
            f"| {item.title} | "
            f"{fvals[0]:.1f} | "
            f"{fvals[1]:.1f} | "
            f"{fvals[2]:.1f} | "
            f"{theory} |"
        )

    lines.append("")
    lines.append("- CSV со сводкой: `results_lab10/csv/summary.csv`")
    lines.append("- CSV с формантами: `results_lab10/csv/formants.csv`")
    lines.append("- CSV с энергетическими областями: `results_lab10/csv/energy_regions.csv`")
    lines.append("")

    for item in results:
        lines.append(f"## {item.title}")
        lines.append("")
        lines.append("| Осциллограмма | Спектрограмма |")
        lines.append("|:--:|:--:|")
        lines.append(
            f"| ![wave](src_lab10/plots/{item.waveform_file}) | "
            f"![spectrogram](src_lab10/plots/{item.spectrogram_file}) |"
        )
        lines.append("")
        lines.append("| Основной тон | Усредненный спектр |")
        lines.append("|:--:|:--:|")
        lines.append(
            f"| ![pitch](src_lab10/plots/{item.pitch_file}) | "
            f"![spectrum](src_lab10/plots/{item.spectrum_file}) |"
        )
        lines.append("")

        related_regions = [r for r in regions if r.recording_key == item.key][:5]

        if related_regions:
            lines.append("### Топ-5 энергетических областей")
            lines.append("")
            lines.append("| Ранг | t0, c | t1, c | f0, Гц | f1, Гц | Энергия |")
            lines.append("|---:|---:|---:|---:|---:|---:|")

            for region in related_regions:
                lines.append(
                    f"| {region.rank} | "
                    f"{region.time_start:.3f} | "
                    f"{region.time_end:.3f} | "
                    f"{region.freq_start:.1f} | "
                    f"{region.freq_end:.1f} | "
                    f"{region.energy:.6e} |"
                )

            lines.append("")

    lines.append("## Вывод")
    lines.append("")
    lines.append(
        "В лабораторной работе были проанализированы записи собственного голоса. "
        "Для каждого звука построены осциллограмма, спектрограмма с окном Ханна, "
        "график основного тона и усредненный спектр. Определены минимальная и "
        "максимальная частота голоса, найден основной тон с наибольшим количеством "
        "обертонов и оценены три наиболее сильные формантные области. Для звуков "
        "«А» и «И» форманты сравниваются с теоретическими ориентирами."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_dirs()

    results: list[RecordingResult] = []
    all_regions: list[EnergyRegion] = []

    for key in REQUIRED_RECORDINGS:
        print(f"Обработка записи: {key}")

        result, regions = process_recording(key)

        results.append(result)
        all_regions.extend(regions)

    save_summary_csv(results)
    save_formants_csv(results)
    save_energy_regions_csv(all_regions)

    shutil.copy2(SUMMARY_CSV, SRC_CSV_DIR / SUMMARY_CSV.name)
    shutil.copy2(FORMANTS_CSV, SRC_CSV_DIR / FORMANTS_CSV.name)
    shutil.copy2(ENERGY_REGIONS_CSV, SRC_CSV_DIR / ENERGY_REGIONS_CSV.name)

    write_report(results, all_regions)

    print("Лабораторная работа №10 выполнена.")
    print(f"Вариант: {VARIANT} — {LAB_TITLE}")
    print(f"Отчет: {REPORT_PATH}")
    print(f"Результаты: {RESULTS_DIR}")
    print("")
    print("Для сдачи запишите файлы:")
    print("input_audio/a.wav")
    print("input_audio/i.wav")
    print("input_audio/animal.wav")


if __name__ == "__main__":
    main()