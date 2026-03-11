import numpy as np
import requests
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import scipy.fft as fft
from scipy.optimize import minimize_scalar
import json
import base64
import sys
import os
import time
import traceback
from flask import Flask, request, jsonify, render_template
import webbrowser
import threading


app = Flask(__name__)

# Default Globals (Can be overridden by the web UI)
REW_API_URL = "http://localhost:4735"
TARGET_SAMPLE_RATE = 48000
FILTER_TAPS = 65536  

APP_STATE = {}

# ==============================================================================
# DSP & REW API FUNCTIONS
# ==============================================================================

def log_smoothed_fast(data, freqs, fraction=3, variable=False):
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    df = freqs[1] - freqs[0]
    cumsum = np.concatenate(([0.0], np.cumsum(data, dtype=np.float64)))
    for i in range(1, len(freqs)):
        f = freqs[i]
        if f == 0: continue
        if variable:
            if f <= 100.0: current_fraction = 48.0
            elif f >= 10000.0: current_fraction = 3.0
            else:
                t = (np.log10(f) - 2.0) / 2.0 
                current_fraction = 48.0 - t * 45.0
        else:
            current_fraction = fraction
            
        w = f * (2**(1.0/(2.0*current_fraction)) - 2**(-1.0/(2.0*current_fraction)))
        bin_w = int(max(1, round(w / df)))
        if bin_w <= 1: 
            smoothed[i] = data[i]
            continue
        start = max(0, i - bin_w // 2)
        end = min(len(data), i + (bin_w // 2) + 1)
        smoothed[i] = (cumsum[end] - cumsum[start]) / (end - start)
    return smoothed

def erb_smoothed_fast(data, freqs):
    """Applies Equivalent Rectangular Bandwidth (ERB) smoothing to magnitude data."""
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    df = freqs[1] - freqs[0]
    if df <= 0: return smoothed
    cumsum = np.concatenate(([0.0], np.cumsum(data, dtype=np.float64)))
    for i in range(1, len(freqs)):
        f = freqs[i]
        if f <= 0: 
            smoothed[i] = data[i]
            continue
        # Moore and Glasberg (1983) ERB formula
        erb_bw = 24.7 * ((4.37 * f / 1000.0) + 1.0)
        bin_w = int(max(1, round(erb_bw / df)))
        if bin_w <= 1:
            smoothed[i] = data[i]
            continue
        start = max(0, i - bin_w // 2)
        end = min(len(data), i + (bin_w // 2) + 1)
        smoothed[i] = (cumsum[end] - cumsum[start]) / (end - start)
    return smoothed

def detect_room_modes(freqs, mag_raw, min_freq=20.0, max_freq=120.0, max_modes=3):
    mag_smoothed = log_smoothed_fast(mag_raw, freqs, fraction=24, variable=False)
    mag_db = 20 * np.log10(np.maximum(mag_smoothed, 1e-12))
    df = freqs[1] - freqs[0]
    if df <= 0: return[]
    dist_bins = max(1, int(5.0 / df))
    peaks, properties = signal.find_peaks(mag_db, prominence=4.0, distance=dist_bins)
    valid_peaks =[]
    for i, p in enumerate(peaks):
        if min_freq <= freqs[p] <= max_freq:
            valid_peaks.append((freqs[p], properties['prominences'][i]))
    valid_peaks.sort(key=lambda x: x[1], reverse=True)
    top_modes = valid_peaks[:max_modes]
    top_modes.sort(key=lambda x: x[0])
    return top_modes

def get_rew_measurements():
    try:
        response = requests.get(f"{REW_API_URL}/measurements")
        response.raise_for_status()
        meas_ids = response.json() 
        detailed_measurements = {}
        for m_id in meas_ids:
            m_id = str(m_id) 
            info_response = requests.get(f"{REW_API_URL}/measurements/{m_id}")
            if info_response.status_code == 200:
                detailed_measurements[m_id] = info_response.json()
            else:
                detailed_measurements[m_id] = {'title': "Unknown Title"}
        return detailed_measurements
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"[!] Could not connect to REW at {REW_API_URL}. Is the API Server enabled?")

def fetch_ir_data(meas_id):
    endpoints =[f"{REW_API_URL}/measurements/{meas_id}/impulse-response", f"{REW_API_URL}/measurements/{meas_id}"]
    data = None
    for ep in endpoints:
        res = requests.get(ep)
        if res.status_code == 200:
            data = res.json()
            break
    if data is None: raise ValueError(f"API returned 404 for ID {meas_id}.")
    if isinstance(data, list): return np.array(data, dtype=np.float32)
    b64_str = None
    if isinstance(data, str): b64_str = data
    elif isinstance(data, dict):
        for key in['impulseResponse', 'ir', 'data', 'samples', 'y']:
            if key in data and isinstance(data[key], str):
                b64_str = data[key]
                break
        if not b64_str:
            for v in data.values():
                if isinstance(v, str) and len(v) > 1000:
                    b64_str = v
                    break
    try:
        return np.frombuffer(base64.b64decode(b64_str), dtype='>f4').astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to decode Base64: {e}")

def fetch_fr_data(meas_id):
    """
    Fetches frequency response data from the REW API.
    Supports linear and log spacing, Base64 or JSON lists, and multiple key variants.
    """
    res = requests.get(f"{REW_API_URL}/measurements/{meas_id}/frequency-response")
    if res.status_code != 200:
        raise ValueError(f"API returned {res.status_code} fetching Frequency Response for ID {meas_id}")
    
    data = res.json()
    
    # --- Format 1: Implicit Frequencies (startFreq + step/ppo) ---
    # This matches the core REW API 'FrequencyResponse' object structure.
    # We make it robust by checking for common key variants (singular, plural, smoothed).
    if isinstance(data, dict) and 'startFreq' in data:
        mag_key = next((k for k in ['magnitude', 'magnitudes', 'smoothedMagnitude'] if k in data), None)
        if mag_key:
            start_freq = float(data['startFreq'])
            
            # Use pointsPerOctave (log) or freqStep (linear)
            if 'freqStep' in data:
                freq_step = float(data['freqStep'])
                is_log = 1.0 < freq_step < 1.1
            elif 'pointsPerOctave' in data or 'ppo' in data:
                ppo_key = 'pointsPerOctave' if 'pointsPerOctave' in data else 'ppo'
                freq_step = 2.0**(1.0 / float(data[ppo_key]))
                is_log = True
            else:
                # Default fallback (unlikely, but prevents crash)
                freq_step = 1.0
                is_log = False

            mag_val = data[mag_key]
            if isinstance(mag_val, str):
                mags = np.frombuffer(base64.b64decode(mag_val), dtype='>f4').astype(np.float32)
            else:
                mags = np.array(mag_val, dtype=np.float32)
            
            if is_log:
                freqs = start_freq * (freq_step ** np.arange(len(mags)))
            else:
                freqs = start_freq + np.arange(len(mags)) * freq_step
            return freqs.astype(np.float32), mags

    # --- Format 2: Explicit Frequencies & Magnitudes in a Dictionary ---
    # e.g. {"frequencies": [...], "magnitudes": [...]} or {"f": "Base64", "m": "Base64"}
    if isinstance(data, dict):
        freq_key = next((k for k in data.keys() if k.lower() in ['freq', 'frequencies', 'freqs', 'f', 'frequency']), None)
        mag_key = next((k for k in data.keys() if k.lower() in ['mag', 'magnitudes', 'mags', 'm', 'magnitude', 'spl', 'smoothedmagnitude']), None)
        
        if freq_key and mag_key:
            f_val = data[freq_key]
            m_val = data[mag_key]
            
            # Handle Base64-encoded arrays inside the keys
            if isinstance(f_val, str):
                freqs = np.frombuffer(base64.b64decode(f_val), dtype='>f4').astype(np.float32)
            else:
                freqs = np.array(f_val, dtype=np.float32)
            
            if isinstance(m_val, str):
                mags = np.frombuffer(base64.b64decode(m_val), dtype='>f4').astype(np.float32)
            else:
                mags = np.array(m_val, dtype=np.float32)
                
            return freqs, mags

    # --- Format 3: List of Point Dictionaries ---
    # e.g. [{"f": 20, "m": 75}, {"f": 21, "m": 76}, ...]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        freq_key = next((k for k in data[0].keys() if k.lower() in ['freq', 'frequency', 'f']), None)
        mag_key = next((k for k in data[0].keys() if k.lower() in ['mag', 'magnitude', 'm', 'spl']), None)
        if freq_key and mag_key:
            return np.array([pt[freq_key] for pt in data], dtype=np.float32), np.array([pt[mag_key] for pt in data], dtype=np.float32)

    # --- Format 4: List of Lists ---
    # e.g. [[20, 75], [21, 76], ...]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) and len(data[0]) >= 2:
        return np.array([pt[0] for pt in data], dtype=np.float32), np.array([pt[1] for pt in data], dtype=np.float32)

    # If we reached here, parsing failed. Log keys for debugging.
    keys = list(data.keys()) if isinstance(data, dict) else f"List (len={len(data)})"
    raise ValueError(f"Unrecognized frequency response format from REW API for ID {meas_id}. Data keys: {keys}")

def parse_rew_house_curve(b64_str, target_freqs):
    """
    Parses a base64 encoded REW-style house curve (.txt) and interpolates it to match target_freqs.
    Returns: A magnitude array (linear, not real dB) representing the house curve shape.
    """
    try:
        content = base64.b64decode(b64_str).decode('utf-8')
        freqs_list = []
        db_list = []
        
        for line in content.split('\n'):
            line = line.strip()
            # Ignore comments and empty lines
            if not line or line.startswith('*') or line.startswith('//') or line.startswith('#'):
                continue
            
            # Extract numbers (handles standard space or tab separation)
            parts = line.split()
            if len(parts) >= 2:
                try:
                    f = float(parts[0])
                    db = float(parts[1])
                    freqs_list.append(f)
                    db_list.append(db)
                except ValueError:
                    continue
        
        if not freqs_list:
            return None
            
        freqs_arr = np.array(freqs_list)
        db_arr = np.array(db_list)
        
        # Sort to ensure monotonic increasing frequencies for numpy.interp
        sort_idx = np.argsort(freqs_arr)
        freqs_arr = freqs_arr[sort_idx]
        db_arr = db_arr[sort_idx]
        
        # Interpolate onto our high-resolution frequency grid
        # Extrapolate flat ends beyond the defined points of the custom curve
        db_interp = np.interp(target_freqs, freqs_arr, db_arr)
        
        # Convert from dB to linear magnitude multiplier
        # Normalize curve so 0 is unity gain anchor
        db_interp = db_interp - np.max(db_interp)
        linear_mag_curve = 10 ** (db_interp / 20.0)
        
        return linear_mag_curve
        
    except Exception as e:
        print(f"Error parsing custom house curve: {e}")
        return None

def generate_crossover(freqs, fc, btype, crossover_type='lr4', phase_type='linear'):
    """
    Generates a crossover filter (minimum or linear phase).
    btype: 'highpass' or 'lowpass'
    crossover_type: e.g., 'lr4', 'bw2', 'bw4', 'none'
    phase_type: 'linear' or 'minimum'
    """
    if crossover_type.lower() == 'none' or crossover_type.lower() == 'bypass':
        return np.ones_like(freqs, dtype=np.complex128)
        
    nyq = TARGET_SAMPLE_RATE / 2.0
    norm_fc = max(min(fc / nyq, 0.99), 1e-5)
    
    # Parse crossover type and order
    if len(crossover_type) < 3:
        ctype = 'lr'
        order = 4
    else:
        ctype = crossover_type[:2].lower()
        try:
            order = int(crossover_type[2:])
        except ValueError:
            order = 4
            
    # Generate the complex response h based on type
    if ctype == 'bw':
        b, a = signal.butter(order, norm_fc, btype=btype, analog=False)
        w, h = signal.freqz(b, a, worN=freqs, fs=TARGET_SAMPLE_RATE)
        
    elif ctype == 'bs':
        b, a = signal.bessel(order, norm_fc, btype=btype, analog=False, norm='phase')
        w, h = signal.freqz(b, a, worN=freqs, fs=TARGET_SAMPLE_RATE)
        
    else: # Linkwitz-Riley (Default)
        bw_order = max(1, order // 2)
        b, a = signal.butter(bw_order, norm_fc, btype=btype, analog=False)
        w, h = signal.freqz(b, a, worN=freqs, fs=TARGET_SAMPLE_RATE)
        h = h * h # Square Butterworth for LR
    
    if phase_type.lower() == 'minimum':
        return h # Return the complex response with its native minimum phase
    else:
        return np.abs(h) + 0j # Zero phase / Linear phase

def get_centered_ir(ir, is_lfe=False):
    peak_idx = np.argmax(np.abs(ir))
    N = FILTER_TAPS
    pre_samples = int(0.100 * TARGET_SAMPLE_RATE) if is_lfe else int(0.005 * TARGET_SAMPLE_RATE)
    post_samples = int(0.500 * TARGET_SAMPLE_RATE) if is_lfe else int(0.015 * TARGET_SAMPLE_RATE)
    
    # Ensure pre+post doesn't exceed FILTER_TAPS (avoids crash at 192kHz/high sample rates)
    total_requested = pre_samples + post_samples
    if total_requested > N:
        ratio = (N - 1) / total_requested
        pre_samples = int(pre_samples * ratio)
        post_samples = int(post_samples * ratio)
        # Final safety check for integer rounding
        if pre_samples + post_samples >= N:
            post_samples = N - pre_samples - 1

    ir_padded = np.pad(ir, (pre_samples, post_samples), mode='constant')
    new_peak = peak_idx + pre_samples
    sliced = ir_padded[new_peak - pre_samples:new_peak + post_samples]
    window = signal.windows.tukey(len(sliced), alpha=0.1 if is_lfe else 0.5)
    sliced = sliced * window
    
    padded = np.zeros(N)
    # Correctly handle potential indexing for pre/post windows
    padded[-pre_samples:] = sliced[:pre_samples]
    padded[:post_samples] = sliced[pre_samples:]
    return padded

def get_fdw_spectrum(ir_centered, freqs, cycles=5.0, fs=48000):
    N = len(ir_centered)
    H_fdw = np.zeros_like(freqs, dtype=np.complex128)
    H_standard = np.fft.rfft(ir_centered)
    for k, f in enumerate(freqs):
        if f < 20.0:
            H_fdw[k] = H_standard[k]
            continue
        win_len_s = cycles / f
        half_win_samples = int((win_len_s / 2.0) * fs)
        if half_win_samples * 2 >= N:
            H_fdw[k] = H_standard[k]
            continue
        t_pos = np.arange(half_win_samples) / fs
        t_neg = np.arange(-half_win_samples, 0) / fs
        t_valid = np.concatenate((t_neg, t_pos))
        ir_valid = np.concatenate((ir_centered[-half_win_samples:], ir_centered[:half_win_samples]))
        win = 0.5 * (1 + np.cos(2 * np.pi * t_valid / win_len_s))
        phasor = np.exp(-1j * 2 * np.pi * f * t_valid)
        H_fdw[k] = np.sum(ir_valid * win * phasor)
    return H_fdw

def generate_final_fir(H_complex, freqs, delay_s, log_func=print, is_lin_phase=True):
    if is_lin_phase and delay_s < 0.05:
        log_func(f"⚠️  Warning: Final FIR delay ({delay_s*1000:.1f} ms) might be too short to safely house sub-bass linear phase pre-ringing!")
    total_samples = delay_s * TARGET_SAMPLE_RATE
    int_shift = int(np.floor(total_samples))
    frac_shift_s = (total_samples - int_shift) / TARGET_SAMPLE_RATE
    H_shifted = H_complex * np.exp(1j * -2.0 * np.pi * freqs * frac_shift_s)
    h_time = fft.irfft(H_shifted, n=FILTER_TAPS)
    h_causal = np.roll(h_time, max(0, min(int_shift, FILTER_TAPS-1)))
    asym_win = np.ones(FILTER_TAPS)
    
    # Only apply fade-in if we have non-zero padding AND we are in linear phase mode
    if is_lin_phase and int_shift > 0:
        fade_in = max(2, int(int_shift * 0.1)) 
        asym_win[:fade_in] = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade_in)))
        
    fade_out = int(TARGET_SAMPLE_RATE * 0.010)
    asym_win[-fade_out:] = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fade_out)))
    return h_causal * asym_win

def get_exact_fractional_peak(H_complex, N):
    h_coarse = fft.irfft(H_complex, n=N)
    h_shifted = np.roll(h_coarse, N // 2)
    peak_idx = np.argmax(np.abs(h_shifted))
    k_mid = np.arange(1, len(H_complex) - 1)
    H_mid = H_complex[1:-1]
    H_0 = np.real(H_complex[0])
    H_Nyq = np.real(H_complex[-1])
    def idft_val(t_shifted):
        t = t_shifted - (N // 2)
        phase = np.exp(1j * 2 * np.pi * k_mid * t / N)
        val = H_0 + 2 * np.real(np.dot(H_mid, phase)) + H_Nyq * np.cos(np.pi * t)
        return -np.abs(val)
    res = minimize_scalar(idft_val, bounds=(peak_idx - 1.0, peak_idx + 1.0), method='bounded')
    return res.x - (N // 2)

def kirkeby_regularized_inverse(H_mag, freqs, target_mag, beta_db=12.0):
    """
    Kirkeby regularized spectral inversion with psychoacoustically-shaped
    frequency-dependent regularization.
    
    Instead of hard-clipping eq = target/measured, uses:
        eq = (target * H) / (H^2 + beta(f)^2)
    
    beta(f) is shaped to match human hearing sensitivity:
      - Below 20 Hz: maximum beta (inaudible, don't correct)
      - 20-200 Hz: moderate beta (room modes are audible)
      - 200-5000 Hz: minimum beta (peak hearing sensitivity, correct aggressively)
      - 5000-10000 Hz: gently rising beta
      - Above 10000 Hz: rising beta (diminishing returns, distortion risk)
      - Above 20000 Hz: maximum beta (inaudible)
    
    H_mag:       measured magnitude spectrum (linear, positive)
    freqs:       frequency array (Hz)
    target_mag:  desired target magnitude spectrum (linear, positive)
    beta_db:     controls overall regularization strength (dB). Default 12.
    
    Returns:     EQ magnitude curve (linear, positive)
    """
    beta_base = 10 ** (-beta_db / 20.0)     # Convert dB to linear floor
    
    # --- Normalize H and target so beta is meaningful ---
    # Without normalization, H_mag (linear SPL) is ~10-1000+, making
    # beta (~0.25) negligible and the regularization a no-op.
    # Using median gives moderate regularization; max would be too aggressive.
    scale = np.median(H_mag[H_mag > 1e-12])
    if scale < 1e-12:
        scale = 1.0
    H_norm = H_mag / scale
    T_norm = target_mag / scale
    
    # --- Build frequency-dependent regularization weight W(f) ---
    # W(f) = 1.0 means full regularization (gentle), W(f) ≈ 0 means minimal (aggressive)
    W = np.ones_like(freqs, dtype=np.float64)
    
    for i, f in enumerate(freqs):
        if f <= 0:
            W[i] = 1.0       # DC: full regularization
        elif f < 20:
            # Ramp from full reg down to moderate over sub-audible range
            W[i] = 1.0 - 0.7 * (f / 20.0)
        elif f < 200:
            # Moderate regularization in bass (room modes are audible but tricky)
            W[i] = 0.3
        elif f < 5000:
            # Minimum regularization: peak hearing sensitivity, correct hard
            t = (f - 200) / (5000 - 200)
            W[i] = 0.3 - 0.25 * t  # 0.3 → 0.05
        elif f < 10000:
            # Rising gently: HF sensitivity decreasing
            t = (f - 5000) / (10000 - 5000)
            W[i] = 0.05 + 0.25 * t  # 0.05 → 0.30
        elif f < 20000:
            # Continues rising
            t = (f - 10000) / (20000 - 10000)
            W[i] = 0.30 + 0.40 * t  # 0.30 → 0.70
        else:
            W[i] = 1.0       # Above audibility: full regularization
    
    # Compute the frequency-dependent beta
    beta_f = beta_base * W
    
    # Kirkeby regularized inverse on normalized values:
    # eq_norm = (T_norm * H_norm) / (H_norm^2 + beta^2)
    H2 = H_norm ** 2
    beta2 = beta_f ** 2
    eq_mag = (T_norm * H_norm) / (H2 + beta2)
    
    return np.maximum(eq_mag, 1e-12)

def mixed_phase_decompose(H_eq_mag, excess_phase, freqs, crossover_hz=500.0):
    """
    True mixed-phase filter design for correction filters.
    
    Below crossover_hz:  minimum-phase EQ (no pre-ringing, causal)
    Above crossover_hz:  linear-phase EQ (preserves transient accuracy)
    Transition:          1-octave Hann crossfade centered at crossover_hz
    
    H_eq_mag:       magnitude of the EQ correction (linear, from kirkeby or clip)
    excess_phase:   the unwrapped excess phase to linearize (radians)
    freqs:          frequency array (Hz)
    crossover_hz:   mixed-phase transition frequency (default 500.0)
    
    Returns: complex H_correction filter (magnitude + mixed phase)
    """
    N = (len(freqs) - 1) * 2  # Reconstruct FILTER_TAPS from rfft length
    
    # --- Crossfade window: 0 = min-phase region, 1 = linear-phase region ---
    # One-octave transition centered at crossover_hz
    f_low = crossover_hz / np.sqrt(2.0)   # ~0.707 * crossover
    f_high = crossover_hz * np.sqrt(2.0)  # ~1.414 * crossover
    
    W_linear = np.zeros_like(freqs)
    W_linear[freqs >= f_high] = 1.0
    idx_trans = (freqs > f_low) & (freqs < f_high)
    if np.any(idx_trans):
        W_linear[idx_trans] = 0.5 * (1 - np.cos(np.pi * (freqs[idx_trans] - f_low) / (f_high - f_low)))
    
    # --- Minimum-phase component (via cepstral liftering) ---
    lifter = np.zeros(N)
    lifter[0] = 1
    lifter[1:N//2] = 2
    if N % 2 == 0:
        lifter[N//2] = 1
    cepstrum = fft.irfft(np.log(np.maximum(H_eq_mag, 1e-12)), n=N)
    min_phase_spectrum = np.exp(fft.rfft(cepstrum * lifter))
    # min_phase_spectrum has magnitude ≈ H_eq_mag and minimum-phase angle
    
    # --- Linear-phase component (zero phase = magnitude only + excess phase correction) ---
    # negative excess phase = correction that linearizes the speaker
    lin_phase_correction = -excess_phase
    H_linear = H_eq_mag * np.exp(1j * lin_phase_correction)
    
    # --- Blend: min-phase in bass, linear-phase in treble ---
    H_mixed = min_phase_spectrum * (1.0 - W_linear) + H_linear * W_linear
    
    return H_mixed

def detect_speaker_rolloff(mag_raw, freqs, threshold_db=-10.0, ref_low=200.0, ref_high=2000.0):
    """
    Detects the natural low-end and high-end rolloff frequencies of a speaker
    by finding where the 1/3-octave-smoothed magnitude drops below threshold_db
    relative to the midband (ref_low–ref_high Hz) average.
    
    Returns (low_rolloff_hz, high_rolloff_hz).
    """
    # Use 1/3-octave smoothing to avoid mistaking dips for rolloff
    mag_smoothed = log_smoothed_fast(mag_raw, freqs, fraction=3, variable=False)
    mag_db = 20 * np.log10(np.maximum(mag_smoothed, 1e-12))
    
    # Selection of reference band: default 200–2000 Hz, or custom for subs
    idx_mid_low = np.argmin(np.abs(freqs - ref_low))
    idx_mid_high = np.argmin(np.abs(freqs - ref_high))
    if idx_mid_high <= idx_mid_low:
        idx_mid_high = idx_mid_low + 1
    midband_level_db = np.mean(mag_db[idx_mid_low:idx_mid_high])
    threshold = midband_level_db + threshold_db  # threshold_db is negative
    
    # Low-end rolloff: scan downward from midband
    low_rolloff_hz = freqs[1] if len(freqs) > 1 else 20.0
    for i in range(idx_mid_low, 0, -1):
        if mag_db[i] < threshold:
            low_rolloff_hz = freqs[i]
            break
    
    # High-end rolloff: scan upward from midband
    high_rolloff_hz = 20000.0
    for i in range(idx_mid_high, len(freqs)):
        if mag_db[i] < threshold:
            high_rolloff_hz = freqs[i]
            break
    
    return float(low_rolloff_hz), float(high_rolloff_hz)

def compute_spatial_variance_weight(position_ids, freqs, fdw_cycles, fs, threshold_db=3.0):
    """
    Compute a frequency-dependent weight W(f) in [0,1] based on cross-seat variance.
    W=1.0 at frequencies where all seats agree, W->0 where variance is high.
    
    position_ids: list of REW measurement IDs (one per listening position)
    threshold_db: std deviation (dB) at which W drops to 0.5
    """
    if len(position_ids) < 2:
        return np.ones_like(freqs)
    
    mags_db = []
    for pid in position_ids:
        try:
            ir = fetch_ir_data(int(pid))
            ir_long = get_centered_ir(ir, is_lfe=False)
            H = get_fdw_spectrum(ir_long, freqs, cycles=fdw_cycles, fs=fs)
            mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
            mag_smooth = log_smoothed_fast(mag_db, freqs, fraction=3)
            mags_db.append(mag_smooth)
        except Exception:
            continue
    
    if len(mags_db) < 2:
        return np.ones_like(freqs)
    
    std_db = np.std(mags_db, axis=0)
    W = 1.0 / (1.0 + (std_db / threshold_db) ** 2)
    return W

def get_crossover_threshold_db(crossover_type):
    """
    Returns the dB point at which a speaker's rolloff should be detected
    for optimal crossover placement with the given crossover type.
    
    Linkwitz-Riley: -6 dB (HPF + LPF each at -6dB sum to unity)
    Butterworth:    -3 dB (each filter is -3dB at fc)
    Bessel:         -3 dB (with norm='phase')
    """
    ctype = crossover_type[:2].lower() if len(crossover_type) >= 2 else 'lr'
    if ctype == 'lr':
        return -6.0
    else:  # bw, bs
        return -3.0

def detect_reflection_gap(ir, fs, threshold_ratio=0.15):
    """
    Detects the time gap between the direct sound peak and the first strong
    early reflection in an impulse response, using the Hilbert envelope.
    
    ir:               raw impulse response array
    fs:               sample rate
    threshold_ratio:  fraction of peak envelope amplitude that counts as a
                      'strong' reflection (default 0.15 = 15% of peak)
    
    Returns: gap_seconds (float). Clamped to [0.5ms, 20ms].
    """
    from scipy.signal import hilbert as hilbert_transform
    
    # Compute analytic signal envelope
    analytic = hilbert_transform(ir)
    envelope = np.abs(analytic)
    
    # Smooth the envelope to avoid false peaks from noise
    smooth_samples = max(1, int(0.0005 * fs))  # 0.5ms smoothing kernel
    kernel = np.ones(smooth_samples) / smooth_samples
    envelope_smooth = np.convolve(envelope, kernel, mode='same')
    
    # Find the direct sound peak
    peak_idx = np.argmax(envelope_smooth)
    peak_val = envelope_smooth[peak_idx]
    
    if peak_val < 1e-12:
        return 0.005  # fallback: 5ms
    
    # Search forward from peak for the first dip below threshold, then the
    # next rise above threshold (= first reflection)
    threshold = peak_val * threshold_ratio
    
    # First, find where envelope drops below threshold after the peak
    found_dip = False
    dip_idx = peak_idx
    for i in range(peak_idx + 1, min(len(envelope_smooth), peak_idx + int(0.030 * fs))):
        if envelope_smooth[i] < threshold:
            found_dip = True
            dip_idx = i
            break
    
    if not found_dip:
        return 0.005  # No clear dip found, fallback

    # Then find where envelope rises back above threshold (= reflection arrival)
    reflection_idx = dip_idx
    for i in range(dip_idx, min(len(envelope_smooth), peak_idx + int(0.030 * fs))):
        if envelope_smooth[i] > threshold:
            reflection_idx = i
            break
    
    gap_s = (reflection_idx - peak_idx) / fs
    
    # Clamp to sensible range
    return float(np.clip(gap_s, 0.0005, 0.020))

def ir_gap_to_fdw_cycles(gap_s, reference_freq=500.0):
    """
    Converts a direct-to-reflection time gap into an optimal FDW cycle count.
    
    The FDW window at a given frequency f has a half-length of:
        t_half = cycles / (2 * f)
    
    We want the full window (2 * t_half = cycles / f) to fit within the gap,
    so:  cycles = gap_s * f
    
    We use a reference frequency in the midband where FDW behavior matters most.
    Result is clamped to [3.0, 10.0].
    """
    cycles = gap_s * reference_freq
    return float(np.clip(cycles, 3.0, 10.0))

def detect_auto_house_curve(H_mains_mags, H_sub_mag, freqs, fc, fs):
    """
    Analyzes the average in-room steady-state magnitude to derive psychoacoustically
    appropriate house curve parameters by measuring the natural room gain slope.
    
    H_mains_mags:  list of linear magnitude arrays from each main speaker
    H_sub_mag:     linear magnitude array from the subwoofer (or None)
    freqs:         frequency array (Hz)
    fc:            crossover frequency (Hz)
    
    Returns: (house_boost_db, house_start_hz, house_end_hz)
    """
    # Build combined in-room magnitude: RMS average of all channels
    all_mags = list(H_mains_mags)
    if H_sub_mag is not None:
        all_mags.append(H_sub_mag)
    
    combined_mag = np.sqrt(np.mean(np.array(all_mags)**2, axis=0))
    combined_mag = np.maximum(combined_mag, 1e-12)
    
    # Smooth with 1/3 octave to get the trend, not individual modes
    combined_db = 20 * np.log10(combined_mag)
    combined_db_smooth = log_smoothed_fast(combined_db, freqs, fraction=3, variable=False)
    
    # Reference level: average in the midband (500-2000 Hz)
    idx_mid_low = np.argmin(np.abs(freqs - 500.0))
    idx_mid_high = np.argmin(np.abs(freqs - 2000.0))
    if idx_mid_high <= idx_mid_low:
        idx_mid_high = idx_mid_low + 1
    mid_level_db = np.mean(combined_db_smooth[idx_mid_low:idx_mid_high])
    
    # Measure how much louder (or quieter) the bass is relative to midband
    # Use the 30-80 Hz region as the "deep bass" reference
    idx_bass_low = np.argmin(np.abs(freqs - 30.0))
    idx_bass_high = np.argmin(np.abs(freqs - 80.0))
    if idx_bass_high <= idx_bass_low:
        idx_bass_high = idx_bass_low + 1
    bass_level_db = np.mean(combined_db_smooth[idx_bass_low:idx_bass_high])
    
    # Natural room gain = how much louder bass is vs. mids
    natural_room_gain = bass_level_db - mid_level_db
    
    # The house curve should follow a psychoacoustically pleasant bass shelf.
    # Research (Harman, Toole) suggests +3 to +6 dB is optimal for music,
    # +6 to +10 dB for cinema. We use the measured room gain as a guide
    # and target a boost that's close to 60-80% of the existing room gain
    # (the room is already doing some of the work).
    
    # If room gain is already high (>8dB), we target less boost (room does the work).
    # If room gain is low (<3dB), we target more boost (room isn't helping).
    if natural_room_gain > 8.0:
        target_boost = max(3.0, natural_room_gain * 0.5)
    elif natural_room_gain > 4.0:
        target_boost = max(4.0, natural_room_gain * 0.7)
    else:
        target_boost = max(4.0, min(8.0, 6.0 + (3.0 - natural_room_gain) * 0.5))
    
    # Clamp to sensible bounds
    house_boost = float(np.clip(target_boost, 2.0, 12.0))
    
    # Find where the room gain slope begins by scanning upward from bass
    # to find where the level crosses the midband reference
    slope_start_hz = 120.0  # default
    for i in range(idx_bass_high, idx_mid_low):
        if combined_db_smooth[i] <= mid_level_db + 1.0:
            slope_start_hz = float(freqs[i])
            break
    slope_start_hz = float(np.clip(slope_start_hz, 80.0, 300.0))
    
    # The slope end (where maximum boost is reached) should be well into the bass
    # Typically around half the crossover frequency or where bass levels off
    slope_end_hz = float(np.clip(fc * 0.8, 20.0, 120.0))
    
    return house_boost, slope_start_hz, slope_end_hz

def detect_schroeder_statistical(mag_raw, freqs, fs=48000, min_f=80.0, max_f=600.0, window_oct=0.25):
    """
    Detects the Schroeder frequency by analyzing the statistical variance
    of the magnitude response. Improved version with percentile-based baseline
    and trend detection to avoid premature floor detection.
    """
    mag_db = 20 * np.log10(np.maximum(mag_raw, 1e-12))
    
    # Calculate rolling variance in log-spaced windows
    variances = []
    test_freqs = []
    
    # Scan with finer steps for better trend detection
    curr_f = min_f
    while curr_f < max_f:
        f_low = curr_f / (2**(window_oct/2))
        f_high = curr_f * (2**(window_oct/2))
        
        idx = (freqs >= f_low) & (freqs <= f_high)
        if np.any(idx) and np.sum(idx) > 3:
            variances.append(np.std(mag_db[idx]))
            test_freqs.append(curr_f)
        
        curr_f *= 1.03 # 3% step for more resolution
        
    if not variances:
        return 200.0 # Fallback
        
    variances = np.array(variances)
    test_freqs = np.array(test_freqs)
    
    # Smooth the variance curve to find the trend
    v_smooth = log_smoothed_fast(variances, test_freqs, fraction=4)
    
    # establish a stochastic baseline: use the 20th percentile 
    # of the higher frequency region (above 300Hz)
    high_freq_idx = test_freqs > 300.0
    if np.any(high_freq_idx):
        baseline_v = np.percentile(v_smooth[high_freq_idx], 20)
    else:
        baseline_v = np.min(v_smooth)
        
    # We search from high to low for where the variance consistently rises above 
    # the baseline by a dynamic threshold (relative to baseline + spread)
    v_spread = np.std(v_smooth[high_freq_idx]) if np.any(high_freq_idx) else 1.0
    threshold = baseline_v + max(2.0, v_spread * 2.0)
    
    fc_detected = 250.0
    trend_count = 0
    required_trend = 3 # Consecutive points above threshold to confirm modal region
    
    for i in range(len(v_smooth)-1, 0, -1):
        if v_smooth[i] > threshold:
            trend_count += 1
        else:
            trend_count = 0
            
        if trend_count >= required_trend:
            # The Schroeder frequency is where it first crossed the threshold 
            # (which is the i + required_trend index)
            detected_idx = min(i + required_trend, len(v_smooth)-1)
            fc_detected = test_freqs[detected_idx]
            break
            
    # Safer fallback: if we still ended up at the floor without a strong trend, 
    # return a reasonable default for a room (200Hz).
    if fc_detected <= min_f + 5.0:
        fc_detected = 200.0
            
    return float(np.clip(fc_detected, min_f, 450.0))

def detect_auto_prc_frequency(H_eq_flat, freqs, smoothed_excess_phase, fs, delay_s, 
                               min_freq=100.0, max_freq=5000.0, step=100.0):
    """
    Iteratively tests PRC cutoff frequencies from TOP DOWN to find the 
    LEAST CONSERVATIVE safe value (highest frequency) that keeps 
    pre-ringing artifacts below psychoacoustic audibility thresholds.
    
    H_eq_flat:              complex EQ spectrum (magnitude + min phase)
    freqs:                  frequency array
    smoothed_excess_phase:  the windowed excess phase to correct
    fs:                     sample rate
    delay_s:                the FIR delay used for generating test FIRs
    min_freq:               lowest PRC freq (Hz)
    max_freq:               highest PRC freq to start searching from (Hz)
    step:                   frequency step between candidates (Hz)
    
    Returns: (best_prc_hz, peak_ratio_db)
    """
    def get_audibility_threshold_db(f):
        # Human ear is less sensitive to slow ringing at low frequencies
        # and extremely sensitive to high frequency pre-echo.
        if f <= 200:
            return -20.0  # 10% peak ratio
        elif f >= 2000:
            return -45.0  # 0.5% peak ratio
        else:
            # Linear interpolation in dB between 200Hz and 2000Hz
            # 200 -> -20, 2000 -> -45
            alpha = (f - 200) / (2000 - 200)
            return -20.0 + alpha * (-45.0 - (-20.0))

    best_prc_hz = 1000.0  # Safe default if search fails
    best_db = 0.0
    
    # Search from highest frequency DOWNWARDS (least conservative first)
    candidates = np.arange(max_freq, min_freq - step, -step)
    
    for prc_hz in candidates:
        # Build PRC window
        W_prc = np.ones_like(freqs)
        fade_start = prc_hz
        fade_end = prc_hz * 2.0
        W_prc[freqs >= fade_end] = 0.0
        idx_prc = (freqs > fade_start) & (freqs < fade_end)
        if np.any(idx_prc):
            W_prc[idx_prc] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_prc] - fade_start) / (fade_end - fade_start)))
        
        target_phase = -smoothed_excess_phase
        # Normalize phase at fade start to prevent wrapping clicks
        idx_fs = np.argmin(np.abs(freqs - fade_start))
        phase_shift = np.round(target_phase[idx_fs] / (2 * np.pi)) * (2 * np.pi)
        target_phase_prc = (target_phase - phase_shift) * W_prc
        
        H_lin = np.exp(1j * target_phase_prc)
        H_candidate = H_eq_flat * H_lin
        
        # Generate test FIR
        test_fir = generate_final_fir(H_candidate, freqs, delay_s, log_func=lambda x: None)
        
        # Evaluate peak-based pre-ringing ratio
        abs_fir = np.abs(test_fir)
        peak_idx = np.argmax(abs_fir)
        main_peak = abs_fir[peak_idx]
        
        if main_peak < 1e-12:
            continue
            
        # Analysis window: more than 2ms before the peak
        pre_margin = int(0.002 * fs)
        pre_window = abs_fir[:max(0, peak_idx - pre_margin)]
        
        if len(pre_window) == 0:
            max_pre_peak = 0.0
        else:
            max_pre_peak = np.max(pre_window)
            
        ratio = max_pre_peak / main_peak
        db = 20 * np.log10(ratio) if ratio > 1e-10 else -100.0
        
        threshold_db = get_audibility_threshold_db(prc_hz)
        
        if db <= threshold_db:
            # Found the highest frequency that is safe
            return float(prc_hz), float(db)
        
        # Track best relative to its frequency threshold
        margin = db - threshold_db
        if 'min_margin' not in locals() or margin < min_margin:
            min_margin = margin
            best_prc_hz = prc_hz
            best_db = db
            
    return float(best_prc_hz), float(best_db)

# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/api/run_phase1', methods=['POST'])
def run_phase1():
    global APP_STATE, REW_API_URL, TARGET_SAMPLE_RATE, FILTER_TAPS
    console_log = []
    def clog(msg):
        console_log.append(msg)
        print(msg)

    try:
        config = request.json
        mains_ids = config.get('mains_ids', [])
        lfe_id = config.get('lfe_id', '')
        fc = float(config.get('fc', 80.0))
        delay_ms = float(config.get('delay_ms', 75.0))
        house_boost = float(config.get('house_boost', 6.0))
        house_start = float(config.get('house_start', 120.0))
        house_end = float(config.get('house_end', 80.0))
        
        # Override Globals with Advanced Configs
        TARGET_SAMPLE_RATE = int(config.get('sample_rate', 48000))
        FILTER_TAPS = int(config.get('filter_taps', 65536))
        REW_API_URL = config.get('rew_api_url', "http://localhost:4735").rstrip('/')
        
        # Load other Advanced Configs
        fdw_cycles = float(config.get('fdw_cycles', 5.0))
        trans_start = float(config.get('trans_start', 200.0))
        trans_end = float(config.get('trans_end', 300.0))
        max_boost_low = float(config.get('max_boost_low', 6.0))
        max_boost_high = float(config.get('max_boost_high', 6.0))
        sub_percentile = float(config.get('sub_percentile', 15.0))
        global_smoothing = config.get('global_smoothing', 'erb')
        vol_match_low = float(config.get('vol_match_low', 500.0))
        vol_match_high = float(config.get('vol_match_high', 2000.0))
        
        mains_eq_enabled = config.get('mains_eq_enabled', True)
        mains_phase_lin_enabled = config.get('mains_phase_lin_enabled', True)
        vol_align_enabled = config.get('vol_align_enabled', True)
        direct_sound_vol_align = config.get('direct_sound_vol_align', False)
        sub_eq_enabled = config.get('sub_eq_enabled', True)
        sub_bandlimit_low_enabled = config.get('sub_bandlimit_low_enabled', True)
        sub_bandlimit_low_hz = float(config.get('sub_bandlimit_low_hz', 10.0))
        sub_bandlimit_high_enabled = config.get('sub_bandlimit_high_enabled', False)
        sub_bandlimit_high_hz = float(config.get('sub_bandlimit_high_hz', 150.0))
        mains_bandlimit_high_enabled = config.get('mains_bandlimit_high_enabled', False)
        mains_bandlimit_high_hz = float(config.get('mains_bandlimit_high_hz', 10000.0))
        preringing_reduction = config.get('preringing_reduction', True)
        prc_freq_hz = float(config.get('prc_freq_hz', 1000.0))
        
        # Regularized Inversion & Mixed-Phase Design
        regularized_inversion = config.get('regularized_inversion', False)
        reg_beta_db = float(config.get('reg_beta_db', 12.0))
        mixed_phase_enabled = config.get('mixed_phase_enabled', False)
        mixed_phase_crossover_hz = float(config.get('mixed_phase_crossover_hz', 500.0))
        sub_gd_eq_enabled = config.get('sub_gd_eq_enabled', False)
        
        # Mixed-phase subsumes PRC — force PRC off when mixed-phase is active
        if mixed_phase_enabled:
            preringing_reduction = False
        
        crossover_mains = config.get('crossover_mains', 'lr4')
        crossover_sub = config.get('crossover_sub', 'lr4')
        phase_mains = config.get('phase_mains', 'linear')
        phase_sub = config.get('phase_sub', 'linear')
        
        # Manual rolloff overrides (0 = auto-detect)
        mains_low_rolloff_hz = float(config.get('mains_low_rolloff_hz', 0))
        mains_high_rolloff_hz = float(config.get('mains_high_rolloff_hz', 0))
        sub_low_rolloff_hz = float(config.get('sub_low_rolloff_hz', 0))
        sub_high_rolloff_hz = float(config.get('sub_high_rolloff_hz', 0))
        
        # New Auto Features
        auto_fdw_enabled = config.get('auto_fdw_enabled', False)
        auto_house_curve_enabled = config.get('auto_house_curve_enabled', False)
        auto_prc_enabled = config.get('auto_prc_enabled', False)
        auto_transition_enabled = config.get('auto_transition_enabled', False)
        
        # Full-range mode detection:
        # Mains are full-range when there's no sub or the mains crossover is bypassed
        is_full_range_mains = (not lfe_id) or (crossover_mains.lower() in ('none', 'bypass'))
        # Sub is full-range when the sub crossover is bypassed
        is_full_range_sub = crossover_sub.lower() in ('none', 'bypass')
        
        auto_align_enabled = config.get('auto_align_enabled', True)
        auto_crossover_enabled = config.get('auto_crossover_enabled', False)
        # Advanced spatial averaging is not available on the ui and is disabled purposefully and to ignore it
        spatial_avg_enabled = False # config.get('spatial_avg_enabled', False)
        spatial_speakers = config.get('spatial_speakers', {})
        spatial_avg_threshold_db = float(config.get('spatial_avg_threshold_db', 3.0))
        
        clog("\n=== XPDRC 1.2 (WEB) ===")
        clog(f"Settings: {TARGET_SAMPLE_RATE}Hz / {FILTER_TAPS} Taps | REW API: {REW_API_URL}")
        clog("Fetching measurements from REW...")
        measurements = get_rew_measurements()
        freqs = fft.rfftfreq(FILTER_TAPS, 1/TARGET_SAMPLE_RATE)
        
        # Common-Ground Crossover Detection
        fc_for_main = {}  # m_id -> fc for that speaker
        if auto_crossover_enabled and not is_full_range_mains:
            clog("\n  -> Auto-detecting and applying optimal common crossover...")
            threshold_db = get_crossover_threshold_db(crossover_mains)
            detected_fcs = []
            for m_id in mains_ids:
                ir_detect = fetch_ir_data(m_id)
                ir_detect_long = get_centered_ir(ir_detect, is_lfe=True)
                H_detect = fft.rfft(ir_detect_long)
                low_rolloff, _ = detect_speaker_rolloff(np.abs(H_detect), freqs, threshold_db=threshold_db)
                det_fc = max(low_rolloff, 20.0)
                detected_fcs.append(det_fc)
                clog(f"     Main ID {m_id}: rolloff at {det_fc:.1f} Hz")
            
            fc_sub = float(np.mean(detected_fcs))
            # Standardize: all speakers + sub use the common average frequency
            for m_id in mains_ids:
                fc_for_main[m_id] = fc_sub
            clog(f"  -> Applied Common Crossover: {fc_sub:.1f} Hz (average of all rolloffs)")
        else:
            for m_id in mains_ids:
                fc_for_main[m_id] = fc
            fc_sub = fc
        
        # Store calculated cross-over state for Phase 2
        APP_STATE['fc'] = fc
        APP_STATE['delay_ms'] = delay_ms
        APP_STATE['fc_sub'] = fc_sub
        APP_STATE['fc_for_main'] = fc_for_main
        APP_STATE['lfe_id'] = lfe_id
        APP_STATE['mains_ids'] = mains_ids
        APP_STATE['is_full_range_mains'] = is_full_range_mains
        APP_STATE['is_full_range_sub'] = is_full_range_sub
        APP_STATE['auto_crossover_enabled'] = auto_crossover_enabled
        APP_STATE['crossover_mains'] = crossover_mains
        APP_STATE['crossover_sub'] = crossover_sub
        APP_STATE['phase_mains'] = phase_mains
        APP_STATE['phase_sub'] = phase_sub
        
        # Step 1: Acoustic Delays
        clog("\n[1] Processing Acoustic Delays...")
        delays_ms_dict = {m_id: 0.0 for m_id in mains_ids}
        if lfe_id: delays_ms_dict[lfe_id] = 0.0
        
        if auto_align_enabled:
            clog("  -> Auto Time-Align: reading IR peak times from REW...")
            for m_id in list(delays_ms_dict.keys()):
                info = measurements.get(m_id, {})
                peak_s = info.get('timeOfIRPeakSeconds')
                if peak_s is not None:
                    delays_ms_dict[m_id] = peak_s * 1000.0
                    clog(f"     ID {m_id} ({info.get('title', '?')}): IR peak at {peak_s*1000:.3f} ms")
                else:
                    clog(f"     ⚠️ ID {m_id}: No IR peak time available from REW, using 0.0 ms")
        else:
            provided_delays = config.get('acoustic_delays', {})
            for k, v in provided_delays.items():
                delays_ms_dict[str(k)] = float(v)

        global_anchor_ms = max(delays_ms_dict.values()) if delays_ms_dict else 0.0
        clog(f"   -> Furthest speaker arrives at {global_anchor_ms:+.3f} ms. This is the global time anchor.")

        
        # VBA Setup
        vba_enabled = config.get('vba_enabled', True)
        vba_modes = []
        vba_attn_db = float(config.get('vba_attn_db', -4.0))
        vba_taps = int(config.get('vba_taps', 4))
        ir_sub_cached = None
        
        if lfe_id and vba_enabled:
            clog("\n  -> Fetching Subwoofer data to automatically detect room modes...")
            ir_sub_cached = fetch_ir_data(lfe_id)
            ir_sub_centered = get_centered_ir(ir_sub_cached, is_lfe=True)
            H_lfe_raw = fft.rfft(ir_sub_centered)
            
            raw_modes_input = config.get('vba_modes_raw', '').strip()
            if raw_modes_input:
                vba_modes = [float(x.strip()) for x in raw_modes_input.split(',') if x.strip()]
                clog(f"  -> Using manual VBA modes: {vba_modes}")
            else:
                detected = detect_room_modes(freqs, np.abs(H_lfe_raw))
                if detected:
                    clog(f"  -> Successfully auto-detected {len(detected)} prominent mode(s):")
                    for i, (mf, mp) in enumerate(detected):
                        clog(f"     Mode {i+1}: {mf:.1f} Hz (Prominence: +{mp:.1f} dB)")
                    vba_modes = [mf for mf, mp in detected]
                else:
                    clog("  -> Could not auto-detect modes. VBA will be skipped.")
        
        # Step 2: MAINS Processing
        
        # Auto FDW cycles detection
        if auto_fdw_enabled and len(mains_ids) > 0:
            try:
                ir_first = fetch_ir_data(mains_ids[0])
                gap_s = detect_reflection_gap(ir_first, fs=TARGET_SAMPLE_RATE)
                fdw_cycles = ir_gap_to_fdw_cycles(gap_s)
                clog(f"  -> Auto FDW: detected {gap_s*1000:.1f}ms reflection gap, set FDW to {fdw_cycles:.1f} cycles")
            except Exception as e:
                clog(f"  -> Auto FDW failed: {e}. Falling back to manual {fdw_cycles:.1f} cycles.")

        # Auto Transition Detection
        if auto_transition_enabled and len(mains_ids) > 0:
            try:
                # Use the vector average magnitude for more stable detection
                H_mains_mag_list = []
                for m_id in mains_ids:
                    ir_tmp = fetch_ir_data(m_id)
                    ir_tmp_long = get_centered_ir(ir_tmp, is_lfe=True)
                    H_mains_mag_list.append(np.abs(fft.rfft(ir_tmp_long)))
                H_mains_rms_mag = np.sqrt(np.mean(np.array(H_mains_mag_list)**2, axis=0))
                
                schroeder_fc = detect_schroeder_statistical(H_mains_rms_mag, freqs, fs=TARGET_SAMPLE_RATE)
                trans_start = schroeder_fc * 0.8  # transition starts slightly below
                trans_end = schroeder_fc * 1.5    # ends well above to ensure stochastic region
                clog(f"  -> Auto Transition: Detected Schroeder Frequency at {schroeder_fc:.1f} Hz. Blending: {trans_start:.0f}Hz - {trans_end:.0f}Hz.")
            except Exception as e:
                clog(f"  -> Auto Transition failed: {e}. Falling back to manual {trans_start:.0f}Hz - {trans_end:.0f}Hz.")

        clog(f"\n[2] Processing Mains ({fdw_cycles}-Cycle FDW, Phase Linearization)...")
        # Full-range rolloff detection from vector average (before per-speaker loop)
        mains_rolloff_low = fc / 2.0  # default: same as standard mode
        mains_rolloff_high = 20000.0
        if is_full_range_mains or auto_house_curve_enabled:
            # Compute RMS magnitude average of all mains (avoids phase cancellation between misaligned speakers)
            H_mains_mag_list = []
            for m_id in mains_ids:
                ir_tmp = fetch_ir_data(m_id)
                ir_tmp_long = get_centered_ir(ir_tmp, is_lfe=True)
                H_mains_mag_list.append(np.abs(fft.rfft(ir_tmp_long)))
            H_mains_rms_mag = np.sqrt(np.mean(np.array(H_mains_mag_list)**2, axis=0))
            
            if is_full_range_mains:
                mains_rolloff_low, mains_rolloff_high = detect_speaker_rolloff(H_mains_rms_mag, freqs)
                if mains_low_rolloff_hz > 0:
                    mains_rolloff_low = mains_low_rolloff_hz
                    clog(f"  -> Mains low rolloff override: {mains_rolloff_low:.1f} Hz")
                else:
                    clog(f"  -> Auto-detected mains low rolloff at {mains_rolloff_low:.1f} Hz (-10dB, from vector avg of {len(mains_ids)} mains)")
                if mains_high_rolloff_hz > 0:
                    mains_rolloff_high = mains_high_rolloff_hz
                    clog(f"  -> Mains high rolloff override: {mains_rolloff_high:.1f} Hz")
                else:
                    clog(f"  -> Auto-detected mains high rolloff at {mains_rolloff_high:.1f} Hz (-10dB, from vector avg of {len(mains_ids)} mains)")
            
            if auto_house_curve_enabled:
                try:
                    sub_mag = None
                    if lfe_id:
                        ir_sub_raw = fetch_ir_data(lfe_id)
                        ir_sub_l = get_centered_ir(ir_sub_raw, is_lfe=True)
                        sub_mag = np.abs(fft.rfft(ir_sub_l))
                    
                    house_boost, house_start, house_end = detect_auto_house_curve(
                        H_mains_mag_list, sub_mag, freqs, fc_sub, fs=TARGET_SAMPLE_RATE
                    )
                    clog(f"  -> Auto House Curve detected: +{house_boost:.1f} dB (Slope: {house_start:.0f} Hz - {house_end:.0f} Hz)")
                except Exception as e:
                    clog(f"  -> Auto House Curve failed: {e}. Falling back to manual parameters.")

        H_correction_dict = {}
        H_fdw_dict = {}
        
        # Build spatial averaging mapping: averaged main_id -> list of raw position IDs
        spatial_positions_for_main = {}
        if spatial_avg_enabled and spatial_speakers:
            mains_channels = [ch for ch in spatial_speakers.keys() if ch.upper() not in ('LFE', 'SUB', 'SW')]
            for i, m_id in enumerate(mains_ids):
                if i < len(mains_channels):
                    ch_label = mains_channels[i]
                    spatial_positions_for_main[m_id] = [str(p) for p in spatial_speakers[ch_label]]
            clog(f"  -> Spatial Averaging enabled for {len(spatial_positions_for_main)} channels")
        
        for m_id in mains_ids:
            ir = fetch_ir_data(m_id)
            ir_short = get_centered_ir(ir, is_lfe=False) 
            ir_long = get_centered_ir(ir, is_lfe=True)   
            H_short = fft.rfft(ir_short)
            H_long = fft.rfft(ir_long)
            
            H_fdw = get_fdw_spectrum(ir_long, freqs, cycles=fdw_cycles, fs=TARGET_SAMPLE_RATE)
            H_fdw_dict[m_id] = H_fdw
            
            lifter = np.zeros(FILTER_TAPS)
            lifter[0] = 1
            lifter[1:FILTER_TAPS//2] = 2
            if FILTER_TAPS % 2 == 0: lifter[FILTER_TAPS//2] = 1


            if mains_eq_enabled:
                mag_fdw_smoothed = log_smoothed_fast(np.abs(H_fdw), freqs, fraction=3, variable=False)
                mag_raw_smoothed = log_smoothed_fast(np.abs(H_long), freqs, fraction=6, variable=False)
                
                idx_start_trans = np.argmin(np.abs(freqs - trans_start))
                idx_end_trans = np.argmin(np.abs(freqs - trans_end))
                if idx_end_trans > idx_start_trans:
                    offset = np.mean(mag_fdw_smoothed[idx_start_trans:idx_end_trans]) / np.maximum(np.mean(mag_raw_smoothed[idx_start_trans:idx_end_trans]), 1e-12)
                    mag_raw_smoothed *= offset
                    
                idx_10k = np.argmin(np.abs(freqs - 10000.0))
                if idx_10k <= idx_end_trans: idx_10k = idx_end_trans + 1
                target_level = np.mean(mag_fdw_smoothed[idx_end_trans:idx_10k])
                
                if regularized_inversion:
                    target_fdw = np.full_like(freqs, target_level)
                    target_raw = np.full_like(freqs, target_level)
                    
                    if auto_crossover_enabled and not is_full_range_mains:
                        H_target_acoustic = generate_crossover(freqs, fc_for_main[m_id], 'highpass', crossover_type=crossover_mains, phase_type='linear')
                        target_fdw = target_fdw * np.abs(H_target_acoustic)
                        target_raw = target_raw * np.abs(H_target_acoustic)
                    
                    eq_mag_fdw = kirkeby_regularized_inverse(mag_fdw_smoothed, freqs, target_fdw, beta_db=reg_beta_db)
                    eq_mag_raw = kirkeby_regularized_inverse(mag_raw_smoothed, freqs, target_raw, beta_db=reg_beta_db)
                    # Safety hard limits on top of regularization
                    eq_mag_fdw = np.clip(eq_mag_fdw, 1e-4, 10 ** (max_boost_high / 20.0))
                    eq_mag_raw = np.clip(eq_mag_raw, 1e-4, 10 ** (max_boost_low / 20.0))
                    clog(f"  -> Mains EQ: Kirkeby Regularized Inversion (β={reg_beta_db:.1f} dB)")
                else:
                    target_final = np.full_like(freqs, target_level)
                    if auto_crossover_enabled and not is_full_range_mains:
                        H_target_acoustic = generate_crossover(freqs, fc_for_main[m_id], 'highpass', crossover_type=crossover_mains, phase_type='linear')
                        target_final = target_final * np.abs(H_target_acoustic)
                        
                    eq_mag_fdw = target_final / np.maximum(mag_fdw_smoothed, 1e-12)
                    eq_mag_raw = target_final / np.maximum(mag_raw_smoothed, 1e-12)
                    eq_mag_fdw = np.clip(eq_mag_fdw, 1e-4, 10 ** (max_boost_high / 20.0))
                    eq_mag_raw = np.clip(eq_mag_raw, 1e-4, 10 ** (max_boost_low / 20.0)) 
                
                W_blend = np.zeros_like(freqs)
                W_blend[freqs > trans_end] = 1.0
                idx_trans = (freqs >= trans_start) & (freqs <= trans_end)
                if np.any(idx_trans):
                    W_blend[idx_trans] = 0.5 * (1 - np.cos(np.pi * (freqs[idx_trans] - trans_start) / (trans_end - trans_start)))
                    
                eq_mag_total = eq_mag_raw * (1.0 - W_blend) + eq_mag_fdw * W_blend
                
                W_fade = np.ones_like(freqs)
                
                if is_full_range_mains:
                    # Full-range: use previously computed rolloff limits
                    f_low_lower_end = mains_rolloff_low
                    f_low_lower_start = f_low_lower_end * 0.75
                    f_high_upper_start = min(mains_rolloff_high, 20000.0)
                    f_high_upper_end = f_high_upper_start * 1.1
                elif auto_crossover_enabled:
                    # Acoustic Target Alignment: stay active down to fc/2 to shape the slope
                    f_low_lower_end = fc_for_main[m_id] / 2.0
                    f_low_lower_start = f_low_lower_end * 0.75
                    f_high_upper_start = 20000.0
                    f_high_upper_end = 22000.0
                    clog(f"  -> Acoustic Target Alignment: Shaping slope at {fc_for_main[m_id]:.1f} Hz")
                else:
                    # Standard mode: sub handles low end, fade below fc/2
                    f_low_lower_end = fc_for_main[m_id] / 2.0
                    f_low_lower_start = f_low_lower_end * 0.75
                    f_high_upper_start = 20000.0
                    f_high_upper_end = 22000.0
                
                idx_ll_start = np.argmin(np.abs(freqs - f_low_lower_start))
                idx_ll_end = np.argmin(np.abs(freqs - f_low_lower_end))
                if idx_ll_end > idx_ll_start:
                    W_fade[:idx_ll_start] = 0.0
                    W_fade[idx_ll_start:idx_ll_end] = 0.5 * (1 - np.cos(np.pi * (freqs[idx_ll_start:idx_ll_end] - f_low_lower_start) / (f_low_lower_end - f_low_lower_start)))
                else:
                    W_fade[:idx_ll_end] = 0.0
                    
                idx_hi_start = np.argmin(np.abs(freqs - f_high_upper_start))
                idx_hi_end = np.argmin(np.abs(freqs - f_high_upper_end))
                if idx_hi_end > idx_hi_start:
                    W_fade[idx_hi_start:idx_hi_end] = W_fade[idx_hi_start:idx_hi_end] * 0.5 * (1 + np.cos(np.pi * (freqs[idx_hi_start:idx_hi_end] - f_high_upper_start) / (f_high_upper_end - f_high_upper_start)))
                    W_fade[idx_hi_end:] = 0.0
                    
                eq_mag_final = (eq_mag_total * W_fade) + (1.0 * (1 - W_fade))
                
                if mains_bandlimit_high_enabled:
                    W_mbl = np.ones_like(freqs)
                    mbl_start = mains_bandlimit_high_hz
                    mbl_end = mains_bandlimit_high_hz * 1.4
                    idx_mbl_s = np.argmin(np.abs(freqs - mbl_start))
                    idx_mbl_e = np.argmin(np.abs(freqs - mbl_end))
                    if idx_mbl_e > idx_mbl_s:
                        W_mbl[idx_mbl_s:idx_mbl_e] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_mbl_s:idx_mbl_e] - mbl_start) / (mbl_end - mbl_start)))
                    W_mbl[idx_mbl_e:] = 0.0
                    eq_mag_final = (eq_mag_final * W_mbl) + (1.0 * (1 - W_mbl))
                    clog(f"  -> Applied Mains Band-Limit High: no correction above {mains_bandlimit_high_hz:.0f} Hz.")
                
                # Apply house curve to full-range mains (bass boost normally from sub)
                if is_full_range_mains:
                    custom_hc_b64 = config.get('custom_house_curve')
                    hc_shape = None
                    if custom_hc_b64:
                        hc_shape = parse_rew_house_curve(custom_hc_b64, freqs)
                        if hc_shape is not None:
                            clog("  -> Applying Custom House Curve to Full-Range Mains.")
                        else:
                            clog("  -> ❌ WARNING: Uploaded Custom House Curve (.txt) was invalid and ignored. Using manual parameters below.")
                    if hc_shape is None:
                        hc_shape = np.ones_like(freqs)
                        idx_hc_end = np.argmin(np.abs(freqs - house_end))
                        idx_hc_start = np.argmin(np.abs(freqs - house_start))
                        if idx_hc_start > idx_hc_end:
                            hc_shape[idx_hc_end:idx_hc_start] = 10 ** (np.linspace(house_boost, 0, idx_hc_start - idx_hc_end) / 20.0)
                        hc_shape[:idx_hc_end] = 10 ** (house_boost / 20.0)
                        clog(f"  -> Applying House Curve to Full-Range Mains (+{house_boost:.1f}dB bass boost).")
                    eq_mag_final = eq_mag_final * hc_shape
                
                
                # Spatial Averaging: attenuate EQ where cross-seat variance is high
                if spatial_avg_enabled and m_id in spatial_positions_for_main:
                    speaker_positions = spatial_positions_for_main[m_id]
                    if len(speaker_positions) >= 2:
                        W_spatial = compute_spatial_variance_weight(
                            speaker_positions, freqs, fdw_cycles, TARGET_SAMPLE_RATE, 
                            threshold_db=spatial_avg_threshold_db
                        )
                        eq_mag_final = eq_mag_final * W_spatial + 1.0 * (1.0 - W_spatial)
                        clog(f"  -> Applied Spatial Averaging (threshold={spatial_avg_threshold_db:.1f} dB, {len(speaker_positions)} positions)")
                
                cepstrum = fft.irfft(np.log(np.maximum(eq_mag_final, 1e-12)), n=FILTER_TAPS)
                H_eq_flat = np.exp(fft.rfft(cepstrum * lifter))
            else:
                H_eq_flat = np.ones_like(freqs, dtype=np.complex128)
                clog("  -> Skipped Mains Magnitude EQ.")

            H_short_flat = H_short * H_eq_flat
            raw_mag = np.maximum(np.abs(H_short_flat), 1e-12)
            
            if mains_phase_lin_enabled:
                cepstrum = fft.irfft(np.log(raw_mag), n=FILTER_TAPS)
                min_phase_complex = fft.rfft(cepstrum * lifter)
                min_phase = np.imag(min_phase_complex)
                
                excess_complex = H_short_flat * np.exp(-1j * min_phase)
                excess_phase = np.unwrap(np.angle(excess_complex))
                
                smoothed_excess_phase = log_smoothed_fast(excess_phase, freqs, fraction=3, variable=False)
                
                if is_full_range_mains:
                    # Full-range: extend phase correction down to double the detected rolloff
                    # Starting at the -10dB point causes too much pre-ringing due to severe phase rotation
                    phase_low_limit = mains_rolloff_low * 2.0
                    phase_low_fade_start = phase_low_limit * 0.75
                    W = np.zeros_like(freqs)
                    W[freqs >= phase_low_limit] = 1.0
                    transition_idx = (freqs > phase_low_fade_start) & (freqs < phase_low_limit)
                    if np.any(transition_idx):
                        W[transition_idx] = 0.5 * (1 - np.cos(np.pi * (freqs[transition_idx] - phase_low_fade_start) / (phase_low_limit - phase_low_fade_start)))
                    clog(f"  -> Full-range phase correction: active from {phase_low_limit:.1f} Hz (rolloff * 2)")
                elif auto_crossover_enabled:
                    # Acoustic Target Alignment: linearize phase through the crossover point
                    phase_low_limit = fc_for_main[m_id] / 2.0
                    phase_low_fade_start = phase_low_limit * 0.75
                    W = np.zeros_like(freqs)
                    W[freqs >= phase_low_limit] = 1.0
                    transition_idx = (freqs > phase_low_fade_start) & (freqs < phase_low_limit)
                    if np.any(transition_idx):
                        W[transition_idx] = 0.5 * (1 - np.cos(np.pi * (freqs[transition_idx] - phase_low_fade_start) / (phase_low_limit - phase_low_fade_start)))
                    clog(f"  -> Linearizing phase through crossover point ({fc_for_main[m_id]:.1f} Hz)")
                else:
                    # Standard: fade out excess phase below crossover (sub handles that region)
                    W = np.zeros_like(freqs)
                    W[freqs >= fc_for_main[m_id]] = 1.0  
                    transition_idx = (freqs > fc_for_main[m_id]/2.0) & (freqs < fc_for_main[m_id])
                    W[transition_idx] = 0.5 * (1 - np.cos(np.pi * (freqs[transition_idx] - fc_for_main[m_id]/2.0) / (fc_for_main[m_id]/2.0)))
                smoothed_excess_phase_windowed = W * smoothed_excess_phase
                
                if mixed_phase_enabled:
                    # True Mixed-Phase: min-phase below crossover, linear-phase above
                    H_mixed = mixed_phase_decompose(
                        np.abs(H_eq_flat), smoothed_excess_phase_windowed, freqs,
                        crossover_hz=mixed_phase_crossover_hz
                    )
                    H_correction_dict[m_id] = H_mixed
                    clog(f"  -> Applied Mixed-Phase Design (crossover: {mixed_phase_crossover_hz:.0f} Hz)")
                else:
                    target_phase = -smoothed_excess_phase_windowed
                    
                    if auto_prc_enabled and preringing_reduction:
                        try:
                            # Use refined top-down psychoacoustic search
                            detected_hz, peak_db = detect_auto_prc_frequency(
                                H_eq_flat, freqs, smoothed_excess_phase_windowed, 
                                TARGET_SAMPLE_RATE, delay_s=(delay_ms/1000.0)
                            )
                            prc_freq_hz = detected_hz
                            clog(f"  -> Auto PRC: selected {prc_freq_hz:.0f} Hz (max pre-peak: {peak_db:.1f} dB)")
                        except Exception as e:
                            clog(f"  -> Auto PRC failed: {e}. Falling back to manual {prc_freq_hz:.0f} Hz.")

                    if preringing_reduction:
                        W_prc = np.ones_like(freqs)
                        fade_start = prc_freq_hz
                        fade_end = prc_freq_hz * 2.0
                        W_prc[freqs >= fade_end] = 0.0
                        idx_prc = (freqs > fade_start) & (freqs < fade_end)
                        if np.any(idx_prc):
                            W_prc[idx_prc] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_prc] - fade_start) / (fade_end - fade_start)))
                        
                        # Normalize phase to nearest 2pi at fade start to prevent unwinding/wrapping during fade-to-zero
                        idx_fs = np.argmin(np.abs(freqs - fade_start))
                        phase_shift = np.round(target_phase[idx_fs] / (2 * np.pi)) * (2 * np.pi)
                        target_phase = (target_phase - phase_shift) * W_prc
                        clog(f"  -> Applied Preringing Reduction (PRC) above {prc_freq_hz} Hz (normalized by {phase_shift/(2*np.pi):.0f}*2pi).")
                        
                    H_lin_phase = np.exp(1j * target_phase)
                    H_candidate = H_eq_flat * H_lin_phase
                    
                    # Iterative ringing validation for full-range mains
                    if is_full_range_mains:
                        max_phase_iters = 10
                        pre_ring_threshold = 0.05  # 5% energy before peak
                        current_phase_floor_hz = phase_low_limit
                        
                        for p_iter in range(max_phase_iters):
                            hpf_test = generate_crossover(freqs, fc_for_main[m_id], 'highpass', crossover_mains, phase_mains)
                            test_H = H_candidate * hpf_test
                            test_fir = generate_final_fir(test_H, freqs, 0.050, log_func=lambda x: None)
                            
                            peak_idx = np.argmax(np.abs(test_fir))
                            pre_margin = int(0.005 * TARGET_SAMPLE_RATE)
                            post_margin = int(0.050 * TARGET_SAMPLE_RATE)
                            total_energy = np.sum(test_fir**2)
                            if total_energy == 0:
                                break
                            pre_energy = np.sum(test_fir[:max(0, peak_idx - pre_margin)]**2)
                            post_energy = np.sum(test_fir[min(len(test_fir), peak_idx + post_margin):]**2)
                            pre_r = pre_energy / total_energy
                            post_r = post_energy / total_energy
                            
                            clog(f"    [Phase Iter {p_iter+1}] floor={current_phase_floor_hz:.1f} Hz | Pre: {pre_r*100:.1f}%, Post: {post_r*100:.1f}%")
                            
                            if pre_r <= pre_ring_threshold:
                                clog("    🟢 Mains phase correction passed ringing validation!")
                                break
                            elif p_iter < max_phase_iters - 1:
                                # Raise the phase correction floor by 10 Hz and rebuild
                                current_phase_floor_hz += 10.0
                                W_retry = np.zeros_like(freqs)
                                W_retry[freqs >= current_phase_floor_hz] = 1.0
                                fade_start_retry = current_phase_floor_hz * 0.75
                                idx_retry = (freqs > fade_start_retry) & (freqs < current_phase_floor_hz)
                                if np.any(idx_retry):
                                    W_retry[idx_retry] = 0.5 * (1 - np.cos(np.pi * (freqs[idx_retry] - fade_start_retry) / (current_phase_floor_hz - fade_start_retry)))
                                retry_phase = W_retry * smoothed_excess_phase
                                target_phase_retry = -retry_phase
                                if preringing_reduction:
                                    # Reuse normalization logic for retry loop
                                    idx_fs = np.argmin(np.abs(freqs - prc_freq_hz))
                                    phase_shift = np.round(target_phase_retry[idx_fs] / (2 * np.pi)) * (2 * np.pi)
                                    target_phase_retry = (target_phase_retry - phase_shift) * W_prc
                                H_lin_phase = np.exp(1j * target_phase_retry)
                                H_candidate = H_eq_flat * H_lin_phase
                            else:
                                clog("    🟡 Max phase iterations reached. Using safest filter.")
                    
                    H_correction_dict[m_id] = H_candidate
            else:
                H_lin_phase = np.ones_like(freqs, dtype=np.complex128)
                clog("  -> Skipped Mains Phase Linearization.")
                H_correction_dict[m_id] = H_eq_flat * H_lin_phase
            clog(f"✅ Processed Main ID {m_id}")

        # Step 3: Volume Alignment
        clog(f"\n[3] Mains Perceptual Volume Alignment ({vol_match_low}Hz-{vol_match_high}Hz)...")
        if vol_align_enabled:
            mains_levels_db = {}
            idx_vol_start = np.argmin(np.abs(freqs - vol_match_low))
            idx_vol_end = np.argmin(np.abs(freqs - vol_match_high))
            
            if direct_sound_vol_align:
                clog("  -> Using Direct Sound (FDW) for volume matching.")
                for m_id in mains_ids:
                    fdw_mag = np.abs(H_fdw_dict[m_id])
                    eq_mag = np.abs(H_correction_dict[m_id])
                    simulated_fdw_db = 20 * np.log10(np.maximum(fdw_mag * eq_mag, 1e-12))
                    mains_levels_db[m_id] = np.mean(simulated_fdw_db[idx_vol_start:idx_vol_end])
            else:
                for m_id in mains_ids:
                    rew_freqs, rew_mag_db = fetch_fr_data(m_id)
                    rew_mag_db_interp = np.interp(freqs, rew_freqs, rew_mag_db)
                    eq_mag = np.abs(H_correction_dict[m_id])
                    eq_mag_db = 20 * np.log10(np.maximum(eq_mag, 1e-12))
                    simulated_mag_db = rew_mag_db_interp + eq_mag_db
                    mains_levels_db[m_id] = np.mean(simulated_mag_db[idx_vol_start:idx_vol_end])
                
            target_mains_level_db = np.mean(list(mains_levels_db.values())) if mains_levels_db else 75.0
            
            for m_id in mains_ids:
                gain_db = target_mains_level_db - mains_levels_db[m_id]
                H_correction_dict[m_id] *= 10 ** (gain_db / 20.0)
                clog(f"  -> Main ID {m_id} Volume Adjusted by {gain_db:+.2f} dB")
        else:
            clog("  -> Skipped Main Volume Alignment.")

        # Step 4: Subwoofer EQ (Cut-Only)
        H_eq_min_phase = None
        H_vba = np.ones_like(freqs, dtype=np.complex128)
        
        if lfe_id:
            clog("\n[4] Generating LFE EQ & Applying Perceptual Post-EQ Alignment...")
            ir_sub = ir_sub_cached if ir_sub_cached is not None else fetch_ir_data(lfe_id)
            ir_sub_centered = get_centered_ir(ir_sub, is_lfe=True)
            H_lfe_raw = fft.rfft(ir_sub_centered)
            
            if vba_enabled and vba_modes:
                vba_gain = 10 ** (-abs(vba_attn_db) / 20.0)
                W_vba = np.ones_like(freqs)
                W_vba[freqs > 150.0] = 0.0
                idx_trans = (freqs >= 100.0) & (freqs <= 150.0)
                if np.any(idx_trans):
                    W_vba[idx_trans] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_trans] - 100.0) / 50.0))
                    
                for mode_freq in vba_modes:
                    vba_delay_s = 1.0 / mode_freq
                    H_vba_mode = np.ones_like(freqs, dtype=np.complex128)
                    for i in range(1, vba_taps + 1):
                        current_gain = (vba_gain ** i) * ((-1) ** i)
                        reflection = current_gain * W_vba * np.exp(-1j * 2 * np.pi * freqs * (vba_delay_s * i))
                        H_vba_mode += reflection
                    H_vba *= H_vba_mode
                    clog(f"  -> Processed Cascade VBA for Mode {mode_freq:.1f} Hz (Delay: {vba_delay_s*1000:.2f} ms)")
                
            H_lfe_simulated = H_lfe_raw * H_vba
            lfe_mag = log_smoothed_fast(np.abs(H_lfe_simulated), freqs, fraction=24, variable=False)
            
            if sub_eq_enabled:
                idx_half = np.argmin(np.abs(freqs - fc_sub/2.0))
                idx_double = np.argmin(np.abs(freqs - fc_sub*2.0))
                lfe_mag_db = 20 * np.log10(np.maximum(lfe_mag, 1e-12))
                
                sub_native_db = np.mean(lfe_mag_db[idx_half:idx_double])
                sub_native_level = 10 ** (sub_native_db / 20.0) 
                
                # Process custom uploaded house curve or default generated one
                custom_hc_b64 = config.get('custom_house_curve')
                hc_shape = None
                if custom_hc_b64:
                    hc_shape = parse_rew_house_curve(custom_hc_b64, freqs)
                    if hc_shape is not None:
                        clog("  -> Applying Custom House Curve to Subwoofer target.")
                    else:
                        clog("  -> ❌ WARNING: Uploaded Custom House Curve (.txt) was invalid and ignored. Using manual parameters below.")
                
                if hc_shape is None:
                    hc_shape = np.ones_like(freqs)
                    idx_hc_end = np.argmin(np.abs(freqs - house_end))
                    idx_hc_start = np.argmin(np.abs(freqs - house_start))
                    if idx_hc_start > idx_hc_end:
                        hc_shape[idx_hc_end:idx_hc_start] = 10 ** (np.linspace(house_boost, 0, idx_hc_start - idx_hc_end) / 20.0)
                    hc_shape[:idx_hc_end] = 10 ** (house_boost / 20.0)
                    clog(f"  -> Applying House Curve (+{house_boost:.1f}dB bass boost).")
                
                target_curve_original = hc_shape * sub_native_level
                
                idx_eval_start = np.argmin(np.abs(freqs - house_end))
                idx_eval_end = np.argmin(np.abs(freqs - fc_sub*2.0))
                
                if idx_eval_end > idx_eval_start:
                    gains_needed = lfe_mag[idx_eval_start:idx_eval_end] / np.maximum(hc_shape[idx_eval_start:idx_eval_end], 1e-12)
                    target_anchor = np.percentile(gains_needed, sub_percentile)
                else:
                    target_anchor = sub_native_level
                    
                target_curve_lowered = hc_shape * target_anchor
                if regularized_inversion:
                    eq_mag = kirkeby_regularized_inverse(lfe_mag, freqs, target_curve_lowered, beta_db=reg_beta_db)
                    eq_mag = np.clip(eq_mag, 1e-4, 1.122)  # Safety: still enforce cut-only ceiling
                    clog(f"  -> Sub EQ: Kirkeby Regularized Inversion (β={reg_beta_db:.1f} dB)")
                else:
                    eq_mag = target_curve_lowered / np.maximum(lfe_mag, 1e-12)
                    eq_mag = np.clip(eq_mag, 1e-4, 1.122) 
                
                if is_full_range_sub:
                    # Crossover bypass: auto-detect sub rolloff limits using a subwoofer-appropriate reference band (30-80 Hz)
                    sub_rolloff_low_auto, sub_rolloff_high_auto = detect_speaker_rolloff(np.abs(H_lfe_raw), freqs, ref_low=30.0, ref_high=80.0)
                    if sub_low_rolloff_hz > 0:
                        sub_rolloff_low_auto = sub_low_rolloff_hz
                        clog(f"  -> Sub low rolloff override: {sub_rolloff_low_auto:.1f} Hz")
                    else:
                        clog(f"  -> Auto-detected sub low rolloff at {sub_rolloff_low_auto:.1f} Hz (-10dB)")
                    if sub_high_rolloff_hz > 0:
                        sub_rolloff_high_auto = sub_high_rolloff_hz
                        clog(f"  -> Sub high rolloff override: {sub_rolloff_high_auto:.1f} Hz")
                    else:
                        clog(f"  -> Auto-detected sub high rolloff at {sub_rolloff_high_auto:.1f} Hz (-10dB)")
                    
                    # High-end fade using detected rolloff
                    fade_start = sub_rolloff_high_auto
                    fade_end = sub_rolloff_high_auto * 1.4
                    W_eq = np.ones_like(freqs)
                    idx_fade_start = np.argmin(np.abs(freqs - fade_start))
                    idx_fade_end = np.argmin(np.abs(freqs - fade_end))
                    if idx_fade_end > idx_fade_start:
                        W_eq[idx_fade_start:idx_fade_end] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_fade_start:idx_fade_end] - fade_start) / (fade_end - fade_start)))
                        W_eq[idx_fade_end:] = 0.0
                    
                    # Low-end fade using detected rolloff
                    low_fade_end = sub_rolloff_low_auto
                    low_fade_start = low_fade_end * 0.75
                    idx_lf_start = np.argmin(np.abs(freqs - low_fade_start))
                    idx_lf_end = np.argmin(np.abs(freqs - low_fade_end))
                    if idx_lf_end > idx_lf_start:
                        W_eq[:idx_lf_start] = 0.0
                        W_eq[idx_lf_start:idx_lf_end] = 0.5 * (1 - np.cos(np.pi * (freqs[idx_lf_start:idx_lf_end] - low_fade_start) / (low_fade_end - low_fade_start)))
                    else:
                        W_eq[:idx_lf_end] = 0.0
                else:
                    # Standard mode: fade above fc*2 to fc*4
                    fade_start = fc_sub * 2.0
                    fade_end = fc_sub * 4.0
                    W_eq = np.ones_like(freqs)
                    idx_fade_start = np.argmin(np.abs(freqs - fade_start))
                    idx_fade_end = np.argmin(np.abs(freqs - fade_end))
                    if idx_fade_end > idx_fade_start:
                        W_eq[idx_fade_start:idx_fade_end] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_fade_start:idx_fade_end] - fade_start) / (fade_end - fade_start)))
                        W_eq[idx_fade_end:] = 0.0
                    
                eq_mag = (eq_mag * W_eq) + (1.0 * (1 - W_eq))
                
                cepstrum = fft.irfft(np.log(eq_mag), n=FILTER_TAPS)
                lifter = np.zeros(FILTER_TAPS)
                lifter[0] = 1
                lifter[1:FILTER_TAPS//2] = 2
                if FILTER_TAPS % 2 == 0: lifter[FILTER_TAPS//2] = 1
                H_eq_min_phase = np.exp(fft.rfft(cepstrum * lifter))
                
                H_sim_sub_eq = H_lfe_simulated * H_eq_min_phase
                actual_post_eq_mag = log_smoothed_fast(np.abs(H_sim_sub_eq), freqs, fraction=3, variable=False)
                actual_post_eq_db = np.mean(20 * np.log10(np.maximum(actual_post_eq_mag[idx_half:idx_double], 1e-12)))
                expected_target_db = np.mean(20 * np.log10(np.maximum(target_curve_original[idx_half:idx_double], 1e-12)))
                
                post_gain_db = expected_target_db - actual_post_eq_db
                H_eq_min_phase *= 10 ** (post_gain_db / 20.0)
                
                # Band-limit: force H_eq_min_phase to unity outside the allowed correction range.
                # Applied AFTER post-gain so the broadband level shift can't leak into excluded bands.
                if sub_bandlimit_low_enabled:
                    W_bl_low = np.ones_like(freqs)
                    bl_low_start = sub_bandlimit_low_hz * 0.7
                    bl_low_end = sub_bandlimit_low_hz
                    idx_bll_s = np.argmin(np.abs(freqs - bl_low_start))
                    idx_bll_e = np.argmin(np.abs(freqs - bl_low_end))
                    W_bl_low[:idx_bll_s] = 0.0
                    if idx_bll_e > idx_bll_s:
                        W_bl_low[idx_bll_s:idx_bll_e] = 0.5 * (1 - np.cos(np.pi * (freqs[idx_bll_s:idx_bll_e] - bl_low_start) / (bl_low_end - bl_low_start)))
                    H_eq_min_phase = H_eq_min_phase * W_bl_low + (1.0 + 0j) * (1 - W_bl_low)
                    clog(f"  -> Applied Sub Band-Limit Low: no correction below {sub_bandlimit_low_hz:.0f} Hz.")
                
                if sub_bandlimit_high_enabled:
                    W_bl_high = np.ones_like(freqs)
                    bl_high_start = sub_bandlimit_high_hz
                    bl_high_end = sub_bandlimit_high_hz * 1.4
                    idx_blh_s = np.argmin(np.abs(freqs - bl_high_start))
                    idx_blh_e = np.argmin(np.abs(freqs - bl_high_end))
                    if idx_blh_e > idx_blh_s:
                        W_bl_high[idx_blh_s:idx_blh_e] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_blh_s:idx_blh_e] - bl_high_start) / (bl_high_end - bl_high_start)))
                    W_bl_high[idx_blh_e:] = 0.0
                    H_eq_min_phase = H_eq_min_phase * W_bl_high + (1.0 + 0j) * (1 - W_bl_high)
                    clog(f"  -> Applied Sub Band-Limit High: no correction above {sub_bandlimit_high_hz:.0f} Hz.")
                
                clog(f"  -> Target Curve dynamically shifted down (Anchor: {sub_percentile}th percentile) to enforce Cut-Only EQ.")
                clog(f"  -> Post-EQ Subwoofer broadband gain adjusted by {post_gain_db:+.2f} dB.")
            else:
                H_eq_min_phase = np.ones_like(freqs, dtype=np.complex128)
                clog("  -> Skipped Subwoofer House Curve & EQ.")

            # --- Sub Group Delay EQ (time-limited excess phase correction) ---
            # Computes excess phase from the already-EQ'd subwoofer, then applies
            # a time-domain-windowed correction to reduce group delay anomalies.
            # The IR is aggressively windowed to limit both pre and post ringing.
            if sub_gd_eq_enabled:
                def evaluate_ringing(fir_array, fs):
                    peak_idx = np.argmax(np.abs(fir_array))
                    pre_margin = int(0.005 * fs)   # Pre-ringing > 5ms before peak
                    post_margin = int(0.050 * fs)  # Post-ringing > 50ms after peak
                    
                    total_energy = np.sum(fir_array**2)
                    if total_energy == 0: return 1.0, 1.0
                    
                    pre_energy = np.sum(fir_array[:max(0, peak_idx - pre_margin)]**2)
                    post_energy = np.sum(fir_array[min(len(fir_array), peak_idx + post_margin):]**2)
                    
                    return pre_energy / total_energy, post_energy / total_energy

                H_sub_eqd_base = H_lfe_simulated * H_eq_min_phase
                
                # Minimum phase from EQ'd magnitude
                eqd_log_mag = np.log(np.maximum(np.abs(H_sub_eqd_base), 1e-12))
                ceps_eqd = fft.irfft(eqd_log_mag, n=FILTER_TAPS)
                lifter_gd = np.zeros(FILTER_TAPS)
                lifter_gd[0] = 1
                lifter_gd[1:FILTER_TAPS//2] = 2
                if FILTER_TAPS % 2 == 0: lifter_gd[FILTER_TAPS//2] = 1
                eqd_min_phase_imag = np.imag(fft.rfft(ceps_eqd * lifter_gd))
                
                # Excess phase
                eqd_phase = np.unwrap(np.angle(H_sub_eqd_base))
                excess_phase_sub_base = eqd_phase - eqd_min_phase_imag
                
                # Frequency Window (fc/2 to 2*fc)
                gd_low = fc_sub / 2.0
                gd_high = fc_sub * 2.0
                W_gd = np.zeros_like(freqs)
                idx_core = (freqs >= gd_low) & (freqs <= gd_high)
                W_gd[idx_core] = 1.0
                
                fade_low_start = gd_low / np.sqrt(2.0)
                idx_fi = (freqs > fade_low_start) & (freqs < gd_low)
                if np.any(idx_fi):
                    W_gd[idx_fi] = 0.5 * (1 - np.cos(np.pi * (freqs[idx_fi] - fade_low_start) / (gd_low - fade_low_start)))
                fade_high_end = gd_high * np.sqrt(2.0)
                idx_fo = (freqs > gd_high) & (freqs < fade_high_end)
                if np.any(idx_fo):
                    W_gd[idx_fo] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_fo] - gd_high) / (fade_high_end - gd_high)))

                omega = 2 * np.pi * freqs
                
                # --- Iterative Group Delay Tuning ---
                clog("\n  -> Optimizing Subwoofer Group Delay via Iterative Ringing Analysis...")
                max_iterations = 10
                best_H_eq = H_eq_min_phase
                
                frac_smooth = 3.0  # Starts with 1/3 octave
                max_phase_caps = [np.pi/2, np.pi/4, np.pi/8] # [0ms, 3ms, 8ms] limits
                pre_ring_threshold = 0.05  # 5% max energy before peak
                
                for i in range(max_iterations):
                    excess_phase_smoothed = log_smoothed_fast(excess_phase_sub_base, freqs, fraction=frac_smooth, variable=False)
                    
                    gd_eqd = np.zeros_like(freqs)
                    gd_eqd[1:] = -np.diff(np.unwrap(np.angle(H_sub_eqd_base))) / np.maximum(np.diff(omega), 1e-20)
                    gd_eqd[0] = gd_eqd[1]
                    gd_eqd_smoothed = log_smoothed_fast(gd_eqd, freqs, fraction=frac_smooth, variable=False)
                    
                    idx_sub = (freqs >= gd_low) & (freqs <= gd_high)
                    gd_baseline = np.median(gd_eqd_smoothed[idx_sub]) if np.any(idx_sub) else 0.0
                    gd_excess_ms = np.maximum(gd_eqd_smoothed - gd_baseline, 0.0) * 1000.0
                    
                    max_phase_per_bin = np.interp(gd_excess_ms, [0.0, 3.0, 8.0], max_phase_caps)
                    
                    phase_raw = excess_phase_smoothed * W_gd
                    phase_correction = np.clip(phase_raw, -max_phase_per_bin, max_phase_per_bin)
                    
                    # Secondary smoothing ensures phase_correction has no jagged edges which directly cause ringing
                    phase_correction = log_smoothed_fast(phase_correction, freqs, fraction=frac_smooth, variable=False)
                    
                    H_gd_allpass = np.exp(-1j * phase_correction)
                    
                    # Validate ringing on the actual eventual output FIR
                    lpf = generate_crossover(freqs, fc_sub, 'lowpass', crossover_sub, phase_sub)
                    test_H = H_eq_min_phase * H_gd_allpass * lpf * H_vba  # Full subwoofer processing chain
                    
                    # Use standard time delay (e.g. 50ms) to ensure peak evaluates safely away from zero
                    test_pad_s = 0.050
                    test_fir = generate_final_fir(test_H, freqs, test_pad_s, log_func=lambda x: None)
                    
                    pre_r, post_r = evaluate_ringing(test_fir, TARGET_SAMPLE_RATE)
                    clog(f"    [Iteration {i+1}] frac={frac_smooth:.1f}, cap={max_phase_caps[0]/np.pi:.2f}π | Pre: {pre_r*100:.1f}%, Post: {post_r*100:.1f}%")
                    
                    if pre_r <= pre_ring_threshold:
                        best_H_eq = H_eq_min_phase * H_gd_allpass
                        clog("    🟢 Subwoofer GD EQ passed ringing validation!")
                        break
                    elif i == max_iterations - 1:
                        if pre_r > 0.14:
                            best_H_eq = H_eq_min_phase
                            clog(f"    🔴 GD EQ ditched entirely due to excessive pre-ringing ({pre_r*100:.1f}% > 14%) at the final iteration.")
                        else:
                            best_H_eq = H_eq_min_phase * H_gd_allpass
                            clog(f"    🟡 Max iterations reached. Using safest filter (Pre: {pre_r*100:.1f}%).")
                        break
                    else:
                        # Relax parameters to reduce sharpness
                        frac_smooth = max(0.5, frac_smooth - 0.8) # 1/2 octave or 1/1 octave
                        max_phase_caps = [c * 0.65 for c in max_phase_caps]
                        pre_ring_threshold += 0.01

                H_eq_min_phase = best_H_eq

        # Step 5: Generate Base FIRs
        clog("\n[5] Generating Base Filters (No Sub Alignment) for REW Processing...")
        base_firs = {}
        target_base_delay_s = delay_ms / 1000.0
        T_target = target_base_delay_s + (global_anchor_ms / 1000.0)
        mains_exact_delays_s = {}
        
        for m_id in mains_ids:
            if auto_crossover_enabled and not is_full_range_mains:
                hpf = np.ones_like(freqs, dtype=np.complex128)
                clog(f"  -> Main ID {m_id}: Using Acoustic Target Alignment (FIR-baked slope)")
            else:
                hpf = generate_crossover(freqs, fc_for_main[m_id], 'highpass', crossover_mains, phase_mains)
            
            FIR_base = H_correction_dict[m_id] * hpf
            ir = fetch_ir_data(m_id)
            ir_centered = get_centered_ir(ir, is_lfe=False)
            H_sys = fft.rfft(ir_centered) * FIR_base
            
            induced_shift_samples = get_exact_fractional_peak(H_sys, FILTER_TAPS)
            induced_shift_s = induced_shift_samples / TARGET_SAMPLE_RATE
            
            T_current = (delays_ms_dict[m_id] / 1000.0) + induced_shift_s
            exact_pad_s = T_target - T_current
            mains_exact_delays_s[m_id] = exact_pad_s
            clog(f"  -> Main ID {m_id} Peak Shift: {induced_shift_s*1000:+.3f} ms | Baked Base Delay: {exact_pad_s*1000:.3f} ms")
            
            base_firs[m_id] = generate_final_fir(FIR_base, freqs, exact_pad_s, log_func=clog)

        pad_s_sub = 0.0
        if lfe_id and H_eq_min_phase is not None:
            pad_s_sub = (global_anchor_ms - delays_ms_dict[lfe_id]) / 1000.0
            lpf = generate_crossover(freqs, fc_sub, 'lowpass', crossover_sub, phase_sub)
            embedded_target_sub = H_eq_min_phase * lpf * H_vba
            base_firs[lfe_id] = generate_final_fir(embedded_target_sub, freqs, target_base_delay_s + pad_s_sub, log_func=clog)
            
        global_max = max([np.max(np.abs(f)) for f in base_firs.values()])
        
        for m_id, generated_fir in base_firs.items():
            normalized_fir = generated_fir / global_max
            safe_title = "".join([c for c in measurements[m_id].get('title', f'ID{m_id}') if c.isalnum()]).strip()
            filename = f"XPDRC_BASE_{safe_title}.wav"
            wavfile.write(filename, TARGET_SAMPLE_RATE, normalized_fir.astype(np.float32))
            clog(f"💾 Saved Base FIR: {filename}")

        # Save to APP_STATE
        APP_STATE.clear()
        APP_STATE.update({
            'mains_ids': mains_ids,
            'lfe_id': lfe_id,
            'freqs': freqs,
            'H_correction_dict': H_correction_dict,
            'H_eq_min_phase': H_eq_min_phase,
            'crossover_mains': crossover_mains,
            'crossover_sub': crossover_sub,
            'phase_mains': phase_mains,
            'phase_sub': phase_sub,
            'H_vba': H_vba,
            'mains_exact_delays_s': mains_exact_delays_s,
            'pad_s_sub': pad_s_sub,
            'target_base_delay_s': target_base_delay_s,
            'measurements': measurements,
            'delays_ms': delays_ms_dict,
            'house_boost': house_boost,
            'house_start': house_start,
            'house_end': house_end,
            'custom_house_curve': config.get('custom_house_curve', ''),
            'sample_rate': TARGET_SAMPLE_RATE,
            'filter_taps': FILTER_TAPS,
            'rew_api_url': REW_API_URL,
            'vol_match_low': vol_match_low,
            'vol_match_high': vol_match_high,
            'sub_percentile': sub_percentile,
            'global_smoothing': global_smoothing,
            'max_boost_high': max_boost_high,
            'regularized_inversion': regularized_inversion,
            'reg_beta_db': reg_beta_db,
            'mains_bandlimit_high_enabled': mains_bandlimit_high_enabled,
            'mains_bandlimit_high_hz': mains_bandlimit_high_hz,
            'sub_bandlimit_low_enabled': sub_bandlimit_low_enabled,
            'sub_bandlimit_low_hz': sub_bandlimit_low_hz,
            'sub_bandlimit_high_enabled': sub_bandlimit_high_enabled,
            'sub_bandlimit_high_hz': sub_bandlimit_high_hz,
            'fc': fc,
            'fc_for_main': fc_for_main,
            'fc_sub': fc_sub,
            'auto_crossover_enabled': auto_crossover_enabled,
            'is_full_range_mains': is_full_range_mains
        })

        return jsonify({'status': 'success', 'message': '\n'.join(console_log)})
    
    except Exception as e:
        err_msg = str(e)
        if "invalid index to scalar variable" in err_msg.lower():
            err_msg += " (Have you entered your measurement IDs or run the setup wizard?)"
        clog(f"❌ Python Error: {err_msg}")
        return jsonify({'status': 'error', 'message': '\n'.join(console_log)})

@app.route('/api/run_phase2', methods=['POST'])
def run_phase2():
    global TARGET_SAMPLE_RATE, FILTER_TAPS
    console_log = []
    def clog(msg):
        console_log.append(msg)
        print(msg)

    try:
        # Restore Phase 1 sample rate and taps
        TARGET_SAMPLE_RATE = APP_STATE.get('sample_rate', 48000)
        FILTER_TAPS = APP_STATE.get('filter_taps', 65536)

        align_lfe_ms = float(request.json.get('align_lfe_ms', 0.0))
        global_id = request.json.get('global_id', '').strip()
        auto_rew_import = request.json.get('auto_rew_import', True)
        auto_target_curve = request.json.get('auto_target_curve', False)
        auto_target_start_hz = float(request.json.get('auto_target_start_hz', 2000.0))
        auto_sub_alignment = request.json.get('auto_sub_alignment', True)
        global_eq_apply_freq_limits = request.json.get('global_eq_apply_freq_limits', False)
        
        clog("\n[6] Baking Phase Alignment Delays into Final Filters...")
        
        shift_s = align_lfe_ms / 1000.0
        align_lfe_s = shift_s if shift_s > 0 else 0.0
        align_mains_s = -shift_s if shift_s < 0 else 0.0

        freqs = APP_STATE['freqs']
        fc = APP_STATE.get('fc', 80.0)
        fc_for_main = APP_STATE.get('fc_for_main', {m_id: fc for m_id in APP_STATE['mains_ids']})
        fc_sub = APP_STATE.get('fc_sub', fc)
        crossover_mains = APP_STATE.get('crossover_mains', 'lr4')
        crossover_sub = APP_STATE.get('crossover_sub', 'lr4')
        phase_mains = APP_STATE.get('phase_mains', 'linear')
        phase_sub = APP_STATE.get('phase_sub', 'linear')
        H_vba = APP_STATE['H_vba']
        measurements = APP_STATE['measurements']
        auto_crossover_enabled = APP_STATE.get('auto_crossover_enabled', False)
        is_full_range_mains = APP_STATE.get('is_full_range_mains', False)
        
        generated_firs = {}
        
        mains_firs_complex = {}
        
        for m_id in APP_STATE['mains_ids']:
            final_delay_s = APP_STATE['mains_exact_delays_s'][m_id] + align_mains_s
            if auto_crossover_enabled and not is_full_range_mains:
                hpf = np.ones_like(freqs, dtype=np.complex128)
            else:
                hpf = generate_crossover(freqs, fc_for_main.get(m_id, fc), 'highpass', crossover_mains, phase_mains)
            embedded_target_mains = APP_STATE['H_correction_dict'][m_id] * hpf
            mains_firs_complex[m_id] = embedded_target_mains
            generated_firs[m_id] = generate_final_fir(embedded_target_mains, freqs, final_delay_s, log_func=clog)

        lfe_id = APP_STATE.get('lfe_id')
        if lfe_id and APP_STATE.get('H_eq_min_phase') is not None:
            # Automatic Subwoofer Alignment via Group Delay matching
            if auto_sub_alignment and mains_firs_complex:
                clog(f"\n  -> Optimizing Subwoofer delay to match mains group delay (crossover region: {fc_sub/2:.0f}-{fc_sub*2:.0f} Hz)...")
                
                # Fetch raw IRs and generate corrected systems for Mains
                H_mains_systems = []
                for m_id in APP_STATE['mains_ids']:
                    ir = fetch_ir_data(m_id)
                    ir_centered = get_centered_ir(ir, is_lfe=False)
                    H_sys_raw = fft.rfft(ir_centered)
                    # Include the baseline exact delay shifts
                    delay_shift_rads = -2 * np.pi * freqs * APP_STATE['mains_exact_delays_s'][m_id]
                    H_mains_sys = H_sys_raw * mains_firs_complex[m_id] * np.exp(1j * delay_shift_rads)
                    H_mains_systems.append(H_mains_sys)
                
                # Vector average of corrected mains
                H_mains_avg = np.mean(H_mains_systems, axis=0)
                
                # --- Compute mains reference group delay (high frequency average) ---
                omega = 2 * np.pi * freqs
                omega_diff = np.maximum(np.diff(omega), 1e-20)
                
                mains_phase = np.unwrap(np.angle(H_mains_avg))
                gd_mains = np.zeros_like(freqs)
                gd_mains[1:] = -np.diff(mains_phase) / omega_diff
                gd_mains[0] = gd_mains[1]
                
                # Reference band: 500 Hz – 2 kHz (stable region well above crossover)
                gd_ref_low, gd_ref_high = 500.0, 2000.0
                idx_ref = (freqs >= gd_ref_low) & (freqs <= gd_ref_high)
                gd_mains_ref = np.mean(gd_mains[idx_ref]) if np.any(idx_ref) else 0.0
                clog(f"     Mains reference GD ({gd_ref_low:.0f}-{gd_ref_high:.0f} Hz): {gd_mains_ref*1000:.3f} ms")
                
                # Fetch raw IR for the Subwoofer
                ir_sub_cached = fetch_ir_data(lfe_id)
                ir_sub_centered = get_centered_ir(ir_sub_cached, is_lfe=True)
                H_sys_sub_raw = fft.rfft(ir_sub_centered)
                
                lpf = generate_crossover(freqs, fc_sub, 'lowpass', crossover_sub, phase_sub)
                embedded_target_sub_base = APP_STATE['H_eq_min_phase'] * lpf * H_vba
                H_sub_sys_corrected = H_sys_sub_raw * embedded_target_sub_base
                target_base_sub_delay = APP_STATE['target_base_delay_s'] + APP_STATE['pad_s_sub']
                
                # Sub evaluation region: crossover band fc/2 to 2*fc
                sub_gd_low = fc_sub / 2.0
                sub_gd_high = fc_sub * 2.0
                idx_sub_band = (freqs >= sub_gd_low) & (freqs <= sub_gd_high)
                
                def gd_match_objective(delta_t_ms):
                    total_sub_delay_s = target_base_sub_delay + (delta_t_ms / 1000.0)
                    delay_shift_rads = -2 * np.pi * freqs * total_sub_delay_s
                    H_sub_shifted = H_sub_sys_corrected * np.exp(1j * delay_shift_rads)
                    
                    phase_sub_unwrapped = np.unwrap(np.angle(H_sub_shifted))
                    gd_sub = np.zeros_like(freqs)
                    gd_sub[1:] = -np.diff(phase_sub_unwrapped) / omega_diff
                    gd_sub[0] = gd_sub[1]
                    
                    if np.any(idx_sub_band):
                        avg_gd_sub = np.mean(gd_sub[idx_sub_band])
                        return (avg_gd_sub - gd_mains_ref) ** 2
                    return float('inf')

                res = minimize_scalar(gd_match_objective, bounds=(-20.0, 20.0), method='bounded')
                if res.success:
                    align_lfe_ms = res.x
                    # Log the final sub GD for verification
                    final_sub_delay_s = target_base_sub_delay + (align_lfe_ms / 1000.0)
                    H_sub_final_check = H_sub_sys_corrected * np.exp(-1j * 2 * np.pi * freqs * final_sub_delay_s)
                    phase_sub_final = np.unwrap(np.angle(H_sub_final_check))
                    gd_sub_final = np.zeros_like(freqs)
                    gd_sub_final[1:] = -np.diff(phase_sub_final) / omega_diff
                    gd_sub_final[0] = gd_sub_final[1]
                    avg_gd_sub_final = np.mean(gd_sub_final[idx_sub_band]) if np.any(idx_sub_band) else 0.0
                    clog(f"     Sub GD ({sub_gd_low:.0f}-{sub_gd_high:.0f} Hz) after alignment: {avg_gd_sub_final*1000:.3f} ms")
                    clog(f"     ✅ Optimal Subwoofer Delay Shift Found: {align_lfe_ms:+.2f} ms (GD error: {abs(avg_gd_sub_final - gd_mains_ref)*1000:.3f} ms)")
                else:
                    clog("     ⚠️ Optimization failed, falling back to 0.0 ms.")
                    align_lfe_ms = 0.0
                
                align_lfe_s = align_lfe_ms / 1000.0
            
            final_delay_s = APP_STATE['target_base_delay_s'] + APP_STATE['pad_s_sub'] + align_lfe_s
            lpf = generate_crossover(freqs, fc_sub, 'lowpass', crossover_sub, phase_sub)
            embedded_target_sub = APP_STATE['H_eq_min_phase'] * lpf * H_vba
            generated_firs[lfe_id] = generate_final_fir(embedded_target_sub, freqs, final_delay_s, log_func=clog)

        global_max = max([np.max(np.abs(f)) for f in generated_firs.values()])
        
        fir_filenames_by_id = {}
        for m_id, generated_fir in generated_firs.items():
            normalized_fir = generated_fir / global_max
            safe_title = "".join([c for c in measurements[m_id].get('title', f'ID{m_id}') if c.isalnum()]).strip()
            filename = f"XPDRC_FILTER_{safe_title}.wav"
            wavfile.write(filename, TARGET_SAMPLE_RATE, normalized_fir.astype(np.float32))
            clog(f"💾 Saved Final FIR: {filename}")
            fir_filenames_by_id[m_id] = os.path.abspath(filename).replace('\\', '/')

        # Clean up base filters now that final filters have been generated
        for m_id in generated_firs.keys():
            safe_title = "".join([c for c in measurements[m_id].get('title', f'ID{m_id}') if c.isalnum()]).strip()
            base_filename = f"XPDRC_BASE_{safe_title}.wav"
            if os.path.exists(base_filename):
                os.remove(base_filename)
                clog(f"🗑️ Deleted Base FIR: {base_filename}")

        if auto_rew_import:
            clog("\n[🚀] Automating REW Post-Processing...")
            def get_meas():
                try:
                    return requests.get(f"{REW_API_URL}/measurements").json()
                except: return []
            
            def wait_new(old, timeout=10):
                start = time.time()
                while time.time() - start < timeout:
                    time.sleep(0.5)
                    curr = get_meas()
                    n = [x for x in curr if x not in old]
                    if n: return n[0]
                return None

            fir_rew_ids = {}
            for mid, fpath in fir_filenames_by_id.items():
                old = get_meas()
                if requests.post(f"{REW_API_URL}/import/impulse-response", json={'path': fpath}).status_code in (200, 202):
                    nid = wait_new(old)
                    if nid:
                        fir_rew_ids[mid] = nid
                        clog(f"  ✅ Imported {os.path.basename(fpath)} as REW ID {nid}")
                    else:
                        clog(f"  ❌ Timeout waiting for REW to import {os.path.basename(fpath)}")
            
            corr_mains = []
            for mid in APP_STATE.get('mains_ids', []):
                if mid in fir_rew_ids:
                    old = get_meas()
                    requests.post(f"{REW_API_URL}/measurements/process-measurements", json={
                        "processName": "Arithmetic", "measurementIndices": [int(mid), int(fir_rew_ids[mid])], "parameters": {"function": "A * B"}
                    })
                    nid = wait_new(old)
                    if nid:
                        corr_mains.append(nid)
                        clog(f"  ✅ Corrected Main {mid} -> REW ID {nid}")

            avg_mains_id = corr_mains[0] if len(corr_mains) == 1 else None
            if len(corr_mains) > 1:
                old = get_meas()
                requests.post(f"{REW_API_URL}/measurements/process-measurements", json={"processName": "Vector average", "measurementIndices": [int(x) for x in corr_mains]})
                avg_mains_id = wait_new(old)
                if avg_mains_id: clog(f"  ✅ Vector Averaged Mains -> REW ID {avg_mains_id}")

            if avg_mains_id:
                global_id = str(avg_mains_id)  # default if no sub exists
                if lfe_id and lfe_id in fir_rew_ids:
                    old = get_meas()
                    requests.post(f"{REW_API_URL}/measurements/process-measurements", json={"processName": "Arithmetic", "measurementIndices": [int(lfe_id), int(fir_rew_ids[lfe_id])], "parameters": {"function": "A * B"}})
                    corr_sub_id = wait_new(old)
                    if corr_sub_id:
                        clog(f"  ✅ Corrected Subwoofer -> REW ID {corr_sub_id}")
                        old = get_meas()
                        requests.post(f"{REW_API_URL}/measurements/process-measurements", json={"processName": "Vector sum", "measurementIndices": [int(avg_mains_id), int(corr_sub_id)]})
                        sum_id = wait_new(old)
                        if sum_id:
                            global_id = str(sum_id)
                            clog(f"  ✅ Vector Summed System (Global) -> REW ID {sum_id}")
            else:
                clog("  ⚠️ Could not determine corrected mains. Global ID unchanged.")

        # ----------------------------------------------------------------------
        # STEP 7: GLOBAL EQ GENERATION (Anchored Standard Inversion)
        # ----------------------------------------------------------------------
        global_eq_enabled = request.json.get('global_eq_enabled', True)
        if global_id and global_eq_enabled:
            clog(f"\n[7] Fetching Global Measurement ID {global_id} Frequency Response for System EQ...")
            
            # Fetch the Frequency Response exactly as seen in REW (returns dB)
            rew_freqs, rew_mag_db = fetch_fr_data(global_id)
            
            # Interpolate to our internal frequency resolution
            rew_mag_db_interp = np.interp(freqs, rew_freqs, rew_mag_db)
            
            # Convert back to linear magnitude for math
            mag_global = 10 ** (rew_mag_db_interp / 20.0)
            mag_global = np.maximum(mag_global, 1e-12)
            
            # Apply chosen user smoothing
            global_smoothing = APP_STATE.get('global_smoothing', 'erb')
            if global_smoothing == 'erb':
                mag_total_smoothed = erb_smoothed_fast(mag_global, freqs)
            elif global_smoothing == 'variable':
                mag_total_smoothed = log_smoothed_fast(mag_global, freqs, variable=True)
            else:
                try:
                    frac = float(global_smoothing)
                    mag_total_smoothed = log_smoothed_fast(mag_global, freqs, fraction=frac, variable=False)
                except ValueError:
                    mag_total_smoothed = erb_smoothed_fast(mag_global, freqs)
            
            # Retrieve parameters for target line
            house_boost = APP_STATE.get('house_boost', 6.0)
            house_start = APP_STATE.get('house_start', 120.0)
            house_end = APP_STATE.get('house_end', 20.0)
            vol_match_low = APP_STATE.get('vol_match_low', 500.0)
            vol_match_high = APP_STATE.get('vol_match_high', 2000.0)
            max_boost_high = APP_STATE.get('max_boost_high', 3.0)
            
            # 1. Base Level derived exactly from volume matching range (mids)
            idx_vol_start = np.argmin(np.abs(freqs - vol_match_low))
            idx_vol_end = np.argmin(np.abs(freqs - vol_match_high))
            
            global_native_level = np.mean(mag_total_smoothed[idx_vol_start:idx_vol_end])
            
            # Process Custom Uploaded House Curve or Default
            custom_hc_b64 = APP_STATE.get('custom_house_curve')
            target_shape = None
            if custom_hc_b64:
                target_shape = parse_rew_house_curve(custom_hc_b64, freqs)
                if target_shape is not None:
                    clog("  -> Applying Custom Uploaded House Curve to Global System EQ.")
                    
            if target_shape is None:
                target_shape = np.ones_like(freqs)
                idx_end = np.argmin(np.abs(freqs - house_end))
                idx_start = np.argmin(np.abs(freqs - house_start))
                if idx_start > idx_end:
                    slope_db = np.linspace(house_boost, 0, idx_start - idx_end)
                    target_shape[idx_end:idx_start] = (10 ** (slope_db / 20.0))
                target_shape[:idx_end] = (10 ** (house_boost / 20.0))
                
            # Scale target shape exactly to mid frequencies
            target_curve = target_shape * global_native_level
            
            if auto_target_curve:
                clog(f"  -> Generating Auto-Target Curve tracking natural HF roll-off above {auto_target_start_hz} Hz...")
                idx_auto_start = np.argmin(np.abs(freqs - auto_target_start_hz))
                idx_20k = np.argmin(np.abs(freqs - 20000.0))
                
                if idx_20k > idx_auto_start:
                    log_freqs = np.log10(np.maximum(freqs[idx_auto_start:idx_20k], 1e-12))
                    mags_db = 20 * np.log10(np.maximum(mag_total_smoothed[idx_auto_start:idx_20k], 1e-12))
                    
                    slope, intercept = np.polyfit(log_freqs, mags_db, 1)
                    octave_slope = slope * np.log10(2)
                    clog(f"     Calculated natural HF slope: {octave_slope:.2f} dB/octave")
                    
                    natural_curve_db = slope * np.log10(np.maximum(freqs, 1e-12)) + intercept
                    natural_curve_mag = 10 ** (natural_curve_db / 20.0)
                    
                    blend_start = auto_target_start_hz * 0.707
                    blend_end = auto_target_start_hz * 1.414
                    
                    W_auto = np.zeros_like(freqs)
                    W_auto[freqs >= blend_end] = 1.0
                    idx_blend = (freqs > blend_start) & (freqs < blend_end)
                    if np.any(idx_blend):
                        W_auto[idx_blend] = 0.5 * (1 - np.cos(np.pi * (freqs[idx_blend] - blend_start) / (blend_end - blend_start)))
                        
                    target_curve = (target_curve * (1 - W_auto)) + (natural_curve_mag * W_auto)
            
            # 3. Generate EQ: Kirkeby regularized or classic inversion
            regularized_inversion = APP_STATE.get('regularized_inversion', True)
            reg_beta_db = APP_STATE.get('reg_beta_db', 12.0)
            
            if regularized_inversion:
                eq_mag_global = kirkeby_regularized_inverse(mag_total_smoothed, freqs, target_curve, beta_db=reg_beta_db)
                eq_mag_global = np.clip(eq_mag_global, 1e-4, 10 ** (max_boost_high / 20.0))
                clog(f"  -> Global EQ: Kirkeby Regularized Inversion (β={reg_beta_db:.1f} dB)")
            else:
                eq_mag_global = target_curve / np.maximum(mag_total_smoothed, 1e-12)
                eq_mag_global = np.clip(eq_mag_global, 1e-4, 10 ** (max_boost_high / 20.0))
            
            # 4. Optionally apply Phase 1 frequency band-limits to Global EQ
            if global_eq_apply_freq_limits:
                fc_p1 = APP_STATE.get('fc', 80.0)
                
                # Mains high band-limit: fade EQ to unity above the cutoff
                if APP_STATE.get('mains_bandlimit_high_enabled', False):
                    mbl_hz = APP_STATE.get('mains_bandlimit_high_hz', 10000.0)
                    W_gl_mbl = np.ones_like(freqs)
                    mbl_start = mbl_hz
                    mbl_end = mbl_hz * 1.4
                    idx_mbl_s = np.argmin(np.abs(freqs - mbl_start))
                    idx_mbl_e = np.argmin(np.abs(freqs - mbl_end))
                    if idx_mbl_e > idx_mbl_s:
                        W_gl_mbl[idx_mbl_s:idx_mbl_e] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_mbl_s:idx_mbl_e] - mbl_start) / (mbl_end - mbl_start)))
                    W_gl_mbl[idx_mbl_e:] = 0.0
                    eq_mag_global = eq_mag_global * W_gl_mbl + 1.0 * (1 - W_gl_mbl)
                    clog(f"  -> Global EQ: applied Mains Band-Limit High ({mbl_hz:.0f} Hz).")
                
                # Sub low band-limit: fade EQ to unity below the cutoff
                if APP_STATE.get('sub_bandlimit_low_enabled', False):
                    bl_low_hz = APP_STATE.get('sub_bandlimit_low_hz', 15.0)
                    W_gl_bll = np.ones_like(freqs)
                    bll_start = bl_low_hz * 0.7
                    bll_end = bl_low_hz
                    idx_bll_s = np.argmin(np.abs(freqs - bll_start))
                    idx_bll_e = np.argmin(np.abs(freqs - bll_end))
                    W_gl_bll[:idx_bll_s] = 0.0
                    if idx_bll_e > idx_bll_s:
                        W_gl_bll[idx_bll_s:idx_bll_e] = 0.5 * (1 - np.cos(np.pi * (freqs[idx_bll_s:idx_bll_e] - bll_start) / (bll_end - bll_start)))
                    eq_mag_global = eq_mag_global * W_gl_bll + 1.0 * (1 - W_gl_bll)
                    clog(f"  -> Global EQ: applied Sub Band-Limit Low ({bl_low_hz:.0f} Hz).")
                
                # Sub high band-limit is ignored for Global EQ because Global EQ must correct the mains high-end.
                # Only Mains High and Sub Low limits are relevant for a system-wide pass.
            else:
                clog("  -> Global EQ: frequency limits bypassed (full-range correction).")
            
            # 5. Convert Global EQ to Minimum Phase
            cepstrum = fft.irfft(np.log(eq_mag_global), n=FILTER_TAPS)
            lifter = np.zeros(FILTER_TAPS)
            lifter[0] = 1
            lifter[1:FILTER_TAPS//2] = 2
            if FILTER_TAPS % 2 == 0: lifter[FILTER_TAPS//2] = 1
            H_global_min_phase = np.exp(fft.rfft(cepstrum * lifter))
            
            clog(f"  -> Global Target Curve anchored perfectly flat to mids ({vol_match_low}Hz-{vol_match_high}Hz).")
            clog(f"  -> Applied {global_smoothing.upper() if global_smoothing in ['erb', 'variable'] else f'1/{global_smoothing} Octave'} smoothing.")
            clog(f"  -> Generated Minimum Phase Global EQ (Unlimited cutting allowed, Max Boost +{max_boost_high}dB).")
            
            # Default Global FIR from Minimum Phase magnitude correction
            H_global_final = H_global_min_phase
            
            # --- Optional Global Phase Linearization ---
            global_phase_lin_enabled = request.json.get('global_phase_lin_enabled', False)
            if global_phase_lin_enabled and global_id:
                clog("\n  -> [Global Phase Linearization] Analyzing system vector sum phase...")
                try:
                    # 1. Fetch system IR and zero-phase it for analysis
                    ir_sys = fetch_ir_data(global_id)
                    fs = TARGET_SAMPLE_RATE
                    
                    # Window: 50ms symmetric Hann centered on peak
                    # This isolates direct sound response for cleaner phase logic
                    win_size = int(0.050 * fs)
                    half_win = win_size // 2
                    peak_idx = np.argmax(np.abs(ir_sys))
                    
                    # Target buffer for Zero Phase analysis (peaking at Sample 0)
                    ir_centered = np.zeros(FILTER_TAPS)
                    hann = np.hanning(win_size)
                    
                    # Correct circular shift: peak goes to 0, pre-energy to end of buffer
                    for i in range(win_size):
                        curr_idx = peak_idx - half_win + i
                        if 0 <= curr_idx < len(ir_sys):
                            # Target is peak (at shift=0) goes to index 0
                            # Pre-peak (shift < 0) goes to indices like FILTER_TAPS - 100
                            shift = i - half_win
                            target_idx = shift % FILTER_TAPS
                            ir_centered[target_idx] = ir_sys[curr_idx] * hann[i]
                    
                    H_sys_complex = fft.rfft(ir_centered)
                    
                    # 2. Extract excess phase (Zero-Delay)
                    # Normalize magnitude for stable log calculation
                    mag_sys = np.maximum(np.abs(H_sys_complex), 1e-12)
                    cep_sys = fft.irfft(np.log(mag_sys), n=FILTER_TAPS)
                    lifter = np.zeros(FILTER_TAPS)
                    lifter[0] = 1
                    lifter[1:FILTER_TAPS//2] = 2
                    if FILTER_TAPS % 2 == 0: lifter[FILTER_TAPS//2] = 1
                    H_min_sys = np.exp(fft.rfft(cep_sys * lifter))
                    
                    # Excess phase spectrum (Now purely the rotation, no bulk delay)
                    H_excess_sys = H_sys_complex / H_min_sys
                    excess_phase_sys = np.unwrap(np.angle(H_excess_sys))
                    
                    # 3. Use top-down detection to find safest broad correction limit on the CORRECTED system
                    # Range: Crossover/2 to 5000Hz (common HF limit)
                    start_freq = max(20.0, fc_sub / 2.0)
                    
                    # Test baseline: Magnitude-Corrected System (Zero-Phase)
                    H_test_base = H_sys_complex * H_global_min_phase
                    
                    detected_hz, peak_db = detect_auto_prc_frequency(
                        H_test_base, freqs, excess_phase_sys, fs, 
                        delay_s=(APP_STATE.get('delay_ms', 0)/1000.0),
                        min_freq=start_freq, max_freq=5000.0
                    )
                    
                    # 4. Generate phase correction window
                    W_ph = np.ones_like(freqs)
                    fade_start = detected_hz
                    fade_end = detected_hz * 2.0
                    W_ph[freqs >= fade_end] = 0.0
                    idx_fade = (freqs > fade_start) & (freqs < fade_end)
                    if np.any(idx_fade):
                        W_ph[idx_fade] = 0.5 * (1 + np.cos(np.pi * (freqs[idx_fade] - fade_start) / (fade_end - fade_start)))
                    
                    # Low-end fade-in (fc/4 to fc/2)
                    low_fade_start = start_freq / 2.0
                    low_fade_end = start_freq
                    W_ph[freqs <= low_fade_start] = 0.0
                    idx_low_fade = (freqs > low_fade_start) & (freqs < low_fade_end)
                    if np.any(idx_low_fade):
                        W_ph[idx_low_fade] = 0.5 * (1 - np.cos(np.pi * (freqs[idx_low_fade] - low_fade_start) / (low_fade_end - low_fade_start)))

                    # 5. Apply phase correction (Inversion)
                    # Correct for the system's excess phase within the window
                    # Normalize phase at the detected fade start frequency
                    idx_norm = np.argmin(np.abs(freqs - fade_start))
                    p_norm = np.round(excess_phase_sys[idx_norm] / (2 * np.pi)) * (2 * np.pi)
                    target_correction = -(excess_phase_sys - p_norm) * W_ph
                    
                    H_ph_corr = np.exp(1j * target_correction)
                    
                    # 6. Cascade filters: (Min-Phase Magnitude) * (Phase Inversion Spectrum)
                    H_global_final = H_global_min_phase * H_ph_corr
                    
                    clog(f"     ✅ Phase Linearization SUCCESS: {detected_hz:.0f} Hz limit (Pre-ringing: {peak_db:.1f} dB)")
                    clog(f"     -> Applied correction from {start_freq:.0f}Hz to {detected_hz:.0f}Hz.")
                except Exception as e:
                    clog(f"     ⚠️ Phase Linearization FAILED: {e}. Falling back to magnitude-only correction.")

            # Build final Global EQ FIR (Time Domain)
            # Use generate_final_fir to shift the peak to delay_ms and apply proper windowing
            # This prevents pre-ringing from wrapping around and warping the magnitude response
            if global_phase_lin_enabled:
                target_delay_s = APP_STATE.get('delay_ms', 75.0) / 1000.0
            else:
                target_delay_s = 0.0
                
            global_fir = generate_final_fir(H_global_final, freqs, target_delay_s, log_func=clog, is_lin_phase=global_phase_lin_enabled)
            
            global_fir_normalized = global_fir / np.max(np.abs(global_fir))
            
            wavfile.write("XPDRC_GLOBAL_EQ.wav", TARGET_SAMPLE_RATE, global_fir_normalized.astype(np.float32))
            clog("💾 Saved Global System EQ FIR: XPDRC_GLOBAL_EQ.wav")
        else:
            clog("\n[7] No Global Measurement ID provided. Skipping Global System EQ generation.")

        return jsonify({'status': 'success', 'message': '\n'.join(console_log)})
    
    except Exception as e:
        err_msg = str(e)
        if "invalid index to scalar variable" in err_msg.lower():
            err_msg += " (Have you entered your measurement IDs or run the setup wizard?)"
        clog(f"❌ Python Error: {err_msg}")
        return jsonify({'status': 'error', 'message': '\n'.join(console_log)})

# ==============================================================================
# SETUP WIZARD ENDPOINTS
# ==============================================================================

@app.route('/api/wizard/get_audio_devices', methods=['GET'])
def get_audio_devices():
    try:
        java_in = requests.get(f"{REW_API_URL}/audio/java/input-devices").json()
        java_out = requests.get(f"{REW_API_URL}/audio/java/output-devices").json()
        res_asio = requests.get(f"{REW_API_URL}/audio/asio/devices")
        asio = res_asio.json() if res_asio.status_code == 200 else []
        return jsonify({'status': 'success', 'java_in': java_in, 'java_out': java_out, 'asio': asio})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/wizard/poll_measurements', methods=['GET'])
def poll_measurements():
    try:
        # Check how many measurements currently exist in REW
        req = requests.get(f"{REW_API_URL}/measurements")
        if req.status_code == 200:
            data = req.json()
            if isinstance(data, list):
                return jsonify({'status': 'success', 'max_measurements': len(data)})
            elif isinstance(data, dict):
                # REW might return an empty dict {} if there are no measurements instead of an empty list []
                return jsonify({'status': 'success', 'max_measurements': len(data.keys())})
            else:
                return jsonify({'status': 'error', 'message': f'Unexpected response format from REW API: {type(data)}'})
        else:
            return jsonify({'status': 'error', 'message': f'Failed to fetch measurements: {req.text}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/wizard/align_and_average', methods=['POST'])
def align_and_average():
    try:
        # Expected input: dict mapping speaker channel to list of dicts:
        # e.g., {'L': [{'pos': 1, 'id': 10}, {'pos': 2, 'id': 11}]}
        config = request.json
        groups = config.get('groups', {})
        
        final_ids = {} # ch -> REW ID of the final vector average
        
        for ch, runs in groups.items():
            if not runs:
                continue
                
            sorted_runs = sorted(runs, key=lambda x: x['pos'])
            
            # Extract REW IDs of the measurements for this speaker
            base_run = sorted_runs[0]
            base_id = base_run['id']
            measurement_ids = [run['id'] for run in sorted_runs]
            
            # REW API expects 1-based measurement indices, but our IDs *are* the 1-based indices from REW.
            
            # Step 1: Tell REW to cross-correlate align these measurements
            # "Cross corr align" aligns them all to the first measurement in the list.
            align_req = {
                "processName": "Cross corr align",
                "measurementIndices": measurement_ids
            }
            res_align = requests.post(f"{REW_API_URL}/measurements/process-measurements", json=align_req)
            if res_align.status_code not in (200, 202):
                raise ValueError(f"Failed to align measurements in REW for {ch}: {res_align.text}")
                
            # Wait briefly for REW to apply alignment shifts
            time.sleep(0.5)

            old_meas = requests.get(f"{REW_API_URL}/measurements").json()

            # Step 2: Tell REW to create a Vector average of the aligned measurements
            avg_req = {
                "processName": "Vector average",
                "measurementIndices": measurement_ids
            }
            
            res_avg = None
            avg_wait_start = time.time()
            while time.time() - avg_wait_start < 45:
                res_avg = requests.post(f"{REW_API_URL}/measurements/process-measurements", json=avg_req)
                if res_avg.status_code in (200, 202):
                    break
                elif "is running" in res_avg.text:
                    time.sleep(1.0)
                else:
                    raise ValueError(f"Failed to average measurements in REW for {ch}: {res_avg.text}")
                    
            if not res_avg or res_avg.status_code not in (200, 202):
                raise ValueError(f"Timeout (45s) waiting for Cross corr align to finish for {ch}.")
            
            # Wait for the new Vector Average measurement to appear in REW
            start_wait = time.time()
            new_id = None
            while time.time() - start_wait < 10:
                time.sleep(0.5)
                curr_meas = requests.get(f"{REW_API_URL}/measurements").json()
                n = [m for m in curr_meas if m not in old_meas]
                if n:
                    # New measurement generated!
                    new_id = n[0]
                    # Rename the new measurement via the REW API
                    requests.put(f"{REW_API_URL}/measurements/{new_id}", json={
                        "title": f"XPDRC_{ch}"
                    })
                    break
                    
            if new_id:
                final_ids[ch] = new_id
            else:
                raise ValueError(f"Timeout waiting for Vector average measurement to be generated in REW for {ch}.")

        return jsonify({'status': 'success', 'final_ids': final_ids})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()})

@app.route('/api/preview_house_curve', methods=['POST'])
def preview_house_curve():
    try:
        config = request.json
        b64_str = config.get('custom_house_curve')
        if not b64_str:
            return jsonify({'status': 'error', 'message': 'No house curve provided.'})
            
        content = base64.b64decode(b64_str).decode('utf-8')
        freqs_list = []
        db_list = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('//') or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    f = float(parts[0])
                    db = float(parts[1])
                    freqs_list.append(f)
                    db_list.append(db)
                except ValueError:
                    continue
                    
        if not freqs_list:
            return jsonify({'status': 'error', 'message': 'Could not parse any valid frequency/dB points.'})
            
        return jsonify({
            'status': 'success',
            'freqs': freqs_list,
            'dbs': db_list
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def open_browser():
    """Opens the local web interface in the default browser."""
    # Wait a short moment for the Flask server to start
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Open the browser only once in the main process (before the reloader starts)
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Thread(target=open_browser, daemon=True).start()

    app.run(debug=True, port=5000)
