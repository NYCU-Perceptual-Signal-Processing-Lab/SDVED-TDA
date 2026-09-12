import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import numpy as np
    import parselmouth
    from parselmouth.praat import call
except ModuleNotFoundError as exc:
    np = None
    parselmouth = None
    call = None
    DEPENDENCY_IMPORT_ERROR = exc
else:
    DEPENDENCY_IMPORT_ERROR = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

from audio_feature_utils import (
    HOP_LENGTH,
    LOW_FREQ_CUTOFF,
    N_FFT,
    N_MFCC,
    SR,

    finite_float,

    load_audio,
    mfcc_matrix,
    onset_strength_mean,
    savgol_delta,
    spectral_centroid_mean,
)
from feature_schema import EXPECTED_FEATURE_DIM, FEATURE_COLUMNS_44D

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "outputs" / "features_44d.csv"


def require_dependencies():
    if DEPENDENCY_IMPORT_ERROR is not None:
        raise SystemExit(
            "feature_extract_filewise.py requires numpy and praat-parselmouth. "
            "soundfile is also required for WAV formats unsupported by Python's wave module.\n"
            "Install the dependencies with: pip install -r requirements.txt\n"
            f"Original import error: {DEPENDENCY_IMPORT_ERROR}"
        )

def _get_praat_features_from_sound(sound):
    """Extract HNR and formants with Praat through Parselmouth."""
    try:
        if sound.get_total_duration() < 0.1:
            return -200.0, 0.0, 0.0

        # --- 1. HNR ---
        harmonicity = sound.to_harmonicity_cc(time_step=0.01)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # --- 2. Formants ---
        formant_obj = sound.to_formant_burg(
            time_step=None, 
            max_number_of_formants=5.0,
            maximum_formant=5500.0
        )
        
        f1 = call(formant_obj, "Get mean", 1, 0, 0, "Hertz")
        f2 = call(formant_obj, "Get mean", 2, 0, 0, "Hertz")
        
        if np.isnan(hnr): hnr = -200.0 
        if np.isnan(f1): f1 = 0.0
        if np.isnan(f2): f2 = 0.0
            
        return hnr, f1, f2

    except Exception as e:
        return -200.0, 0.0, 0.0


def get_praat_features(file_path):
    """
    Extract HNR and formants from an audio file with Praat/Parselmouth.
    """
    try:
        abs_path = os.path.abspath(file_path)
        sound = parselmouth.Sound(abs_path)
        return _get_praat_features_from_sound(sound)
    except Exception as e:
        return -200.0, 0.0, 0.0


def get_praat_features_from_audio(y, sr):
    try:
        sound = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=float(sr))
        return _get_praat_features_from_sound(sound)
    except Exception:
        return -200.0, 0.0, 0.0


def extract_44_features_from_audio(y, sr, praat_features=None):
    """Extract the public 44-D feature set used by both training scripts."""
    # 1. Spectral Centroid
    mean_centroid = spectral_centroid_mean(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # 2. Onset Strength (Flux)
    mean_flux = onset_strength_mean(y, sr, hop_length=HOP_LENGTH, n_fft=N_FFT, fmax=LOW_FREQ_CUTOFF)

    # --- 3. MFCC + Delta + Delta-Delta ---
    mfcc = mfcc_matrix(y, sr, n_mfcc=N_MFCC)
    mfcc_delta = savgol_delta(mfcc, order=1)
    mfcc_delta2 = savgol_delta(mfcc, order=2)

    mfcc_mean = np.mean(mfcc, axis=1)
    delta_mean = np.mean(mfcc_delta, axis=1)
    delta2_mean = np.mean(mfcc_delta2, axis=1)

    if praat_features is None:
        hnr_praat, f1_praat, f2_praat = get_praat_features_from_audio(y, sr)
    else:
        hnr_praat, f1_praat, f2_praat = praat_features

    if hnr_praat is None: hnr_praat = -200.0
    if f1_praat is None: f1_praat = 0.0
    if f2_praat is None: f2_praat = 0.0
    data = {
        "centroid_mean": finite_float(mean_centroid, 0.0),
        "flux_mean_low": finite_float(mean_flux, 0.0),
        "hnr_praat": finite_float(hnr_praat, -200.0),
        "f1_praat": finite_float(f1_praat, 0.0),
        "f2_praat": finite_float(f2_praat, 0.0),
    }

    for i in range(N_MFCC):
        data[f"mfcc_{i+1}_mean"] = finite_float(mfcc_mean[i], 0.0)
    for i in range(N_MFCC):
        data[f"mfcc_delta_{i+1}_mean"] = finite_float(delta_mean[i], 0.0)
    for i in range(N_MFCC):
        data[f"mfcc_delta2_{i+1}_mean"] = finite_float(delta2_mean[i], 0.0)

    if len(data) != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_DIM} features, but extracted {len(data)}."
        )
    return data


def get_features(file_info):
    full_path, rel_path = file_info
    try:
        # ==========================================
        # Part A: Hand-written spectral features, without librosa.
        # ==========================================
        y, sr = load_audio(full_path, target_sr=SR)

        # ==========================================
        # Part B: Praat acoustic features.
        # ==========================================
        praat_features = get_praat_features(full_path)

        # ==========================================
        # Combine all features into one output row.
        # ==========================================
        data = {"filename": rel_path}
        data.update(extract_44_features_from_audio(y, sr, praat_features=praat_features))

        return data

    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
        return None

def feature_header():
    header = ["filename", *FEATURE_COLUMNS_44D]
    if len(header) - 1 != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            f"Expected a {EXPECTED_FEATURE_DIM}-D header, got {len(header) - 1}."
        )
    return header


def scan_wav_files(root_folder):
    file_list = []
    for current_root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                full_path = os.path.join(current_root, file)
                rel_path = os.path.relpath(full_path, root_folder)
                file_list.append((full_path, rel_path))
    file_list.sort(key=lambda item: item[1])
    return file_list


def process_folder_parallel(root_folder, output_csv, workers=None, max_files=None):
    root_folder = Path(root_folder).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    if not root_folder.is_dir():
        raise SystemExit(f"Audio folder does not exist or is not a directory: {root_folder}")

    print("Scanning files...")
    file_list = scan_wav_files(root_folder)
    if max_files is not None:
        file_list = file_list[:max_files]

    print(f"Found {len(file_list)} files. Starting processing...")

    if not file_list:
        print("No WAV files found.")
        return

    if workers == 1:
        iterator = map(get_features, file_list)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(file_list))
        results = list(iterator)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(get_features, file_list)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(file_list))
            results = list(iterator)

    results = [r for r in results if r is not None]

    if not results:
        print("No results generated.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feature_header())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Done! Saved to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the 44-D acoustic feature set from a folder of WAV files."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Your root audio folder. WAV files are discovered recursively.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N files after sorting.")
    return parser.parse_args()
    
 
def main():
    args = parse_args()
    require_dependencies()
    if args.workers is not None and args.workers <= 0:
        raise SystemExit("--workers must be a positive integer.")
    if args.max_files is not None and args.max_files <= 0:
        raise SystemExit("--max-files must be a positive integer.")
    process_folder_parallel(args.folder, args.output, workers=args.workers, max_files=args.max_files)


if __name__ == "__main__":
    main()
