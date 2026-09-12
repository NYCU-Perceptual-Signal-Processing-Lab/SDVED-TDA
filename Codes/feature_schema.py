"""Shared schema for the public 44-dimensional acoustic feature set."""

N_MFCC = 13

FEATURE_COLUMNS_44D = [
    "centroid_mean",
    "flux_mean_low",
    "hnr_praat",
    "f1_praat",
    "f2_praat",
]
FEATURE_COLUMNS_44D += [f"mfcc_{index}_mean" for index in range(1, N_MFCC + 1)]
FEATURE_COLUMNS_44D += [
    f"mfcc_delta_{index}_mean" for index in range(1, N_MFCC + 1)
]
FEATURE_COLUMNS_44D += [
    f"mfcc_delta2_{index}_mean" for index in range(1, N_MFCC + 1)
]

EXPECTED_FEATURE_DIM = 44
assert len(FEATURE_COLUMNS_44D) == EXPECTED_FEATURE_DIM
