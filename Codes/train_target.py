import argparse
import os
import random
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
AMPHION_ROOT = Path(os.environ.get("AMPHION_ROOT", PROJECT_ROOT / "Amphion")).expanduser()
if AMPHION_ROOT.is_dir() and str(AMPHION_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(AMPHION_ROOT.parent))
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "timbre_numba_cache")
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm

from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
from feature_schema import EXPECTED_FEATURE_DIM, FEATURE_COLUMNS_44D

# Suppress warnings
warnings.filterwarnings("ignore")

########## Configuration ##########
class Config:
    # Path Settings
    LABEL_FILE = None
    AUDIO_ROOT = None
    CHECKPOINT_DIR = str(PROJECT_ROOT / "outputs" / "target_checkpoints")
    
    # Column Mapping
    COL_FILENAME = "filename"
    COL_SCORE = "score"
    COL_SPEAKER = "speaker_id"
    # Audio settings
    TARGET_SR = 16000
    TARGET_LEN = 2 * TARGET_SR  # Use deterministic two-second segments.
    
    # Training settings
    BATCH_SIZE = 128
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Model settings
    TIMBRE_DIM = 256  # FACodec output dimension.
    MANUAL_DIM = EXPECTED_FEATURE_DIM
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Saving
    EARLY_STOP_PATIENCE = 15

    # Trim only leading/trailing silence based on the whole file. Middle pauses stay intact.
    ENABLE_EDGE_VAD_TRIM = True
    VAD_FRAME_MS = 50
    VAD_HOP_MS = 25
    VAD_RELATIVE_DB = -40.0
    VAD_ABSOLUTE_RMS = 1e-4
    VAD_MIN_EDGE_SILENCE_SEC = 0.75
    VAD_PAD_SEC = 0.15

########## Data Alignment & Processing ##########
class DataAligner:
    """Align labels with audio files and normalize the manual features."""
    @staticmethod
    def align_data(label_path: str, audio_root: str) -> Tuple[List[Dict], int]:
        print(f"Loading data...\nSource: {label_path}")
        
        # 1. Read the label table.
        if label_path.endswith('.csv'):
            df = pd.read_csv(label_path)
        else:
            df = pd.read_excel(label_path)

        # Check that all required metadata columns exist.
        required_cols = [Config.COL_FILENAME, Config.COL_SCORE, Config.COL_SPEAKER]
        missing_cols = [column for column in required_cols if column not in df.columns]
        if missing_cols:
            raise ValueError(f"The CSV is missing required columns: {missing_cols}")

        # 2. Use the fixed public 44-D schema; ignore unrelated CSV columns.
        missing_features = [c for c in FEATURE_COLUMNS_44D if c not in df.columns]
        if missing_features:
            raise ValueError(f"The CSV is missing 44-D feature columns: {missing_features}")
        feature_cols = FEATURE_COLUMNS_44D
        num_features = len(feature_cols)
        print(f"Using the fixed {num_features}-D manual feature set")

        # Remove non-finite values before calculating normalization statistics.
        # Invalid labels make the loss non-finite, while invalid features can
        # contaminate an entire normalized feature column.
        numeric_scores = pd.to_numeric(df[Config.COL_SCORE], errors='coerce')
        numeric_features = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        score_values = numeric_scores.to_numpy(dtype=np.float64, copy=False)
        feature_values = numeric_features.to_numpy(dtype=np.float64, copy=False)
        valid_score_mask = np.isfinite(score_values)
        valid_feature_mask = np.isfinite(feature_values).all(axis=1)
        valid_row_mask = valid_score_mask & valid_feature_mask

        if not valid_row_mask.all():
            invalid_indices = np.flatnonzero(~valid_row_mask)
            invalid_rows = df.iloc[invalid_indices][
                [Config.COL_FILENAME, Config.COL_SPEAKER, Config.COL_SCORE]
            ].copy()
            invalid_rows['invalid_label'] = ~valid_score_mask[invalid_indices]
            invalid_feature_cells = ~np.isfinite(feature_values[invalid_indices])
            invalid_rows['invalid_features'] = [
                ', '.join(
                    feature_cols[column_index]
                    for column_index in np.flatnonzero(row_mask)
                )
                for row_mask in invalid_feature_cells
            ]
            print(
                f"Removing {len(invalid_rows)} rows containing NaN/Inf "
                "labels or features:"
            )
            print(invalid_rows.to_string(index=False))

        df = df.loc[valid_row_mask].copy()
        if df.empty:
            raise ValueError("No usable training rows remain after removing NaN/Inf")
        df[Config.COL_SCORE] = numeric_scores.loc[valid_row_mask]
        df[feature_cols] = numeric_features.loc[valid_row_mask]

        # Calculate the normalization mean and standard deviation.
        vals = df[feature_cols].values # (N_samples, N_features)
        mean = np.mean(vals, axis=0)
        #print(mean)
        std = np.std(vals, axis=0) + 1e-8  # Avoid division by zero.
        #print(std)
        # Apply Z-score normalization: (x - mean) / std.
        normalized_vals = (vals - mean) / std
        if not np.isfinite(normalized_vals).all():
            raise ValueError("Normalized manual features still contain NaN/Inf")
        
        # Store normalized values back in the DataFrame.
        df_norm = df.copy()
        df_norm[feature_cols] = normalized_vals
        # --------------------

        # 3. Scan audio files recursively.
        audio_root_path = Path(audio_root)
        audio_files_map = {}
        print("Scanning audio files...")
        for f in audio_root_path.rglob("*"):
            if f.suffix.lower() in ['.wav', '.mp3']:
                audio_files_map[f.stem] = str(f)
        
        aligned_data = []
        missing_count = 0
        
        # 4. Match table rows to audio files.
        print("Matching labels with audio files...")
        for _, row in tqdm(df_norm.iterrows(), total=len(df)):
            fname_str = str(row[Config.COL_FILENAME])
            fname_stem = Path(fname_str).stem
            
            if fname_stem in audio_files_map:
                # Read the normalized feature vector for this row.
                feat_vector = row[feature_cols].values.astype(np.float32)
                speaker_id = str(row[Config.COL_SPEAKER])
                aligned_data.append({
                    'path': audio_files_map[fname_stem],
                    'score': float(row[Config.COL_SCORE]),
                    'manual_feats': feat_vector,
                    'orig_name': fname_str,
                    'speaker': speaker_id 
                })
            else:
                missing_count += 1

        print(f"Alignment complete: matched={len(aligned_data)}, missing={missing_count}")
        return aligned_data, num_features

def load_and_lufs_normalize(path: str, out_sr: int = 16000) -> Optional[torch.Tensor]:
    try:
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if not torch.isfinite(wav).all():
            print(f"Error loading {path}: audio contains NaN/Inf samples")
            return None
        
        wav = wav / (wav.abs().max() + 1e-8) * 0.95
        
        if sr != out_sr:
            wav = torchaudio.transforms.Resample(sr, out_sr)(wav)
            
        return wav
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def detect_active_edge_bounds(wav: torch.Tensor, target_sr: int = 16000) -> Tuple[int, int]:
    """Return active start/end sample offsets after trimming only file edges."""
    total_samples = wav.size(1)
    if not Config.ENABLE_EDGE_VAD_TRIM or total_samples <= 0:
        return 0, total_samples

    frame_len = max(1, int(target_sr * Config.VAD_FRAME_MS / 1000))
    hop_len = max(1, int(target_sr * Config.VAD_HOP_MS / 1000))
    if total_samples < frame_len:
        return 0, total_samples

    mono = wav.squeeze(0)
    frames = mono.unfold(0, frame_len, hop_len)
    frame_rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-12)
    max_rms = frame_rms.max()
    if float(max_rms) <= 1e-8:
        return 0, total_samples

    relative_threshold = max_rms * (10 ** (Config.VAD_RELATIVE_DB / 20.0))
    threshold = max(float(relative_threshold), Config.VAD_ABSOLUTE_RMS)
    active = frame_rms >= threshold
    active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
    if active_idx.numel() == 0:
        return 0, total_samples

    first_frame = int(active_idx[0].item())
    last_frame = int(active_idx[-1].item())
    pad_samples = int(Config.VAD_PAD_SEC * target_sr)
    active_start = max(0, first_frame * hop_len - pad_samples)
    active_end = min(total_samples, last_frame * hop_len + frame_len + pad_samples)

    min_edge_silence = int(Config.VAD_MIN_EDGE_SILENCE_SEC * target_sr)
    if active_start < min_edge_silence:
        active_start = 0
    if total_samples - active_end < min_edge_silence:
        active_end = total_samples

    if active_end <= active_start:
        return 0, total_samples
    return active_start, active_end


class HybridDataset(Dataset):
    def __init__(self, data_list: List[Dict]):
        self.data = data_list

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        target_len = Config.TARGET_LEN # 32000 samples
        
        # 1. Load the complete audio file.
        wav = load_and_lufs_normalize(item['path'], Config.TARGET_SR)
        
        if wav is None:
            # Fall back to another random sample when decoding fails.
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        
        # 2. Create a deterministic segment using the precomputed offset.
        T = wav.size(1)
        start = int(item.get('start_offset', 0))
        trim_end = item.get('trim_end_offset')
        
        # Pad audio that is shorter than the target segment.
        if T < target_len:
            pad = target_len - T
            wav_segment = F.pad(wav, (0, pad))
        else:
            start = min(max(0, start), max(0, T - 1))
            end = start + target_len
            if trim_end is not None:
                end = min(end, int(trim_end))
            end = min(end, T)
            wav_segment = wav[:, start:end]
            
            # Pad again if boundary clipping made the segment too short.
            if wav_segment.size(1) < target_len:
                pad = target_len - wav_segment.size(1)
                wav_segment = F.pad(wav_segment, (0, pad))

        # 3. Return model inputs and the target score.
        manual_feats = torch.tensor(item['manual_feats'], dtype=torch.float32)
        label = torch.tensor([item['score']], dtype=torch.float32)
        
        return {
            'audio': wav_segment.squeeze(0),
            'manual_feats': manual_feats,
            'labels': label
        }

########## Model (Feature Fusion Architecture) ##########
class HybridTimbreHead(nn.Module):
    def __init__(self, embedding_dim=256, manual_dim=EXPECTED_FEATURE_DIM, dropout=0.1):
        super().__init__()
        
        # Path A: FACodec Embedding
        # Normalize the pretrained FACodec representation.
        self.embed_norm = nn.LayerNorm(embedding_dim)
        
        # Path B: Manual Features
        # Normalize the manual features; DataAligner validates the dimension.
        self.manual_net = nn.Sequential(
            #nn.Linear(manual_dim, 64),
            nn.LayerNorm(manual_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion Layer
        # Input dimension = FACodec embedding + manual features.
        fusion_dim = embedding_dim + manual_dim
        
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(16, 4),
            nn.GELU(),
            nn.Dropout(dropout),
                        
            nn.Linear(4, 1) # Output Score
        )
    

    def forward(self, facodec_embed, manual_feats):
        """
        facodec_embed: (batch, 256) representation from FACodec.
        manual_feats: (batch, manual_dim) features from the CSV.
        """
        x_emb = self.embed_norm(facodec_embed)
        x_man = self.manual_net(manual_feats)
        
        # Concatenate the two feature branches.
        # 
        combined = torch.cat([x_emb, x_man], dim=1)
        
        # Predict the regression score.
        return self.fusion_net(combined)
    

########## Trainer ##########
class Trainer:
    def __init__(self, config: Config, manual_feature_dim: int, pretrained_path: str = None, freeze_level: int = 2):
        self.config = config
        self.device = config.DEVICE
        
        # Use the feature dimension validated by DataAligner.
        self.manual_feature_dim = manual_feature_dim
        
        self.setup_models()
        if pretrained_path:
            self.load_pretrained(pretrained_path)
            
            # Apply the requested transfer-learning freeze level.
            self.set_freeze_level(level=freeze_level)
        ft_lr = config.LEARNING_RATE * 0.6 if pretrained_path else config.LEARNING_RATE
        
        # Give the optimizer only parameters that remain trainable.
        trainable_params = [p for p in self.hybrid_head.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(trainable_params, lr=ft_lr, weight_decay=config.WEIGHT_DECAY)
        
        # Optional alternative: differential learning rates for full fine-tuning.
        """
        self.optimizer = AdamW([
            {'params': self.hybrid_head.manual_net.parameters(), 'lr': 1e-6},
            {'params': self.hybrid_head.fusion_net.parameters(), 'lr': 1e-5}
        ], weight_decay=config.WEIGHT_DECAY)
        """
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        #self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    def load_pretrained(self, path):
        """Load pretrained weights from an architecture-compatible checkpoint."""
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        # strict=True ensures that the source and target architectures match.
        self.hybrid_head.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("Weights loaded successfully!")
    def set_freeze_level(self, level: int):
        """
        level 0: Keep every layer trainable.
        level 1: Freeze the embedding and manual-feature preprocessing layers.
        level 2: Also freeze the first fusion block and train later blocks only.
        """
        print(f"Setting freeze level to: {level}")

        # Helper for toggling gradients on an entire module.
        def freeze_module(module, freeze=True):
            for param in module.parameters():
                param.requires_grad = not freeze

        # Reset by making every layer trainable.
        freeze_module(self.hybrid_head, freeze=False)

        if level == 0:
            return  # All layers remain trainable.

        # Level 1+: freeze input preprocessing layers.
        if level >= 1:
            freeze_module(self.hybrid_head.embed_norm, freeze=True)
            freeze_module(self.hybrid_head.manual_net, freeze=True)

        # Level 2+: freeze the first fusion block as well.
        if level >= 2:
            # Select layers by their Sequential indices.
            # Keep the later fusion blocks trainable while freezing the first block.
            frozen_indices = [0, 1, 2]
            # frozen_indices = [0, 1, 2, 3, 4, 5]  # Freeze two fusion blocks.
            
            for idx, layer in enumerate(self.hybrid_head.fusion_net):
                if idx in frozen_indices:
                    freeze_module(layer, freeze=True)
                    
        # Display the parameters that remain trainable.
        print("=== Current Trainable Parameters ===")
        for name, param in self.hybrid_head.named_parameters():
            if param.requires_grad:
                print(f"[trainable] {name}")
            # else:
            #     print(f"[frozen] {name}")
                
    def setup_models(self):
        print("Setting up models...")
        # 1. Load the frozen FACodec backbone.
        self.fa_encoder = FACodecEncoder(ngf=32, up_ratios=[2,4,5,5], out_channels=256)
        self.fa_decoder = FACodecDecoder(
            in_channels=256, upsample_initial_channel=1024, ngf=32,
            up_ratios=[5,5,4,2], vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3,
            vq_dim=256, codebook_dim=8, codebook_size_prosody=10, codebook_size_content=10, codebook_size_residual=10,
            use_gr_residual_f0=True, use_gr_residual_phone=True, use_gr_x_timbre=True,
        )
        
        try:
            enc_ckpt = hf_hub_download("amphion/naturalspeech3_facodec", "ns3_facodec_encoder.bin")
            dec_ckpt = hf_hub_download("amphion/naturalspeech3_facodec", "ns3_facodec_decoder.bin")
            self.fa_encoder.load_state_dict(torch.load(enc_ckpt, map_location="cpu"))
            self.fa_decoder.load_state_dict(torch.load(dec_ckpt, map_location="cpu"))
        except Exception as e:
            print(f"Warning: Could not download FACodec weights automatically. {e}")

        self.fa_encoder.eval().to(self.device)
        self.fa_decoder.eval().to(self.device)
        
        # 2. Hybrid Head (Trainable)
        print(f"Initializing Hybrid Model with Manual Dim: {self.manual_feature_dim}")
        self.hybrid_head = HybridTimbreHead(
            embedding_dim=self.config.TIMBRE_DIM, 
            manual_dim=self.manual_feature_dim
        ).to(self.device)

    def extract_timbre(self, audio):
        # Input shape: (batch, time).
        with torch.no_grad():
            audio_in = audio.unsqueeze(1) # (Batch, 1, Time)
            z = self.fa_encoder(audio_in)
            _, _, _, _, spk = self.fa_decoder(z, vq=True, eval_vq=True)
            # The speaker embedding is normally shaped (batch, 256).
        return spk

    def save_weights_only(self, filename):
        save_dict = {
            'model_state_dict': self.hybrid_head.state_dict(),
            # Store metadata so inference can reconstruct the manual branch.
            'config_info': {
                'attr': self.config.COL_SCORE,
                'type': 'hybrid', 
                'manual_dim': self.manual_feature_dim
            }
        }
        torch.save(save_dict, os.path.join(self.config.CHECKPOINT_DIR, filename))

    def train_epoch(self, loader):
        self.hybrid_head.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            audio = batch['audio'].to(self.device)
            manual_feats = batch['manual_feats'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 1. Extract the FACodec embedding.
            spk = self.extract_timbre(audio)
            
            # 2. Run the two-input regression head.
            pred = self.hybrid_head(spk, manual_feats)
            
            # Loss Calculation
            loss_mse = F.mse_loss(pred, labels)
            loss_l1 = F.l1_loss(pred, labels)
            loss = loss_mse + 0.5 * loss_l1

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Detected a non-finite training loss before the optimizer update."
                    f" audio_finite={bool(torch.isfinite(audio).all())},"
                    f" manual_feats_finite={bool(torch.isfinite(manual_feats).all())},"
                    f" labels_finite={bool(torch.isfinite(labels).all())},"
                    f" prediction_finite={bool(torch.isfinite(pred).all())}"
                )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return total_loss / len(loader)

    def validate(self, loader):
        self.hybrid_head.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                audio = batch['audio'].to(self.device)
                manual_feats = batch['manual_feats'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                spk = self.extract_timbre(audio)
                
                # Run the two-input regression head.
                pred = self.hybrid_head(spk, manual_feats)
                
                loss = F.mse_loss(pred, labels) + 0.5 * F.l1_loss(pred, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Detected a non-finite validation loss."
                        f" audio_finite={bool(torch.isfinite(audio).all())},"
                        f" manual_feats_finite={bool(torch.isfinite(manual_feats).all())},"
                        f" labels_finite={bool(torch.isfinite(labels).all())},"
                        f" prediction_finite={bool(torch.isfinite(pred).all())}"
                    )
                total_loss += loss.item()
        return total_loss / len(loader)

    def train(self, train_loader, val_loader):
        print(f"Start training for {self.config.NUM_EPOCHS} epochs...")
        patience = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            self.scheduler.step()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience = 0
                self.save_weights_only("best_hybrid_model.pth")
            else:
                patience += 1
                if patience >= self.config.EARLY_STOP_PATIENCE:
                    print("Early stopping!")
                    break
        
        self.save_weights_only("final_hybrid_model.pth")
    def evaluate_mae(self, loader):
        """Load the best checkpoint and calculate mean absolute error."""
        print("\n[Final Evaluation] Loading the best checkpoint for MAE...")
        # Use the same filename created by save_weights_only.
        best_path = os.path.join(self.config.CHECKPOINT_DIR, "best_hybrid_model.pth")
        
        if not os.path.exists(best_path):
            print("Could not find the best model checkpoint.")
            return

        # Restore the best model state.
        checkpoint = torch.load(best_path, map_location=self.device)
        
        # Restore the hybrid regression head.
        self.hybrid_head.load_state_dict(checkpoint['model_state_dict'])
        self.hybrid_head.eval()

        total_abs_error = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Calculating MAE"):
                # The hybrid model requires both audio and manual features.
                audio = batch['audio'].to(self.device)
                manual_feats = batch['manual_feats'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 1. Extract the FACodec timbre embedding.
                spk = self.extract_timbre(audio)
                
                # 2. Predict from the embedding and manual features.
                pred = self.hybrid_head(spk, manual_feats)
                
                # Accumulate absolute error over all samples.
                abs_error = torch.abs(pred - labels)
                total_abs_error += abs_error.sum().item()
                total_samples += labels.size(0)

        final_mae = total_abs_error / total_samples
        print(f"\n" + "="*30)
        print("Final result (best hybrid model):")
        print(f"Mean absolute error (MAE): {final_mae:.4f}")
        print("="*30)
        return final_mae
    
def split_by_speaker(all_data: List[Dict], val_ratio=0.2):
    """
    Split training and validation data by speaker identity.
    """
    # 1. Collect unique speaker identities.
    speakers = list(set(d['speaker'] for d in all_data))
    speakers.sort()  # Sort before shuffling for reproducibility.
    random.seed(42)
    random.shuffle(speakers)
    
    # 2. Determine the speaker split point.
    n_val = int(len(speakers) * val_ratio)
    val_speakers = set(speakers[:n_val])
    train_speakers = set(speakers[n_val:])
    
    print(f"Total speakers: {len(speakers)}")
    print(f"Validation Speakers ({len(val_speakers)}): {list(val_speakers)[:5]}...")
    
    # 3. Assign samples according to speaker identity.
    train_data = [d for d in all_data if d['speaker'] in train_speakers]
    val_data = [d for d in all_data if d['speaker'] in val_speakers]
    
    print(f"Train samples (file-level): {len(train_data)}")
    print(f"Val samples (file-level): {len(val_data)}")
    
    return train_data, val_data

def expand_to_segments(data_list: List[Dict], target_sr=16000, target_len_samples=32000):
    """
    Represent each file as non-overlapping two-second segments.

    This does not create new audio files. It adds a ``start_offset`` to copied
    metadata. When VAD is enabled, only long leading and trailing silence is
    trimmed; pauses within the recording are preserved.
    """
    expanded_data = []
    trimmed_files = 0
    total_leading_trim_sec = 0.0
    total_trailing_trim_sec = 0.0
    print("Expanding audio files into two-second segments...")
    
    for item in tqdm(data_list):
        path = item['path']
        try:
            if Config.ENABLE_EDGE_VAD_TRIM:
                wav = load_and_lufs_normalize(path, target_sr)
                if wav is None:
                    continue
                total_samples = wav.size(1)
                trim_start, trim_end = detect_active_edge_bounds(wav, target_sr)
            else:
                # Read metadata only; this is faster than decoding the file.
                info = torchaudio.info(path)
                orig_sr = info.sample_rate
                orig_frames = info.num_frames
                duration_sec = orig_frames / orig_sr
                total_samples = int(duration_sec * target_sr)
                trim_start, trim_end = 0, total_samples

            leading_trim_sec = trim_start / target_sr
            trailing_trim_sec = max(0, total_samples - trim_end) / target_sr
            if trim_start > 0 or trim_end < total_samples:
                trimmed_files += 1
                total_leading_trim_sec += leading_trim_sec
                total_trailing_trim_sec += trailing_trim_sec
            
            # Calculate the number of complete non-overlapping segments.
            active_samples = max(0, trim_end - trim_start)
            num_segments = active_samples // target_len_samples
            
            # Keep one segment for short files; the dataset will pad it.
            if num_segments == 0:
                num_segments = 1
            
            for i in range(num_segments):
                # Copy the score, manual features, and speaker metadata.
                new_item = item.copy()
                # Store the segment start in samples at the target rate.
                new_item['start_offset'] = trim_start + i * target_len_samples
                new_item['trim_start_offset'] = trim_start
                new_item['trim_end_offset'] = trim_end
                expanded_data.append(new_item)
                
        except Exception as e:
            print(f"Error checking {path}: {e}")
            
    print(f"Segmentation complete: {len(data_list)} files -> {len(expanded_data)} segments")
    if Config.ENABLE_EDGE_VAD_TRIM:
        print(
            "VAD edge trimming: "
            f"{trimmed_files}/{len(data_list)} files, "
            f"leading={total_leading_trim_sec:.1f}s, "
            f"trailing={total_trailing_trim_sec:.1f}s"
        )
    return expanded_data
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune the 44-D timbre regression model on a target domain."
    )
    parser.add_argument(
        "--labels", required=True, help="CSV/XLSX containing labels and 44-D features."
    )
    parser.add_argument(
        "--audio-root", required=True, help="Root folder containing target audio files."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=Config.CHECKPOINT_DIR,
        help=f"Directory for model checkpoints (default: {Config.CHECKPOINT_DIR}).",
    )
    parser.add_argument(
        "--pretrained",
        help="Optional source-domain best_hybrid_model.pth. Omit to train from scratch.",
    )
    parser.add_argument("--filename-column", default=Config.COL_FILENAME)
    parser.add_argument("--score-column", default=Config.COL_SCORE)
    parser.add_argument("--speaker-column", default=Config.COL_SPEAKER)
    parser.add_argument("--freeze-level", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def configure_paths_and_columns(args):
    label_path = Path(args.labels).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()

    if not label_path.is_file():
        raise SystemExit(f"Label file does not exist: {label_path}")
    if not audio_root.is_dir():
        raise SystemExit(f"Audio root does not exist or is not a directory: {audio_root}")
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio must be between 0 and 1.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be zero or greater.")

    pretrained_path = None
    if args.pretrained:
        pretrained_path = Path(args.pretrained).expanduser().resolve()
        if not pretrained_path.is_file():
            raise SystemExit(f"Pretrained checkpoint does not exist: {pretrained_path}")

    Config.LABEL_FILE = str(label_path)
    Config.AUDIO_ROOT = str(audio_root)
    Config.CHECKPOINT_DIR = str(checkpoint_dir)
    Config.COL_FILENAME = args.filename_column
    Config.COL_SCORE = args.score_column
    Config.COL_SPEAKER = args.speaker_column
    return str(pretrained_path) if pretrained_path else None


########## Main ##########
def main():
    args = parse_args()
    pretrained_weights = configure_paths_and_columns(args)
    print(f"Compute device: {Config.DEVICE}")
    if not torch.cuda.is_available():
        print("Warning: no GPU detected; training will be slow.")

    # 1. Load and align the target-domain data.
    try:
        all_data, detected_feat_dim = DataAligner.align_data(
            label_path=Config.LABEL_FILE, 
            audio_root=Config.AUDIO_ROOT
        )
    except Exception as e:
        print(f"Failed to load training data: {e}")
        return

    # 2. Split by speaker to prevent speaker leakage between subsets.
    train_data_raw, val_data_raw = split_by_speaker(
        all_data, val_ratio=args.val_ratio
    )
    
    # 3. Expand each file into two-second segment metadata.
    train_data_segments = expand_to_segments(train_data_raw)
    val_data_segments = expand_to_segments(val_data_raw)
    
    # 4. Build datasets and shuffle only the training loader.
    train_dataset = HybridDataset(train_data_segments)
    val_dataset = HybridDataset(val_data_segments)
    
    print(f"Final training segments: {len(train_dataset)}")
    print(f"Final validation segments: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # 5. Fine-tune and evaluate the model.
    trainer = Trainer(
        Config(), 
        manual_feature_dim=detected_feat_dim,
        pretrained_path=pretrained_weights,
        freeze_level=args.freeze_level,
    )
    
    trainer.train(train_loader, val_loader)
    trainer.evaluate_mae(val_loader)
if __name__ == "__main__":
    main()
