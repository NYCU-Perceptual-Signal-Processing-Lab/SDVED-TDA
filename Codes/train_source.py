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
    CHECKPOINT_DIR = str(PROJECT_ROOT / "outputs" / "source_checkpoints")
    
    # Column Mapping
    COL_FILENAME = "filename"
    COL_SCORE = "score"
    COL_SPEAKER = "speaker_id"
    
    # Audio settings
    TARGET_SR = 16000
    TARGET_LEN = 2 * TARGET_SR  # Randomly crop two seconds during training.
    
    # Training settings
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 10*1e-4
    WEIGHT_DECAY = 1e-5
    VAL_RATIO = 0.2
    SPLIT_SEED = 42
    
    # Model settings
    TIMBRE_DIM = 256  # FACodec output dimension.
    MANUAL_DIM = EXPECTED_FEATURE_DIM
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Saving
    EARLY_STOP_PATIENCE = 10

########## Data Alignment & Processing ##########
class DataAligner:
    """Align label rows with audio files and collect manual features."""
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
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The label table is missing required columns: {missing_cols}")

        # 2. Use the fixed public 44-D schema; ignore unrelated CSV columns.
        missing_features = [c for c in FEATURE_COLUMNS_44D if c not in df.columns]
        if missing_features:
            raise ValueError(f"The CSV is missing 44-D feature columns: {missing_features}")
        feature_cols = FEATURE_COLUMNS_44D
        num_features = len(feature_cols)
        print(f"Using the fixed {num_features}-D manual feature set")

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
        for _, row in tqdm(df.iterrows(), total=len(df)):
            fname_str = str(row[Config.COL_FILENAME])
            fname_stem = Path(fname_str).stem
            
            if fname_stem in audio_files_map:
                # Keep raw values; normalize after the sample split.
                feat_vector = row[feature_cols].values.astype(np.float32)
                
                aligned_data.append({
                    'path': audio_files_map[fname_stem],
                    'score': float(row[Config.COL_SCORE]),
                    'manual_feats': feat_vector,
                    'orig_name': fname_str,
                    'speaker': str(row[Config.COL_SPEAKER])
                })
            else:
                missing_count += 1

        print(f"Alignment complete: matched={len(aligned_data)}, missing={missing_count}")
        return aligned_data, num_features


def split_by_sample(data: List[Dict], val_ratio: float, seed: int):
    """Randomly split individual aligned samples into train and validation sets."""
    if len(data) < 2:
        raise ValueError("A sample-level split requires at least two samples")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    shuffled_data = list(data)
    rng = random.Random(seed)
    rng.shuffle(shuffled_data)
    num_val_samples = max(1, int(len(shuffled_data) * val_ratio))
    num_val_samples = min(num_val_samples, len(shuffled_data) - 1)

    val_data = shuffled_data[:num_val_samples]
    train_data = shuffled_data[num_val_samples:]

    train_speakers = {item['speaker'] for item in train_data}
    val_speakers = {item['speaker'] for item in val_data}

    print(
        f"Sample-level split (seed={seed}): "
        f"train={len(train_data)} samples, "
        f"validation={len(val_data)} samples, "
        f"speaker_overlap={len(train_speakers & val_speakers)}"
    )
    return train_data, val_data


def normalize_manual_features(train_data: List[Dict], val_data: List[Dict]):
    """Fit Z-score statistics on training samples and apply them to both sets."""
    train_values = np.stack([item['manual_feats'] for item in train_data])
    mean = np.mean(train_values, axis=0)
    std = np.std(train_values, axis=0) + 1e-8

    for item in train_data:
        item['manual_feats'] = ((item['manual_feats'] - mean) / std).astype(np.float32)
    for item in val_data:
        item['manual_feats'] = ((item['manual_feats'] - mean) / std).astype(np.float32)

    print("Manual features normalized with training-set mean and standard deviation")
    return mean.astype(np.float32), std.astype(np.float32)

def load_and_lufs_normalize(path: str, out_sr: int = 16000) -> Optional[torch.Tensor]:
    try:
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        wav = wav / (wav.abs().max() + 1e-8) * 0.95
        
        if sr != out_sr:
            wav = torchaudio.transforms.Resample(sr, out_sr)(wav)
            
        return wav
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

class HybridDataset(Dataset):
    def __init__(self, data_list: List[Dict]):
        self.data = data_list

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        wav = load_and_lufs_normalize(item['path'], Config.TARGET_SR)
        if wav is None:
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        
        # Apply a random crop during training.
        T = wav.size(1)
        if T >= Config.TARGET_LEN:
            start = torch.randint(0, T - Config.TARGET_LEN + 1, (1,)).item()
            wav = wav[:, start:start + Config.TARGET_LEN]
        else:
            pad = Config.TARGET_LEN - T
            wav = F.pad(wav, (0, pad))
        
        # Convert the manual features and target score to tensors.
        manual_feats = torch.tensor(item['manual_feats'], dtype=torch.float32)
        label = torch.tensor([item['score']], dtype=torch.float32)
        
        return {
            'audio': wav.squeeze(0),
            'manual_feats': manual_feats,
            'labels': label,
            'path': item['path']
        }

########## Model (Feature Fusion Architecture) ##########
class HybridTimbreHead(nn.Module):
    def __init__(
        self, embedding_dim=256, manual_dim=EXPECTED_FEATURE_DIM, dropout=0.1
    ):
        super().__init__()
        
        # Path A: FACodec Embedding
        # Normalize the pretrained FACodec representation.
        self.embed_norm = nn.LayerNorm(embedding_dim)
        
        # Path B: Manual Features
        # Normalize the 44-D manual feature vector from the CSV.
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
                        
            nn.Linear(4, 1),
            nn.GELU() # Output Score
        )
    # sigmoid version
    def forward(self, facodec_embed, manual_feats):
        x_emb = self.embed_norm(facodec_embed)
        x_man = self.manual_net(manual_feats)
        combined = torch.cat([x_emb, x_man], dim=1)
        
        return self.fusion_net(combined)
    # w/o sigmoid
    """
    def forward(self, facodec_embed, manual_feats):
        
        x_emb = self.embed_norm(facodec_embed)
        x_man = self.manual_net(manual_feats)
        
        # Concatenate the two feature branches.
        # 
        combined = torch.cat([x_emb, x_man], dim=1)
        
        # Predict the regression score.
        return self.fusion_net(combined)
    """
########## Trainer ##########
class Trainer:
    def __init__(self, config: Config, manual_feature_dim: int):
        self.config = config
        self.device = config.DEVICE
        
        # Use the feature dimension validated by DataAligner.
        self.manual_feature_dim = manual_feature_dim
        
        self.setup_models()
        
        self.optimizer = AdamW(self.hybrid_head.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    def print_model_parameters(model, name="Model"):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"--- {name} parameter summary ---")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("-" * 30)

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

        Trainer.print_model_parameters(self.fa_encoder, "FACodec Encoder")
        Trainer.print_model_parameters(self.fa_decoder, "FACodec Decoder")
        if hasattr(self.fa_decoder, 'timbre_encoder'):
            timbre_branch = self.fa_decoder.timbre_encoder
            t_total = sum(p.numel() for p in timbre_branch.parameters())
            print("--- FACodec timbre encoder ---")
            print(f"Parameters: {t_total:,}")
            print("-" * 40)
        else:
            print("Could not find timbre_encoder; check the FACodec implementation.")
        self.fa_encoder.eval().to(self.device)
        self.fa_decoder.eval().to(self.device)
        # 2. Hybrid Head (Trainable)
        print(f"Initializing Hybrid Model with Manual Dim: {self.manual_feature_dim}")
        self.hybrid_head = HybridTimbreHead(
            embedding_dim=self.config.TIMBRE_DIM, 
            manual_dim=self.manual_feature_dim
        ).to(self.device)
        Trainer.print_model_parameters(self.hybrid_head, "Hybrid Timbre Head")
        
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

    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the 44-D source-domain timbre regression model."
    )
    parser.add_argument(
        "--labels", required=True, help="CSV/XLSX containing labels and 44-D features."
    )
    parser.add_argument(
        "--audio-root", required=True, help="Root folder containing source audio files."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=Config.CHECKPOINT_DIR,
        help=f"Directory for model checkpoints (default: {Config.CHECKPOINT_DIR}).",
    )
    parser.add_argument("--filename-column", default=Config.COL_FILENAME)
    parser.add_argument("--score-column", default=Config.COL_SCORE)
    parser.add_argument("--speaker-column", default=Config.COL_SPEAKER)
    return parser.parse_args()


def configure_paths_and_columns(args):
    label_path = Path(args.labels).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()

    if not label_path.is_file():
        raise SystemExit(f"Label file does not exist: {label_path}")
    if not audio_root.is_dir():
        raise SystemExit(f"Audio root does not exist or is not a directory: {audio_root}")

    Config.LABEL_FILE = str(label_path)
    Config.AUDIO_ROOT = str(audio_root)
    Config.CHECKPOINT_DIR = str(checkpoint_dir)
    Config.COL_FILENAME = args.filename_column
    Config.COL_SCORE = args.score_column
    Config.COL_SPEAKER = args.speaker_column


########## Main ##########
def main():
    args = parse_args()
    configure_paths_and_columns(args)
    print(f"Compute device: {Config.DEVICE}")
    if not torch.cuda.is_available():
        print("Warning: no GPU detected; training will be slow.")

    # 1. Align data and validate the feature dimension.
    try:
        # Returns the aligned rows and feature count.
        all_data, detected_feat_dim = DataAligner.align_data(
            label_path=Config.LABEL_FILE, 
            audio_root=Config.AUDIO_ROOT
        )
    except Exception as e:
        print(f"Failed to load training data: {e}")
        return

    # 2. Random sample-level split; speakers may occur in both subsets.
    train_data, val_data = split_by_sample(
        all_data,
        val_ratio=Config.VAL_RATIO,
        seed=Config.SPLIT_SEED,
    )
    normalize_manual_features(train_data, val_data)
    
    # Build datasets containing both audio and manual features.
    train_dataset = HybridDataset(train_data)
    val_dataset = HybridDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. Train with the validated feature dimension.
    trainer = Trainer(Config(), manual_feature_dim=detected_feat_dim)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
