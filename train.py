import os
import sys
import warnings
import logging
import argparse
import re
import math
import random
import numpy as np
import pandas as pd
import torch
import sacrebleu
import gc

warnings.filterwarnings("ignore")

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================
# Global Configuration & Paths
# ============================================================
MODEL_NAME = "google/byt5-large"

# Kaggle Output Directory (saves to your working directory so it's downloadable)
OUTPUT_DIR = "/kaggle/working/byt5-large-draft"

# Kaggle Input Directory
# IMPORTANT: Update "your-dataset-folder" to the actual name of your dataset in Kaggle's input panel.
DATA_DIR = "/kaggle/input/datasets/priyanshkeshari/deep-past-dataset" 

# Note: Fallbacks are provided just in case it's placed directly in /kaggle/working/
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv") if os.path.exists(os.path.join(DATA_DIR, "train.csv")) else "train.csv"
LEXICON_PATH = os.path.join(DATA_DIR, "OA_Lexicon_eBL.csv") if os.path.exists(os.path.join(DATA_DIR, "OA_Lexicon_eBL.csv")) else "OA_Lexicon_eBL.csv"
DICT_PATH = os.path.join(DATA_DIR, "eBL_Dictionary.csv") if os.path.exists(os.path.join(DATA_DIR, "eBL_Dictionary.csv")) else "eBL_Dictionary.csv"

SEED = 42

# Input length kept large to fit Dictionary Draft injections
MAX_INPUT_LENGTH = 1536 
MAX_TARGET_LENGTH = 512
EVAL_MAX_LENGTH = 384
EPOCHS = 12

# H100 80GB Constraints:
# 16 batch_size * 2 grad_accum = 32 Effective Batch Size
BATCH_SIZE = 16        
GRAD_ACCUM = 2        
EVAL_BATCH_SIZE = 32   

LR = 2e-4             
LABEL_SMOOTHING = 0.1
LEX_SUBSAMPLE = 1400


# ============================================================
# 0. Data Prep & Helper Functions (Formerly shared.data_prep)
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_split(data_dir):
    """Loads the raw training data and splits it into 90/10 Train/Val"""
    logger.info(f"Loading data from {TRAIN_FILE}...")
    df = pd.read_csv(TRAIN_FILE)
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_size = max(1, int(len(df) * 0.1))
    
    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)
    return train_df, val_df

def create_bidirectional_pairs(df):
    """Creates bidirectional training pairs: Akkadian -> English and English -> Akkadian."""
    inputs = []
    targets = []
    weights = []
    
    for _, row in df.iterrows():
        src = str(row['transliteration']).strip()
        tgt = str(row['translation']).strip()
        w = row.get('seq_weight', 1.0)
        
        # Forward pair
        inputs.append(f"translate: akkadian to english: {src}")
        targets.append(tgt)
        weights.append(w)
        
        # Backward pair
        inputs.append(f"translate: english to akkadian: {tgt}")
        targets.append(src)
        weights.append(w)
        
    return pd.DataFrame({"input_text": inputs, "target_text": targets, "seq_weight": weights})

def create_dictionary_pairs():
    """Creates standalone bidirectional training pairs directly from the Lexicon and eBL Dictionary."""
    dfs = []
    if os.path.exists(LEXICON_PATH):
        df_lex = pd.read_csv(LEXICON_PATH).dropna(subset=['form', 'lexeme'])
        df_lex = df_lex.rename(columns={"form": "transliteration", "lexeme": "translation"})
        dfs.append(df_lex[['transliteration', 'translation']])
    
    if os.path.exists(DICT_PATH):
        df_ebl = pd.read_csv(DICT_PATH).dropna(subset=['word', 'definition'])
        df_ebl = df_ebl.rename(columns={"word": "transliteration", "definition": "translation"})
        dfs.append(df_ebl[['transliteration', 'translation']])
        
    if not dfs:
        return pd.DataFrame(columns=["input_text", "target_text", "seq_weight"])
        
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined["seq_weight"] = 1.0
    return create_bidirectional_pairs(df_combined).drop_duplicates().reset_index(drop=True)

def compute_metrics(eval_preds, tokenizer):
    """Calculates BLEU, chrF++, and the Geometric Mean for evaluation."""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Replace -100 masking with pad token ID for decoding
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
    chrf = sacrebleu.corpus_chrf(decoded_preds, decoded_labels, word_order=2)
    
    geo_mean = math.sqrt((bleu.score + 1e-8) * (chrf.score + 1e-8))
    
    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "geo_mean": geo_mean
    }


# ============================================================
# 1. Text Sanitization Utilities
# ============================================================
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})
_ALLOWED_FRACS = [(1/6, "0.16666"), (1/4, "0.25"), (1/3, "0.33333"), (1/2, "0.5"), (2/3, "0.66666"), (3/4, "0.75"), (5/6, "0.83333")]
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")
_WS_RE = re.compile(r"\s+")
_GAP_UNIFIED_RE = re.compile(r"<\s*big[\s_\-]*gap\s*>|<\s*gap\s*>|\bbig[\s_\-]*gap\b|\bx(?:\s+x)+\b|\.{3,}|…+|\[\.+\]|\[\s*x\s*\]|\(\s*x\s*\)|(?<!\w)x{2,}(?!\w)|(?<!\w)x(?!\w)|\(\s*large\s+break\s*\)|\(\s*break\s*\)|\(\s*\d+\s+broken\s+lines?\s*\)", re.I)
_CHAR_TRANS = str.maketrans({"ḫ":"h","Ḫ":"H","ʾ":"","₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9","—":"-","–":"-"})
_SUB_X = "ₓ"
_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")
_PN_RE = re.compile(r"\bPN\b")
_KUBABBAR_RE = re.compile(r"KÙ\.B\.")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {"0.8333": "⅚", "0.6666": "⅔", "0.3333": "⅓", "0.1666": "⅙", "0.625": "⅝", "0.75": "¾", "0.25": "¼", "0.5": "½"}
_SOFT_GRAM_RE = re.compile(r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)(?:\.\s*(?:plur|plural|sing|singular))?\.?\s*[^)]*\)", re.I)
_BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE = re.compile(r"\(\?\)")
_CURLY_DQ_RE = re.compile("[\u201c\u201d]")
_CURLY_SQ_RE = re.compile("[\u2018\u2019]")
_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12}
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")
_FORBIDDEN_TRANS = str.maketrans("", "", '——<>⌈⌋⌊[]+ʾ;')
_COMMODITY_RE = re.compile(r'(?<=\s)-(gold|tax|textiles)\b')
_COMMODITY_REPL = {"gold": "pašallum gold", "tax": "šadduātum tax", "textiles": "kutānum textiles"}
_SHEKEL_REPLS = [
    (re.compile(r'5\s+11\s*/\s*12\s+shekels?', re.I), '6 shekels less 15 grains'),
    (re.compile(r'5\s*/\s*12\s+shekels?', re.I), '⅓ shekel 15 grains'),
    (re.compile(r'7\s*/\s*12\s+shekels?', re.I), '½ shekel 15 grains'),
    (re.compile(r'1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?', re.I), '15 grains'),
]
_SLASH_ALT_RE = re.compile(r'(?<![0-9/])\s+/\s+(?![0-9])\S+')
_STRAY_MARKS_RE = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')
_MULTI_GAP_RE = re.compile(r'(?:<gap>\s*){2,}')
_EXTRA_STRAY_RE = re.compile(r'(?<!\w)(?:\.\.+|xx+)(?!\w)')
_HACEK_TRANS = str.maketrans({"ḫ": "h", "Ḫ": "H"})

def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))
    if abs(frac - best[0]) <= 2e-3:
        dec = best[1]
        if ip == 0: return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    return f"{x:.5f}".rstrip("0").rstrip(".")

def _frac_repl(m: re.Match) -> str: return _EXACT_FRAC_MAP[m.group(0)]
def _commodity_repl(m: re.Match) -> str: return _COMMODITY_REPL[m.group(1)]
def _month_repl(m: re.Match) -> str: return f"Month {_ROMAN2INT.get(m.group(1).upper(), m.group(1))}"
def _normalize_gaps_vec(ser: pd.Series) -> pd.Series: return ser.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)
def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz", "š").replace("SZ", "Š").replace("s,", "ṣ").replace("S,", "Ṣ").replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    return _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)

class OptimizedPreprocessor:
    def preprocess_batch(self, texts: list[str]) -> list[str]:
        ser = pd.Series(texts).fillna("").astype(str)
        ser = ser.apply(_ascii_to_diacritics)
        ser = ser.str.replace(_DET_UPPER_RE, r"\1", regex=True)
        ser = ser.str.replace(_DET_LOWER_RE, r"{\1}", regex=True)
        ser = _normalize_gaps_vec(ser)
        ser = ser.str.translate(_CHAR_TRANS)
        ser = ser.str.replace(_SUB_X, "", regex=False)
        ser = ser.str.replace(_KUBABBAR_RE, "KÙ.BABBAR", regex=True)
        ser = ser.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        ser = ser.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()

class VectorizedPostprocessor:
    def postprocess_batch(self, translations: list[str]) -> list[str]:
        s = pd.Series(translations).fillna("").astype(str)
        s = _normalize_gaps_vec(s)
        s = s.str.replace(_PN_RE, "<gap>", regex=True)
        s = s.str.replace(_COMMODITY_RE, _commodity_repl, regex=True)
        for pat, repl in _SHEKEL_REPLS:
            s = s.str.replace(pat, repl, regex=True)
        s = s.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        s = s.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
        s = s.str.replace(_SOFT_GRAM_RE, " ", regex=True)
        s = s.str.replace(_BARE_GRAM_RE, " ", regex=True)
        s = s.str.replace(_UNCERTAIN_RE, "", regex=True)
        s = s.str.replace(_STRAY_MARKS_RE, "", regex=True)
        s = s.str.replace(_EXTRA_STRAY_RE, "", regex=True)
        s = s.str.replace(_SLASH_ALT_RE, "", regex=True)
        s = s.str.replace(_CURLY_DQ_RE, '"', regex=True)
        s = s.str.replace(_CURLY_SQ_RE, "'", regex=True)
        s = s.str.replace(_MONTH_RE, _month_repl, regex=True)
        s = s.str.replace(_MULTI_GAP_RE, "<gap>", regex=True)
        
        # Protect <gap> markers before stripping forbidden chars
        s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
        s = s.str.translate(_FORBIDDEN_TRANS)
        s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
        
        s = s.str.translate(_HACEK_TRANS)
        s = s.str.replace(_REPEAT_WORD_RE, r"\1", regex=True)
        for n in range(4, 1, -1):
            pat = r"\b((?:\w+\s+){" + str(n-1) + r"}\w+)(?:\s+\1\b)+"
            s = s.str.replace(pat, r"\1", regex=True)
        s = s.str.replace(_PUNCT_SPACE_RE, r"\1", regex=True)
        s = s.str.replace(_REPEAT_PUNCT_RE, r"\1", regex=True)
        s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
        return s.tolist()


# ============================================================
# 2. Data Augmentor & Leakage Prevention
# ============================================================
class AkkadianDataAugmentor:
    def __init__(self):
        self.anchors = {
            r'\bKIŠIB\b': 'Seal', r'\bDUMU\b': 'son',
            r'\bKÙ\.BABBAR\b': 'silver', r'\bAN\.NA\b': 'tin',
            r'\bTÚG\b': 'textile', r'\bGÚ\b': 'talent'
        }

    def resolve_math_and_anchors(self, text: str) -> str:
        if not isinstance(text, str): return text
        # Make regex strictly require digits (e.g., "0.5" or "5", not ".")
        math_pattern = r'(?:(\d+(?:\.\d+)?)\s*ma-na)?(?:\s*(\d+(?:\.\d+)?)\s*GÍN)?'
        
        def math_replacer(match):
            mina_str, shekel_str = match.groups()
            if not mina_str and not shekel_str: return match.group(0)
            try:
                minas = float(mina_str) if mina_str else 0.0
                shekels = float(shekel_str) if shekel_str else 0.0
            except ValueError:
                return match.group(0)
                
            total_shekels = round((minas * 60) + shekels)
            if total_shekels == 0: return match.group(0)
            return f"{match.group(0)} [MATH: {total_shekels} shekels]"

        text = re.sub(math_pattern, math_replacer, text)
        for pat, meaning in self.anchors.items():
            text = re.sub(pat, f"{pat} [ANCHOR: {meaning}]", text)
        return text

    def augment_tablet_damage(self, source: str, target: str) -> dict:
        """Simulates tablet damage by randomly injecting <gap> into the source."""
        words = source.split()
        if len(words) < 3:
            return {'transliteration': source, 'translation': target, 'is_synthetic': False}
            
        # Randomly replace 1 to 3 words with <gap>
        num_gaps = random.randint(1, min(4, len(words) // 2))
        gap_indices = random.sample(range(len(words)), num_gaps)
        for idx in gap_indices:
            words[idx] = "<gap>"
            
        new_source = " ".join(words)
        return {'transliteration': new_source, 'translation': target, 'is_synthetic': True}

    def remove_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        train_src = train_df['transliteration'].astype(str).str.strip().str.lower()
        train_tgt = train_df['translation'].astype(str).str.strip().str.lower()
        val_src = val_df['transliteration'].astype(str).str.strip().str.lower()
        val_tgt = val_df['translation'].astype(str).str.strip().str.lower()
        
        leak_mask = train_tgt.isin(val_tgt) | train_src.isin(val_src)
        leaks_found = leak_mask.sum()
        
        if leaks_found > 0:
            logger.warning(f"Data Leakage Guard: Dropping {leaks_found} leaking rows from training data.")
            train_df = train_df[~leak_mask].reset_index(drop=True)
        return train_df

    def calculate_seq_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        lengths = df['transliteration'].astype(str).str.len()
        df['seq_weight'] = lengths.astype(float)
        logger.info(f"Sequence weights calculated. Min: {df['seq_weight'].min():.1f}, Max: {df['seq_weight'].max():.1f}")
        return df

    def build_training_dataset(self, train_data, val_data=None, num_augments: int = 3) -> pd.DataFrame:
        df = pd.read_csv(train_data) if isinstance(train_data, str) else train_data.copy()
        if val_data is not None:
            val_df = pd.read_csv(val_data) if isinstance(val_data, str) else val_data
            df = self.remove_leakage(df, val_df)
        
        all_rows = []
        logger.info(f"Applying robust data augmentation (Tablet Damage Denoising Autoencoder) (x{num_augments} multiplier)...")
        for _, row in df.iterrows():
            source = str(row['transliteration'])
            target = str(row['translation'])
            all_rows.append({'transliteration': source, 'translation': target, 'is_synthetic': False})
            for _ in range(num_augments):
                all_rows.append(self.augment_tablet_damage(source, target))
            
        aug_df = pd.DataFrame(all_rows)
        logger.info(f"Dataset expanded to {len(aug_df)} rows.")
        
        logger.info("Injecting Base-60 math resolutions and Sumerogram anchors...")
        aug_df['transliteration'] = aug_df['transliteration'].apply(self.resolve_math_and_anchors)
        aug_df = self.calculate_seq_weights(aug_df)
        
        return aug_df


# ============================================================
# 3. Dataset & Weighted Trainer
# ============================================================
class DraftMTDataset:
    def __init__(self, df, tokenizer, max_input_len, max_target_len):
        self.inputs = df["input_text"].tolist()
        self.targets = df["target_text"].tolist()
        # Default weight to 1.0 if not assigned (e.g., for Lexicon/Val data)
        self.weights = df["seq_weight"].tolist() if "seq_weight" in df.columns else [1.0] * len(df)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.tokenizer(self.inputs[idx], max_length=self.max_input_len, truncation=True)
        y = self.tokenizer(self.targets[idx], max_length=self.max_target_len, truncation=True)
        
        # Mask padding tokens so loss is not calculated on padded bytes
        labels = y["input_ids"]
        labels = [l if l != self.tokenizer.pad_token_id else -100 for l in labels]
        x["labels"] = labels
        
        # Add weights for custom loss calculation
        x["weights"] = torch.tensor(self.weights[idx], dtype=torch.float32)
        return x

class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss function that scales the standard cross-entropy loss 
        by the normalized sequence length weights.
        """
        weights = inputs.pop("weights", None)
        outputs = model(**inputs)
        loss = outputs.loss
        
        if weights is not None:
            # Normalize weights by the batch mean to prevent massive gradient explosion
            # while still assigning relative importance to longer sequences in the batch
            normalized_weights = weights / (weights.mean() + 1e-8)
            loss = (loss * normalized_weights).mean()
            
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        """
        Override prediction_step to safely remove our custom "weights" key.
        If we don't remove it, model.generate() crashes because it doesn't recognize "weights".
        Removing it during eval also means our Validation Loss is standard cross-entropy.
        """
        if "weights" in inputs:
            # Create a new dictionary without the weights key to pass to the model
            inputs = {k: v for k, v in inputs.items() if k != "weights"}
            
        return super().prediction_step(
            model, 
            inputs, 
            prediction_loss_only, 
            ignore_keys=ignore_keys, 
            **gen_kwargs
        )


# ============================================================
# 4. Main Pipeline
# ============================================================
def main(dry_run: bool = False):
    seed_everything(SEED)

    # 1. Load Data
    logger.info("Loading train and validation splits ...")
    train_df, val_df = load_split(DATA_DIR)
    
    # 2. Text Sanitization (Run Before Augmentation)
    preprocessor = OptimizedPreprocessor()
    postprocessor = VectorizedPostprocessor()
    
    # Assuming 'transliteration' is your input column in the raw CSV
    # If it is 'input_text', we rename them to transliteration for the preprocessor
    if 'input_text' in train_df.columns:
        train_df = train_df.rename(columns={"input_text": "transliteration"})
        val_df = val_df.rename(columns={"input_text": "transliteration"})
        
    train_df["transliteration"] = preprocessor.preprocess_batch(train_df["transliteration"].tolist())
    val_df["transliteration"] = preprocessor.preprocess_batch(val_df["transliteration"].tolist())
    train_df["translation"] = postprocessor.postprocess_batch(train_df["translation"].tolist())
    val_df["translation"] = postprocessor.postprocess_batch(val_df["translation"].tolist())

    # 3. Data Augmentation & Leakage Prevention
    logger.info("Initializing Data Augmentor...")
    augmentor = AkkadianDataAugmentor()
    
    # Passing `val_df` here automatically drops any leaking sentences before multiplying!
    train_df = augmentor.build_training_dataset(
        train_data=train_df, 
        val_data=val_df, 
        num_augments=3 if not dry_run else 1
    )

    # 4. Create Bidirectional Pairs
    logger.info("Creating Bidirectional Pairs...")
    train_draft = create_bidirectional_pairs(train_df)
    val_draft = create_bidirectional_pairs(val_df)

    dict_df = create_dictionary_pairs()
    n_lex = min(LEX_SUBSAMPLE, len(dict_df))
    dict_sub = dict_df.sample(n=n_lex, random_state=SEED).reset_index(drop=True)

    # Append dict_sub but ensure it has the necessary seq_weight column (default to 1.0)
    if "seq_weight" not in dict_sub.columns:
        dict_sub["seq_weight"] = 1.0

    train_full = pd.concat([train_draft, dict_sub], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    if dry_run:
        train_full = train_full.head(32)
        val_draft = val_draft.head(16)

    logger.info(f"Final Train size (augmented sentences + lexicon): {len(train_full)}")
    logger.info(f"Final Val size: {len(val_draft)}")

    # 5. Load ByT5-Large Model
    logger.info(f"Loading {MODEL_NAME} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_dataset = DraftMTDataset(train_full, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    val_dataset = DraftMTDataset(val_draft, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    # 6. Training Arguments (H100 80GB Optimized)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "phase1"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,                # FIX: Prevents massive optimizer state writes!
        save_total_limit=1,                  # FIX: Only keeps 1 recent + 1 best checkpoint to save 15GB disk space
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        num_train_epochs=1 if dry_run else EPOCHS,
        weight_decay=0.01,
        label_smoothing_factor=LABEL_SMOOTHING,
        predict_with_generate=True,
        generation_max_length=EVAL_MAX_LENGTH,
        fp16=False,
        bf16=True,                           # H100 excels at BF16
        tf32=True,                           # H100 Hopper TF32 speedup
        dataloader_num_workers=2,            # Reduced to prevent dataloader thread deadlocks
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",           # Fused AdamW is extremely fast on H100
        lr_scheduler_type="cosine",
        warmup_steps=150,                    # Replaced deprecated warmup_ratio
        load_best_model_at_end=True,
        metric_for_best_model="geo_mean",
        report_to="none",                    # Handled by WANDB_DISABLED as well
        logging_steps=50,
        remove_unused_columns=False,
    )

    trainer = WeightedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        compute_metrics=lambda ep: compute_metrics(ep, tokenizer),
    )

    class MetricLogger(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                ep = metrics.get("epoch", "?")
                gm = metrics.get("eval_geo_mean", 0)
                bl = metrics.get("eval_bleu", 0)
                ch = metrics.get("eval_chrf", 0)
                logger.info(">>> Epoch %s  |  geo_mean: %.2f  |  BLEU: %.2f  |  chrF++: %.2f", ep, gm, bl, ch)

            # Prevent memory fragmentation and lingering hooks after evaluation
            gc.collect()
            torch.cuda.empty_cache()

    trainer.add_callback(MetricLogger())

    # 7. Execute Training
    logger.info(f"Starting Training for {MODEL_NAME}...")
    trainer.train()

    best_path = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_path)
    logger.info(f"Best model saved to {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Quick test with tiny subset")
    # Use parse_known_args() so Kaggle's internal -f arguments are safely ignored!
    args, _ = parser.parse_known_args()
    main(dry_run=args.dry_run)
# Final commit update
