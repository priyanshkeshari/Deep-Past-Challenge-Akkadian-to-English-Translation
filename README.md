# Deep Past Challenge: Akkadian-to-English Translation

This repository contains the training and inference pipeline for decoding **4,000-year-old Old Assyrian Akkadian** transliterations into English. This pipeline was built for the [Kaggle Deep Past Initiative Challenge](https://www.kaggle.com/), aimed at addressing intense low-resource constraints, heavy morphological variability, and broken/damaged semantic contexts in ancient clay tablets.

Our solution consists of two fundamental parts:
1. `train.py`: High-performance model training pipeline utilizing `ByT5-large` with custom sequences scaling, bidirectional grounding, and Tablet Damage Autoencoding.
2. `inference.ipynb`: Minimum Bayes Risk (MBR) multi-model sampling generator.



## 🛠 Engineering & Data Science Justifications

### 1. Model Selection: `google/byt5-large`
Translating ancient Akkadian introduces deep out-of-vocabulary (OOV) challenges. Transliterations consist of unpredictable syllables, non-standard diacritical variants, and fragmented character chunks. 
- Standard subword tokenizers (like SentencePiece used in T5/BART) split these syllables arbitrarily, isolating functional suffixes and heavily fragmenting meaning.
- We opted for **ByT5**, a completely *byte-level* (character-level) model. By bypassing the subword dictionary entirely, the model handles out-of-distribution variations, special unicode Cuneiform representations, and partial words seamlessly without failing back to `<unk>` tokens. 

### 2. Denoising Tablet Damage (Data Augmentation)
Since we are dealing with thousands of unearthed, chipped, and physically damaged clay tablets, data augmentation couldn't just rely on standard NLP tricks like synonym replacement (which doesn't exist broadly for Akkadian). 

We engineered a **Tablet Damage Simulator** inside `AkkadianDataAugmentor`:
- During training batch generation, `1-4` sequential or random tokens are entirely wiped and replaced with `<gap>`.
- The Loss function acts as a **Denoising Autoencoder**, aggressively teaching the model to interpolate the missing semantic context to still arrive at the pristine English target translation.
- This creates massive robustness against noisy or incomplete evaluation sets.

### 3. Bidirectional Anchoring
Low-resource linguistic tasks often suffer from "catastrophic forgetting", where the model learns overarching statistical grammar (sentence generation) but starts hallucinating hard vocab terms.
- We merged **standalone exact word mappings** from both `OA_Lexicon_eBL` and `eBL_Dictionary`.
- We fed these standalone pairs into the dataset **Bidirectionally**:
  - `translate: akkadian to english: [word] -> [def]`
  - `translate: english to akkadian: [def] -> [word]`
- By forcing the model to continuously auto-regress backward and forward on literal translations, we act as a severe regularization mechanism keeping the generation statistically rooted to academic dictionaries.

### 4. Custom Sequence-Weighted Loss Function
The Deep Past Competition is evaluated on the geometric mean of `BLEU` and `chrF++`. 
- By default, standard Cross-Entropy Loss averages error across all tokens equally. This routinely results in the model getting "lazy" by perfectly memorizing easy one-word dictionary definitions to lower perplexity, while failing drastically on complex 20-word multi-clause administrative texts.
- **Solution (`WeightedSeq2SeqTrainer`)**: We applied a dynamic mathematical penalty scaling factor corresponding to the **character length** of the source sequence. The longer and more syntactically complex the row, the heavier the gradient penalty on a mistake. This explicitly prioritizes `BLEU` coherence along complex sentences.

### 5. Deterministic Pre/Post-processing
We completely eliminated dataset discrepancy noise:
- **Transcription Alignment**: Automatically standardized fraction representations (e.g., standardizing erratic translations of `5/12 shekel` into canonical competition forms `⅓ shekel 15 grains`).
- **Sumerogram Protection**: Unifying `KÙ.BABBAR` logic uniformly. 
- **Stray Marks**: Cleared stray regex artifacts (`..`, `xx`, `<< >>`) that unfairly penalized `chrF++` matching.

### 6. Inference Strategy: Cross-Model Ensemble MBR Decoding
While `train.py` establishes our robust neural weights, our highest gains were realized in the decoding suite (`inference.ipynb`). Because Akkadian translation is highly fragmented, committing to an absolute "highest probability map" (Standard Greedy/Beam Search) is exceptionally brittle. Instead, we adopted an **Ensemble Minimum Bayes Risk (MBR)** architecture, which essentially crowdsources the correct translation across multiple models and probabilistic pathways.

#### A. Multi-Model Candidate Pooling
Instead of relying on one model, our inference pools structural candidates from two distinctly trained architectures:
- **Model A**: Our custom `train.py` fine-tuned `ByT5-large` model.
- **Model B**: A strong, publicly available community model.
By cross-pollinating both models into the same candidate pool space, we allow diverse dialectic translations to battle for structural dominance.

#### B. Hyper-Diverse Candidate Generation
For every single test instance, we generate up to **34 distinct translation candidates** (capped heavily at a 32 deduplicated MBR pool capacity):
- **Standard Beam Candidates**: Top 6 sequences generated using standard `num_beams=10`.
- **Diverse Beam Candidates**: 5 candidates forced through grouped Beam Search (`num_beam_groups=4`, `diversity_penalty=0.9`) to physically push the search tree down alternate syntactic routes.
- **Multi-Temperature Sampling**: 6 fully stochastic candidates injected by sampling the probability distributions across 3 elevated temperatures (`temps: [0.5, 0.7, 0.9]` $\times$ 2 generations each). 

#### C. Optimized Throughput: Bucket Batching & AMP
With up to 34 heavy `ByT5` generations running sequentially, inference times can timeout Kaggle limits. 
- We grouped inference instances into **6 length-based Buckets** (`BucketBatchSampler`), preventing the Transformer pad-wastage bottleneck.
- Loaded models via `BF16` AMP and executed BetterTransformer (where available) scaling down the raw mathematical footprint.

#### D. MBR Reranking Selection Metrics
Once the massive pool of candidates is generated for a single row, the notebook iterates through the pool and intrinsically evaluates every candidate *against every other candidate*. We score consensus overlap utilizing a custom weighted rubric prioritizing the exact competition goals:
- **chrF++ (weight 55%)**: Heavy bias towards morphological character overlap.
- **BLEU (weight 25%)**: Penalizes completely rogue syntax pathways.
- **Jaccard Index (weight 20%)**: Enforces rigid vocabulary sets.
- **Length Constraint (10%)**: Minor penalty pushing toward optimal lengths.

By selecting the specific candidate that possesses the **highest aggregate consensus score** across the generated pool, we systematically eliminate "grammatical dead ends". We select the most probabilistically secure translation that firmly structurally aligns with the majority of other parallel translations.

---




<!-- Final tracking -->
