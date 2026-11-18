# Movie Genre Classification

**MSDS 565 Unit 3 · Team3**

This project classifies movie plot synopses as Comedy, Horror and Action. Each genre is an independent binary target, so a movie can receive multiple labels. The project follows CRISP-DM through exploratory analysis, text representation, class-weighted logistic regression, LLM knowledge distillation and model interpretation.

The main question is how held-out performance changes when a small local LLM supplies the training labels instead of IMDB. All models use **plot_synopsis only**. Movie titles, year, runtime, ratings and vote counts are retained for analysis or audit, rather than used as predictive features.

## Results

The supplied local run produced the following held-out test macro-F1 scores. Every row averages the same three genre-specific scores over the same 2,330 test movies.

| Rank | Training labels | Representation | Comedy | Horror | Action | Mean |
|---:|---|---|---:|---:|---:|---:|
| 1 | IMDB | tfidf | 0.7162 | 0.8494 | 0.7789 | 0.7815 |
| 2 | IMDB | emb-minilm | 0.6927 | 0.7565 | 0.7700 | 0.7398 |
| 3 | LLM | emb-minilm | 0.5730 | 0.5259 | 0.4616 | 0.5202 |
| 4 | LLM | tfidf | 0.6604 | 0.3374 | 0.1962 | 0.3980 |

The teacher's positive-class F1 on the **299-movie annotation sample** is Comedy 0.6481, Horror 0.6626 and Action 0.6445; mean 0.6518. These annotation-sample scores are a separate reference and are not ranked against held-out test scores.

IMDB-trained TF-IDF has the strongest mean test macro-F1 in this run. LLM students use 299 enriched training movies, compared with 9,365 movies for the IMDB models. Sample size, sample composition and label source change together, so the gap does not isolate annotation noise.

## Repository structure

The repository root contains exactly these four entries:

| Location | Contents |
|---|---|
| `README.md` | Project overview, methods, results and reproduction instructions |
| `environment.yml` | Actual export from the Conda environment used for this project |
| `models/` | Six logistic regression classifiers and the fitted preprocessing artifacts needed by Notebook 3 |
| `notebooks/` | The three numbered Team3 notebooks with saved outputs |

## Notebook sequence

### 1 EDA

`notebooks/P3_MovieGenre-1_EDA_Team3.ipynb`

Explores release year, runtime, rating and votes; compares ratings by genre; audits the 100-vote minimum; shows every genre's counts, rates and correlations; and studies text lengths and vocabulary. It writes its configuration and filter audit into the external work folder. It does not fit vectorizers or models.

### 2 Vectorization Modeling and LLM Distillation

`notebooks/P3_MovieGenre-2_Modeling_Team3.ipynb`

Fits TF-IDF and scalers on training text, caches MiniLM embeddings outside the repository, and trains logistic regression for every representation/genre pair. Each search compares five class-weight settings with five stratified folds and macro-F1 scoring. The notebook shows classification reports and confusion matrices, runs the schema-constrained teacher, measures annotation accuracy, trains six pseudo-label students, compares all configurations and exports the required IMDB artifacts.

### 3 Interpretability

`notebooks/P3_MovieGenre-3_xAI_Team3.ipynb`

Reloads the saved TF-IDF artifacts, plots ten positive and ten negative word coefficients per genre, and computes full-test Comedy SHAP explanations. It includes mean-absolute-SHAP and beeswarm plots, ranking comparisons and two mechanically selected waterfall cases. This package refreshes those interpretation outputs against the included latest models; the three TF-IDF scores were reconciled against the supplied local metrics.

## Dataset and split

Use the instructor-supplied `mpst_imdb_movies.csv` from the class Datasets folder. It contains 11,824 joined movies. A minimum of **100 votes** retains **11,695** movies: **9,365 train + validation** and **2,330 test**. The original split is preserved, and Comedy, Horror and Action each exceed the required 5% positive rate in both partitions.

The source links are background information; the supplied joined CSV is the project input:

- [MPST Movie Plot Synopses with Tags](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags)
- [IMDB Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/)

## Text representations and classifiers

- **TF-IDF:** English stopwords, unigrams, `min_df=3`, `max_df=0.98`, sublinear term frequency and at most 8,000 terms. Its scaler uses `with_mean=False` to retain sparse matrices.
- **Sentence embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, **384 dimensions**, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Native input length is 256 wordpieces; longer synopses use the prefix. The matrix and cache identity live outside GitHub.
- **Classifier:** logistic regression for all experiments. The class-weight grid is `None`, `balanced`, and positive-class weights 0.5, 2 and 4 with negative weight 1. Five-fold stratified CV uses `scoring='f1_macro'`. There is no SMOTE or other resampling.
- **Scaling:** fitted on training inputs only, then reused for test data and the pseudo-label sample.

## Ollama teacher

The teacher is **qwen2.5:3b**, using the Ollama Python SDK. The supplied local record reports Ollama **0.40.16**, SDK **0.6.2**, and model digest `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`.

For each genre, 100 positive training examples are sampled with seed 42. Their deduplicated union contains **299 movies**. The prompt sends a single plot and the three genre names per movie. It excludes the title and true labels. A JSON schema requires exactly three boolean fields and forbids extra properties. Synopses are capped at the first 12,000 Unicode characters with an 8,192-token context and up to 256 output tokens; truncation is audited.

The supplied teacher/student stages are **completed**. Validated responses, `pseudo_labeled.csv`, metadata and generated tables remain in the external `MovieGenre_work` folder. Matching checkpoint responses can be reused without calling the teacher again for completed movies.

To reproduce teacher inference, install/start Ollama and pull the selected model:

```cmd
ollama pull qwen2.5:3b
```

For the reported Windows CUDA failure, quit the existing Ollama tray application and start its CPU server in a separate Command Prompt:

```cmd
set "CUDA_VISIBLE_DEVICES=-1"
set "OLLAMA_VULKAN=0"
set "LLAMA_ARG_CACHE_RAM=0"
ollama serve
```

Keep that server window open. The notebook's current request uses `num_gpu=0` and a longer timeout for CPU inference.

## Existing GitHub repository and environment

Use your existing **MSDS565-MovieGenre-Team3** repository. Upload `README.md`, `environment.yml`, `models/` and `notebooks/` directly to its root. Upload the contents of the extracted project folder, so there is no second `MSDS565-MovieGenre-Team3` folder inside the GitHub repository.

Keep your actual exported `environment.yml` if you have already generated it locally. The environment file distributed in this archive is a bootstrap specification based on observed package versions, and must be replaced by your actual Conda export before submission. In Windows Command Prompt, with the repository at the path below, export directly using:

```cmd
"%USERPROFILE%\anaconda3\Scripts\conda.exe" env export --name msds565 --no-builds --file "%USERPROFILE%\P3_GitHub\MSDS565-MovieGenre-Team3\environment.yml"
```

Use the actual repository location if you extracted it elsewhere. This command uses the canonical `environment.yml` filename supported by Conda's environment exporter.

## Run or review the notebooks locally

Open the saved notebooks to review their existing outputs. For execution, extract the separately supplied **MovieGenre_work.zip** beside the repository: the two folders must share a parent. Keep its CSVs, embedding matrix and teacher JSONL outside the repo. The notebook defaults use that sibling folder; `P3_WORK_DIR` and `P3_DATA_PATH` can select another external location.

Launch Jupyter in your actual course environment, select `Python (msds565)`, and execute **1 → 2 → 3** when changing inputs or dependencies. Save outputs. The retained notebook execution counts have gaps from interactive use, which have been preserved rather than relabeled. Their code cells are in execution order and have saved outputs without errors. Remove regenerated `.ipynb_checkpoints` folders before uploading.

## Model artifacts

- `P3_N2_emb-minilm_logreg_action.pkl`
- `P3_N2_emb-minilm_logreg_comedy.pkl`
- `P3_N2_emb-minilm_logreg_horror.pkl`
- `P3_N2_emb-minilm_scaler.pkl`
- `P3_N2_manifest.json`
- `P3_N2_tfidf_logreg_action.pkl`
- `P3_N2_tfidf_logreg_comedy.pkl`
- `P3_N2_tfidf_logreg_horror.pkl`
- `P3_N2_tfidf_scaler.pkl`
- `P3_N2_tfidf_vectorizer.pkl`

The six classifiers use `P3_N2_<rep>_logreg_<genre>.pkl` exactly. Notebook 3 requires the fitted TF-IDF vectorizer and scaler, and verifies its movie order against the manifest. Load only the project's own trusted pickle files, using the environment that produced them.

## Interpretation limits

The supplied text audit flags 121 test movies whose normalized plots also occur in training, plus 169 plot groups with conflicting selected-genre labels affecting 379 movies. These source-quality issues may influence measured scores. The supplied split and labels are retained. MiniLM reads a short prefix, while TF-IDF uses the full text. Teacher and students also differ in input coverage and training sample size. SHAP contributions explain model behavior in log-odds and are not causal claims.

## Course submission

Use the existing team repository **MSDS565-MovieGenre-Team3**. Keep all three exact notebook filenames, required model artifacts, saved outputs and the actual exported environment. Raw data, exported CSVs, cached matrices and checkpoint notebooks do not belong in GitHub. Meeting minutes and the individual peer evaluations are separate graded items and should follow the instructor's submission route.
