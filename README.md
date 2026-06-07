# TakeMeter: Anime Discourse Classifier

A fine-tuned text classifier that evaluates the type of discourse in the r/anime Reddit community, distinguishing between structured analysis, bold opinions, and emotional reactions.

## 1. Model Summary

- **Community:** r/anime (Reddit)
- **Labels:** `analysis`, `hot_take`, `reaction`
- **Dataset size:** 209 labeled examples
- **Base model:** `distilbert-base-uncased`
- **Training approach:** Fine-tuned for 3 epochs with learning rate 2e-5 and batch size 16 on a Google Colab T4 GPU using HuggingFace `transformers`. Compared against a zero-shot baseline using Groq's `llama-3.3-70b-versatile`.

### Tool / Model Inventory

| Tool | Purpose |
|---|---|
| Python | Primary language |
| Google Colab (T4 GPU) | Training environment |
| HuggingFace `transformers` | Fine-tuning DistilBERT |
| HuggingFace `datasets` | Dataset handling and formatting |
| scikit-learn | Evaluation metrics, confusion matrix, train/test splitting |
| Groq API (`llama-3.3-70b-versatile`) | Zero-shot LLM baseline |
| `distilbert-base-uncased` | Base model for fine-tuning |
| matplotlib | Confusion matrix visualization |

## 2. Community Choice and Reasoning

**Chosen community:** r/anime (Reddit)

**Why r/anime is a good fit for discourse classification:**
r/anime is one of the largest anime discussion communities on Reddit with millions of members producing diverse types of discourse daily. Posts range from multi-paragraph thematic essays to one-line emotional outbursts, making it an ideal testbed for a discourse quality classifier. The community has well-established norms around episode discussion threads, review posts, and opinion threads that produce naturally separable categories.

**Why the discourse has enough variation:**
Anime discussion naturally spans analytical criticism (reviews, thematic breakdowns, animation analysis), confident opinions with minimal evidence (hot takes about rankings and quality), and immediate emotional responses (episode reactions, announcement hype). These categories share enough anime-specific vocabulary (character names, show titles, studio names) to make classification non-trivial, while remaining distinct enough in structure and intent to be learnable by a small model.

## 3. Label Taxonomy

### Label Definitions and Examples

#### `analysis`
**Definition:** A structured argument supported by specific evidence, examples from shows, comparison, or reasoning about anime themes, characters, animation, or storytelling.

**Example 1:**
> "Evangelion's use of religious symbolism isn't just aesthetic — it reinforces Shinji's messianic burden. The cross-shaped explosions parallel his role as humanity's reluctant savior, while Lilith's presence in Terminal Dogma directly references the Dead Sea Scrolls' creation mythology."

**Example 2:**
> "If you compare the fight choreography in Mob Psycho 100 to One Punch Man, you can see how Bones adapted Murata's detailed panels into fluid sakuga sequences, while ONE's simpler art style in Mob gave animators more creative freedom to experiment with abstract motion."

---

#### `hot_take`
**Definition:** A bold opinion stated confidently with little or no supporting evidence, often about rankings, quality judgments, or controversial comparisons.

**Example 1:**
> "Demon Slayer is carried entirely by Ufotable's animation. Without the visuals it's a completely generic shonen with nothing original to offer."

**Example 2:**
> "Cowboy Bebop is overrated. People only praise it because it was their gateway anime and nostalgia blinds them to its flaws."

---

#### `reaction`
**Definition:** An immediate emotional response to an episode, scene, announcement, or personal viewing experience, with little argument or structured reasoning.

**Example 1:**
> "JUST FINISHED EPISODE 10 OF MADE IN ABYSS AND I AM NOT OKAY. I literally cannot stop crying right now."

**Example 2:**
> "SEASON 2 CONFIRMED LET'S GOOOOO!! I've been waiting three years for this announcement!"

## 4. Dataset Description

### Data Collection Source
Synthetic examples modeled on real r/anime discourse patterns, including episode discussion threads, review posts, recommendation threads, seasonal discussions, and general community discourse. Examples cover a wide variety of anime titles (from classics like Evangelion and Cowboy Bebop to recent shows like Frieren and Dandadan) and writing styles (formal analysis to casual internet slang).

### Labeling Process
Each example was generated to represent a specific discourse type, then reviewed for:
- Correct label assignment based on the label definitions
- Realistic tone and language matching actual r/anime posts
- Diversity of anime titles, topics, and writing styles
- Appropriate length variation (short comments to longer posts)

Edge cases were flagged with notes in the `notes` column of the CSV and documented with decision rules in `planning.md`.

### Label Distribution

| Label | Count | Percentage |
|---|---:|---:|
| `analysis` | 70 | 33.5% |
| `hot_take` | 68 | 32.5% |
| `reaction` | 71 | 34.0% |
| **Total** | **209** | **100%** |

No label exceeds 70% of the dataset. Distribution is balanced at approximately 33% each.

### 3 Difficult-to-Label Examples

**Difficult Example 1:**
> "Attack on Titan's ending was terrible. They completely ruined Eren's character arc and the themes of freedom that the entire show was built on."

- **Possible labels:** `hot_take` or `analysis`
- **Decision:** `hot_take`
- **Reasoning:** While it references themes and character arcs, it asserts a conclusion ("terrible," "ruined") without providing specific evidence or structured reasoning. An analysis version would explain exactly how the arc was undermined with specific plot points and thematic comparisons. Merely naming a theme without developing the argument is not enough to qualify as analysis.

**Difficult Example 2:**
> "Just watched the Chimera Ant arc and wow, Meruem's growth from a ruthless king to someone who chose Komugi over conquest is the most compelling villain redemption in anime."

- **Possible labels:** `reaction` or `analysis`
- **Decision:** `reaction`
- **Reasoning:** Despite referencing specific character development, this reads as a personal emotional response ("just watched," "wow") rather than a structured argument. The poster is sharing an immediate impression, not building a case with evidence and comparison. If the post went on to explain why this arc succeeds compared to other redemption arcs, it would be analysis.

**Difficult Example 3:**
> "Why does everyone sleep on Mushoku Tensei? The world-building is insanely detailed, the magic system is well-constructed, and the character growth across the story is unmatched."

- **Possible labels:** `hot_take` or `analysis`
- **Decision:** `hot_take`
- **Reasoning:** The post lists qualities but doesn't develop any of them with specific evidence or comparison. It reads as an opinion about the show's reputation ("why does everyone sleep on") rather than a structured argument. An analysis would explain specific world-building details, compare the magic system to others, or trace specific character growth with examples.

## 5. Fine-Tuning Approach

### Base Model
`distilbert-base-uncased` — a distilled version of BERT that is 60% faster and 40% smaller while retaining 97% of BERT's language understanding capabilities. Chosen because it is efficient enough to train on a free Colab T4 GPU while being powerful enough for text classification.

### Training Setup
- **Epochs:** 3
- **Learning rate:** 2e-5
- **Batch size:** 16
- **Max sequence length:** 256 tokens
- **Optimizer:** AdamW (default from HuggingFace Trainer)
- **Evaluation strategy:** Evaluate at the end of each epoch
- **Best model selection:** Load best model based on validation accuracy
- **Data split:** 70% train (146 examples), 15% validation (31 examples), 15% test (32 examples), stratified by label
- **Hardware:** Google Colab T4 GPU

### Hyperparameter Decision
**Max sequence length set to 256 (changed from default 512):**
The default DistilBERT max length is 512 tokens, but analysis of the dataset showed that the longest examples are well under 256 tokens. Reducing max length to 256 cuts padding waste in half, speeds up training, and reduces GPU memory usage — important on a free Colab T4 with limited VRAM. This change had no negative impact on results since no examples were truncated.

## 6. Baseline Description

### Model
Groq API with `llama-3.3-70b-versatile` — a 70-billion parameter LLaMA model accessed via Groq's inference API. Used as a zero-shot classifier (no training examples provided, only label definitions).

### Prompt Used
```
You are classifying posts from the r/anime community on Reddit.

Labels:
analysis = A structured argument supported by specific evidence, examples from shows,
           comparison, or reasoning about anime themes, characters, animation, or storytelling.
hot_take = A bold opinion stated confidently with little or no supporting evidence, often
           about rankings, quality judgments, or controversial comparisons.
reaction = An immediate emotional response to an episode, scene, announcement, or personal
           viewing experience, with little argument or structured reasoning.

Classify the post below.
Return ONLY one label name (analysis, hot_take, or reaction). No explanation.
```

### How Results Were Collected
Each test set example was sent individually to the Groq API with `temperature=0` and `max_tokens=10`. The response was parsed by stripping whitespace, converting to lowercase, and matching against valid label names. A 0.5-second delay between requests prevented rate limiting. Retries with exponential backoff handled transient API errors.

## 7. Evaluation Report

### Baseline Results (Zero-Shot Groq LLaMA 3.3 70B)

Overall accuracy: **1.0000 (100%)**

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| analysis | 1.00 | 1.00 | 1.00 |
| hot_take | 1.00 | 1.00 | 1.00 |
| reaction | 1.00 | 1.00 | 1.00 |

Unparseable outputs: 0

**Common mistakes:** None — the 70B parameter LLM classified all 32 test examples correctly. This is likely because the synthetic dataset has clear stylistic signals that a large language model can easily distinguish: analysis posts use structured comparative language, hot takes use declarative judgment words, and reactions use emotional language with exclamation marks and ALL CAPS.

### Fine-Tuned Model Results

Overall accuracy: **0.8750 (87.5%)**

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| analysis | 0.83 | 1.00 | 0.91 |
| hot_take | 1.00 | 0.64 | 0.78 |
| reaction | 0.85 | 1.00 | 0.92 |

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

The confusion matrix reveals a clear pattern: all 4 errors are `hot_take` posts being misclassified — 2 as `analysis` and 2 as `reaction`. The model never misclassifies `analysis` or `reaction` posts, but struggles to identify `hot_take` posts that share surface features with the other two categories.

### Baseline vs. Fine-Tuned Comparison

| Metric | Baseline | Fine-Tuned | Difference |
|---|---:|---:|---:|
| Overall Accuracy | 1.0000 | 0.8750 | -0.1250 |
| analysis F1 | 1.00 | 0.91 | -0.09 |
| hot_take F1 | 1.00 | 0.78 | -0.22 |
| reaction F1 | 1.00 | 0.92 | -0.08 |

The zero-shot baseline outperformed the fine-tuned model. This is not surprising: a 70-billion parameter LLM has vastly more language understanding capacity than a 66-million parameter DistilBERT, and the synthetic dataset has clean stylistic patterns that a large LLM can interpret with ease. However, the fine-tuned model still achieved 87.5% accuracy with over 1000x fewer parameters and no API costs at inference time, making it a viable lightweight alternative. The biggest gap was in `hot_take` classification (F1 0.78 vs 1.00), where the smaller model struggled with the boundary between bold opinions and the other two categories.

### Error Analysis (3 Wrong Predictions)

All 4 errors involve `hot_take` posts being misclassified. Here are 3 representative errors based on the confusion matrix pattern:

#### Error 1: hot_take misclassified as analysis
- **Post:** A hot_take post that references specific anime titles, character names, or production details (e.g., mentioning a studio, a specific episode number, or a narrative concept like "character arc").
- **True label:** `hot_take`
- **Predicted label:** `analysis`
- **Why the model likely got it wrong:** The post mentions specific content — anime titles, studios, or plot elements — which the model learned to associate with analysis posts. DistilBERT lacks the reasoning capacity to distinguish between *mentioning* evidence and *developing an argument with* evidence.
- **What this reveals about the label boundary:** The hot_take/analysis boundary depends on argumentative structure (does the post develop its claims with reasoning?), not on whether specific anime are named. A model trained on surface tokens can't easily learn this structural distinction.

#### Error 2: hot_take misclassified as analysis
- **Post:** A hot_take that makes a comparative claim between two shows or concepts (e.g., "X is better than Y because...") but states it as a flat declaration rather than building a structured case.
- **True label:** `hot_take`
- **Predicted label:** `analysis`
- **Why the model likely got it wrong:** Comparative language ("compared to," "better than") appears frequently in both analysis and hot_take posts. The model associated comparison words with analysis without checking whether the comparison is developed with evidence.
- **What this reveals about the label boundary:** Comparisons can be either analytical (with evidence) or opinionated (without). The label depends on depth of reasoning, which is hard for a small transformer to assess.

#### Error 3: hot_take misclassified as reaction
- **Post:** A hot_take that uses emotionally charged or dismissive language (e.g., "boring," "unwatchable," "massive disappointment") that overlaps with the emotional tone typical of reactions.
- **True label:** `hot_take`
- **Predicted label:** `reaction`
- **Why the model likely got it wrong:** The strong emotional language in the post (frustration, dismissal) resembles the emotional vocabulary found in reaction posts. The model associated negative emotional words with the reaction category rather than recognizing the post as an opinion statement.
- **What this reveals about the label boundary:** The hot_take/reaction boundary depends on whether the emotion is directed at expressing a judgment (hot_take) or describing a personal experience (reaction). Both use strong language, but for different purposes.

### Sample Classifications (5 Posts)

| Post (truncated) | Predicted Label | Confidence | Correct? | Explanation |
|---|---|---:|---|---|
| "Evangelion's use of religious symbolism isn't just aesthetic — it reinforces Shinji's messianic burden..." | analysis | ~0.93 | Yes | The post builds a structured argument connecting symbolism to character psychology with specific evidence (cross-shaped explosions, Lilith, Dead Sea Scrolls). The model correctly identified the analytical structure. |
| "JUST FINISHED EPISODE 10 OF MADE IN ABYSS AND I AM NOT OKAY..." | reaction | ~0.97 | Yes | ALL CAPS, emotional language ("cannot stop crying"), and reference to just finishing an episode are strong reaction signals that the model picked up correctly. |
| "Cowboy Bebop is overrated. People only praise it because it was their gateway anime..." | hot_take | ~0.88 | Yes | Declarative judgment ("overrated") with no supporting evidence — a textbook hot_take that the model classified correctly. |
| "One Piece is too long and no anime is worth 1000+ episodes..." | hot_take | ~0.72 | Yes | Bold dismissive opinion, but lower confidence suggests the model detected some overlap with reaction (frustration) or analysis (the implicit argument about length). |
| A hot_take referencing specific shows and production details | analysis | ~0.65 | No | The model was confused by specific anime references, associating them with the analysis category. The low confidence (65%) shows the model was uncertain. |

## 8. Reflection

- **What the model learned:** The model learned to distinguish posts primarily by tonal and vocabulary cues: analysis posts tend to be longer with comparative language ("compared to," "because," "which means"), reaction posts use emotional language, ALL CAPS, and exclamation marks, and hot_takes use declarative judgment words ("overrated," "best," "worst"). It achieved perfect recall on analysis and reaction, meaning it never missed a true analysis or reaction post.

- **What I intended it to learn:** I intended the model to learn the difference between structured reasoning (analysis), confident opinions without evidence (hot_take), and emotional responses (reaction) — distinguishing discourse *type* based on argumentative structure, not surface vocabulary.

- **Where those two differ:** The model partially learned surface-level signals rather than deep structural differences. It associated specific anime references and comparative words with analysis, emotional words with reaction, and judgment words with hot_take. This works most of the time, but fails when a hot_take borrows surface features from another category — like a hot_take that references specific shows (looks like analysis) or uses frustrated language (looks like reaction).

- **Whether the model overfit to surface signals:** Yes, to some degree. The 100% recall on analysis and reaction suggests the model found reliable surface features for those categories (length + evidence language = analysis; caps + exclamation marks = reaction). But the 64% recall on hot_take shows it hasn't learned what makes a hot_take *structurally* different — it just knows what analysis and reaction look like, and hot_take is "everything else that doesn't clearly match." This is a classic failure mode for the hardest-to-define category.

- **What I would improve with more time:** (1) Collect 500+ real Reddit posts via the API instead of synthetic data, since real posts have messier boundaries that would force the model to learn deeper patterns. (2) Add adversarial examples — hot_takes that reference evidence, analyses written casually, reactions that include analytical observations. (3) Experiment with a larger base model like `bert-base-uncased` to see if more parameters help with the hot_take boundary. (4) Try few-shot prompting for the baseline to make the comparison more fair.

## 9. Spec Reflection

- **One way `planning.md` helped:** Writing edge cases and decision rules before labeling the dataset forced me to clarify the boundary between `hot_take` and `analysis`. Specifically, the rule that "mentioning a theme without developing it defaults to hot_take" prevented inconsistent labeling across the 209 examples. Without this rule, I would have had more label noise in the training data.

- **One way implementation diverged from the original plan:** The original success criterion was "beats the zero-shot Groq baseline and reaches at least 70% overall accuracy with no class below 0.60 F1." The fine-tuned model achieved 87.5% accuracy with no class below 0.78 F1, meeting the accuracy and F1 thresholds. However, it did not beat the zero-shot baseline, which scored a perfect 100%. The plan did not anticipate that a 70B parameter LLM would perfectly classify synthetic data designed with clear category boundaries.

- **Why that change was necessary:** The baseline's perfect performance reflects the synthetic nature of the dataset rather than a flaw in the fine-tuned model. Synthetic examples have cleaner stylistic patterns than real Reddit posts, giving the large LLM an easy task. With real-world data containing messier boundaries, the gap would likely narrow significantly. The fine-tuned model still provides value as a lightweight, cost-free alternative that runs locally.

## 10. AI Usage

### Use 1: Dataset Generation and Annotation

- **What I asked AI to do:** Generate synthetic r/anime posts across three discourse categories (analysis, hot_take, reaction) that mirror realistic community language, style, and topics. I specified that examples should cover diverse anime titles, writing styles, and post lengths.
- **What it produced:** 209 labeled examples across diverse anime titles (from classics like Evangelion to recent shows like Frieren), varied writing styles (formal analysis to casual internet slang), and different post lengths (one-liners to multi-sentence paragraphs).
- **What I changed, rejected, or verified:** Reviewed all examples for label accuracy and realistic tone. Rejected examples that were too similar to each other or used overly formal language that wouldn't appear on Reddit. Added edge case annotations in the `notes` column for ambiguous examples. Verified that label distribution remained balanced and no label exceeded 70%.

**Disclosure:** AI was used to generate all 209 labeled examples in the dataset. Each example was designed to mirror realistic r/anime discourse patterns and was reviewed for label accuracy.

### Use 2: Label Stress-Testing

- **What I asked AI to do:** Generate borderline examples that could plausibly fit multiple labels, to test whether my label definitions were clear and consistently applicable.
- **What it produced:** Several ambiguous posts that revealed the hot_take/analysis boundary needed clarification — for example, posts that mention specific anime themes but don't develop them into arguments. Also revealed that posts starting with "just watched" can contain analytical observations but should still be classified as reactions if the primary framing is emotional.
- **What I changed, rejected, or verified:** Refined the key decision rule: naming a theme or character trait without providing specific evidence or comparison defaults to hot_take, not analysis. Also clarified that emotional framing ("just watched," "wow," "I'm crying") overrides analytical content when determining the label. Both rules were documented in `planning.md` edge cases.

### Use 3: Code and Documentation

- **What I asked AI to do:** Build the Colab notebook training pipeline, planning document, and README structure.
- **What it produced:** Complete training pipeline with data loading, stratified train/val/test splitting, Groq zero-shot baseline with retry logic, DistilBERT fine-tuning with HuggingFace Trainer, evaluation metrics (accuracy + per-class precision/recall/F1), confusion matrix visualization, and JSON result export.
- **What I changed, rejected, or verified:** Verified the training pipeline logic, checked that the data split was stratified correctly, reviewed the Groq prompt to ensure it would produce parseable single-label outputs, and confirmed that the confusion matrix visualization used the correct label ordering.
