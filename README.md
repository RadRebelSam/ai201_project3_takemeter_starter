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
- **Data split:** 70% train (~146 examples), 15% validation (~31 examples), 15% test (~32 examples), stratified by label
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
Each test set example was sent individually to the Groq API with `temperature=0` and `max_tokens=10`. The response was parsed by stripping whitespace, converting to lowercase, and matching against valid label names. A 0.5-second delay between requests prevented rate limiting. Unparseable outputs (responses that didn't match any label) were tracked separately. Retries with exponential backoff handled transient API errors.

## 7. Evaluation Report

### Baseline Results (Zero-Shot Groq LLaMA 3.3 70B)

<!-- FILL AFTER RUNNING NOTEBOOK -->

Overall accuracy: _[fill after running notebook]_

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| analysis | _[fill]_ | _[fill]_ | _[fill]_ |
| hot_take | _[fill]_ | _[fill]_ | _[fill]_ |
| reaction | _[fill]_ | _[fill]_ | _[fill]_ |

Unparseable outputs: _[fill]_

**Common mistakes:** _[fill — describe which labels the baseline confuses most, e.g. "The baseline frequently classified hot_takes as analysis when the post mentioned specific anime titles or character names, suggesting the LLM equated referencing content with structured reasoning."]_

### Fine-Tuned Model Results

Overall accuracy: _[fill after running notebook]_

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| analysis | _[fill]_ | _[fill]_ | _[fill]_ |
| hot_take | _[fill]_ | _[fill]_ | _[fill]_ |
| reaction | _[fill]_ | _[fill]_ | _[fill]_ |

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### Baseline vs. Fine-Tuned Comparison

| Metric | Baseline | Fine-Tuned | Difference |
|---|---:|---:|---:|
| Overall Accuracy | _[fill]_ | _[fill]_ | _[fill]_ |
| analysis F1 | _[fill]_ | _[fill]_ | _[fill]_ |
| hot_take F1 | _[fill]_ | _[fill]_ | _[fill]_ |
| reaction F1 | _[fill]_ | _[fill]_ | _[fill]_ |

_[Fill with 2-3 sentences: Which model performed better overall? Which labels improved most with fine-tuning? Which labels remained difficult? Was fine-tuning worth it given the small dataset?]_

### Error Analysis (3 Wrong Predictions)

#### Error 1
- **Post:** _[fill — copy the text from notebook output]_
- **True label:** _[fill]_
- **Predicted label:** _[fill]_
- **Why the model likely got it wrong:** _[fill — e.g. "The post mentions specific anime titles and character traits, which the model may have associated with analysis. However, it uses them to support a bold claim rather than building a structured argument."]_
- **What this reveals about the label boundary:** _[fill — e.g. "Mentioning evidence doesn't make a post analysis — it's about whether the evidence is developed into a structured argument."]_

#### Error 2
- **Post:** _[fill]_
- **True label:** _[fill]_
- **Predicted label:** _[fill]_
- **Why the model likely got it wrong:** _[fill]_
- **What this reveals about the label boundary:** _[fill]_

#### Error 3
- **Post:** _[fill]_
- **True label:** _[fill]_
- **Predicted label:** _[fill]_
- **Why the model likely got it wrong:** _[fill]_
- **What this reveals about the label boundary:** _[fill]_

### Sample Classifications (3–5 Posts)

| Post (truncated) | Predicted Label | Confidence | Explanation |
|---|---|---:|---|
| _[fill]_ | _[fill]_ | _[fill]_ | _[fill — for at least one correct prediction, explain why it makes sense]_ |
| _[fill]_ | _[fill]_ | _[fill]_ | _[fill]_ |
| _[fill]_ | _[fill]_ | _[fill]_ | _[fill]_ |
| _[fill]_ | _[fill]_ | _[fill]_ | _[fill]_ |
| _[fill]_ | _[fill]_ | _[fill]_ | _[fill]_ |

## 8. Reflection

- **What the model learned:** _[fill after running — e.g. "The model learned to distinguish posts by structural and tonal cues: analysis posts tend to be longer with comparative language ('compared to', 'because', 'which means'), hot_takes use declarative judgment words ('overrated', 'best', 'worst', 'always'), and reactions use emotional language, ALL CAPS, and exclamation marks."]_

- **What I intended it to learn:** I intended the model to learn the difference between structured reasoning (analysis), confident opinions without evidence (hot_take), and emotional responses (reaction) in anime discourse — distinguishing discourse *type* based on argumentative structure, not topic.

- **Where those two differ:** _[fill — e.g. "The model may have partially learned surface-level signals rather than deep structural differences. For example, it might classify any post with ALL CAPS as reaction regardless of content, or any post mentioning multiple anime titles as analysis regardless of whether it builds an argument."]_

- **Whether the model overfit to surface signals:** _[fill — e.g. "Some evidence of surface-level learning: the model likely associates exclamation marks and caps with reaction, specific show names with analysis, and judgment words like 'overrated' with hot_take. These are correlated with the labels but aren't the actual defining features."]_

- **What I would improve with more time:** _[fill — e.g. "Collect more data (500+ examples), include real Reddit posts via the API, add adversarial examples that have surface features of one class but belong to another (e.g. an analysis written in ALL CAPS), and experiment with a larger base model like bert-base-uncased."]_

## 9. Spec Reflection

- **One way `planning.md` helped:** _[fill — e.g. "Writing edge cases and decision rules in planning.md before labeling the dataset forced me to clarify the boundary between hot_take and analysis. Specifically, the rule that 'mentioning a theme without developing it defaults to hot_take' prevented inconsistent labeling across the 209 examples."]_

- **One way implementation diverged from the original plan:** _[fill — e.g. "The original plan targeted 210 examples (70 per label), but the final dataset has 209 examples with a slight imbalance (70/68/71). Some examples were removed during review because they were too similar to other examples or didn't clearly fit any label."]_

- **Why that change was necessary:** _[fill — e.g. "Removing borderline examples improved label consistency, which matters more for a small dataset than hitting an exact target count."]_

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
