# TakeMeter Planning Document

## 1. Community

**Chosen community:** r/anime (Reddit)

**Why it is a good fit for classification:**
r/anime is one of the largest anime discussion communities on Reddit with millions of members producing diverse types of discourse daily. Posts range from multi-paragraph thematic essays to one-line emotional outbursts, making it an ideal testbed for a discourse quality classifier. The community has well-established norms around episode discussion threads, review posts, and opinion threads that produce naturally separable categories of discourse.

**Why the discourse has enough variation:**
Anime discussion naturally spans analytical criticism (reviews, thematic breakdowns, animation analysis), confident opinions with minimal evidence (hot takes about rankings and quality), and immediate emotional responses (episode reactions, announcement hype). These categories share enough anime-specific vocabulary to make classification non-trivial while remaining distinct enough in structure and intent to be learnable.

## 2. Labels

### Label: analysis
**Definition:** A structured argument supported by specific evidence, examples from shows, comparison, or reasoning about anime themes, characters, animation, or storytelling.

**Clear Example 1:**
"Evangelion's use of religious symbolism isn't just aesthetic — it reinforces Shinji's messianic burden. The cross-shaped explosions parallel his role as humanity's reluctant savior, while Lilith's presence in Terminal Dogma directly references the Dead Sea Scrolls' creation mythology."

**Clear Example 2:**
"If you compare the fight choreography in Mob Psycho 100 to One Punch Man, you can see how Bones adapted Murata's detailed panels into fluid sakuga sequences, while ONE's simpler art style in Mob actually gave animators more creative freedom to experiment with abstract motion."

---

### Label: hot_take
**Definition:** A bold opinion stated confidently with little or no supporting evidence, often about rankings, quality judgments, or controversial comparisons.

**Clear Example 1:**
"Demon Slayer is carried entirely by Ufotable's animation. Without the visuals, it's a completely generic shonen with nothing original to offer."

**Clear Example 2:**
"Cowboy Bebop is overrated. People only praise it because it was their gateway anime and nostalgia blinds them to how episodic and unfocused the middle episodes are."

---

### Label: reaction
**Definition:** An immediate emotional response to an episode, scene, announcement, or personal viewing experience, with little argument or structured reasoning.

**Clear Example 1:**
"JUST FINISHED EPISODE 10 OF MADE IN ABYSS AND I AM NOT OKAY. I literally cannot stop crying. Why would they do this to us??"

**Clear Example 2:**
"SEASON 2 CONFIRMED LET'S GOOOOO!! I've been waiting three years for this announcement. Today is the best day of my life."

## 3. Hard Edge Cases

### Edge Case 1
**Ambiguous example:** "Attack on Titan's ending was terrible. They completely ruined Eren's character arc and the themes of freedom that the entire show was built on."
**Possible labels:** hot_take, analysis
**Final decision:** hot_take
**Decision rule:** While it references themes and character arcs, it asserts a conclusion ("terrible," "ruined") without providing specific evidence or structured reasoning. An analysis version would explain exactly how the arc was undermined with specific plot points and thematic comparisons. Merely naming a theme without developing the argument is not enough to qualify as analysis.

### Edge Case 2
**Ambiguous example:** "Just watched the Chimera Ant arc and wow, Meruem's growth from a ruthless king to someone who chose Komugi over conquest is the most compelling villain redemption in anime."
**Possible labels:** reaction, analysis
**Final decision:** reaction
**Decision rule:** Despite referencing specific character development, this reads as a personal emotional response ("just watched," "wow") rather than a structured argument. The poster is sharing an immediate impression, not building a case with evidence and comparison. If the post went on to explain why this arc succeeds compared to other redemption arcs, it would be analysis.

### Edge Case 3
**Ambiguous example:** "Why does everyone sleep on Mushoku Tensei? The world-building is insanely detailed, the magic system is well-constructed, and the character growth across the story is unmatched."
**Possible labels:** hot_take, analysis
**Final decision:** hot_take
**Decision rule:** The post lists qualities but doesn't develop any of them with specific evidence or comparison. It reads as an opinion about the show's reputation ("why does everyone sleep on") rather than a structured argument. An analysis would explain specific world-building details, compare the magic system to others, or trace specific character growth with examples.

### Edge Case 4
**Ambiguous example:** "Frieren is a 10/10 masterpiece and I'm so emotional after that last episode. The way they handled the exam arc was brilliant."
**Possible labels:** reaction, hot_take
**Final decision:** reaction
**Decision rule:** The emotional language ("so emotional," "10/10") and reference to a recent viewing experience ("that last episode") signal an immediate reaction rather than a considered opinion. Hot takes are typically about reputation or rankings in the abstract, while reactions are tied to a specific viewing moment. The praise of the exam arc is vague enough that it doesn't constitute analysis.

## 4. Data Collection Plan

**Source:** Synthetic examples modeled on real r/anime discourse patterns, including episode discussion threads, review posts, recommendation threads, seasonal discussion, and general community discourse.

**Target count:** 210 examples total
- analysis: ~70 examples
- hot_take: ~70 examples
- reaction: ~70 examples

**What I will do if one label is underrepresented:** Generate additional examples for that category, focusing on underrepresented subtypes. For example, if analysis examples are underrepresented, I will add more character analysis, animation analysis, and thematic comparison posts.

**How I will avoid one label exceeding 70%:** Target equal distribution (~33% per label). After initial creation, count examples per label and rebalance. The balanced three-label design makes it unlikely any single label would reach 70%.

## 5. Evaluation Metrics

- **Accuracy:** Overall percentage of correct predictions across all labels. Provides a single summary number for model performance.
- **Per-class precision:** Of all posts the model predicts as a given label, how many actually belong to that label. Important because mislabeling a reaction as analysis would misrepresent discourse quality.
- **Per-class recall:** Of all posts that truly belong to a label, how many does the model correctly identify. Important for ensuring the model doesn't systematically miss one type of discourse.
- **Per-class F1:** Harmonic mean of precision and recall per label, providing a balanced single metric that penalizes models that trade off precision for recall or vice versa.
- **Confusion matrix:** Visual representation of which labels are being confused with each other, revealing systematic error patterns (e.g., hot_take being confused with analysis).

These metrics are appropriate because this is a multi-class classification task where all three classes matter equally. Accuracy alone could be misleading if the model favors one label, so per-class metrics ensure balanced performance across all discourse types.

## 6. Definition of Success

The classifier is successful if it beats the zero-shot Groq baseline and reaches at least 70% overall accuracy with no class below 0.60 F1.

## 7. AI Tool Plan

**Label stress-testing:** I will give my label definitions to an AI tool and ask it to generate borderline examples that could plausibly fit multiple labels. If I cannot classify them consistently, I will revise the label definitions before annotating the full dataset.

**Annotation assistance:** AI will help generate synthetic examples that mirror real r/anime discourse patterns. All generated examples will be reviewed and edited to ensure they are realistic, varied in style and length, and correctly labeled.

**Failure analysis:** After training, I will use AI to analyze misclassified examples and identify patterns in errors — for example, whether the model confuses hot takes with analysis when opinion posts mention character names, or whether short reactions get misclassified. This will help determine whether labels need refinement or the model needs different training data.
