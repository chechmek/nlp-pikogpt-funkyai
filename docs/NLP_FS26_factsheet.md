# PikoGPT Challenge Fact Sheet V1.1

DS-NLP Lab | Spring Semester 2026

## Natural Language Processing with Large Language Models

## The Concept: Startup Simulation

We are simulating an early-stage startup phase. You are the founders; we are the investors.

- **The Challenge:** Go from zero to a fully trained LLM with a chat frontend in 12 weeks.
- **Why?** LLMs are transforming the tech landscape and driving the startup ecosystem. Building them yourself from scratch provides the best understanding.
- **Support:** We provide mentoring and GPU resources for final training runs.

Prof. Handschuh and the TAs act as Lead Investor, making final "investment" decisions (grading).

## 1. Organization

- **Professor:** Prof. Siegfried Handschuh
- **Teaching Assistants:** Götz-Henrik Wiegand and Lorena Raichle
- **Communication:** Microsoft Teams or email

### Group Formation

- Form groups of 4 students. If you cannot find a group, the staff will assist.
- Choose a startup name.
- Plan monthly milestones until the end of the semester.
- Submit your group members, startup name, and milestones today.
- Start with data preprocessing and EDA.

### Group Plan / Phases

- Your startup consists of 4 students.
- After pretraining, for post-training, split into subgroups of 2 students each.
- After post-training, merge results into one tech report and one poster.
- All 4 people work on one codebase with clear separation during post-training.

## 2. Roadmap

The exercise sessions are your dedicated Startup Lab. After a short "Code Input" of about 15 minutes connecting theory to practice, the rest of the session is reserved for onsite team work and assistance from the TAs.

| Date | Room | Exercise Content | Milestones |
|---|---|---|---|
| 17.02 | Säntis - 1.OG | Motivation and Challenge Intro | Startup Registration |
| 24.02 | Säntis - 1.OG |  |  |
| 03.03 | Hilti Inn. R1 - 1.OG |  |  |
| 10.03 | Winterg. - 1.OG |  |  |
| 17.03 | Hilti Inn. R2 - 1.OG |  |  |
| 24.03 | Rosenb. Rotm. - EG |  |  |
| Semester Break |  |  |  |
| Semester Break |  |  |  |
| 13.04 |  | Slide submission via Canvas |  |
| 14.04 | Vadian Gallus - EG | Midterm Presentation |  |
| 21.04 | Vadian Gallus - EG |  |  |
| 28.04 | Bodensee - 1.OG |  |  |
| 05.05 | C, 61 61-070 |  |  |
| 11.05 |  | Poster submission via Canvas |  |
| 12.05 | Hilti Inn. R2 - 1.OG |  |  |
| 19.05 | Hilti Inn. R1 - 1.OG | Pseudo-Conference Demo and Poster |  |
| 02.06 |  | Project Submission | Code and Report Submission |

## 3. Grading

The final grade is separated into the following percentages:

- **10%** Active participation in lecture and exercise
- **65%** Project
- Repository: code, documentation, README
- Benchmarks: leaderboard results, pre-training evaluations, loss, perplexity
- Tech report
- Project work and participation
- Individual contributions statement via Canvas
- **5%** Midterm presentation
- **20%** Pseudo conference
- Poster
- Demo session
- Conference participation, including poster discussions

## 4. The Project Task: Build an LLM from Scratch

If you have questions or uncertainties, contact the teaching assistants directly via MS Teams or email.

Any changes or deviations from this fact sheet will be announced in lecture or exercise.

### 4.1 PikoGPT Template

You are welcome to use the provided template project:

<https://github.com/unisg-ics-dsnlp/PikoGPT_Template>

### 4.2 Technical Constraints (The Rules)

#### Hard Constraints and Requirements

- **Pretraining Data:** Provided OpenWebText subset
- **Tokenizer:** GPT-2 tokenizer

```python
from transformers import GPT2TokenizerFast
tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
```

- **Architecture:** You are encouraged to design your own architecture within the following constraints:
- Decoder-only, no MoE
- Context size: 1024
- Max decoder layers: 24
- Model size: max 40M parameters
- **Compute Budget:** 2x 24h on 8xV100 or similar
- **Tech Stack:** Python and PyTorch. High-level frameworks for the main training parts of the repository are forbidden. If unsure, ask the TAs.
- **Forbidden:** External checkpoints, knowledge distillation, and similar shortcuts

### 4.3 Evaluation and Benchmarking

Your startup will be evaluated based on three pillars:

1. **General Model Quality**
   - Loss and perplexity on the WikiText-103 test dataset:
     <https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-raw-v1/test>
   - Loss and perplexity on a provided OpenWebText test split:
     <https://drive.switch.ch/index.php/s/6TLGQFEIkAPJ72K>
   - Important: test datasets must not appear in training in any form.

2. **Leaderboard (Post-Training Evaluation)**
   - LAMBADA: next-word prediction in context
   - HellaSwag and WinoGrande: multiple choice / commonsense reasoning
   - OpenBookQA: general knowledge / reasoning-light
   - Additional hidden benchmarks

3. **Systems Engineering**
   - Reproducibility: one-line run script, deterministic seeds/configs, well-defined stages, `uv` package management, etc.
   - Efficiency: implementation quality
   - General code standards: clean code, documentation, and similar

### 4.4 Leaderboard: Implementation Requirements

Your model will be evaluated using an automated benchmark runner. The runner treats your project as a black box and interacts with it only via the command line.

#### Key Idea

- Your model code will not be imported directly.
- The repository will be cloned and evaluated by running a script.
- The model is run via `python main.py --stage inference`.
- Each benchmark example corresponds to one inference call.
- All students are evaluated under identical conditions.

#### Inference Contract (Mandatory)

The inference stage must be deterministic, including seed and `temperature = 0`.

Your repository must support the following command:

```bash
python main.py --stage inference \
  --checkpoint CKPT.pt \
  --prompt "..." \
  --max-tokens N \
  --temperature 0 \
  --device auto \
  --leaderboard \
  --seed 0
```

This is the only interface used for evaluation.

#### Required Behavior

- `--stage inference` loads the model and runs text generation
- `--checkpoint` specifies the trained model checkpoint
- `--prompt` is the input text
- `--max-tokens` limits the number of generated tokens
- `--temperature 0` must perform greedy decoding by argmax
- `--device auto` selects the best device in order `CUDA > MPS > CPU`
- `--seed 0` ensures deterministic behavior

#### Leaderboard Mode

When the `--leaderboard` flag is set:

- Your program must print only the generated text to stdout
- No logging, banners, or explanations are allowed

Correct example output:

```text
B
```

Incorrect output:

```text
Loading checkpoint...
The answer is B
```

Any additional output will cause the example to be marked invalid.

#### Benchmarks

Your model will be evaluated on the following public benchmarks:

| Benchmark | Task Type | Metric |
|---|---|---|
| HellaSwag | Multiple choice (A-D) | Accuracy |
| WinoGrande | Binary choice (A/B) | Accuracy |
| OpenBookQA | Multiple choice (A-D) | Accuracy |
| LAMBADA | Next-word prediction | Accuracy |

Additional hidden benchmarks will also be used.

#### Important Notes

- You can submit code and model checkpoints for evaluation multiple times.
- Do it early enough to not miss an evaluation before the final pseudo conference.
- Leaderboard results are mandatory for the tech report.
- You may structure your code freely; only the inference contract is mandatory.
- If the command fails or the output is malformed, the example is invalid.

### 4.5 Demo Interface: Requirements

To showcase final models and checkpoints, you should implement a simple chat interface. This is part of the pseudo-conference demo session. Different prompts will be used to compare models and showcase behavior.

Important: the chat interface is meant as a short demo and inference frontend. You may use high-level frameworks such as Gradio for this part.

The goal is not to spend much time on interface development. It should just work for demonstration.

Grading: the chat interface is graded only on basic usability and base requirements.

#### Requirements

- Some form of GUI: browser-based, standalone, or terminal UI
- User text input box
- Simple contextual chat where a new prompt uses current conversation context plus user text
- Button for a new conversation that clears context
- Option to load different model checkpoints
