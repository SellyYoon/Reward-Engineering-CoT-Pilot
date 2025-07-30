# Reward-Engineering-CoT-Pilot

## 1. Project Overview
This project is a pilot experiment designed to validate the effectiveness of a Reward Engineering framework aimed at mitigating common issues in Large Language Models (LLMs), such as Hallucination, Reward Hacking, and Sycophancy.

Through meticulously designed prompts and a real-time feedback system, this framework guides models to generate responses that are more honest, logical, and capable of self-verification.

## 2. Experimental Design
The experiment adopts a 2x2 Factorial Design to measure the independent and interactive effects of two core variables.
- **Variable A: Feedback Mechanism (Scoring Method)**
	- **Post-hoc Scoring:** All problems are solved first and scored collectively afterward (No intermediate feedback).
	- **Real-time Scoring:** Feedback (reward) is provided after each problem, and this feedback is injected as context into the prompt for the subsequent problem (Sliding Window).
- **Variable B: Prompt Type**
	- **Basic Prompt:** Contains standard Chain-of-Thought (CoT) instructions.
	- **Specialized Prompt:** Includes sophisticated guardrails such as role-playing, quality checklists, forbidden phrases, and self-critique instructions.

| Condition | Feedback Mechanism | Prompt Type | Primary Measurement Goal                            |
|-----------|--------------------|-------------|-----------------------------------------------------|
| A         | Post-hoc           | Basic       | Baseline model performance.                         |
| B         | Post-hoc           | Specialized | Effect of prompt enhancements.                      |
| C         | Real-time          | Basic       | Effect of real-time feedback (in-context learning). |
| D         | Real-time          | Specialized | Synergistic effect of prompt and feedback.          |

## 3. Project Structure
```
.
├── analysis/               # Jupyter Notebook for results analysis
│   └── analysis.ipynb
├── configs/                # Configuration files (prompts, models, schemas)
│   ├── prompts.py
│   ├── schemas.py
│   └── settings.py
├── datasets/               # Dataset creation and loading scripts
│   └── dataset.py
├── logs/                   # Directory for experiment result logs (generated)
├── src/                    # Core application logic
│   ├── evaluator.py
│   ├── model_caller.py
│   ├── reward_system.py
│   ├── session_manager.py
│   ├── trial_runner.py
│   └── turn_manager.py
├── .env                    # Environment variables for API keys
├── main.py                 # Main execution script
├── Dockerfile
└── docker-compose.yml
```

## 4. Setup and Execution
### 4.1. Environment Configuration
1. Create a `.env` file in the project root.
2. Populate it with the necessary API keys, referencing `configs/eviroment_settings.md`.
```env
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-..."
GROK_API_KEY="..."
GOOGLE_API_KEY="..."
HF_API_KEY="hf_..."
```

### 4.2. Running with Docker (Recommended)
1. Ensure Docker and Docker Compose are installed.
2. From the project root directory, run the following command:
```bash
docker-compose up --build
```
This command will build the container, install all dependencies, and execute the `main.py` script to run the full suite of experiments as defined in `configs/settings.py`.

### 4.3. Running with a Local Conda Environment
1. Create and activate the Conda environment using the provided file:
```bash
conda env create -f environment.yml
conda activate cot-pilot
```

2. Run the main script:
```bash
python main.py
```

## 5. Analyzing Results
Upon completion of the experiments, the results will be saved as .jsonl files in the logs/ directory.

The analysis/analysis.ipynb Jupyter Notebook can be used to load these log files, perform statistical analysis, and visualize the results to validate the experimental hypotheses.