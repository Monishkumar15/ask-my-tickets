"""
One-time setup: wires this project's own eval/judge logic into Langfuse
Cloud's hosted eval features (LLM connection, dataset, LLM-as-judge
evaluator, evaluation rule) -- the same architectural pattern found live on
a reference Langfuse project ("Rag Support"), inspected directly via the
API before writing this (see chat history: lf.api.evaluators/evaluation_rules/
datasets/llm_connections.list() against that project).

Unlike that reference project, this one reuses OUR OWN already-validated
pipeline logic instead of a fresh prompt:
  - The LLM-as-judge evaluator's rubric is evals/judge.py's RUBRIC verbatim
    -- the same rubric validated across 4 rounds in FINDINGS_week6.md, not
    a newly invented prompt.
  - The dataset is this project's own EVAL_QUESTIONS + REGRESSION_CASES
    (evals/eval_questions.py) -- the same 20 cases run_eval.py already uses.
  - Rule-based checks (evals/assertions.py) are NOT hosted evaluators here --
    they run as local, free, no-LLM-call SDK evaluators inside
    run_langfuse_experiment.py's run_experiment() call instead, since they
    need the actual retrieved chunks (page numbers, source dedup) that a
    hosted evaluator's {{input}}/{{output}} template variables can't reach
    the same way. run_experiment()'s local evaluators still post real
    Scores to Langfuse automatically (confirmed by reading
    langfuse/_client/client.py's run_experiment implementation -- each
    Evaluation return value becomes a self.create_score(...) call).

Safe to re-run: dataset/dataset-item creation upserts by name/id, and the
evaluator/rule/connection creates are one-off additions, not required to
run more than once per project.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langfuse import Langfuse  # noqa: E402
from langfuse.api.evaluation_commons.types.evaluation_rule_filter import (  # noqa: E402
    EvaluationRuleFilter_Boolean,
    EvaluationRuleFilter_StringOptions,
)
from langfuse.api.evaluation_commons.types.evaluator_output_definition import (  # noqa: E402
    EvaluatorOutputDefinition_Numeric,
)
from langfuse.api.evaluation_commons.types.prompt_variable_mapping_input import (  # noqa: E402
    PromptVariableMappingInput,
)
from langfuse.api.evaluation_rules.types.evaluation_rule_evaluator_assignment_input import (  # noqa: E402
    EvaluationRuleEvaluatorAssignmentInput,
)
from langfuse.api.evaluators.types.create_evaluator_request import CreateEvaluatorRequest_LlmAsJudge  # noqa: E402
from langfuse.api.evaluators.types.evaluator_chat_message import EvaluatorChatMessage  # noqa: E402
from langfuse.api.evaluators.types.evaluator_model_config import EvaluatorModelConfig  # noqa: E402

import config  # noqa: E402
from evals.eval_questions import EVAL_QUESTIONS, REGRESSION_CASES  # noqa: E402
from evals.judge import RUBRIC  # noqa: E402
from evals.trajectory_judge import RUBRIC as TRAJECTORY_RUBRIC  # noqa: E402

DATASET_NAME = "ask_my_tickets_eval_set"
LLM_CONNECTION_PROVIDER = "ask-my-tickets-gemini"
EVALUATOR_NAME = "ticket-reply-judge"

# Week 8: a SEPARATE dataset for agent trajectory runs, not the same
# experiment-root-span filter as the fixed-pipeline judge above -- keeps
# the two hosted judges from cross-firing on each other's traces, the same
# separation the reference project used (rag_regression_tests vs
# w5_eval_cases -- two datasets, two rules, each targeted at one judge).
TRAJECTORY_DATASET_NAME = "ask_my_tickets_agent_trajectory_set"
TRAJECTORY_EVALUATOR_NAME = "agent-trajectory-judge"

# Langfuse's hosted LLM-as-judge convention expects the model's JSON response
# to use "score"/"reason" keys (confirmed live against the reference
# project's own working evaluators -- e.g. reply-quality-judge-v2), unlike
# judge.py's local "verdict"/"reason" -- adapted here, but the rubric BODY
# (what counts as GOOD vs POOR) is judge.py's RUBRIC verbatim, unchanged.
JUDGE_PROMPT = f"""{RUBRIC}
Context (the retrieved source material):
{{{{context}}}}

Customer question: {{{{question}}}}

Assistant's reply: {{{{answer}}}}

Respond with ONLY a JSON object, no other text:
{{"score": 1.0, "reason": "one sentence"}}  (use score 1.0 for GOOD, 0.0 for POOR)
"""


def _strip_suffix(url, suffix):
    return url[: -len(suffix)] if url.endswith(suffix) else url


def setup_llm_connection(lf):
    base_url = _strip_suffix(config.GEMINI_API_URL, "/chat/completions")
    conn = lf.api.llm_connections.upsert(
        provider=LLM_CONNECTION_PROVIDER,
        adapter="openai",  # Gemini's OpenAI-compatible endpoint -- same adapter type the reference project used for Groq's OpenAI-compatible endpoint
        secret_key=config.GEMINI_API_KEY,
        base_url=base_url,
        custom_models=[config.GEMINI_MODEL_ID],
        with_default_models=False,
    )
    print(f"[llm_connection] provider={conn.provider} base_url={conn.base_url} models={conn.custom_models}")
    return conn


def setup_dataset(lf):
    existing = {d.name: d for d in lf.api.datasets.list().data}
    if DATASET_NAME in existing:
        dataset = existing[DATASET_NAME]
        print(f"[dataset] already exists: {dataset.id}")
    else:
        dataset = lf.api.datasets.create(
            name=DATASET_NAME,
            description="ask-my-tickets eval set: EVAL_QUESTIONS (15) + REGRESSION_CASES (5) from evals/eval_questions.py, the same cases run_eval.py already uses locally.",
        )
        print(f"[dataset] created: {dataset.id}")

    all_cases = EVAL_QUESTIONS + REGRESSION_CASES
    for case in all_cases:
        lf.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=f"case-{case['id']}",  # stable id -> re-running this script upserts, doesn't duplicate
            input=case,
            expected_output=None,
            metadata={"problem_type": case.get("problem_type")},
        )
    lf.flush()
    print(f"[dataset] upserted {len(all_cases)} items")
    return dataset


def setup_evaluator(lf):
    existing = {e.name: e for e in lf.api.evaluators.list().data}
    if EVALUATOR_NAME in existing:
        print(f"[evaluator] already exists: {existing[EVALUATOR_NAME].id}")
        return existing[EVALUATOR_NAME]

    evaluator = lf.api.evaluators.create(
        request=CreateEvaluatorRequest_LlmAsJudge(
            name=EVALUATOR_NAME,
            description="Hosted version of evals/judge.py's validated GOOD/POOR rubric (FINDINGS_week6.md, 4 validation rounds) -- scores 1.0 for GOOD, 0.0 for POOR.",
            prompt=[EvaluatorChatMessage(role="user", content=JUDGE_PROMPT)],
            variable_mapping=[
                PromptVariableMappingInput(variable="question", source="input", json_path="$.question"),
                PromptVariableMappingInput(variable="context", source="output", json_path="$.context"),
                PromptVariableMappingInput(variable="answer", source="output", json_path="$.answer"),
            ],
            model_config_=EvaluatorModelConfig(provider=LLM_CONNECTION_PROVIDER, model=config.GEMINI_MODEL_ID),
            output_definition=EvaluatorOutputDefinition_Numeric(
                min_value=0.0,
                max_value=1.0,
                score_value_instructions="1.0 if the rubric's GOOD criteria are met, 0.0 if POOR.",
                score_reasoning_instructions="One sentence citing the context, matching judge.py's own reason style.",
            ),
        )
    )
    print(f"[evaluator] created: {evaluator.id}")
    return evaluator


def setup_evaluation_rule(lf, dataset, evaluator, rule_name=None):
    rule_name = rule_name or f"Experiment evaluators for {dataset.name}"
    existing = {r.name: r for r in lf.api.evaluation_rules.list().data}
    if rule_name in existing:
        print(f"[evaluation_rule] already exists: {existing[rule_name].id}")
        return existing[rule_name]

    rule = lf.api.evaluation_rules.create(
        name=rule_name,
        enabled=True,
        sampling=1.0,
        filter=[
            EvaluationRuleFilter_StringOptions(column="datasetId", operator="any of", value=[dataset.id]),
            EvaluationRuleFilter_Boolean(column="isExperimentItemRootSpan", operator="=", value=True),
        ],
        evaluator_assignments=[EvaluationRuleEvaluatorAssignmentInput(evaluator_id=evaluator.id)],
    )
    print(f"[evaluation_rule] created: {rule.id}")
    return rule


# Week 8's trajectory judge (evals/trajectory_judge.py) grades the agent's
# step-by-step reasoning trail, not the final answer -- adapted here the
# same way judge.py's RUBRIC was: same rubric body, "score"/"reason" JSON
# keys instead of "verdict"/"reason" to match Langfuse's hosted parsing
# convention.
TRAJECTORY_JUDGE_PROMPT = f"""{TRAJECTORY_RUBRIC}
Customer question: {{{{question}}}}

Agent's step-by-step trail:
{{{{trajectory_text}}}}

Agent's final answer: {{{{answer}}}}

Respond with ONLY a JSON object, no other text:
{{"score": 1.0, "reason": "one sentence pointing at the specific step, if any, where the reasoning broke down"}}  (use score 1.0 for GOOD, 0.0 for POOR)
"""


def setup_trajectory_dataset(lf):
    existing = {d.name: d for d in lf.api.datasets.list().data}
    if TRAJECTORY_DATASET_NAME in existing:
        dataset = existing[TRAJECTORY_DATASET_NAME]
        print(f"[trajectory_dataset] already exists: {dataset.id}")
    else:
        dataset = lf.api.datasets.create(
            name=TRAJECTORY_DATASET_NAME,
            description="ask-my-tickets agent trajectory eval set: same 20 cases as ask_my_tickets_eval_set, run through agent.py's run_agent() instead of the fixed pipeline (Week 8).",
        )
        print(f"[trajectory_dataset] created: {dataset.id}")

    all_cases = EVAL_QUESTIONS + REGRESSION_CASES
    for case in all_cases:
        lf.create_dataset_item(
            dataset_name=TRAJECTORY_DATASET_NAME,
            id=f"traj-case-{case['id']}",
            input=case,
            expected_output=None,
            metadata={"problem_type": case.get("problem_type")},
        )
    lf.flush()
    print(f"[trajectory_dataset] upserted {len(all_cases)} items")
    return dataset


def setup_trajectory_evaluator(lf):
    existing = {e.name: e for e in lf.api.evaluators.list().data}
    if TRAJECTORY_EVALUATOR_NAME in existing:
        print(f"[trajectory_evaluator] already exists: {existing[TRAJECTORY_EVALUATOR_NAME].id}")
        return existing[TRAJECTORY_EVALUATOR_NAME]

    evaluator = lf.api.evaluators.create(
        request=CreateEvaluatorRequest_LlmAsJudge(
            name=TRAJECTORY_EVALUATOR_NAME,
            description="Hosted version of evals/trajectory_judge.py's validated GOOD/POOR rubric -- grades the agent's reasoning PATH, not just its final answer.",
            prompt=[EvaluatorChatMessage(role="user", content=TRAJECTORY_JUDGE_PROMPT)],
            variable_mapping=[
                PromptVariableMappingInput(variable="question", source="input", json_path="$.question"),
                PromptVariableMappingInput(variable="trajectory_text", source="output", json_path="$.trajectory_text"),
                PromptVariableMappingInput(variable="answer", source="output", json_path="$.answer"),
            ],
            model_config_=EvaluatorModelConfig(provider=LLM_CONNECTION_PROVIDER, model=config.GEMINI_MODEL_ID),
            output_definition=EvaluatorOutputDefinition_Numeric(
                min_value=0.0,
                max_value=1.0,
                score_value_instructions="1.0 if the reasoning path is GOOD (sound, evidence-grounded), 0.0 if POOR.",
                score_reasoning_instructions="One sentence pointing at the specific step where reasoning broke down, if any.",
            ),
        )
    )
    print(f"[trajectory_evaluator] created: {evaluator.id}")
    return evaluator


def main():
    lf = Langfuse()
    conn = setup_llm_connection(lf)
    dataset = setup_dataset(lf)
    evaluator = setup_evaluator(lf)
    setup_evaluation_rule(lf, dataset, evaluator)

    traj_dataset = setup_trajectory_dataset(lf)
    traj_evaluator = setup_trajectory_evaluator(lf)
    setup_evaluation_rule(lf, traj_dataset, traj_evaluator)

    print("\nSetup complete. Run evals/run_langfuse_experiment.py (fixed pipeline) "
          "or evals/run_langfuse_agent_experiment.py (agent trajectory) next.")


if __name__ == "__main__":
    main()
