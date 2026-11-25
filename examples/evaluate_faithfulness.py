from llm_evaluator.judges.ollama import OllamaJudge
from llm_evaluator.metrics.faithfulness import FaithfulnessEvaluator


judge = OllamaJudge(model="llama3.2")

evaluator = FaithfulnessEvaluator(judge=judge)

result = evaluator.evaluate(
    question="Como o usuário pode redefinir sua senha?",
    context=(
        "O sistema permite redefinir a senha por meio de um link "
        "enviado ao e-mail cadastrado. O link é válido por 24 horas."
    ),
    answer=(
        "O usuário pode redefinir sua senha por um link enviado "
        "ao e-mail cadastrado, válido por 24 horas."
    ),
)

print("Evaluation:", result.evaluator)
print("Faithfulness:", result.metrics["faithfulness"])
print("Reasoning:", result.details["reasoning"])

result.save_json("results/faithfulness_ollama.json")
