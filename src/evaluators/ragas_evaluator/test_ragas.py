from datasets import Dataset
from ragas.metrics import answer_correctness
from ragas import evaluate
from langchain_aws import ChatBedrock
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import embedding_factory

# Set up Bedrock LLM
llm_bedrock = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-west-2"
)
llm = LangchainLLMWrapper(llm_bedrock)

# Set up Bedrock embeddings using LiteLLM provider
# Amazon Titan embedding model via Bedrock
embeddings = embedding_factory(
    "litellm",
    model="bedrock/cohere.embed-english-v3",
)

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}
dataset = Dataset.from_dict(data_samples)

# Configure the metric to use your custom LLM and embeddings
answer_correctness.llm = llm
answer_correctness.embeddings = embeddings

# Run evaluation with explicit embeddings parameter
score = evaluate(dataset, metrics=[
                 answer_correctness], llm=llm, embeddings=embeddings)

df_scores = score.to_pandas()
print(df_scores)
