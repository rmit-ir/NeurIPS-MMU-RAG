from datasets import Dataset
from ragas.metrics import answer_correctness
from ragas import evaluate
from langchain_aws import ChatBedrock
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LiteLLMEmbeddings
from ragas.cache import DiskCacheBackend
from ragas.dataset_schema import EvaluationResult


cacher = DiskCacheBackend(cache_dir="/tmp/ragas_cache")

# Set up Bedrock LLM
llm_bedrock = ChatBedrock(
    # model="arn:aws:bedrock:us-west-2:403959104676:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region="us-west-2",
    # model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    # region_name="us-west-2"
)
llm = LangchainLLMWrapper(llm_bedrock, cache=cacher)

embeddings = LiteLLMEmbeddings(
    model="bedrock/cohere.embed-english-v3",
    cacher=cacher
)

data_samples = {
    'question': ['how to remove unwanted people from a photo in photoshop'],
    'answer': ["""To remove unwanted people from a photo in Photoshop, follow these steps based on the tools and techniques described in the search results:  

1. **Duplicate the Background Layer**: Press `Ctrl + J` (or `Cmd + J` on Mac) to create a backup of the original image, ensuring you don’t alter the original permanently [4].  
2. **Select the Lasso Tool**: Use the Lasso Tool (`L`) to manually draw a selection around the person you want to remove. For smoother edges, use the **Quick Selection Tool** (`W`) or **Object Selection Tool** [2].  
3. **Refine the Selection**:  
   - **Feather the Selection**: Go to `Select > Modify > Feather` and set a value (1–5 pixels) to soften the edges [4].  
   - **Adjust the Selection**: Use the **Subtract from Selection** option if the selection includes unwanted background areas [2].  
4. **Remove the Selected Area**:  
   - **Content-Aware Fill**: Press `Shift + F5` to use Photoshop’s automatic fill feature, which replaces the selected area with surrounding content [2].  
   - **Patch Tool**: Select the Patch Tool (`J`), then drag the selection to an area of the image to replace the person [2].  
   - **Clone Stamp Tool**: For manual refinement, use the Clone Stamp Tool (`S`) to sample and paint over the gap with surrounding pixels [4].  
5. **Clean Up Edges**: Use the **Healing Brush Tool** or **Clone Stamp Tool** to blend the removed area with the background, ensuring seamless results [2].  
6. **Save the Image**: Go to `File > Save As` and choose your desired format (e.g., JPEG, PNG) [2].  

For complex cases, consider using the **Content-Aware Fill** feature (`Edit > Content-Aware Fill`) for automated adjustments [2]. Always test edits on a copy of the image to avoid irreversible changes [4]."""],
    'ground_truth': ["""To remove unwanted people from a photo in Photoshop, use the Remove Tool, Content-Aware Fill, or the Clone Stamp Tool. For the Remove Tool, brush over the unwanted person and Photoshop will attempt to remove them using AI. Content-Aware Fill, accessed via Edit, allows you to select the person and fill the area with surrounding pixels. The Clone Stamp Tool lets you manually sample and replace the area with other parts of the image."""]
}
dataset = Dataset.from_dict(data_samples)

# Configure the metric to use your custom LLM and embeddings
answer_correctness.llm = llm
answer_correctness.embeddings = embeddings


async def main():
    # Run evaluation with explicit embeddings parameter
    score = evaluate(dataset, metrics=[answer_correctness],
                     llm=llm, embeddings=embeddings, return_executor=False)

    if isinstance(score, EvaluationResult):
        df_scores = score.to_pandas()
        print(df_scores)

if __name__ == "__main__":
    import asyncio
    # to avoid Event Loop is already closed error
    asyncio.run(main())
