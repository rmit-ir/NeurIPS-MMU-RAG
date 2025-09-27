import os
from datasets import Dataset
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ragas.metrics import answer_correctness
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LiteLLMEmbeddings
from ragas.cache import DiskCacheBackend
from ragas.executor import Executor as RagasExecutor
from langchain.globals import set_debug

set_debug(True)

cacher = DiskCacheBackend(cache_dir="/tmp/ragas_cache")


# # Set up Bedrock LLM
llm_mmu = ChatOpenAI(
    base_url="https://mmu-gpu-server-llm-proxy.rankun.org/v1",
    api_key=SecretStr(os.environ.get("MMU_OPENAI_API_KEY", "")),
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0.0,
)

llm_mmu = LangchainLLMWrapper(llm_mmu, cache=cacher)

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
    'ground_truth': ["""To remove unwanted people from a photo in Photoshop, use tools like **Generative Fill**, **Remove Tool**, or **Clone Stamp**—these are designed to erase people and blend the background seamlessly[3]. The process has become much easier in recent Photoshop versions, especially with the introduction of AI-powered features.

**Common methods to remove people:**

- **Generative Fill (Photoshop 2024 and later):**
  1. Select the **Lasso Tool** from the toolbar.
  2. Draw a selection around the person you want to remove, including some background for better blending.
  3. Right-click inside the selection and choose **Generative Fill**.
  4. You can leave the text prompt empty or type something specific if you want.
  5. Click **Generate** and Photoshop's AI will fill in the selected area with new, matching background[3].

- **Clone Stamp or Healing Brush Tools:**
  - Manually paint over people with sampled pixels from nearby areas to blend them out. This is more advanced and requires patience for realistic results.

**Tips for best results:**
- Always work on a duplicate layer or a new transparent layer for non-destructive editing.
- For complex backgrounds, use a combination of Generative Fill and Clone Stamp for touch-ups.
- After removal, zoom in and check edges for any obvious artifacts.

Modern Photoshop (2023/2024+) automates much of the process, especially with Generative Fill—older methods like manual cloning may still be needed for fine details or tricky areas[3]."""]
}
dataset = Dataset.from_dict(data_samples)

# Configure the metric to use your custom LLM and embeddings
answer_correctness.llm = llm_mmu
answer_correctness.embeddings = embeddings


async def main():
    # Run evaluation with explicit embeddings parameter
    executor = evaluate(dataset, metrics=[answer_correctness],
                        return_executor=True)

    if isinstance(executor, RagasExecutor):
        score = await executor.aresults()
        # df_scores = score.to_pandas()
        print(score)

if __name__ == "__main__":
    import asyncio
    # to avoid Event Loop is already closed error
    asyncio.run(main())
