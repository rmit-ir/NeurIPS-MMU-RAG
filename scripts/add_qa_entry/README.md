# Self-Donated Q&A Entry Tool

A simple and user-friendly script to add question-answer pairs to your self-donated dataset in proper JSONL format.

**This script handles multi-line questions and answers.**
 
Simply copy and paste what you want to donate, and it will take care of the rest.

## 🚀 How to Run

### Running the Script

**The script automatically finds the project root, so you can run it from anywhere within the project:**

```bash
# From project root:
python scripts/add_qa_entry/add_qa_entry.py

# From any subdirectory:
python ../../scripts/add_qa_entry/add_qa_entry.py

# Or using absolute path:
python /full/path/to/scripts/add_qa_entry/add_qa_entry.py
```

1. **Follow the interactive prompts:**
   - Enter your filename (e.g., "shuoqi")
   - Enter your question (multiline supported)
   - Enter your answer (multiline supported)
   - Enter the source
   - Confirm and save


### 📊 Output Format

Each entry is saved as a single line of JSON with these fields:

```json
{
  "id": "2",
  "question": "Your question here...",
  "answer": "Your detailed answer here...",
  "source": "Source of the information"
}
```
### 📁 File Structure

Your Q&A entries will be saved as:
```
data/self-donated-topics/{your_name}_self-donated_qa.jsonl
```

For example, if you enter "shuoqi" as the filename, it creates:
```
data/self-donated-topics/shuoqi_self-donated_qa.jsonl
```


## Step-by-Step Example
Input has been quoted using `[]'.

```bash
$ python scripts/add_qa_entry/add_qa_entry.py

=== Self-Donated Q&A Entry Tool ===

Enter the filename (without extensions, will create as {filename}_self-donated_qa.jsonl): 
[shuoqi]

Next entry ID will be: 2

Enter your question:
(Type 'DONE' on a new line when finished)
--------------------------------------------------
[What are the key benefits of using multimodal AI systems?]
[DONE]

Enter your answer:
(Type 'DONE' on a new line when finished)
--------------------------------------------------
[Multimodal AI systems offer several key benefits]
DONE

Enter the source:
[ChatGPT]

============================================================
PREVIEW OF ENTRY TO BE ADDED:
============================================================
ID: 2
Question: What are the key benefits of using multimodal AI systems in real-world applications?
Answer: Multimodal AI systems offer several key benefits
============================================================

Add this entry? (y/n): 
[y]

✅ Entry successfully added to data/self-donated-topics/shuoqi_self-donated_qa.jsonl
Entry ID: 2
```