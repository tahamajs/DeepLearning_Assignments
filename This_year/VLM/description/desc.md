

# 📄 Project Title: Detecting Object Hallucinations in Vision-Language Models (VLMs) via Logit Analysis

## 1\. Executive Summary

Vision-Language Models (VLMs) like LLaVA are powerful but suffer from "Object Hallucinations"—confidently claiming an object exists in an image when it does not. This project aims to build a lightweight detection mechanism that flags potential hallucinations by analyzing the model's internal **logit probabilities** (uncertainty scores) rather than relying solely on the generated text.

**Goal:** Differentiate between *correct responses* and *hallucinations* by setting a confidence threshold on the model's output.

-----

## 2\. The Dataset: POPE (Polling Object Probe Evaluation)

We will use the **POPE Benchmark** (specifically the COCO-Random or COCO-Adversarial subset). This is the standard academic dataset for this task.

  * **Why this dataset?** It consists of simple binary questions: *"Is there a [object] in the image?"*
  * **Structure:**
      * **Image:** References images from the COCO dataset.
      * **Question:** e.g., "Is there a dining table in the image?"
      * **Ground Truth Label:** `yes` (object exists) or `no` (object does not exist).
  * **The Challenge:** The dataset intentionally asks about objects that are **not** present to trigger hallucinations.

-----

## 3\. Technical Stack & Requirements

  * **Environment:** Google Colab (Free Tier - T4 GPU, 12GB RAM).
  * **Model:** `llava-hf/llava-1.5-7b-hf` (Hugging Face version).
  * **Optimization:** 4-bit Quantization (`bitsandbytes`) to reduce VRAM usage from \~14GB to \~5GB.
  * **Libraries:** `transformers`, `accelerate`, `torch`, `pillow`, `scikit-learn` (for metrics).

-----

## 4\. Methodology (The Algorithm)

We will use a **Logit-Based Uncertainty Approach**. Unlike complex methods that require generating text multiple times (which is slow), this method looks at the raw mathematical probability of the first generated token.

### Step 1: Image & Prompt Loading

We load an image and format the prompt specifically for LLaVA:
`"USER: <image>\nIs there a dog in this image? Answer with Yes or No.\nASSISTANT:"`

### Step 2: Forward Pass (Inference)

We feed the data into the model. Instead of just asking for the text, we request the `scores` (logits) for the very first token generated.

### Step 3: Uncertainty Extraction

The model will predict the probability for the token "Yes" vs. the token "No".

  * If the model predicts "Yes" with **99% probability**, it is confident.
  * If the model predicts "Yes" with **55% probability** (and 45% "No"), it is "guessing" or hallucinating.

### Step 4: Thresholding & Classification

We compare the confidence score against a threshold (e.g., 0.75).

  * **Score \< Threshold:** Flag as Potential Hallucination.
  * **Score \> Threshold:** Flag as Reliable Response.

-----

## 5\. Implementation Roadmap (For Colab)

Here is the logic you will implement. I have broken it down so you can visualize the code flow.

### Phase A: Setup

1.  Install dependencies (`transformers`, `bitsandbytes`).
2.  Load the tokenizer and model using `load_in_4bit=True`.

### Phase B: Data Pipeline

1.  Download the `coco_pope_random.json` file from the official GitHub.
2.  Create a function to download the actual images from COCO URLs on the fly (since we cannot download the full 20GB COCO dataset to Colab).

### Phase C: The Detection Loop

*Pseudocode Logic:*

```python
results = []

for item in dataset[:50]: # Run on first 50 items for test
    1. Load Image from URL.
    2. Ask Question: "Is there a [object]?"
    3. Run Model -> Get Logits for "Yes" (token_id: 3869) and "No" (token_id: 1939).
    4. Apply Softmax to get Probability (0.0 to 1.0).
    5. Compare Model Answer vs. Ground Truth (Did it hallucinate?).
    6. Store the Probability score and the Result.
```

### Phase D: Analysis

1.  Calculate **Accuracy**: How often was the model right?
2.  Calculate **Hallucination Rate**: How often did it say "Yes" when the answer was "No"?
3.  **Visualization:** Plot a histogram showing the Confidence Scores of *Correct Answers* vs. *Hallucinations*. (Ideally, hallucinations should have lower scores).

-----

## 6\. Expected Outcome & Deliverables

By the end of this project, you will have:

1.  A working Colab Notebook.
2.  A statistical analysis proving that **hallucinations often correlate with lower model confidence.**
3.  A system that prints: *"Warning: The model says there is a cat, but confidence is low (60%). Possible hallucination detected."*

-----

### Would you like me to generate the Python Code block for "Phase A" and "Phase B" now so you can start the implementation in Colab?