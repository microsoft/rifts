## Navigating Rifts in Human-LLM Grounding: Study and Benchmark

This repository contains source code for the paper **Navigating Rifts in Human-LLM Grounding: Study and Benchmark** (ACL 2025) by [Omar Shaikh](https://oshaikh.com/), [Hussein Mozannar](https://husseinmozannar.github.io/), [Gagan Bansal](https://gagb.github.io/), [Adam Fourney](https://www.adamfourney.com/), and [Eric Horvitz](https://erichorvitz.com/)

This is not a software release, but source code to help in reproducibility of the paper.

Feel free to reach out to [Omar Shaikh](https://oshaikh.com/) with any questions!

[[Paper]](https://arxiv.org/abs/2503.13975)

**If you're here for the dataset, it's hosted on Hugging Face here!**
[[Dataset]](https://huggingface.co/datasets/microsoft/rifts)

### *Abstract* 

Language models excel at following instructions but often struggle with the collaborative aspects of conversation that humans naturally employ. This limitation in grounding---the process by which conversation participants establish mutual understanding---can lead to outcomes ranging from frustrated users to serious consequences in high-stakes scenarios. To systematically study grounding challenges in human-LLM interactions, we analyze logs from three human-assistant datasets: WildChat, MultiWOZ, and Bing Chat. We develop a taxonomy of grounding acts and build models to annotate and forecast grounding behavior. Our findings reveal significant differences in human-human and human-LLM grounding: LLMs were three times less likely to initiate clarification and sixteen times less likely to provide follow-up requests than humans. Additionally, early grounding failures predicted later interaction breakdowns. Building on these insights, we introduce RIFTS: a benchmark derived from publicly available LLM interaction data containing situations where LLMs fail to initiate grounding. We note that current frontier models perform poorly on RIFTS, highlighting the need to reconsider how we train and prompt LLMs for human interaction. To this end, we develop a preliminary intervention that mitigates grounding failures.

### *Repository Structure*

This project consists of two major components: a labeler and a forecaster.

The first component is a robust, validated labeler of grounding acts for any conversational dataset between a user and assistant. To run the labeler, start by opening gpt-label and reading the readme inside.

The forecaster predicts grounding patterns before they happen. To train forecasting models, evaluate them, and analyze the forecasts they produce across a range of prompts, open the interaction-bench folder. To train models, we additionally provide a modified version of torchtune.

### *How do I cite this work?* 

Feel free to use the following BibTeX entry.

**BibTeX:**

```tex
@misc{shaikh2025navigatingriftshumanllmgrounding,
      title={Navigating Rifts in Human-LLM Grounding: Study and Benchmark}, 
      author={Omar Shaikh and Hussein Mozannar and Gagan Bansal and Adam Fourney and Eric Horvitz},
      year={2025},
      eprint={2503.13975},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.13975}, 
}
```
