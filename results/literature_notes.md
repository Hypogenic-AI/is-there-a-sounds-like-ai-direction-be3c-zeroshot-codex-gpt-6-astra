# Sources inspected
Retrieved 2026-10-02. Full HTML downloads are in ignored data/literature/.
- https://arxiv.org/abs/2608.24780 and HTML: Quaremba et al., representation probing and shared machine-text direction; does not by itself establish a generation intervention.
- https://arxiv.org/abs/2606.07313 and v2 HTML: SV-Detect uses layerwise alignments for a detector; evaluation primarily detection and transfer.
- https://arxiv.org/abs/2503.03601 and HTML: Kuznetsov et al. explicitly DO perform generative interventions on SAE decoder features, not just detection. Sections on feature steering and Appendices G/H use GPT-4o to interpret resulting changes. Therefore do not claim first causal manipulation of detection-related features. Narrow question: whether a dense paired authorship readout transfers causally with independent detector agreement, quality controls and nuisance comparisons.
- https://arxiv.org/abs/2601.10387 and https://www.anthropic.com/research/assistant-axis: persona direction; our small assistant/storyteller contrast is only a proxy, not a reproduction of their multi-persona axis.
- HC3: https://huggingface.co/datasets/Hello-SimpleAI/HC3 ; CC-BY-SA-4.0; original raw JSONL has no id; preserve original line offsets.
- HAP-E: https://huggingface.co/datasets/browndw/human-ai-parallel-corpus ; MIT dataset card; human chunk 2 is correct matched continuation. Prefix chunk 1 is NOT a valid human continuation comparator.
- https://huggingface.co/openai-community/roberta-base-openai-detector : GPT-2 detector, historically trained and not a valid universal authorship oracle. Independently calibrate on actual test sources.
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct : public model; residual blocks used rather than SAE.
