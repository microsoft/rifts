
# python -m src.gpt_label \
#     --batch_size 50 \
#     --pickle "./labeled_data/copilot_sample.pkl" \
#     --output "./labeled_data/copilot_sample_labeled.json" 


# python -m src.gpt_label \
#     --batch_size 10 \
#     --pickle "./labeled_data/wildchat_sample.pkl" \
#     --output "./labeled_data/wildchat_sample_labeled.json" \
#     --resume

python -m src.gpt_label \
    --batch_size 10 \
    --pickle "./labeled_data/multiwoz_sample.pkl" \
    --output "./labeled_data/multiwoz_sample_labeled.json" \
    --resume