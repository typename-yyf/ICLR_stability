# python3 bert_2.py \
#     --config /home/mychen/Stability/bert/config/bert_wikitext103.json \
#     --log-dir /home/mychen/Stability/bert/log/8-23 \
#     $*

python3 bert_2.py \
    --config /home/mychen/ER_TextSpeech/Stability_2/config/bert_wikitext103.json \
    --log-dir /home/mychen/ER_TextSpeech/Stability_2/log/9-3 \
    --seed 1 \
    $*

# python3 bert_explosion_analysis.py \
#     --config /home/mychen/Stability/bert/config/bert_wikitext103.json \
#     --log-dir /home/mychen/Stability/bert/log \
#     --tag debug \
#     --seed 1 \
#     --lr 5e-4 \
#     --gpu 7 \
#     $*