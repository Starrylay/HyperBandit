# !/bin/bash
# nohup bash -c '
for b in "HyperBandit"; do
    for d in "NYC"; do
        for f in "LLM_with_attribute_pca"; do
            for e in "polar"; do 
                for r in -1; do
                    for t in 4000; do
                        for s in 0.4; do
                            for poolsize in 25; do
                                for data_size in 40000; do
                                    python bandit_main.py \
                                        --gpu=0 \
                                        --warm_start \
                                        --baseline="$b" \
                                        --train_window="$t" \
                                        --time_embedding="$e" \
                                        --sample_rate="$s" \
                                        --rank="$r" \
                                        --dataset="$d" \
                                        --feature="$f" \
                                        --isupdate \
                                        --is_hypernet \
                                        --pool_size="$poolsize" \
                                        --llm_dada_size="$data_size" \
                                        --result_path="./results"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
# ' > /dev/null 2>&1 &

