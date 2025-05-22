#!/bin/bash

# Define L40 nodes (dlc2gpu01–08 and dlc2gpu10–16)
L40_NODES=$(seq -f "dlc2gpu%02g" 1 8; seq -f "dlc2gpu%02g" 10 16)
TOTAL_GPUS_PER_NODE=8

declare -A user_gpu_count
used=0
total=0

echo "📊 L40 GPU Usage per Node:"
echo "---------------------------"

# Loop through each node
for node in $L40_NODES; do
    node_used=0
    # Get list of users and how many GPUs they use on this node
    while IFS= read -r line; do
        user=$(echo $line | awk '{print $1}')
        gpus=$(echo $line | grep -o "gres/gpu:[0-9]\+" | cut -d: -f2)
        gpus=${gpus:-0}
        (( user_gpu_count[$user] += gpus ))
        (( node_used += gpus ))
    done < <(squeue -w $node -h -o "%u %b")

    echo "$node: $node_used GPUs in use"
    (( used += node_used ))
    (( total += TOTAL_GPUS_PER_NODE ))
done

echo "---------------------------"
echo "Total L40 GPUs: $total"
echo "Used  L40 GPUs: $used"
echo "Free  L40 GPUs: $((total - used))"

echo
echo "👥 GPU Usage per User:"
echo "---------------------------"
for user in "${!user_gpu_count[@]}"; do
    printf "%-12s : %d GPUs\n" "$user" "${user_gpu_count[$user]}"
done
