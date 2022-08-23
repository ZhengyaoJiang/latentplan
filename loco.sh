export PYTHONPATH=.:$PYTHONPATH

name=T-1
datasets=(hopper-medium-replay-v2 hopper-medium-expert-v2)

for round in {1..5}; do
  for data in ${datasets[@]}; do
    python scripts/train.py --dataset $data --exp_name $name-$round --tag development --seed $round
    python scripts/trainprior.py --dataset $data --exp_name $name-$round
    for i in {1..20};
    do
       python scripts/plan.py --test_planner beam_prior --dataset $data --exp_name $name-$round --suffix $i --beam_width 64 --n_expand 4 --horizon 15
    done
  done
done

for data in ${datasets[@]}; do
  for round in {1..5}; do
    python plotting/read_results.py --exp_name $name-$round --dataset $data
  done
  python plotting/read_results.py --exp_name $name --dataset $data
done


