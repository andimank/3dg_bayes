#!/usr/bin/bash

sed -i'' -e "s/run_closure = False/run_closure = True/g" configurations.py

for i in {0..9}; do
  sed -i'' -e "s/closure_val_pt =.*/closure_val_pt = $i/g" configurations.py
    #./bayes_mcmc.py --nwalkers 300 --nburnsteps 20000 20000 #|| exit
    #./src/emulator_truth_vs_dob.py #|| exit
    ./plot_posteriors.py
done
