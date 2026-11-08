# Campaign night-20261108-1846

## Base model (mean ± std over seeds)

| experiment | seeds | val f1 | test precision | test recall | test f1 | test f2 | syn rare_hard recall | syn rare_hard f1 | syn long recall | syn long f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| A_real | 3 | 0.936 ± 0.007 | 0.921 ± 0.008 | 0.923 ± 0.003 | 0.922 ± 0.004 | 0.953 ± 0.007 | 0.731 ± 0.014 | 0.722 ± 0.007 | 0.724 ± 0.071 | 0.748 ± 0.019 |
| B_mix_full | 3 | 0.929 ± 0.009 | 0.909 ± 0.009 | 0.929 ± 0.003 | 0.919 ± 0.003 | 0.959 ± 0.006 | 0.844 ± 0.004 | 0.804 ± 0.004 | 0.820 ± 0.019 | 0.801 ± 0.020 |
| D_staged_1ep | 3 | 0.938 ± 0.004 | 0.926 ± 0.006 | 0.939 ± 0.004 | 0.932 ± 0.002 | 0.963 ± 0.002 | 0.798 ± 0.007 | 0.786 ± 0.009 | 0.801 ± 0.004 | 0.789 ± 0.004 |
| C_mix_half | 3 | 0.934 ± 0.003 | 0.926 ± 0.003 | 0.936 ± 0.011 | 0.931 ± 0.006 | 0.952 ± 0.007 | 0.848 ± 0.009 | 0.823 ± 0.014 | 0.825 ± 0.008 | 0.810 ± 0.005 |
| E_mix_quarter | 3 | 0.939 ± 0.008 | 0.924 ± 0.009 | 0.926 ± 0.002 | 0.925 ± 0.004 | 0.942 ± 0.008 | 0.838 ± 0.034 | 0.808 ± 0.042 | 0.816 ± 0.031 | 0.793 ± 0.029 |
| F_staged_2ep | 3 | 0.942 ± 0.004 | 0.936 ± 0.012 | 0.937 ± 0.004 | 0.936 ± 0.005 | 0.960 ± 0.008 | 0.793 ± 0.009 | 0.780 ± 0.008 | 0.794 ± 0.009 | 0.791 ± 0.011 |


## Full pipeline on the fake test set (mean over seeds, rules on)

| experiment | seeds | masked | exact | leak spans | traps hit | extra predictions |
|---|---|---|---|---|---|---|
| A_real | 3 | 0.925 | 0.799 | 54.7 | 1.0 | 2.3 |
| B_mix_full | 3 | 0.952 | 0.813 | 35.3 | 2.3 | 3.7 |
| D_staged_1ep | 3 | 0.953 | 0.814 | 34.0 | 1.0 | 3.0 |
| C_mix_half | 3 | 0.956 | 0.818 | 32.3 | 1.7 | 1.3 |
| E_mix_quarter | 3 | 0.950 | 0.810 | 36.7 | 2.7 | 1.7 |
| F_staged_2ep | 3 | 0.946 | 0.814 | 39.0 | 1.0 | 1.3 |

Recipes: **A_real**: real data only (baseline); **B_mix_full**: real + all 3000 synthetic, shuffled together; **D_staged_1ep**: 1 epoch on synthetic first, then real; **C_mix_half**: real + half of the synthetic (1500), shuffled; **E_mix_quarter**: real + a quarter of the synthetic (750), shuffled; **F_staged_2ep**: 2 epochs on synthetic first, then real

Model folders and per-seed numbers: `registry.md` / `registry.csv`.
