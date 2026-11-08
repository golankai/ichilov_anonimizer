# Campaign night-20261108-1846: models and results

Model folders are under `training/models/` (git-ignored, on the GPU machine). `model_dir` is relative to the repo.

| experiment | seed | status | model_dir | best ep | val F1 | test F1 | ambig seen | ambig held-out | pipe masked | pipe exact | leaks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_real | 42 | done | training\models\night-20261108-1846\A_real\seed_42 | 4.0 | 0.929 | 0.924 | 0.819 | 0.879 | 0.889 | 0.786 | 81 |
| A_real | 43 | done | training\models\night-20261108-1846\A_real\seed_43 | 6.0 | 0.943 | 0.925 | 0.791 | 0.840 | 0.940 | 0.802 | 44 |
| A_real | 44 | done | training\models\night-20261108-1846\A_real\seed_44 | 5.0 | 0.936 | 0.917 | 0.813 | 0.863 | 0.947 | 0.808 | 39 |
| B_mix_full | 42 | done | training\models\night-20261108-1846\B_mix_full\seed_42 | 6.0 | 0.932 | 0.921 | 0.968 | 0.968 | 0.952 | 0.812 | 35 |
| B_mix_full | 43 | done | training\models\night-20261108-1846\B_mix_full\seed_43 | 2.0 | 0.919 | 0.915 | 0.966 | 0.954 | 0.949 | 0.815 | 37 |
| B_mix_full | 44 | done | training\models\night-20261108-1846\B_mix_full\seed_44 | 6.0 | 0.937 | 0.920 | 0.979 | 0.961 | 0.953 | 0.813 | 34 |
| D_staged_1ep | 42 | done | training\models\night-20261108-1846\D_staged_1ep\seed_42 | 6.0 | 0.936 | 0.933 | 0.933 | 0.946 | 0.958 | 0.815 | 31 |
| D_staged_1ep | 43 | done | training\models\night-20261108-1846\D_staged_1ep\seed_43 | 6.0 | 0.942 | 0.934 | 0.913 | 0.946 | 0.948 | 0.816 | 38 |
| D_staged_1ep | 44 | done | training\models\night-20261108-1846\D_staged_1ep\seed_44 | 5.0 | 0.935 | 0.930 | 0.892 | 0.917 | 0.955 | 0.812 | 33 |
| C_mix_half | 42 | done | training\models\night-20261108-1846\C_mix_half\seed_42 | 5.0 | 0.930 | 0.924 | 0.966 | 0.961 | 0.955 | 0.816 | 33 |
| C_mix_half | 43 | done | training\models\night-20261108-1846\C_mix_half\seed_43 | 4.0 | 0.937 | 0.934 | 0.963 | 0.962 | 0.955 | 0.820 | 33 |
| C_mix_half | 44 | done | training\models\night-20261108-1846\C_mix_half\seed_44 | 6.0 | 0.934 | 0.936 | 0.960 | 0.948 | 0.958 | 0.819 | 31 |
| E_mix_quarter | 42 | done | training\models\night-20261108-1846\E_mix_quarter\seed_42 | 1.0 | 0.931 | 0.929 | 0.944 | 0.954 | 0.944 | 0.804 | 41 |
| E_mix_quarter | 43 | done | training\models\night-20261108-1846\E_mix_quarter\seed_43 | 6.0 | 0.947 | 0.922 | 0.955 | 0.955 | 0.948 | 0.815 | 38 |
| E_mix_quarter | 44 | done | training\models\night-20261108-1846\E_mix_quarter\seed_44 | 6.0 | 0.939 | 0.924 | 0.961 | 0.967 | 0.958 | 0.812 | 31 |
| F_staged_2ep | 42 | done | training\models\night-20261108-1846\F_staged_2ep\seed_42 | 6.0 | 0.944 | 0.940 | 0.909 | 0.939 | 0.949 | 0.812 | 37 |
| F_staged_2ep | 43 | done | training\models\night-20261108-1846\F_staged_2ep\seed_43 | 5.0 | 0.938 | 0.931 | 0.921 | 0.953 | 0.945 | 0.819 | 40 |
| F_staged_2ep | 44 | done | training\models\night-20261108-1846\F_staged_2ep\seed_44 | 5.0 | 0.944 | 0.938 | 0.895 | 0.926 | 0.945 | 0.811 | 40 |
