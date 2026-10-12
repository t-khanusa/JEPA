---
name: lyapunov-latent-planning
description: Goal, v1 design, evaluation metrics, key experiments, current status and future directions for replacing the temporal-straightening (TS) aggcos curvature loss with a ControlJEPA Lyapunov loss on aggregated latents for latent planning. Use when editing models/lyapunov/, the lyapreach/lyaptube straighten modes in VWorldModel, launching or analysing aggcos-vs-Lyapunov runs, or deciding what to try next.
---

# Lyapunov loss cho latent planning (ControlJEPA → TS)

> File này phải được lưu UTF-8. Công cụ Write của agent từng ghi lossy (mất dấu tiếng Việt); nếu cần sửa, ghi qua Python với `encoding="utf-8"` rồi kiểm tra bằng `file SKILL.md`.

## 1. Mục tiêu

- Thay `training.straighten=aggcos1e-1` bằng một Lyapunov loss (ControlJEPA, NeurIPS26) **trên cùng agg latents**, không đổi gì khác, và đạt **planning success cao hơn** aggcos.
- Nguyên tắc v1: **drop-in replacement, train và phân tích output trước khi thêm bất kỳ thành phần nào.**
- Người phán xử cuối cùng là planning success (`plan.py`), không phải training loss.

## 2. Thiết kế v1 (code: `temporal-straightening/models/lyapunov/ts_lyapunov.py`)

Mỗi training window \(h=\mathrm{agg}(\mathrm{visual\_only}(z))\in\mathbb R^{B\times T\times 128}\), với \(T=\text{num\_hist}+\text{num\_pred}=4\) (frameskip 5, tức 15 env steps), là một đoạn start→goal:

\[
c=h_{T-1}-h_0,\; d_t=h_t-h_0,\; p_t=\langle d_t,\hat c\rangle,\; e_t=d_t-p_t\hat c,\; L=\|c\|
\]
\[
V_t=\alpha\frac{\|e_t\|^2}{L^2}+\beta\Big[\tfrac{t}{T-1}-\tfrac{p_t}{L}\Big]_+^2,\qquad
\mathcal L=\operatorname{mean}_t\,\mathrm{act}\big(V_{t+1}-\gamma V_t-\tau\big)
\]

- `lyapreach<λ>`: ControlJEPA `LyapunovReachabilityLoss` (transverse_norm=L, schedule_asym).
- `lyaptube<λ>`: β=0, đúng `LyapunovControlLoss(norm_mode="v_geo")`. Đã kiểm tra khớp reference tới 4e-9.
- `lyapheading<λ>` (v2, đã train; planning 34/50 ở cả λ=0.1 và λ=0.2): heading error của từng bước so với dây cung, \(V_t=1-\cos(h_{t+1}-h_t,\,c)\), \(t=0..T-2\), neo \(V_{-1}=0\), cùng recurrence ISS. Bị chặn trong [0,2], bất biến với scale của từng bước và với độ dài dây cung, phạt đi lùi (backtracking). Log thêm `lyap_heading_cos`, `lyap_backtrack_rate`.
- Tổng loss: `z_loss + λ·L_lyap`, cùng vị trí với aggcos trong `VWorldModel.forward`. Gradient đi vào projector và AggMLP (DINO frozen).
- Window có \(L^2<\) `min_L_sq` (frame tĩnh) bị loại khỏi loss.
- Config `training.lyapunov`: γ=0.95, τ=1e-4, α=1, β=0.5, min_L_sq=1e-8, use_softplus=False, log_geometry=True.
- Prefix `lyap*` được parse trước `aggcos`/`cos`. Hydra dir khác nhau theo chuỗi `straighten`, nên các run không đè nhau.

So với adapter LLM cũ (`lyapunov_planning.py`), v1 sửa ba điểm:
1. Schedule đúng `[0, 1/3, 2/3, 1]`. Ánh xạ user/assistant span cũ đếm endpoint hai lần, cho ra `[0, 1/4, 1/2, 1]`.
2. Activation là config tường minh. Bản cũ hardcode `use_softplus=False` dù adapter truyền True.
3. Vectorized, không loop Python theo batch.

Tính chất đại số cần nhớ khi đọc kết quả:
- \(V_0=0\) do neo tại \(h_0\), nên với T=4 ISS suy biến thành ống bán kính cố định (\(V_1\le\tau\), \(V_2\le\gamma\tau+\tau\)). Đây là ràng buộc thẳng hàng trong window, không phải "recovery" từ nhiễu.
- **Tube mù thứ tự**: đi tới rồi lùi dọc dây cung có tube loss 0, còn cosine cho giá trị tối đa 2. Reach chỉ bắt được một phần qua lag.
- Lag-only nên không phạt teleport (\(h_1\approx h_{T-1}\)). Theo dõi `lyap_p_over_L` / `geo_p_over_L_mid` (≈0.5 tốt, →1 là teleport).
- **Scale-invariant trên agg nhưng không cố định scale của projector output z.** Đây là failure mode chính đã đo được (mục 5).

## 3. Metrics đánh giá (theo thứ tự ưu tiên)

1. **Planning success** cùng planner, seed, goal set. Hai protocol:
   - Repo này: `plan.py --config-name plan_gd.yaml` (goal_H=25, goal_source=dset, GD lr 0.1, 100 opt steps, objective `last` = visual MSE + α·proprio), `n_evals=50`. Báo cáo paired success.
   - Protocol đã khoá ở repo cũ (`/project/khanhnt/temporal-straightening/research/`): held-out episodes 1800–1999, H∈{5,10,15}, n=100, Adam 100 steps. Tiêu chí "thắng TS": success cao hơn ở H15 với exact McNemar p<0.05 trên paired episodes, ≥3 seed.
2. **Rollout error đã chuẩn hoá theo scale** (`scripts/diagnose_scale.py --run <run_dir> --epoch <e>`, eval-only, cần `source /project/khanhnt/temporal-straightening/setup_env.sh`). Rollout 5 model steps từ 1 frame với action thật:
   - `feature_var`: variance của visual z giữa các state. Đây là scale của model.
   - `nmse_rollout_H`: raw MSE chia cho `feature_var`. Giá trị >1 nghĩa là sai số dự đoán lớn hơn khoảng cách điển hình giữa các state.
   - `imagined_demo_cost_ratio`: cost tưởng tượng còn lại sau khi rollout action demo (TS ≈0.1–0.3).
   - `demo_over_zero_cost`: cost với action demo so với action 0. Càng thấp thì planner càng dễ phân biệt action tốt.
3. Geometry trên agg, log giống nhau ở mọi run (`train_/val_geo_*`): `geo_goal_monotone_rate` (→1), `geo_V_transverse_mean/max` (↓), `geo_curvature` = 1−cos, `geo_progress_monotone_rate`, `geo_chord_over_path` (→1), `geo_latent_std`.
4. Nội bộ Lyapunov: `lyap_viol_rate`, `lyap_lag_rate`, `lyap_V_transverse`, `lyap_V_progress`, `lyap_p_over_L`, `lyap_valid_frac`, `lyap_chord_norm`.
5. **Không** so trực tiếp các giá trị raw `val_z_visual_err_rollout*`, `val_z_visual_loss`, `val_z_visual_err_pred/full` hay `val_loss` giữa các run. Chúng là MSE thô và giảm theo bình phương scale của feature. Một model nén scale sẽ "thắng" các đường này nhưng thua planning.
6. Bỏ qua `z_visual_err_next1`: slice `[-1:0]` rỗng khi num_pred=1, nên luôn NaN.

## 4. Thí nghiệm quan trọng (v1, matched setup)

Cùng `encoder=dino_channel`, seed, lr, 20 epochs, batch 32, decoder như baseline; chỉ đổi `training.straighten`.

| ID | straighten | Câu hỏi |
|---|---|---|
| E0 | `aggcos1e-1` | Baseline TS |
| E1 | `lyapreach1e-1` | Cùng λ: Reach đã sửa schedule có bằng aggcos không? |
| E2 | `lyapreach5e-1` | Cùng độ lớn loss: aggcos ≈1.2, Reach ≈0.25 ở đầu training |
| E3 | `lyaptube1e-1` | Ablation thứ tự: bỏ progress (β=0) |
| E4 | `lyapreach1e-1` + `training.lyapunov.use_softplus=True` | Activation |

Trước khi chạy E2–E4, cần cân nhắc lại vì đã phát hiện nén scale (mục 5). λ lớn hơn có thể làm nén mạnh hơn.

```bash
cd temporal-straightening && conda activate ts && source /project/khanhnt/temporal-straightening/setup_env.sh
GPUS="1 2" LYAP_RUNS="lyapreach1e-1" bash scripts/run_aggcos_vs_lyap.sh        # E0 || E1
# planning: 10 seeds x n_evals=5 (GPU 0 only has ~9.5 GB free; n_evals=10 OOMs)
for s in 100 200 300 400 500 600 700 800 900 1000; do
  WANDB_MODE=offline python plan.py --config-name plan_gd.yaml ckpt_base_path=./checkpoints/test \
    model_name=umaze_lyapreach1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05 model_epoch=20 n_evals=5 seed=$s
done
# mechanism / scale diagnostics (eval only)
python scripts/diagnose_scale.py     --run checkpoints/test/<run> --epoch 20
python scripts/diagnose_mechanism.py --run checkpoints/test/<run> --epoch 20 --batch 16
python scripts/screen_energies.py    --run checkpoints/test/<run> --epoch 20
```

`plan.py` dùng `SerialVectorEnv` cho point_maze (MuJoCo crash trong forked workers) và resolve `ckpt_base_path` tương đối theo cwd gốc (port từ repo cũ).

Môi trường:
- conda env `ts` (Python 3.9). DINOv2 được pin `facebookresearch/dinov2:b48308a` trong `models/dino.py`, vì upstream main dùng cú pháp `X | None`.
- `DATASET_DIR=/project/khanhnt/temporal-straightening/data`, `PYTHONNOUSERSITE=1`. Script train tự set hai biến này.
- Planning và `diagnose_scale.py` cần MuJoCo env: `source /project/khanhnt/temporal-straightening/setup_env.sh`.
- GPU 0 thường bị job khác chiếm. Dùng GPU 1, 2 để train.

## 5. Trạng thái hiện tại (2026-10-12)

### 5.1 Kết quả v1 (matched, 20 epochs)

- Planning (`plan_gd.yaml`, goal_H=25, epoch 20, 10 seed × `n_evals=5` = 50 episode paired, GPU 0): **aggcos 44/50, lyapreach 37/50**. aggcos thắng riêng 10 episode, lyapreach thắng riêng 3, exact McNemar p=0.092. Lặp lại kết quả repo cũ (45 vs 33): sửa bug schedule không cứu được Reach.
- Nén scale (`diagnose_scale.py`, 64 val windows, H=5 model steps):

| Run | var(z) | raw MSE_H | NMSE_H | imagined cost ratio | demo/zero |
|---|---:|---:|---:|---:|---:|
| aggcos e1 / e4 / e8 | 0.036 / 0.046 / 0.033 | 0.080 / 0.031 / 0.012 | 2.24 / 0.67 / 0.37 | 1.53 / 0.58 / 0.27 | 0.79 / 0.50 / 0.24 |
| lyapreach e1 / e4 / e8 | 0.008 / 0.004 / 0.003 | 0.073 / 0.008 / 0.004 | 8.85 / 1.91 / 1.30 | 4.88 / 1.24 / 0.93 | 0.77 / 0.85 / 0.68 |
| aggcos e20 (cũ, 45/50) | 0.042 | 0.008 | 0.19 | 0.12 | 0.11 |
| Reach e20 (cũ, 33/50) | 0.004 | 0.004 | 0.87 | 0.61 | 0.61 |

### 5.2 Cơ chế: hai loss KHÔNG hoạt động giống nhau (`diagnose_mechanism.py`, `screen_energies.py`, epoch 19)

Đo trên 512 val windows dài bằng training window (4 frame), cùng checkpoint, đối chiếu với trạng thái vật lý (agent xy):

1. **Reach đo độ dài dây cung, không đo độ cong.** Do chia cho \(L^2\), loss tập trung vào window agent đi chậm (dây cung ngắn):
   - Top 10% window chiếm 80% (model Reach) đến 99.8% (model aggcos) tổng Reach loss; aggcos chỉ 19–32%.
   - Spearman(Reach loss, độ dài dây cung) = −0.80. Spearman với độ cong vật lý chỉ 0.18–0.62, trong khi aggcos đạt 0.69–0.83.
2. **Gradient Reach bùng nổ và lấn át JEPA.** Trên projector, \(0.1\cdot\|g_{\text{reach}}\|/\|g_{\text{JEPA}}\|\) = 6.3 (e1) và 10.4 (e19); aggcos là 0.8 và 3.0. Đánh giá Reach trên checkpoint aggcos cho gradient lớn ~750× aggcos.
3. **Hướng gradient gần như trực giao, thậm chí ngược nhau.** cos(\(g_{\text{aggcos}}, g_{\text{reach}}\)) trên projector = −0.11 đến −0.19 ở e19, nên hai loss đẩy encoder theo hai hướng khác nhau.
4. **Reach còn kém trên chính chỗ cần thẳng.** Trên window đi thẳng về mặt vật lý, agg của model aggcos thẳng hơn hẳn (aggcos loss 0.026 vs 0.33), và thấp hơn cả trên Reach loss (0.005 vs 0.014). Patch trajectory (không gian của planner) cũng thẳng hơn: cos loss 0.94 vs 1.11.
5. **Thông tin không bị mất.** Probe tuyến tính agent xy từ patch: R² = 0.999 ở cả hai. Spearman(khoảng cách patch tới goal, khoảng cách vật lý) 0.91 vs 0.90. Khác biệt nằm ở khả năng dự đoán của predictor (NMSE) và scale, không phải ở nội dung thông tin.
6. Với T=4, recurrence chỉ còn \(V_3=0\) và \(V_0=0\), nên Reach ≈ \(\max(V_1, V_2+0.05V_1)/3\) cộng hinge (~50% transition không có gradient). Năng lượng được chọn và cách chuẩn hoá quyết định mọi thứ, không phải cấu trúc ISS.

Kết luận: nguyên nhân thực sự là **năng lượng V chuẩn hoá theo \(L^2\)**. Nó tạo loss đuôi nặng do window chậm chi phối, gradient lớn gấp nhiều lần JEPA, hướng gradient không khớp với độ cong thật, và kéo theo nén scale cùng predictor kém.

### 5.3 Screen các năng lượng ứng viên (offline, không train)

| Năng lượng (cùng recurrence ISS) | Spearman với độ cong vật lý | Top-10% share | 0.1·grad / JEPA | cos(grad, aggcos) |
|---|---:|---:|---:|---:|
| aggcos (tham chiếu) | 0.69–0.83 | 0.15–0.32 | 0.8–5.2 | 1 |
| reach_L (v1) | 0.18–0.62 | 0.65–1.00 | 6–2283 | −0.19–0.29 |
| tube_dt (ControlJEPA `d_t` norm) | −0.02–0.19 | 0.17–0.83 | 0.3–1.9 | −0.35–0.65 |
| heading_goal (bước so với hướng còn lại tới goal) | 0.36–0.48 | 0.23–0.38 | 0.4–3.0 | 0.08–0.98 |
| **heading_chord → `lyapheading`** | **0.64–0.79** | **0.18–0.25** | **0.4–2.2** | **0.81–0.93 trên ckpt aggcos** |

`lyapheading` là ứng viên duy nhất giữ được cả ba tính chất của aggcos (bám độ cong thật, không đuôi nặng, gradient cùng cỡ và cùng hướng), đồng thời thêm hai tính chất cosine không có: tham chiếu toàn cục theo dây cung và phạt đi lùi.

### 5.4 Planning `lyapheading` (epoch 20, 2026-10-12)

Train xong 2026-10-10 17:01 / 17:11 UTC (`logs/aggcos_vs_lyap_20261010_015554/`). Planning cùng protocol mục 5.1: `plan_gd.yaml`, epoch 20, seed 100..1000, `n_evals=5`. GPU 0 chỉ còn ~4.3 GB nên bỏ. `lyapheading1e-1` trên GPU 1 (~4.9 GB), `lyapheading2e-1` trên GPU 2 (đỉnh ~6.1 GB). Log: `logs/plan/umaze_lyapheading{1e-1,2e-1}_e20_s*.log`.

| Run | Success | Thua riêng / thắng riêng so với aggcos | exact McNemar |
|---|---:|---:|---:|
| aggcos λ=0.1 | 44/50 | — | — |
| lyapreach λ=0.1 | 37/50 | 10 / 3 | p=0.092 |
| lyapheading λ=0.1 | **34/50** | 11 / 1 | **p=0.006** |
| lyapheading λ=0.2 | **34/50** | 13 / 3 | **p=0.021** |

So với Reach: heading λ=0.1 thua riêng 7 / thắng riêng 10 (p=0.63); λ=0.2 thua riêng 8 / thắng riêng 11 (p=0.65). Hai λ heading hoà nhau 9/9 (p=1). Screen offline (mục 5.3) không dự báo được planning: heading thua aggcos có ý nghĩa và không hơn Reach.

Đối chiếu setting (2026-10-12): cả 40 log có cùng `plan_gd.yaml` (mtime 2026-10-07), `model_epoch=20`, `eval_seed = seed·n+1` cho n=0..4, thư mục `gH25_dset/..._gd_lr0.1_an0_opt100_objlast_initzero`. Train chỉ khác chuỗi `training.straighten`; block `lyapunov` giống hệt, λ=0.2 nằm trong chuỗi `lyapheading2e-1`. Episode được `random.randint` sau `seed(planning_seed)`; split val dùng `torch.Generator` cục bộ seed 42. `plan_targets.pkl` của seed 1000 (ảnh, state, action) giống nhau từng bit giữa Reach, heading λ=0.1 và λ=0.2. File của aggcos trên đĩa là seed 100 vì seed đó được chạy lại sau cùng, nên không so ảnh trực tiếp được; đường sample không phụ thuộc weight.

Hình học agg ở epoch 20 (wandb summary), cùng metric `geo_*`:

| | latent std | cos bước liên tiếp | chord/path | V_transverse | V_progress |
|---|---:|---:|---:|---:|---:|
| aggcos | 0.204 | 0.464 | 0.787 | — | — |
| lyapreach | 0.102 | 0.135 | 0.668 | 0.281 | 0.058 |
| lyapheading λ=0.1 | 0.043 | 0.018 | 0.593 | 0.354 | — |
| lyapheading λ=0.2 | 0.041 | 0.012 | 0.589 | 0.361 | — |

Progress của Reach đã đạt lịch (val `p/L` = 0.505, lag rate 0.51) nhưng nhỏ hơn transverse ~5 lần. Heading không có số hạng quãng đường; val `lyap_heading_cos` = 0.55 và backtrack còn 6.8%.

Lịch sử từ repo cũ (`/project/khanhnt/temporal-straightening`, xem `LYAPUNOV_TONG_HOP.md`, `research/LYAPUNOV_DESIGN_EXPERIMENT_SUMMARY.md`, `research/NEXT_LYAPUNOV_BEAT_TS_2026_10_09.md`, `research/results/frozen_pilot_20260905_v2/REPORT.md`):

| Thiết kế | Kết quả | Ghi chú |
|---|---|---|
| Reach trên agg (`aggreach1e-1`) | 33/50 vs TS 45/50 | Bug schedule + ReLU; variance patch nhỏ hơn TS ~11× |
| RolloutTube (P2) | 7/50 | Suy biến; variance ~2e-6 (gần collapse) |
| Excess-energy tube + guard (patch) | 40/50 vs TS 46/50 | Có ích so với JEPA+guard (25/50), chưa bằng cosine |
| Dir-Lya (patch) | 41/50 vs patch-cos 45/50 | Về đại số ≈ năng lượng cosine có trọng số |
| CLM (goal metric trên data) | H15 34/100 vs TS 78/100 | Encoder metric tốt nhưng phá predictor |
| Optimizer-step Lyapunov | Không có gain riêng | Gain được giải thích bởi prediction-only |

Insight chung: success bám theo **imagined progress** (rollout của predictor hạ cost về goal), không bám raw MSE hay monotonicity của encoder metric.

## 6. Hướng đề xuất tiếp theo

1. **`lyapheading` không đạt mục tiêu.** 34/50 ở cả hai λ, thua aggcos 44/50 (McNemar p=0.006 và 0.021) và không khác Reach. Trên agg nó còn kém Reach: std 0.043 so với 0.102 (aggcos 0.204), cosine bước liên tiếp 0.02 so với 0.13 (aggcos 0.46). Progress term của Reach không phải chỗ hỏng — nó đã bám lịch; transverse/`L^2` mới là phần lớn của V. Đo `diagnose_scale.py` trên patch (không gian planner) vẫn còn thiếu nếu muốn nối std của agg với NMSE của predictor.
2. Không chạy ablation γ / softplus / heading-so-với-bước-liền-trước khi chưa có một năng lượng ≥ aggcos. Nếu vẫn muốn tách đóng góp sau này:
   - recurrence (γ) so với năng lượng thuần (γ=0, tức mean của V): Lyapunov có thêm giá trị gì ngoài năng lượng?
   - softplus so với ReLU;
   - heading theo dây cung so với heading theo bước liền trước (tương đương cosine).
3. Không dùng lại năng lượng chuẩn hoá theo \(L^2\) (reach_L, tube) hoặc theo \(\|d_t\|^2\) (tube_dt) cho TS, trừ khi có trọng số/eligibility chống đuôi nặng.
4. Sau khi có bản drop-in ≥ aggcos: PCL (CLF trên rollout của predictor, kèm fidelity anchor), ISS trên sai số rollout đo theo NMSE (γ≈1), CLF trên đúng cost của planner, window dài / đa thang đo.
5. Câu chuyện lý thuyết cho paper: planning success ⇐ (CLF trên cost) + (ISS của world model), với sai số đo **tương đối theo scale**, và năng lượng Lyapunov phải **bị chặn và bám độ cong thật** để không bị window suy biến chi phối.
