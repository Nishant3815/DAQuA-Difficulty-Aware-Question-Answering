### Logging for `query_ftune.py`

Wandb Login API key for team **daqua**

`b579458371e8c12eda5d0a6fb66604bfdfc635e0`

### Steps to add the team API key

```
import wandb
wandb.login()
<paste API key given above>
```



URL: [https://wandb.ai/daqua/query-fine-tuning](https://wandb.ai/daqua/query-fine-tuning)

- Setup and Config

  ```python
  wandb.init(project = "query-fine-tuning",
          notes = 'Query fine tuning for MHop retrieval',
          tags = ["trial"],
          config = args,
          entity = "daqua")
  ```

- Training

  - Warmup

    - Epoch Level

      ```python
      
      ```

    - Pbar Level

      ```python
      wandb.log({
                      "Epoch/warmup_train": ep_idx + 1,
                      "avg_train_loss/warmup_train": total_loss / step_idx,
                      "acc@1/warmup_train": sum(total_accs) / len(total_accs),
                      f"acc@{args.top_k}/warmup_train": sum(total_accs_k) / len(total_accs_k)}
                 )
      ```

  - Warmup Dev Eval

    Calls Evaluate from `eval_phrase_retrieval.py` with `firsthop=True`

    ```python
    wandb.log({
                    "Epoch/warmup_dev_eval": ep_idx + 1,
                    "dev_acc@1/warmup_dev_eval": dev_em,
                    "dev_f1@1/warmup_dev_eval": dev_f1,
                    "dev_evidence_acc@1/warmup_dev_eval": evid_dev_em,
                    "dev_evidence_f1@1/warmup_dev_eval": evid_dev_f1,
                    "dev_set_joint_f1@1/warmup_dev_eval": joint_substr_f1}
                    )
    ```

  - Joint Training

    - Epoch Level

      ```python
      
      ```

    - PBar Level

      ```python
      wandb.log({
                "Epoch/joint_train": ep_idx + 1,
                "avg_train_loss/joint_train": total_loss / step_idx,
                "first_hop_acc@1/joint_train": np.mean(total_accs),
                f"first_hop_acc@{args.top_k}/joint_train": np.mean(total_accs_k),
                "(second_hop_acc@1/joint_train": np.mean(total_u_accs),
                f"(second_hop_acc@{args.top_k}/joint_train": np.mean(total_u_accs_k)}
                )
      ```

  - Dev Evaluation

    Calls Evaluate from `eval_phrase_retrieval.py` with `multihop=True`

    ```python
    wandb.log({
                "Epoch/joint_dev_eval": ep_idx + 1,
                "dev_acc@1/joint_dev_eval": dev_em,
                "dev_f1@1/joint_dev_eval": dev_f1}
                )
    ```

    **Best Model Summary**

    ```python
    wandb.run.summary["best_accuracy_joint"] = best_acc
    wandb.run.summary["best_epoch_joint"] = best_epoch
    wandb.run.summary["best_model_joint"] = save_path
    ```

- Final Evaluation

  N/A
