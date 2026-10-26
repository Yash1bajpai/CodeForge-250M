# Project safety
- Notify the owner and obtain explicit approval before any model training or GPU launch; migration must not submit additional CPU preparation.
- Never set CF_TRAINING_APPROVED=1 in controller setup, shell startup, tmux startup or services.
- Only Kaggle owner yashbajpai2027 is enabled; preserve other account files without activating rotation.
- Keep credentials, private session archives and signed URLs outside Git and public logs.
- Preserve old checkpoints and the Windows copy; use non-destructive cloud copies, never rclone sync.
