# Worklog

Short note to remember what we did.

- Trained the large baseline model for 3 epochs.
- Trained the LLaMA-style model as well.
- Compared the two and the large baseline performed better overall.
- Continued training the large model up to 8 epochs / 50k steps.
- After evaluation, the best saved result on the OWT-style comparison was the 5-epoch large checkpoint.
- Added a simple local chat stage to test checkpoints from a browser.
- Added a standalone evaluation stage so checkpoints can be evaluated directly on WikiText and OpenWebText test data.
- Added OWT evaluation support and used it to compare the main large-model checkpoints.

Current practical conclusion:

- Main model to keep: large baseline
- Best saved checkpoint for OWT-style comparison: 5 epochs
