| DA-v1 | Adam     | lr=0.001     | Train: 97.20, Val: 50.20, Best: 50.55     |
| ----- | -------- | ------------ | ----------------------------------------- |
| DA-v1 | SGD      | lr=0.0001    | Train: 69.49, Val: 33.03, Best: 33.03     |
| DA-v2 | SGD      | lr=0.001     | Train: 99.99, Val: 55.77, Best: 56.16     |
| DA-v2 | Adam     | lr=0.0004    | Train: 97.85, Val: 54.26, Best: 54.62     |

DA-V1: Additional steps are introduced: resizing the images to 36x36 and center-cropping them to 32x32 in order to standardize the input.

DA-v2: This includes data augmentation techniques to improve generalization and performance.
