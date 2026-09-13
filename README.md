---
library_name: transformers
tags:
- maternal
- fetal plane
- classification
- CNN
metrics:
- accuracy
model-index:
- name: FP_Classifcation-V1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# FP_Classifcation-V1

This model was trained from scratch on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3715
- Accuracy: 0.8824
- Precision Macro: 0.8529
- Recall Macro: 0.8867
- F1 Macro: 0.8671

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 4
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 50
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | Precision Macro | Recall Macro | F1 Macro |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:---------------:|:------------:|:--------:|
| 1.2695        | 1.0   | 483   | 0.9169          | 0.6594   | 0.6276          | 0.6618       | 0.6148   |
| 1.2483        | 2.0   | 966   | 0.7571          | 0.7515   | 0.7120          | 0.7200       | 0.7148   |
| 1.1007        | 3.0   | 1449  | 0.7602          | 0.6659   | 0.6788          | 0.7617       | 0.6503   |
| 0.9167        | 4.0   | 1932  | 0.6425          | 0.7782   | 0.7402          | 0.8037       | 0.7553   |
| 0.8257        | 5.0   | 2415  | 0.6162          | 0.7947   | 0.7688          | 0.7845       | 0.7715   |
| 0.8868        | 6.0   | 2898  | 0.5976          | 0.7576   | 0.7269          | 0.8177       | 0.7407   |
| 0.8947        | 7.0   | 3381  | 0.5502          | 0.8044   | 0.7672          | 0.8258       | 0.7793   |
| 0.9688        | 8.0   | 3864  | 0.5810          | 0.8121   | 0.7919          | 0.8017       | 0.7796   |
| 0.7785        | 9.0   | 4347  | 0.5259          | 0.8137   | 0.7860          | 0.8361       | 0.7912   |
| 0.7221        | 10.0  | 4830  | 0.4798          | 0.8412   | 0.8117          | 0.8399       | 0.8181   |
| 0.8350        | 11.0  | 5313  | 0.5120          | 0.8360   | 0.8242          | 0.8174       | 0.8180   |
| 0.6168        | 12.0  | 5796  | 0.5778          | 0.8263   | 0.8110          | 0.8311       | 0.8088   |
| 0.7170        | 13.0  | 6279  | 0.4234          | 0.8497   | 0.8175          | 0.8607       | 0.8337   |
| 0.5224        | 14.0  | 6762  | 0.4244          | 0.8453   | 0.8147          | 0.8648       | 0.8250   |
| 0.6574        | 15.0  | 7245  | 0.4858          | 0.8606   | 0.8472          | 0.8480       | 0.8464   |
| 0.6560        | 16.0  | 7728  | 0.4204          | 0.8529   | 0.8228          | 0.8723       | 0.8409   |
| 0.5167        | 17.0  | 8211  | 0.3715          | 0.8683   | 0.8335          | 0.8751       | 0.8498   |
| 0.6732        | 18.0  | 8694  | 0.3831          | 0.8570   | 0.8204          | 0.8725       | 0.8392   |
| 0.6153        | 19.0  | 9177  | 0.3715          | 0.8824   | 0.8529          | 0.8867       | 0.8671   |
| 0.5594        | 20.0  | 9660  | 0.3744          | 0.8655   | 0.8290          | 0.8803       | 0.8500   |
| 0.4291        | 21.0  | 10143 | 0.3573          | 0.8853   | 0.8623          | 0.8848       | 0.8723   |
| 0.6011        | 22.0  | 10626 | 0.3675          | 0.8719   | 0.8445          | 0.8842       | 0.8582   |
| 0.4130        | 23.0  | 11109 | 0.3563          | 0.8816   | 0.8561          | 0.8792       | 0.8664   |
| 0.5596        | 24.0  | 11592 | 0.3789          | 0.8828   | 0.8629          | 0.8782       | 0.8697   |


### Framework versions

- Transformers 5.17.0
- Pytorch 2.11.0+cu128
- Datasets 5.0.1
- Tokenizers 0.23.1
