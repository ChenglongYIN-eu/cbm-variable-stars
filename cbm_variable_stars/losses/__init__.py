"""
Loss functions for CBM variable star classification.

Provides:
    - CBMJointLoss:         Joint training loss (alpha * L_concept + beta * L_cls)
    - CBMCalibrationLoss:   Plan B loss with partial concept GT support
    - CBMSequentialLoss:    Two-stage sequential training loss (Plan B only)
    - CBMIndependentLoss:   Independent concept + label training with gradient detach
    - compute_class_weights: Class weight calculation utilities
"""
