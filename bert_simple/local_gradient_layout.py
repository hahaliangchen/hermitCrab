"""Feature layout that keeps relation-local gradients strong and background tiny."""

from __future__ import annotations

from typing import List, Sequence

import torch

from examples import bert_mlm_dimension_combinations as base


class LocalFeatureIndexLayout(base.FeatureIndexLayout):
    """Use the relation mask on every parameter, including non-feature tensors.

    The original layout left parameters without a hidden-size axis untouched,
    which meant their full gradients still passed through during a relation
    update.  This layout scales those tensors by the same small background
    weight as inactive hidden dimensions.
    """

    def mask_gradients(
        self,
        gradients: Sequence[torch.Tensor],
        feature_mask: torch.Tensor,
        background_weight: float,
    ) -> List[torch.Tensor]:
        if feature_mask.numel() != self.hidden_size:
            raise ValueError("feature mask width does not match hidden size")
        weights = background_weight + (1.0 - background_weight) * feature_mask.float()
        result: List[torch.Tensor] = []
        for parameter, gradient in zip(self.parameters, gradients):
            if parameter.ndim == 1 and parameter.numel() == self.hidden_size:
                result.append(gradient * weights)
            elif parameter.ndim >= 2 and parameter.shape[-1] == self.hidden_size:
                view_shape = [1] * gradient.ndim
                view_shape[-1] = self.hidden_size
                result.append(gradient * weights.view(view_shape))
            elif parameter.ndim >= 2 and parameter.shape[0] == self.hidden_size:
                view_shape = [self.hidden_size] + [1] * (gradient.ndim - 1)
                result.append(gradient * weights.view(view_shape))
            else:
                # Non-feature parameters are part of the shared glue too, but
                # must not receive an unmasked full relation gradient.
                result.append(gradient * float(background_weight))
        return result
