import torch

from ivsf_pipeline.model import SixLayerCNNGTSRB


def test_gtsrb_six_layer_cnn_outputs_43_logits() -> None:
    model = SixLayerCNNGTSRB()

    logits = model(torch.zeros(2, 3, 32, 32))

    assert logits.shape == (2, 43)