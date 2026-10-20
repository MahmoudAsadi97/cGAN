import pytest
import torch
from torch import nn

from fluvius_cgan.models import Discriminator, Generator


@pytest.mark.parametrize('classes', [64, 0])
def test_reference_topology_and_gradients(classes):
    generator, discriminator = Generator(num_classes=classes), Discriminator(num_classes=classes)
    assert [(m.in_features, m.out_features) for m in generator.modules() if isinstance(m, nn.Linear)] == [(24 + classes, 64), (64, 128), (128, 64), (64, 24)]
    assert [(m.in_features, m.out_features) for m in discriminator.modules() if isinstance(m, nn.Linear)] == [(24 + classes, 64), (64, 128), (128, 256), (256, 1)]
    assert [m.negative_slope for m in generator.modules() if isinstance(m, nn.LeakyReLU)] == [.2] * 3
    assert [m.p for m in discriminator.modules() if isinstance(m, nn.Dropout)] == [.4]
    assert not any(isinstance(m, (nn.Sigmoid, nn.Tanh, nn.BatchNorm1d)) for m in generator.modules())
    assert not any(isinstance(m, nn.Sigmoid) for m in discriminator.modules())
    labels = torch.tensor([0, 1, 63])
    output = generator(torch.randn(3, 24), labels)
    assert output.shape == (3, 24)
    logits = discriminator(output, labels)
    assert logits.shape == (3, 1)
    nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in generator.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in discriminator.parameters())


@pytest.mark.parametrize('labels', [torch.tensor([-1]), torch.tensor([64]), torch.tensor([1.5]), torch.tensor([[1]])])
def test_invalid_integer_condition_labels_fail(labels):
    with pytest.raises((ValueError, TypeError), match='label'):
        Generator()(torch.zeros(1, 24), labels)


def test_output_is_linear_and_conditions_are_one_hot():
    generator = Generator()
    captured = []
    hook = generator.model[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    generator(torch.zeros(2, 24), torch.tensor([2, 63]))
    hook.remove()
    assert captured[0].shape == (2, 88)
    torch.testing.assert_close(captured[0][:, 24:].sum(1), torch.ones(2))
    assert captured[0][0, 26] == 1 and captured[0][1, 87] == 1
    with torch.no_grad():
        generator.model[-1].weight.zero_()
        generator.model[-1].bias.fill_(-2)
    assert torch.equal(generator(torch.zeros(1, 24), torch.tensor([0])), torch.full((1, 24), -2.))
