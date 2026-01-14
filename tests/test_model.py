import pytest
import torch   
from my_project.model import MyAwesomeModel

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
        model(torch.randn(1,1,28,29))
    assert len(train_dataset) == N_train, "Dataset did not have the correct number of samples"

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)