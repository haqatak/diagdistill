import random
import sys
from unittest.mock import patch, MagicMock
import pytest

# Mocking dependencies for the entire module because they are not available in the environment.
# This allows importing utils.misc without ImportError.
@pytest.fixture(scope="module", autouse=True)
def mock_env_dependencies():
    mock_torch = MagicMock()
    with patch.dict(sys.modules, {
        'numpy': MagicMock(),
        'torch': mock_torch,
        'torch.cuda': mock_torch.cuda
    }):
        yield

def test_set_seed_functional_random():
    """Verify that set_seed actually works for the standard random module."""
    from utils.misc import set_seed
    seed = 123

    set_seed(seed)
    val1 = random.random()

    set_seed(seed)
    val2 = random.random()

    assert val1 == val2

    set_seed(seed + 1)
    val3 = random.random()
    assert val1 != val3

def test_set_seed_calls_libraries():
    """Verify that set_seed calls the expected functions in numpy and torch."""
    # We patch the imports in utils.misc specifically
    with patch('utils.misc.np.random.seed') as mock_np_seed, \
         patch('utils.misc.torch.manual_seed') as mock_torch_seed, \
         patch('utils.misc.torch.cuda.manual_seed_all') as mock_torch_cuda_seed, \
         patch('utils.misc.torch.use_deterministic_algorithms') as mock_deterministic, \
         patch('utils.misc.random.seed') as mock_random_seed:

        from utils.misc import set_seed
        seed = 456

        # Test default behavior
        set_seed(seed, deterministic=False)
        mock_random_seed.assert_called_once_with(seed)
        mock_np_seed.assert_called_once_with(seed)
        mock_torch_seed.assert_called_once_with(seed)
        mock_torch_cuda_seed.assert_called_once_with(seed)
        mock_deterministic.assert_not_called()

        # Test deterministic=True
        set_seed(seed, deterministic=True)
        mock_deterministic.assert_called_once_with(True)
