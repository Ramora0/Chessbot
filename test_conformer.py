"""
Validation tests for conformer implementation.

Tests verify:
1. Token splitting (64 board + 6 metadata)
2. Spatial reshape (64 tokens → 8×8 → 64 tokens)
3. End-to-end forward pass
4. Gradient flow to conv modules
"""

import torch
import torch.nn as nn
from transformers import LlamaConfig

from model import ChessPolicyValueModel
from conformer_layers import (
    SpatialConvolutionModule,
    ChessConformerLayer,
    ChessConformerModel,
)


def test_spatial_reshape():
    """Test that 64 tokens reshape to 8×8 correctly and back."""
    print("\n" + "="*70)
    print("TEST 1: Spatial Reshape (64 tokens → 8×8 → 64 tokens)")
    print("="*70)

    batch_size = 4
    hidden_size = 768
    board_tokens = torch.randn(batch_size, 64, hidden_size)

    # Create spatial convolution module
    conv_module = SpatialConvolutionModule(hidden_size)

    # Forward pass
    output = conv_module(board_tokens)

    # Verify shape is preserved
    assert output.shape == (batch_size, 64, hidden_size), \
        f"Expected shape {(batch_size, 64, hidden_size)}, got {output.shape}"

    # Verify output is different (convolution was applied)
    assert not torch.allclose(output, board_tokens), \
        "Output should be different from input after convolution"

    print(f"✓ Input shape: {board_tokens.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Spatial convolution applied successfully")
    print("✓ TEST 1 PASSED")


def test_token_split():
    """Test that board/metadata split works correctly in ChessConformerLayer."""
    print("\n" + "="*70)
    print("TEST 2: Token Split (64 board + 6 metadata)")
    print("="*70)

    batch_size = 4
    seq_len = 70  # 64 board + 6 metadata
    hidden_size = 768

    # Create minimal config
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=768,
        max_position_embeddings=256,
    )

    # Create conformer layer
    layer = ChessConformerLayer(config, layer_idx=0)
    layer.eval()  # Set to eval mode to avoid batch norm issues

    # Create input with 70 tokens
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    with torch.no_grad():
        output = layer(hidden_states)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, hidden_size), \
        f"Expected shape {(batch_size, seq_len, hidden_size)}, got {output.shape}"

    print(f"✓ Input shape: {hidden_states.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Token split and recombination successful")
    print("✓ TEST 2 PASSED")


def test_forward_pass():
    """Test end-to-end forward pass through ChessPolicyValueModel."""
    print("\n" + "="*70)
    print("TEST 3: End-to-End Forward Pass")
    print("="*70)

    batch_size = 4
    seq_len = 70
    vocab_size = 1000
    policy_dim = 1858

    # Create config
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=256,  # Smaller for faster testing
        num_hidden_layers=2,  # Just 2 layers for testing
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        max_position_embeddings=256,
        attention_dropout=0.1,
        hidden_dropout=0.1,
    )
    config.policy_dim = policy_dim
    config.use_conformer = True  # Enable conformer

    # Create model
    print("Creating ChessPolicyValueModel with conformer...")
    model = ChessPolicyValueModel(config)
    model.eval()  # Set to eval mode

    # Create random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass (without labels, just inference)
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)

    # Verify outputs
    assert outputs.policy_logits is not None, "policy_logits should not be None"
    assert outputs.wdl_logits is not None, "wdl_logits should not be None"
    assert outputs.policy_logits.shape == (batch_size, policy_dim), \
        f"Expected policy shape {(batch_size, policy_dim)}, got {outputs.policy_logits.shape}"
    assert outputs.wdl_logits.shape == (batch_size, 128), \
        f"Expected wdl shape {(batch_size, 128)}, got {outputs.wdl_logits.shape}"

    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Policy logits shape: {outputs.policy_logits.shape}")
    print(f"✓ WDL logits shape: {outputs.wdl_logits.shape}")
    print("✓ TEST 3 PASSED")


def test_conv_gradients():
    """Test that gradients flow to conv modules."""
    print("\n" + "="*70)
    print("TEST 4: Gradient Flow to Conv Modules")
    print("="*70)

    batch_size = 2
    seq_len = 70
    vocab_size = 1000
    policy_dim = 1858

    # Create config
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=128,  # Very small for faster testing
        num_hidden_layers=1,  # Just 1 layer
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=256,
        attention_dropout=0.1,
    )
    config.policy_dim = policy_dim
    config.use_conformer = True

    # Create model
    model = ChessPolicyValueModel(config)
    model.train()  # Set to train mode

    # Create random input and targets
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    policy_target = torch.randn(batch_size, policy_dim)  # Random target for testing
    wdl_target = torch.randn(batch_size, 128)

    # Forward pass with loss computation
    print("Running forward pass with loss...")
    outputs = model(
        input_ids=input_ids,
        policy=policy_target,
        wdl=wdl_target,
    )

    # Verify loss exists
    assert outputs.loss is not None, "Loss should not be None"
    print(f"✓ Loss computed: {outputs.loss.item():.4f}")

    # Backward pass
    print("Running backward pass...")
    outputs.loss.backward()

    # Check that conv module parameters have gradients
    conv_params_with_grad = 0
    conv_params_total = 0

    for name, param in model.named_parameters():
        if 'conv_module' in name:
            conv_params_total += 1
            if param.grad is not None:
                conv_params_with_grad += 1
                # Verify gradient is not zero
                if param.grad.abs().sum() > 0:
                    print(f"✓ Gradient flows to: {name} (norm: {param.grad.norm().item():.4f})")

    assert conv_params_with_grad > 0, \
        "At least some conv module parameters should have gradients"

    print(f"✓ {conv_params_with_grad}/{conv_params_total} conv parameters have gradients")
    print("✓ TEST 4 PASSED")


def test_parameter_count():
    """Test that conformer adds ~20% more parameters."""
    print("\n" + "="*70)
    print("TEST 5: Parameter Count Comparison")
    print("="*70)

    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=768,
        num_hidden_layers=20,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=768,
        max_position_embeddings=256,
    )
    config.policy_dim = 1858

    # Create transformer model (baseline)
    config.use_conformer = False
    transformer_model = ChessPolicyValueModel(config)
    transformer_params = sum(p.numel() for p in transformer_model.parameters())

    # Create conformer model
    config.use_conformer = True
    conformer_model = ChessPolicyValueModel(config)
    conformer_params = sum(p.numel() for p in conformer_model.parameters())

    # Calculate increase
    param_increase = (conformer_params - transformer_params) / transformer_params * 100

    print(f"✓ Transformer parameters: {transformer_params:,}")
    print(f"✓ Conformer parameters: {conformer_params:,}")
    print(f"✓ Parameter increase: {param_increase:.1f}%")

    # Verify increase is reasonable (should be around 15-25%)
    assert 10 < param_increase < 30, \
        f"Parameter increase should be 10-30%, got {param_increase:.1f}%"

    print("✓ TEST 5 PASSED")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("CONFORMER VALIDATION TESTS")
    print("="*70)

    try:
        test_spatial_reshape()
        test_token_split()
        test_forward_pass()
        test_conv_gradients()
        test_parameter_count()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nConformer implementation is ready for training!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
