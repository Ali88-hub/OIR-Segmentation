import segmentation_models_pytorch as smp


def build_model(config):
    """Build U-Net with pretrained encoder and multi-channel sigmoid output.

    Each output channel predicts one mask type independently (multi-label).
    Returns raw logits — apply sigmoid in loss/inference.
    """
    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=config.num_classes,
        activation=None,
        decoder_attention_type=config.decoder_attention,
    )
    return model
