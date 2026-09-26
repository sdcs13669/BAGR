"""Classifier backbone layers (GINE).

To add a backbone: implement an encoder class whose constructor kwargs are a
subset of its BACKBONE_KWARGS entry plus an ``out_dim`` attribute, register it
in BACKBONES, and keep the plus/ blocks bit-exact no-ops when disabled."""

from .gin import GINEEncoder, GINELayer

BACKBONES = {
    'gin': GINEEncoder,

}

BACKBONE_KWARGS = {
    'gin': frozenset({'edge_feat_dim', 'hidden_dims', 'dropouts',
                      'virtual_node', 'residual', 'node_feat_dim',
                      'gnn_norm', 'readout', 'edge_encoder_kind',
                      'ffn', 'ffn_dropout', 'pos_enc', 'pos_enc_ksteps'}),
}


def get_backbone(name: str):
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone '{name}', available: {sorted(BACKBONES)}")
    return BACKBONES[name]


def backbone_init_kwargs(backbone: str, config) -> dict:
    """ClassifierConfig -> constructor kwargs whitelisted for the backbone."""
    full = dict(
        edge_feat_dim=config.edge_feat_dim,
        hidden_dims=config.hidden_dims,
        dropouts=config.dropouts,
        virtual_node=config.virtual_node,
        residual=config.residual,
        node_feat_dim=config.node_feat_dim,
        gnn_norm=config.gnn_norm,
        readout=config.readout,
        edge_encoder_kind=config.edge_encoder,
        ffn=config.ffn,
        ffn_dropout=config.ffn_dropout,
        pos_enc=config.pos_enc,
        pos_enc_ksteps=config.pos_enc_ksteps,
    )
    supported = BACKBONE_KWARGS[backbone]
    return {k: v for k, v in full.items() if k in supported}


__all__ = ['GINEEncoder', 'GINELayer',
           'BACKBONES', 'BACKBONE_KWARGS', 'get_backbone',
           'backbone_init_kwargs']
