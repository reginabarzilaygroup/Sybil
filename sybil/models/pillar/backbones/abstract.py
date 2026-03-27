from sybil.models.pillar.abstract_model import AbstractModel


class AbstractBackbone(AbstractModel):
    def forward(self, x, **extras):
        """
        Args:
            For 3D images: x: tensor of shape (B, C, T, H, W)
        Returns:
            output: dict with keys "activ"
            For 3D images "activ" shape is: [batch_size, backbone_hidden_dim, D', H', W']
        """
        output = {"activ": x}
        pass
