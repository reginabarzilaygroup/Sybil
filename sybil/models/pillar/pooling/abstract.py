from sybilx.models.pillar.abstract_model import AbstractModel


class AbstractPooling(AbstractModel):
    def forward(self, x, **extras):
        """
        Args:
            For 3D images: x: tensor of shape (B, C, T, H, W)
        Returns:
            output: dict with keys "hidden"
            For 3D images "hidden" shape is: [batch_size, C]
        """
        output = {"hidden": x}
        pass
