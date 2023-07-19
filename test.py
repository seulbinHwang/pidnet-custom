class FullModel(nn.Module):

    def __init__(self, model, sem_loss, bd_loss):
        super(FullModel, self).__init__()
        self.model = model  # PIDNet
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss

    def pixel_acc(self, pred, label):
        """

        Args:
            pred: (batch_size, 2, 1024, 1024)
            label: [batch_size, height, width]

        Returns:
        """
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc