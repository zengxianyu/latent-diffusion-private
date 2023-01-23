import torch
import pdb
import torch.nn.functional as F
import torchvision

class CLSLoss(torch.nn.Module):
    def __init__(self, weights_path, num_bit=1):
        super().__init__()
        print("create classifier")
        net_cls = torchvision.models.resnet34(pretrained=False)
        net_cls.fc = torch.nn.Linear(net_cls.fc.in_features,num_bit)
        torch.nn.init.kaiming_normal_(net_cls.fc.weight)
        net_cls.load_state_dict(torch.load(weights_path))
        net_cls.eval()
        self.net = net_cls
        target = torch.Tensor([1])
        self.register_buffer("target", target, persistent=False)

    def forward(self,img, y_in=None):
        bsize = img.size(0)
        #img = (img+1)/2
        #img = (img-self.meanv)/self.stdv
        logit = self.net(img)
        target = self.target.expand_as(logit) if y_in is None else y_in
        loss = F.binary_cross_entropy_with_logits(logit, target, reduce=False)
        return torch.squeeze(loss)
