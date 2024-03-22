from gatys_utils import *
from matplotlib.pyplot import imshow


class Gatys:
    # pre and post processing for images
    img_size = 512
    prep = transforms.Compose([transforms.Resize((img_size, img_size)),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                               ])
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                      std=[1, 1, 1]),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                                 ])
    postpb = transforms.Compose([transforms.ToPILImage()])

    # define layers, loss functions, weights and compute optimization targets
    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
#    if torch.cuda.is_available():
#       loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    # these are good weights settings:
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    def postp(self, tensor):  # to clip results in the range [0,1]
        t = self.postpa(tensor)
        t[t > 1] = 1
        t[t < 0] = 0
        img = self.postpb(t)
        return img

    def __init__(self, weights='DEFAULT'):
        self.model = VGG()
        state_dict = get_VGG19_state_dict(self.model, weights=weights)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def transfer_style(self, content, style, max_iter=500, show_iter=50, opt_img=None):
        # preprocessing
        content_image = self.prep(content).unsqueeze(0).to(self.device)
        style_image = self.prep(style).unsqueeze(0).to(self.device)

        if opt_img is None:
            # random init
            opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True)
        elif opt_img == "content":
            opt_img = Variable(content_image.data.clone(), requires_grad=True)
        elif opt_img == "style":
            opt_img = Variable(style_image.data.clone(), requires_grad=True)
        else:
            opt_img = Variable(opt_img.to(self.device), requires_grad=True)

        # compute optimization targets
        style_targets = [GramMatrix()(A).detach() for A in self.model(style_image, self.style_layers)]
        content_targets = [A.detach() for A in self.model(content_image, self.content_layers)]
        targets = style_targets + content_targets

        # run style transfer
        optimizer = optim.LBFGS([opt_img])
        n_iter = [0]

        while n_iter[0] <= max_iter:

            def closure():
                optimizer.zero_grad()
                out = self.model(opt_img, self.loss_layers)
                layer_losses = [self.weights[a] * self.loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
                loss = sum(layer_losses)
                loss.backward()
                n_iter[0] += 1
                # print loss
                if show_iter > 0 and n_iter[0] % show_iter == (show_iter - 1):
                    print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
                return loss

            optimizer.step(closure)

        # display result
        return opt_img

    def get_content_from_image(self, image, pil=True):
        if pil:
            content_image = self.prep(image).unsqueeze(0).to(self.device)
        else:
            content_image = image.to(self.device)
        return [A.detach() for A in self.model(content_image, self.content_layers)]

    def get_style_from_image(self, image, pil=True):
        if pil:
            style_image = self.prep(image).unsqueeze(0).to(self.device)
        else:
            style_image = image.to(self.device)
        return [GramMatrix()(A).detach() for A in self.model(style_image, self.style_layers)]

    def postprocess(self, image):
        return self.postp(image.data[0].cpu().squeeze())

    def display_image(self, image, pil=False):
        if not pil:
            out_img = self.postprocess(image)
        else:
            out_img = image
        imshow(out_img)