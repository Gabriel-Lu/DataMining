class DEDG_AugMix(ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(DEDG_AugMix, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, device)
        self.hparams = hparams
        self.class_balance=hparams['class_balanced']
        self.iteration = 0
        self.id_featurizer = self.featurizer
        self.dis_id = self.classifier
        self.gen = networks.AdaINGen(1, self.id_featurizer.n_outputs, hparams) if not hparams['is_mnist'] else networks.VAEGen()
        self.dis_img = networks.MsImageDis(hparams=hparams) 
        
        def to_gray(half=False): #simple
            def forward(x):
                x = torch.mean(x, dim=1, keepdim=True)
                if half:
                    x = x.half()
                return x
            return forward
        self.single = to_gray(False)
        self.optimizer_gen = torch.optim.Adam([p for p in list(self.gen.parameters())  if p.requires_grad], lr=self.hparams['lr_g'], betas=(0, 0.999), weight_decay=self.hparams['weight_decay_g'])
        self.jsd = True

        self.id_criterion = nn.CrossEntropyLoss()
        

    def recon_criterion(self, input, target):
            diff = input - target.detach()
            return torch.mean(torch.abs(diff[:]))
    
    def train_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
            print('there has bn')

    def forward(self, x_a, x_b, xp_a, xp_b):
        '''
            inpus:
                x_a, x_b: image from dataloader a,b
                xp_a, xp_b: positive pair of x_a, x_b
        '''
        b = xp_a.shape[0]
        s_a = self.gen.encode(self.single(x_a))# v for x_a
        s_b = self.gen.encode(self.single(x_b))# v for x_b
        f_a, x_fa = self.id_featurizer(x_a) # f_a: detached s for x_a, x_fa: s for x_a
        p_a = self.dis_id(x_fa)             # identity classification result for x_a
        f_b, x_fb = self.id_featurizer(x_b)
        p_b = self.dis_id(x_fb)
        fp_a, xp_fa = self.id_featurizer(xp_a)
        pp_a = self.dis_id(xp_fa)
        fp_b, xp_fb = self.id_featurizer(xp_b)
        pp_b = self.dis_id(xp_fb)
        # self-reconstruction
        x_a_recon = self.gen.decode(s_a[:b], f_a[:b]) # generate from identity and style of a
        x_b_recon = self.gen.decode(s_b[:b], f_b[:b])
        # style-preserving reconstructi
        x_a_recon_p = self.gen.decode(s_a[:b], fp_a) # generate from identity of p_a and styld of a
        x_b_recon_p = self.gen.decode(s_b[:b], fp_b)

        return x_fa, x_fb, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p    
 

    def gen_update(self, xf_a, xf_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, x_a, x_b,  l_a, l_b, hparams):
        '''
            inputs:
                x_ab: generated from identity of b and style of a
                x_ba: generated from identity of a and style of b
                s_a, s_b: style factors for x_a, x_b
                f_a, f_b: detached semantic factors for x_a, x_b
                p_a, p_b: identity prediction results for x_a, x_b
                pp_a, pp_b: identity prediction results for the positive pair of x_a, x_b
                x_a_recon, x_b_recon: reconstruction of x_a, x_b
                x_a_recon_p, x_b_recon_p: reconstruction of the positive pair of x_a, x_b
                x_a, x_b,  l_a, l_b: images and identity labels
                hparams: parameters
        '''
        b = x_a_recon.shape[0]
        self.optimizer_gen.zero_grad()
        self.optimizer.zero_grad()

        #################################
        # auto-encoder image reconstruction
        self.loss_gen_recon_x = self.recon_criterion(x_a_recon, x_a[:b])+self.recon_criterion(x_b_recon, x_b[:b]) + self.recon_criterion(x_a_recon_p, x_a[:b])+ self.recon_criterion(x_b_recon_p, x_b[:b])

        # Emprical Loss
        self.loss_id = self.id_criterion(p_a[:b], l_a[:b]) + self.id_criterion(p_b[:b], l_b[:b]) +  self.id_criterion(pp_a, l_a[:b]) + self.id_criterion(pp_b, l_b[:b])
        self.loss_aug_id =  self.id_criterion(p_a[b:], l_a[b:]) + self.id_criterion(p_b[b:], l_b[b:])

        if self.jsd:
            logits_clean, logits_aug1, logits_aug2 = torch.split(xf_a, b)
            logits_clean_b, logits_aug1_b, logits_aug2_b = torch.split(xf_b, b)
            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                        logits_aug2, dim=1)
            p_clean_b, p_aug1_b, p_aug2_b = F.softmax(
                logits_clean_b, dim=1), F.softmax(
                    logits_aug1_b, dim=1), F.softmax(
                        logits_aug2_b, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            p_mixture_b = torch.clamp((p_clean_b + p_aug1_b + p_aug2_b) / 3., 1e-7, 1).log()
            self.jsd_loss = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean') +
                            F.kl_div(p_mixture_b, p_clean_b, reduction='batchmean') +
                            F.kl_div(p_mixture_b, p_aug1_b, reduction='batchmean') +
                            F.kl_div(p_mixture_b, p_aug2_b, reduction='batchmean')) / 3.
        else:
            self.jsd_loss = torch.tensor(0)
        # total loss
        self.loss_gen_total = self.loss_id + \
                    hparams['recon_x_w'] * self.loss_gen_recon_x + \
                    hparams['recon_id_w'] * self.loss_aug_id +\
                    self.jsd_loss

        self.loss_gen_total.backward()
        self.optimizer_gen.step()
        self.optimizer.step()

    def update(self, minibatches, minibatches_neg, pretrain_model=None, unlabeled=None, iteration=0):
        images_a = torch.cat([x for x, y, pos in minibatches])
        labels_a = torch.cat([y for x, y, pos in minibatches])
        pos_a_ = torch.cat([pos for x, y, pos in minibatches])
        images_b = torch.cat([x for x, y, pos in minibatches_neg])
        labels_b = torch.cat([y for x, y, pos in minibatches_neg])
        pos_b_ = torch.cat([pos for x, y, pos in minibatches_neg])
        device = "cuda" if minibatches[0][0].is_cuda else "cpu" 
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.preprocess = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        self.TO_pil = transforms.ToPILImage()
        if self.jsd:
            for image_a, image_b, label_a, label_b in zip(images_a, images_b, labels_a, labels_b):
                image_a, x_ba1, x_ba2= Augmix(self.TO_pil(image_b.cpu()), self.preprocess, no_jsd=not self.jsd)
                image_b, x_ab1, x_ab2= Augmix(self.TO_pil(image_a.cpu()), self.preprocess, no_jsd=not self.jsd)
                images_a = torch.cat([images_a, x_ba1.to(device).unsqueeze(0), x_ba2.to(device).unsqueeze(0)], dim=0)
                images_b = torch.cat([images_b, x_ab1.to(device).unsqueeze(0), x_ab2.to(device).unsqueeze(0)], dim=0)
                labels_a = torch.cat([labels_a, label_a.unsqueeze(0), label_a.unsqueeze(0)], 0)
                labels_b = torch.cat([labels_b, label_b.unsqueeze(0), label_b.unsqueeze(0)], 0)
        else:
            images_b = Augmix(images_b, self.preprocess, no_jsd=not self.jsd)
            images_a = Augmix(images_a, self.preprocess, no_jsd=not self.jsd)
        
        pos_a, pos_b = [], []
        for image_a, image_b in zip(pos_a_, pos_b_):
            x_pos = Augmix(self.TO_pil(image_b.cpu()), self.preprocess, no_jsd=True)
            x_pos_b = Augmix(self.TO_pil(image_a.cpu()), self.preprocess, no_jsd=True)
            pos_a.append(x_pos.to(device).unsqueeze(0))
            pos_b.append(x_pos_b.to(device).unsqueeze(0))
    
        pos_a, pos_b =torch.cat(pos_a), torch.cat(pos_b)
        xf_a, xf_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p = self.forward(images_a, images_b, pos_a, pos_b)

        self.gen_update(xf_a, xf_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, images_a, images_b, labels_a, labels_b, self.hparams)
        return {'loss_total': self.loss_gen_total.item(), 
                'loss_cls': self.loss_id.item(), 
                'loss_cls_aug': self.loss_aug_id.item(), 
                'loss_recon_x': self.loss_gen_recon_x.item(),
                'loss_jsd': self.jsd_loss.item()}
                    
    def sample(self, x_a, x_b, pretrain_model=None):
        device = "cuda" if x_a.is_cuda else "cpu" 
        x_as, x_bs, x_a_aug, x_b_aug, x_a_aug1, x_b_aug1 = [], [], [], [], [], []
        for image_a, image_b in zip(x_a, x_b):
            x_b_, x_ab1, x_ab2= Augmix(self.TO_pil(image_b.cpu()), self.preprocess, no_jsd=not self.jsd)
            x_a_, x_ba1, x_ba2= Augmix(self.TO_pil(image_a.cpu()), self.preprocess, no_jsd=not self.jsd)
            x_a_aug.append(x_ba1.to(device).unsqueeze(0)); x_a_aug1.append(x_ba2.to(device).unsqueeze(0))
            x_b_aug.append(x_ab1.to(device).unsqueeze(0)); x_b_aug1.append(x_ab2.to(device).unsqueeze(0))
            x_as.append(x_a_.to(device).unsqueeze(0)); x_bs.append(x_b_.to(device).unsqueeze(0))
        x_a_aug, x_a_aug1=torch.cat(x_a_aug), torch.cat(x_a_aug1)
        x_b_aug, x_b_aug1 = torch.cat(x_b_aug), torch.cat(x_b_aug1)
        x_as, x_bs = torch.cat(x_as), torch.cat(x_bs)
        return x_as, x_a_aug, x_a_aug1, x_bs, x_b_aug, x_b_aug1

    def predict(self, x):
        return self.dis_id(self.id_featurizer(x)[-1])

    def step(self):
        self.iteration += 1
