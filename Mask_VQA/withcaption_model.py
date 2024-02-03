import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import datetime
import torchvision

#deformable conv2d
class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator
                                          )
        return x

##################modify resnet for image encoder #####################

class Modify_Resnet(nn.Module):
    def __init__(self,embed_size):
        super(Modify_Resnet,self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.layer4.deform1= DeformableConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4.deform2= DeformableConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4.deform3 = DeformableConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = nn.Linear(512, embed_size)


    def forward(self,x):
        x = self.model(x)

        return x


#####img encoder for vqa ################3
class ImgEncoder(nn.Module):

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        self.model = Modify_Resnet(embed_size)

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
     #   img_feature = self.fc(img_feature)                   # [batch_size, embed_size]

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return img_feature

#########caption encoder ######################
class CaptionEncoder(nn.Module):
    def __init__(self,embed_size):
        super(CaptionEncoder, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True

        return self.dropout(self.relu(features))

class CaptionDecoder(nn.Module):
    def __init__(self,embed_size,word_embed_size,hidden_size,cap_vocab_size,num_layers):
        super(CaptionDecoder, self).__init__()
        self.word2vec = nn.Embedding(cap_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)  # 2 for hidden and cell states

    def forward(self, caption):
        cap_vec = self.word2vec(caption)  # [batch_size, max_qst_length=100, word_embed_size=300]
        cap_vec = self.tanh(cap_vec)
        cap_vec = cap_vec.transpose(0, 1)  # [max_qst_length=30, batch_size, word_embed_size=300]

        _, (hidden, cell) = self.lstm(cap_vec)  # [num_layers=2, batch_size, hidden_size=512]
        cap_feature = torch.cat((hidden, cell), 2)  # [num_layers=2, batch_size, 2*hidden_size=1024]
        cap_feature = cap_feature.transpose(0, 1)  # [batch_size, num_layers=2, 2*hidden_size=1024]
        cap_feature = cap_feature.reshape(cap_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        cap_feature = self.tanh(cap_feature)
        cap_feature = self.fc(cap_feature)  # [batch_size, embed_size]

        return cap_feature

############qst encoder for vqa###################3
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]

        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature

class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size,cap_vocab_size, num_layers, hidden_size):

        super(VqaModel, self).__init__()
        #self.CaptionEncoder = CaptionEncoder(embed_size)
        #self.CaptionDecoder = CaptionDecoder(embed_size, hidden_size, cap_vocab_size, num_layers)

        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.cap_encoder = CaptionDecoder(cap_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

        self.fc1_cap = nn.Linear(embed_size, cap_vocab_size)
        self.fc2_cap = nn.Linear(cap_vocab_size, cap_vocab_size)

    def forward(self, img, qst,cap):
        #
        # cap_img_feature = self.img_encoder(img)
        # caption_feature = self.qst_encoder(cap)
        # combined_feature = torch.mul(cap_img_feature, caption_feature)  # [batch_size, embed_size]
        # combined_feature = self.tanh(combined_feature)
        # combined_feature = self.dropout(combined_feature)
        # combined_feature = self.fc1_cap(combined_feature)  # [batch_size, ans_vocab_size=1000]
        # combined_feature = self.tanh(combined_feature)
        # combined_feature = self.dropout(combined_feature)
        # caption = self.fc2_cap(combined_feature)



        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return combined_feature

    def visualization_vqa(self,img,qst,vocab):

        result_answer = []

        img_feature = self.img_encoder(img.unsqueeze(0))  # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst.unsqueeze(0))  # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)  # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)  # [batch_size, ans_vocab_size=1000]
        predicted = combined_feature.argmax(1)

        result_answer.append(predicted.item())

        return [vocab.idx2word(idx) for idx in result_answer]

