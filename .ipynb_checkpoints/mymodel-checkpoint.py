import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

def get_ResNet():
    """获得主干网络ResNet50"""
    model = resnet50(pretrained=True)
    # 设置模型的输出通道,fc为ResNet中的最后一层，它的in_features即为输出的类别，就是输出通道，为2048
    output_channels = model.fc.in_features
    #   将网络中的所有子网络放入sequential，然后除去ResNet中最后的池化层和线性层，只保留了主干网络和前面的一些网络
    #   list(model.children())[:-2]的输出如下
    # [Conv2d(3, 64),
    # BatchNorm2d(64),
    # ReLU(),
    # MaxPool2d(kernel_size=3, stride=2, padding=1),
    # Sequential(),
    # Sequential(),
    # Sequential(),
    # Sequential()]
    # 计划就是在sequential之间穿插自制的MMCA模块
    ##
    model = list(model.children())[:-2]
    return model, output_channels

#   从小模块开始做起，先是构建一个核大小自定义的卷积模块
class My_attention(nn.Module):
    """自定义核大小卷积核"""
    #   输入需要：输入通道，核大小（默认1）
    def __init__(self, input_channels, kernel_size=1) -> None:
        super().__init__()
        self.my_attention = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

    #   作为一个卷积模块，那么需要输入是必然的，所以forward函数必需要有输入数据集
    def forward(self, x):
        return self.my_attention(x)


class MMCA_module(nn.Module):
    """构建MMCA模块，MMCA包括DR和MRA，DR需要降维因子reduction[]，还有DR的层数level，故需要参数：输入通道，降维因子reduction，层数level
        注：MMCA并不改编通道数"""
    def __init__(self, input_channels, reduction=[16], level=1) -> None:
        super().__init__()
        #   先构建DR部分
        #   设置模块
        modules = []
        for i in range(level):  # DR的个数由level来定
            #   先是确定输出维度
            output_channels = input_channels // reduction[i]
            #   在modules里添加卷积层、BN曾、激活层
            modules.append(nn.Conv2d(input_channels, output_channels, kernel_size = 1))  # 默认1x1卷积
            modules.append(nn.BatchNorm2d(output_channels))
            modules.append(nn.ReLU())
            input_channels = output_channels

        # MRA层， 包括了三个卷积层，分别是1x1,3x3，5x5， 然后是ReLU，这个包括在了my_attention里面
        # 先将底层的DR包括进去
        self.DR = nn.Sequential(*modules)
        self.MRA1 = My_attention(input_channels, 1)
        # self.MRA1 = nn.Sequential(
        #     nn.Conv2d(input_channels, 1, kernel_size=1),
        #     nn.ReLU()
        # )
        self.MRA3 = My_attention(input_channels, 3)
        self.MRA5 = My_attention(input_channels, 5)
        # 三者Concat的操作放在forward里面，接下来就是卷积层+激活层，这里的激活函数用Sigmoid
        self.last_conv = nn.Sequential(
            #   三个my_attention的输出通道都是1，所以这里的通道数为3
            nn.Conv2d(3, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #   由于需要利用互补注意力公式：F*(1-A)，这里需要先存储一下输入F
        input = x
        #   显示进入DR层
        x = self.DR(x)
        #   将MRA1、MRA3、MRA5的输出cancat在一起
        x = torch.cat([self.MRA1(x), self.MRA3(x), self.MRA5(x)], dim=1)  # 在第二个维度Concat起来，也就是说，使[(a, b), (c,d)]->[(a, b, c, d)]

        x = self.last_conv(x)
        #   F*(1-A) = F - F*A
        return (1 - x), input * (1 - x)
        # return input - input * x

# 主要任务:生成输入GA模块的feature_map、将feature_map处理后的texture，对性别数据进行编码后的gender_encode
# 初始化参数：性别编码长度（文中用的32）， 主干网络，输出通道
# 2.21 漏了一个环节，由于同时训练contextual和texture的话，会极大地增加训练模型的难度，所以在这里先就对GA之前的模块训练
# 然后再训练GA
class MMANet_BeforeGA(nn.Module):
    """主模型MMANet的在输入到GA前的部分"""
    # 不在类内定义主干网络是因为怕梯度损失吗
    def __init__(self, genderSize, backbone, out_channels) -> None:
        super().__init__()
        # self.resnet50 = get_ResNet()
        # 共有四块MMCA，所以这里分成四块来写，每块的主干部分和MMCA分开
        # 注意点：resnet总共四个sequential，输出通道分别是256, 512, 1024, 2048，这也确定MMCA的输入通道，但经过四层后高宽除以32
        # ResNet的前五层分别为：线性层conv2d，bn，ReLU，maxpooling，和第一个sequential
        self.out_channels = out_channels
        self.backbone1 = nn.Sequential(*backbone[0:5])
        self.MMCA1 = MMCA_module(256)
        self.backbone2 = backbone[5]
        self.MMCA2 = MMCA_module(512, reduction=[4, 8], level=2)
        self.backbone3 = backbone[6]
        self.MMCA3 = MMCA_module(1024, reduction=[8, 8], level=2)
        self.backbone4 = backbone[7]
        self.MMCA4 = MMCA_module(2048, reduction=[8, 16], level=2)
        # MMCA中的的降维因子的总乘积随着通道数的翻倍，也跟着翻倍，但为什么变成两个，或者为什么大的放后面，这就无从考究了

        # 性别编码
        self.gender_encoder = nn.Linear(1, genderSize)
        # 由于标签变成了独热，估改输入为2
        # self.gender_encoder = nn.Linear(2, genderSize)
        self.gender_BN = nn.BatchNorm1d(genderSize)

        # 2.21新增，在GA模块之前就对resnet+MMCA进行训练，所以这里就添加MLP层
        # self.MLP = nn.Sequential(
        #     nn.Linear(out_channels + genderSize, 1024),
        #     # nn.Linear(out_channels, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     # 3_20改，将结果输出为一个长为230的向量，而不是一个单独的数字
        #     nn.Linear(512, 1)
        #     # nn.Linear(512, 230),
        #     # nn.BatchNorm1d(230),
        #     # nn.ReLU(),
        #     # nn.Linear(230, 1)
        #     # nn.Softmax()
        self.FC0 = nn.Linear(out_channels + genderSize, 1024)
        self.BN0 = nn.BatchNorm1d(1024)

        self.FC1 = nn.Linear(1024, 512)
        self.BN1 = nn.BatchNorm1d(512)

        self.output = nn.Linear(512, 240)

    # 前馈函数，需要输入一个图片，以及性别，不仅需要输出feature map，还需要加入MLP输出分类结果
    def forward(self, image, gender):
    # # def forward(self, image):
        # 第一步：用主干网络生成feature_map
        AM1, x = self.MMCA1(self.backbone1(image))
        AM2, x = self.MMCA2(self.backbone2(x))
        AM3, x = self.MMCA3(self.backbone3(x))
        AM4, x = self.MMCA4(self.backbone4(x))
        # x = self.backbone1(image)
        # x = self.backbone2(x)
        # x = self.backbone3(x)
        # x = self.backbone4(x)
        # 由于MMCA不改变通道数，所以x的shape由原来的NCHW -> N(2048)(H/32)(W/32)
        feature_map = x

        # 第二步：将feature_map降维成texture，这里采用自适应平均池化
        x = F.adaptive_avg_pool2d(x, 1) # N(2048)(H/32)(W/32) -> N(2048)(1)(1)
        # 把后面两个1去除，用torch.squeeze
        x = torch.squeeze(x)
        # 调整x的形状，使dim=1=输出通道的大小
        x = x.view(-1, self.out_channels)
        texture = x

        # 第三步，对性别进行编码，获得gender_encode
        gender_encode = self.gender_encoder(gender)
        gender_encode = self.gender_BN(gender_encode)
        gender_encode = F.relu(gender_encode)
        # feature_map.shape=N(2048)(H/32)(W/32)
        # texture.shape = N(2048)
        # gender_encode.shape = N(32)

        # 2.21 第四步，为这一层的训练做准备，使texture+gender作为输入，放入MLP
        x = torch.cat([x, gender_encode], dim=1)
        # output_beforeGA = self.MLP(x)
        # 拆分MLP
        x = F.relu(self.BN0(self.FC0(x)))
        x = F.relu(self.BN1(self.FC1(x)))
        output_beforeGA = self.output(x)
        output_beforeGA = F.softmax(output_beforeGA)
        distribute = torch.arange(0, 240)
        output_beforeGA = (output_beforeGA*distribute).sum(dim=1)
        # return AM1, AM2, AM3, AM4, feature_map, texture, gender_encode, output_beforeGA
        # return AM1, AM2, AM3, AM4, output_beforeGA
        return output_beforeGA
    # 加入微调函数
    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)

class myres(nn.Module):
    """主模型MMANet的在输入到GA前的部分"""
    # 不在类内定义主干网络是因为怕梯度损失吗
    def __init__(self, genderSize, backbone, out_channels) -> None:
        super().__init__()
        # self.resnet50 = get_ResNet()
        # 共有四块MMCA，所以这里分成四块来写，每块的主干部分和MMCA分开
        # 注意点：resnet总共四个sequential，输出通道分别是256, 512, 1024, 2048，这也确定MMCA的输入通道，但经过四层后高宽除以32
        # ResNet的前五层分别为：线性层conv2d，bn，ReLU，maxpooling，和第一个sequential
        self.out_channels = out_channels
        self.backbone = nn.Sequential(*backbone)
        # MMCA中的的降维因子的总乘积随着通道数的翻倍，也跟着翻倍，但为什么变成两个，或者为什么大的放后面，这就无从考究了

        # 性别编码
        self.gender_encoder = nn.Linear(1, genderSize)
        # 由于标签变成了独热，估改输入为2
        # self.gender_encoder = nn.Linear(2, genderSize)
        self.gender_BN = nn.BatchNorm1d(genderSize)

        # 2.21新增，在GA模块之前就对resnet+MMCA进行训练，所以这里就添加MLP层
        # self.MLP = nn.Sequential(
        #     nn.Linear(out_channels + genderSize, 1024),
        #     # nn.Linear(out_channels, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     # 3_20改，将结果输出为一个长为230的向量，而不是一个单独的数字
        #     nn.Linear(512, 1)
        #     # nn.Linear(512, 230),
        #     # nn.BatchNorm1d(230),
        #     # nn.ReLU(),
        #     # nn.Linear(230, 1)
        #     # nn.Softmax()
        self.FC0 = nn.Linear(out_channels + genderSize, 1024)
        self.BN0 = nn.BatchNorm1d(1024)

        self.FC1 = nn.Linear(1024, 512)
        self.BN1 = nn.BatchNorm1d(512)

        # self.output = nn.Linear(512, 1)
        self.output = nn.Linear(512, 240)

    # 前馈函数，需要输入一个图片，以及性别，不仅需要输出feature map，还需要加入MLP输出分类结果
    def forward(self, image, gender, mean, div):
    # # def forward(self, image):
        # 第一步：用主干网络生成feature_map
        x = self.backbone(image)
        # x = self.backbone1(image)
        # x = self.backbone2(x)
        # x = self.backbone3(x)
        # x = self.backbone4(x)
        # 由于MMCA不改变通道数，所以x的shape由原来的NCHW -> N(2048)(H/32)(W/32)
        feature_map = x

        # 第二步：将feature_map降维成texture，这里采用自适应平均池化
        x = F.adaptive_avg_pool2d(x, 1) # N(2048)(H/32)(W/32) -> N(2048)(1)(1)
        # 把后面两个1去除，用torch.squeeze
        x = torch.squeeze(x)
        # 调整x的形状，使dim=1=输出通道的大小
        x = x.view(-1, self.out_channels)
        texture = x

        # 第三步，对性别进行编码，获得gender_encode
        gender_encode = self.gender_encoder(gender)
        gender_encode = self.gender_BN(gender_encode)
        gender_encode = F.relu(gender_encode)
        # feature_map.shape=N(2048)(H/32)(W/32)
        # texture.shape = N(2048)
        # gender_encode.shape = N(32)

        # 2.21 第四步，为这一层的训练做准备，使texture+gender作为输入，放入MLP
        x = torch.cat([x, gender_encode], dim=1)
        # output_beforeGA = self.MLP(x)
        # 拆分MLP
        x = F.relu(self.BN0(self.FC0(x)))
        x = F.relu(self.BN1(self.FC1(x)))
        output_beforeGA = self.output(x)

        output_beforeGA = F.softmax(output_beforeGA)
        distribute = ((torch.arange(0, 240)-mean)/div).cuda()
        output_beforeGA = (output_beforeGA*distribute).sum(dim=1)
        # return AM1, AM2, AM3, AM4, feature_map, texture, gender_encode, output_beforeGA
        # return AM1, AM2, AM3, AM4, output_beforeGA
        return output_beforeGA
    # 加入微调函数
    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)
