import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

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
        return (1 - x), (input - input * x)

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
        self.MLP = nn.Sequential(
            nn.Linear(out_channels + genderSize, 1024),
            # nn.Linear(out_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # 3_20改，将结果输出为一个长为230的向量，而不是一个单独的数字
            # nn.Linear(512, 1).
            nn.Linear(512, 230),
            nn.BatchNorm1d(230),
            nn.ReLU(),
            nn.Linear(230, 1)
            # nn.Softmax()
        )


    # 前馈函数，需要输入一个图片，以及性别，不仅需要输出feature map，还需要加入MLP输出分类结果
    def forward(self, image, gender):
    # def forward(self, image):
        # 第一步：用主干网络生成feature_map
        AM1, x = self.MMCA1(self.backbone1(image))
        AM2, x = self.MMCA2(self.backbone2(x))
        AM3, x = self.MMCA3(self.backbone3(x))
        AM4, x = self.MMCA4(self.backbone4(x))
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
        output_beforeGA = self.MLP(x)

        # return AM1, AM2, AM3, AM4, feature_map, texture, gender_encode, output_beforeGA
        return output_beforeGA
    # 加入微调函数
    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)
# GA模块。目的是对主干网络学习到的feature_map，经过重映射的方式，学习不同层之间的上下文关系
# class GA(nn.Module):
#
#     def __init__(self) -> None:
#         super().__init__()
#
#     def forward(self, x, out_channels):
#         x1 = F.adaptive_avg_pool2d(x, 1)
#         x2 = torch.squeeze(x1)
#         x3 = x2.view(-1, out_channels)
#
#         return x1, x2, x3

class GA_attention(nn.Module):
    """主要目标是生成Q矩阵和K矩阵， 然后和传入的feature_map进行矩阵相乘，得到真正的Q和K,QK相乘得到注意力图A
    需要注意的是，在传入feature_map前，对其进行了reshape，使N(2048)(H/32)(W/32) -> N(2048)(H*W/1024)
    所以feature_channels指的是通道数，attention_size指的是每个通道下的面积大小
    公式中是X的转置乘上矩阵Q(或K)，所以实际目的是重映射x的通道数"""
    def __init__(self, feature_channels, attention_size) -> None:
        super().__init__()
        # 首先是生成Q矩阵和K矩阵，先随机生成矩阵，再经过凯明初始化
        # torch.empty() 创建任意数据类型的张量，括号内是维度
        self.Qmatrix = nn.Parameter(torch.empty(feature_channels, attention_size))
        nn.init.kaiming_uniform_(self.Qmatrix)

        self.Kmatrix = nn.Parameter(torch.empty(feature_channels, attention_size))
        nn.init.kaiming_uniform_(self.Kmatrix)

        # 定义激活函数leakyReLU
        self.leakyReLU = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 先将x转置，再与Q、K相乘，然后再将得到的两个矩阵再进行相乘，就能得到注意力图
        # x.shape = N(2048)(H*W/1024)，转置只转后两个维度
        x = x.transpose(1, 2)
        Q = self.leakyReLU(torch.matmul(x, self.Qmatrix))
        K = self.leakyReLU(torch.matmul(x, self.Kmatrix))
        return self.softmax(torch.matmul(Q, K.transpose(1, 2)))

class GA(nn.Module):
    """"""
    # 目的是将输入的注意力图A与feature_map作矩阵乘法，学习不同通道之间的上下文关系，最后经过一个可学习的权重矩阵W
    # 最后是输出contextual部分
    # 2.21 小trick，在放入MMANet一起训练之前，经过一个MLP层得到另外输出，与经过MMANet训练后的结果相加除二作为最终输出
    # 此外，还将训练好的MMCA作为backbone输入其中，并将参数冻结，同样在MMANet中也将GA这么做，如此一来训练的难度有所下降吧？
    def __init__(self, attention_size, feature_channels, output_channels, backbone) -> None:
        super().__init__()
        self.attention_size = attention_size
        self.feature_channels = feature_channels
        self.output_channels = output_channels
        #
        # 先从获得注意力图
        self.get_attention_map = GA_attention(feature_channels, attention_size)
        # 创建可学习的权重矩阵W
        self.weight = nn.Parameter(torch.empty(feature_channels, output_channels))
        nn.init.kaiming_uniform_(self.weight)

        self.leaky_ReLU = nn.LeakyReLU()

        # 2.21 将MMCA作为主干网络输入
        self.backbone = backbone
        # 冻结其参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2.21修改新增MLP层，使该层直接输出与性别的结果，作为下一个模块的正则项
        self.Linear = nn.Linear(1024+32, 1024)
        self.BN = nn.BatchNorm1d(1024)
        # self.output = nn.Linear(1024, 230)
        self.output = nn.Linear(1024, 1)

    def forward(self, data, gender): # 注意力公式 A*x的转置
        # 2.21 将数据放入训练好的MMCA中生成feature_map，texture，gender_encode，output_beforeGA
        AM1, AM2, AM3, AM4, feature_map, texture, gender_encode, output_beforeGA = self.backbone(data, gender)
        # 先对feature_map进行处理
        x = feature_map.view(-1, self.feature_channels, self.attention_size)
        A = self.get_attention_map(x)
        x = torch.matmul(A, x.transpose(1, 2))
        x = (torch.matmul(x, self.weight)).transpose(1, 2)
        x = self.leaky_ReLU(x)
        x = torch.squeeze(F.adaptive_avg_pool1d(x, 1))
        # 记录contextual，contextual.shape = N(1024)
        contextual = x

        # 2.21与性别cat一起，输出该层的结果
        x = torch.cat([x, gender_encode], dim=1)
        x = F.relu(self.BN(self.Linear(x)))
        output_afterGA = self.output(x)

        # 2.21 不仅输出上下文关系，还输出之前的所有信息，以及经过MLP的结果作为正则项
        return contextual, texture, gender_encode, (output_afterGA, output_beforeGA), A

    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)
        self.backbone.eval()

# 到此为止，三者都已经具备了，由MMANet生成的texture， gender_encode， 还有GA生成的contextual
# 接下来就是将以上三者concat在一起，经过MLP输出最终结果了。
# 顺序就是(
#      ResNet50
#         ↓
#   MMANet_beforeGA
#         ↓
#       the GA
#         ↓
#      MMANet
# )
class MMANet(nn.Module):

    def __init__(self, backbone) -> None:
        super().__init__()
        # MMANet的前部分
        self.backbone = backbone
        # 将其梯度冻结
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # 定义线性层MLP

        self.MLP = nn.Sequential(
            nn.Linear(1024 + 2048 + 32, 512), # contextual+texture+gender_encode
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Linear(512, 230)
            nn.Linear(512, 1)
        )

    def forward(self, image, gender):
        # 先将图片输入到MMANet中得到contextual, texture, gender_encode, output
        contextual, texture, gender_encode, output, A = self.backbone(image, gender)
        # 将output作为正则项加入到其中
        # return contextual, texture, gender_encode, (self.MLP(torch.concat([contextual, texture, gender_encode], dim=1)) + output[0])/2
        return (self.MLP(torch.concat([contextual, texture, gender_encode], dim=1)) + output[0]) / 2, A

    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)
        self.backbone.fine_tune(need_fine_tune=need_fine_tune)
        # self.backbone.eval()

if __name__ == '__main__':
    # model, out_channels = get_ResNet()
    # mymodel = nn.Sequential(*model)
    # MMCA = MMCA_module(256)
    # my_MMANet = MMCANet_BeforeGA(32, *get_ResNet())
    # GA_model = GA_attention(2048, 256)
    # GA_model = GA_model.to(device=try_gpu())
    #   输入格式为 BxCxHxW
    x = torch.rand((10, 3, 512, 512))
    x = x.to(device=try_gpu())
    # my_MMANet = my_MMANet.to(device=try_gpu())
    # print(MMCA)
    # print(MMCA(x).shape)
    # print(*model[5:])
    gender = torch.ones((10, 1)).cuda()
    # feature_map, texture, gender_encode = my_MMANet(x, gender)
    # print(feature_map.shape, texture.shape, gender_encode.shape)
    # print(GA_model)
    # node_feature = feature_map.view(-1, 2048, 16 * 16)
    # print(node_feature.shape)
    # print(GA_model(node_feature).shape)
    # GA_output = GA(16*16, 2048, 1024)
    # GA_output.to(device=try_gpu())
    # print(GA_output(feature_map).shape)
    # print(mymodel(x).shape)

    MMANet = MMANet()
    MMANet = MMANet.to(device=try_gpu())
    print(sum(p.numel() for p in MMANet.parameters()))
    # print(MMANet(x, gender))