import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=4, padding=1)
		self.bn1=nn.BatchNorm2d(ndf)
		self.conv3 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*2, ndf*2, kernel_size=4, padding=1)
		self.bn2=nn.BatchNorm2d(ndf*2)
		self.conv5 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv6 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, padding=1)
		self.bn3=nn.BatchNorm2d(ndf*4)
		self.conv7 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv8 = nn.Conv2d(ndf*8, ndf*8, kernel_size=4, padding=1)
		self.bn4=nn.BatchNorm2d(ndf*8)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.bn1(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.bn2(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		x = self.conv6(x)
		x = self.bn3(x)
		x = self.leaky_relu(x)

		x = self.conv7(x)
		x = self.leaky_relu(x)
		x = self.conv8(x)
		x = self.bn4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x
