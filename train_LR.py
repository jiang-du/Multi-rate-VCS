import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import cv2
import numpy as np

use_cuda = True
device = torch.device("cuda:1" if (
    use_cuda and torch.cuda.is_available()) else "cpu")
path = "./DIV2K_HR/"
save_name = "MR_01c_res3_l1.pth"
num_epoch = 500
learning_rates = (1e-4, 2e-5)


class CSNet(torch.nn.Module):
    def __init__(self):
        super(CSNet, self).__init__()
        k_stride = 20   # for 1920x1080, 24 can be devided; for 720p, choose 20
        # MR = 0.01
        color_channel = 3
        mr = 12  # int(MR * 576) -- or 400; color *3
        self.conv0 = torch.nn.Conv2d(
            in_channels=color_channel, out_channels=mr, kernel_size=(2*k_stride), stride=k_stride, padding=k_stride)
        self.deconv0 = torch.nn.ConvTranspose2d(
            in_channels=mr, out_channels=color_channel, kernel_size=(2*k_stride), stride=k_stride, padding=k_stride)

        self.conv1_1 = torch.nn.Conv2d(
            in_channels=color_channel, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = torch.nn.Conv2d(
            in_channels=32, out_channels=color_channel, kernel_size=7, stride=1, padding=3)

        self.conv2_1 = torch.nn.Conv2d(
            in_channels=color_channel, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_3 = torch.nn.Conv2d(
            in_channels=32, out_channels=color_channel, kernel_size=7, stride=1, padding=3)

        self.conv3_1 = torch.nn.Conv2d(
            in_channels=color_channel, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv3_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3_3 = torch.nn.Conv2d(
            in_channels=32, out_channels=color_channel, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        measurement = self.conv0(x)
        y0 = self.deconv0(measurement)
        y = torch.nn.functional.relu(self.conv1_1(y0))
        y = torch.nn.functional.relu(self.conv1_2(y))
        y1 = y0 + self.conv1_3(y)

        y = torch.nn.functional.relu(self.conv2_1(y1))
        y = torch.nn.functional.relu(self.conv2_2(y))
        y2 = y1 + self.conv2_3(y)

        y = torch.nn.functional.relu(self.conv3_1(y2))
        y = torch.nn.functional.relu(self.conv3_2(y))
        y = y2 + self.conv3_3(y)
        return measurement, y  # y0, y1, y2, y


def main():
    model = CSNet().to(device)

    trans = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(720)])

    trainset = ImageFolder(path + "train", trans)
    valset = ImageFolder(path + "test", trans)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        valset, batch_size=2, shuffle=True)

    criterion = torch.nn.L1Loss()  # MSELoss or L1Loss

    # pre-train
    # model.load_state_dict(torch.load(save_name))

    # --------------------------------------------------
    # ----- step 1: preliminary -----
    # freeze params
    for i in range(3):
        for j in range(3):
            for para in eval("model.conv%d_%d.parameters()" % (i+1, j+1)):
                para.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rates[0])

    for epoch in range(num_epoch):  # num_epoch
        # ----- train -----
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y0, y1, y2, y = model(inputs)
            outputs = y0
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('Stage 0: Train -- Epoch=%3d, Iter=%3d, loss=%.5f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        # ----- validation -----
        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            y0, y1, y2, y = model(inputs)
            outputs = y0
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 7 == 6:    # print every epoch
                print('Stage 0: Validation -- Epoch=%3d, loss=%.5f' %
                      (epoch + 1, running_loss / 7))
                running_loss = 0.0
                rec = outputs[1, 0].detach().cpu().numpy()
                rec = rec * 255
                # cv2.imshow("test", rec)
                # cv2.waitKey(1)
                cv2.imwrite("train_show/stage0_%d.png" %
                            epoch, rec.astype(np.uint8))
        if epoch % 10 == 9:
            torch.save(model.state_dict(), save_name)
            print("Model saved for stage 0, epoch=%d." % (epoch + 1))

    # --------------------------------------------------
    # ----- step 2: res1 -----
    # freeze params
    for para in model.conv0.parameters():
        para.requires_grad = False
    for para in model.deconv0.parameters():
        para.requires_grad = False
    for i in range(3):
        for j in range(3):
            for para in eval("model.conv%d_%d.parameters()" % (i+1, j+1)):
                para.requires_grad = True if (i == 0) else False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rates[0])

    for epoch in range(num_epoch):  # num_epoch
        # ----- train -----
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y0, y1, y2, y = model(inputs)
            outputs = y1
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('Stage 1: Train -- Epoch=%3d, Iter=%3d, loss=%.5f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        # ----- validation -----
        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            y0, y1, y2, y = model(inputs)
            outputs = y1
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 7 == 6:    # print every epoch
                print('Stage 1: Validation -- Epoch=%3d, loss=%.5f' %
                      (epoch + 1, running_loss / 7))
                running_loss = 0.0
                rec = outputs[1, 0].detach().cpu().numpy()
                rec = rec * 255
                # cv2.imshow("test", rec)
                # cv2.waitKey(1)
                cv2.imwrite("train_show/stage1_%d.png" %
                            epoch, rec.astype(np.uint8))
        if epoch % 10 == 9:
            torch.save(model.state_dict(), save_name)
            print("Model saved for stage 1, epoch=%d." % (epoch + 1))

    # --------------------------------------------------
    # ----- step 3: res2 -----
    # freeze params
    for i in range(3):
        for j in range(3):
            for para in eval("model.conv%d_%d.parameters()" % (i+1, j+1)):
                para.requires_grad = True if (i == 1) else False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rates[0])

    for epoch in range(num_epoch):  # num_epoch
        # ----- train -----
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y0, y1, y2, y = model(inputs)
            outputs = y2
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('Stage 2: Train -- Epoch=%3d, Iter=%3d, loss=%.5f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        # ----- validation -----
        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            y0, y1, y2, y = model(inputs)
            outputs = y2
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 7 == 6:    # print every epoch
                print('Stage 2: Validation -- Epoch=%3d, loss=%.5f' %
                      (epoch + 1, running_loss / 7))
                running_loss = 0.0
                rec = outputs[1, 0].detach().cpu().numpy()
                rec = rec * 255
                # cv2.imshow("test", rec)
                # cv2.waitKey(1)
                cv2.imwrite("train_show/stage2_%d.png" %
                            epoch, rec.astype(np.uint8))
        if epoch % 10 == 9:
            torch.save(model.state_dict(), save_name)
            print("Model saved for stage 2, epoch=%d." % (epoch + 1))

    # --------------------------------------------------
    # ----- step 4: rse3 -----
    # freeze params
    for i in range(3):
        for j in range(3):
            for para in eval("model.conv%d_%d.parameters()" % (i+1, j+1)):
                para.requires_grad = True if (i == 2) else False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rates[0])

    for epoch in range(num_epoch):  # num_epoch
        # ----- train -----
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y0, y1, y2, y = model(inputs)
            outputs = y
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('Stage 3: Train -- Epoch=%3d, Iter=%3d, loss=%.5f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        # ----- validation -----
        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            y0, y1, y2, y = model(inputs)
            outputs = y
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 7 == 6:    # print every epoch
                print('Stage 3: Validation -- Epoch=%3d, loss=%.5f' %
                      (epoch + 1, running_loss / 7))
                running_loss = 0.0
                rec = outputs[1, 0].detach().cpu().numpy()
                rec = rec * 255
                # cv2.imshow("test", rec)
                # cv2.waitKey(1)
                cv2.imwrite("train_show/stage3_%d.png" %
                            epoch, rec.astype(np.uint8))
        if epoch % 10 == 9:
            torch.save(model.state_dict(), save_name)
            print("Model saved for stage 3, epoch=%d." % (epoch + 1))

    # --------------------------------------------------
    # ----- step 5: fine tune -----
    # freeze params
    for para in model.parameters():
        para.requires_grad = True
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rates[1])

    for epoch in range(num_epoch):  # num_epoch
        # ----- train -----
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y0, y1, y2, y = model(inputs)
            outputs = y
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('Stage 4: Fine tune -- Epoch=%3d, Iter=%3d, loss=%.5f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        # ----- validation -----
        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            y0, y1, y2, y = model(inputs)
            outputs = y
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 7 == 6:    # print every epoch
                print('Stage 4: Validation -- Epoch=%3d, loss=%.5f' %
                      (epoch + 1, running_loss / 7))
                running_loss = 0.0
                rec = outputs[1, 0].detach().cpu().numpy()
                rec = rec * 255
                # cv2.imshow("test", rec)
                # cv2.waitKey(1)
                cv2.imwrite("train_show/stage4_%d.png" %
                            epoch, rec.astype(np.uint8))
        if epoch % 10 == 9:
            torch.save(model.state_dict(), save_name)
            print("Model saved for stage 4, epoch=%d." % (epoch + 1))

    torch.save(model.state_dict(), save_name)
    print('Finished Training.')


if __name__ == '__main__':
    main()
