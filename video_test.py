import torch
import cv2
import numpy as np
import json

from train_LR import CSNet, save_name
from train_HR import CSNet as HmrCSNet
from train_HR import save_name as hr_save_name

use_cuda = True
device = torch.device("cuda:1" if (
    use_cuda and torch.cuda.is_available()) else "cpu")
dir_name = "./video_1.mp4"


def psnr(A, B):
    mse = np.sum(np.square(A.flatten() - B.flatten())) / np.size(A)
    p = 10*np.log(1/mse)/np.log(10)
    return p, mse


def main():
    model = CSNet().to(device)
    model.load_state_dict(torch.load(
        save_name, map_location=torch.device(device)))
    model.eval()
    model_H = HmrCSNet().to(device)
    model_H.load_state_dict(torch.load(
        hr_save_name, map_location=torch.device(device)))
    model_H.eval()

    # statistics
    total_p = .0
    total_m = .0
    total_c = 0
    m0 = None
    cap = cv2.VideoCapture(dir_name)
    # video writer
    #writer = cv2.VideoWriter(filename="vis.mp4", fourcc=cv2.VideoWriter_fourcc(
    #    *'x264'), fps=25, frameSize=(1280, 720), isColor=True)
    lib_psnr = list()
    lib_mr = list()
    lib_roi_psnr = list()
    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        # size of frame is assumed 1920x1080x3
        # convert to 720p
        frame = cv2.resize(frame, (1280, 720))
        I = frame.astype(np.float32)/255
        # convert to 1*3*w*h (channel first)
        test_in = torch.from_numpy(I).permute(
            (2, 0, 1)).unsqueeze(0).to(device)
        m, y0 = model(test_in)
        if m0 is None:
            # first frame: just capture
            m0 = m[0].detach()
            continue

        # judge difference
        delta = torch.abs(m[0].detach() - m0)
        # print(m0.shape) # 12x37x65
        delta = torch.sum(delta, dim=0) / 12
        print(torch.max(delta))
        threshold = 0.3
        # delta = torch.nn.functional.relu(delta - threshold) + threshold
        delta = delta.cpu().numpy()
        # show delta and count HR rate
        num_hr = 0
        for i in range(37):
            for j in range(65):
                if delta[i, j]<threshold:
                    delta[i, j] = 0
                else:
                    delta[i, j] = 1
                    num_hr += 1
        print(num_hr/(36.0*64.0))
        
        # erode and dilate (denoise)
        kernel = np.ones((2, 2))
        delta = cv2.erode(delta, kernel, iterations=1)
        kernel = np.ones((3, 3))
        delta = cv2.dilate(delta, kernel, iterations=2)
        # delta = cv2.dilate(delta, kernel, iterations=1)
        mr = 0.01 + 0.2*np.sum(delta)/(36.0*64)
        print("mr=%f"%mr)
        lib_mr.append(mr)

        # apply mask
        # delta = delta.astype(np.uint8)
        delta = cv2.resize(delta, (1280+20, 720+20), interpolation=cv2.INTER_NEAREST)#, interpolation=cv2.INTER_NEAREST)
        delta = delta[10:-10, 20:]
        delta = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
        hr_input = delta * frame

        # process hr
        hr_input = hr_input.astype(np.float32)/255
        test_in = torch.from_numpy(hr_input).permute(
            (2, 0, 1)).unsqueeze(0).to(device)
        y1 = model_H(test_in)

        # combine
        lr_rec = y0[0].permute((1, 2, 0)).detach().cpu().numpy()
        hr_rec = y1[0].permute((1, 2, 0)).detach().cpu().numpy()
        # refine the mask
        delta = cv2.resize(delta, (128, 72))
        kernel = np.ones((4, 4))
        delta = cv2.erode(delta, kernel, iterations=1)
        delta = cv2.resize(delta, (1280, 720))
        total_img = delta * hr_rec + (1-delta) * lr_rec
        
        # low resolution
        Psnr, Mse = psnr(I, total_img)
        print("%s, PNSR=%.4f, MSE=%.6f" % (total_c, Psnr, Mse))
        total_p += Psnr
        total_m += Mse

        lib_psnr.append(Psnr)
        if mr>0.0100000001:
            Psnr, Mse = psnr(delta*I, delta*total_img)
            print("%s, ROI PNSR=%.4f, MSE=%.6f" % (total_c, Psnr, Mse))
        else:
            Psnr = 0
        lib_roi_psnr.append(Psnr)

        # for visualization
        """
        lr_rec[lr_rec>1]=1
        lr_rec[lr_rec<0]=0
        hr_rec[hr_rec>1]=1
        hr_rec[hr_rec<0]=0
        delta[delta>1]=1
        delta[delta<0]=0
        lr_rec = (lr_rec * 255).astype(np.uint8)
        hr_rec = (hr_rec * 255).astype(np.uint8)
        delta = (delta * 255).astype(np.uint8)
        cv2.imwrite("lr.png", lr_rec)
        cv2.imwrite("hr.png", hr_rec)
        cv2.imwrite("mask.png", delta)
        """
        
        # visualize large image
        total_img[total_img>1]=1
        total_img[total_img<0]=0
        total_img = total_img * 255
        total_img = total_img.astype(np.uint8)
        #writer.write(total_img)
        total_img = cv2.putText(total_img, "Video 1, frame "+str(total_c), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite("results/"+str(total_c)+".png", total_img)
        cv2.imshow("test", total_img)
        cv2.waitKey(10)
        total_c += 1

    total_p /= total_c
    total_m /= total_c
    #writer.release()
    cap.release()
    with open('vid1.json', 'w') as f:
        json.dump((lib_psnr, lib_mr, lib_roi_psnr), f)
    print("Average PSNR=%.4f, MSE=%.6f" % (total_p, total_m))


if __name__ == '__main__':
    main()
