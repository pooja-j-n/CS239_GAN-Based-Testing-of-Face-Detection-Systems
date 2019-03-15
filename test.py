"""
Get Final Test Image
"""
import torch
from torch.autograd import Variable
import torchvision.utils
import os
import argparse
import pdb
import model
import dataloader


def save_result(result_path, G_MF, G_FM, img, label, nrow, num, num_img):
    if os.path.exists(result_path) is False:
        os.mkdir(result_path)
        
    full_img_name = result_path + '/' + num + '.png'
    full_img_name_2 = result_path + '/' + "transformed/" + str(num_img) + '.png'
    # img = Variable with size of (N, 3, 128, 128)
    N, C, H, W = img.size()
    result_img = torch.FloatTensor(2*N, C, H, W)


    for i in range(N):
        tranformed_img = torch.FloatTensor(1, C, H, W)
        ori_img = img.data[i].unsqueeze(0)
        result_img[2 * i] = ori_img
        
        ori_img = Variable(ori_img)
        
        if label[i] == 0:
            fake_img = G_MF(ori_img)
        else:
            fake_img = G_FM(ori_img)   
        
        result_img[(2 * i) + 1] = fake_img.data.cpu()

        tranformed_img[0] = fake_img.data.cpu()



        original_img = torch.FloatTensor(1, C, H, W)
        original_img[0] = ori_img
        torchvision.utils.save_image(original_img, "result/newresult/" + str(num_img) + "0_original.png", normalize=True, scale_each=True)

        torchvision.utils.save_image(fake_img, "result/newresult/" + str(num_img) + "1_transformed.png", normalize=True, scale_each=True)
        num_img = num_img+1

    torchvision.utils.save_image(result_img, full_img_name, nrow=nrow, padding=4,
                                 normalize=True, scale_each=True)

def main(args):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    dloader, data_num = dataloader.getDataLoader(args.root_dir, args.sub_dir, batch=args.batch_size, 
                                                 shuffle=False)

    G_MF = model.Generator().type(dtype)
    G_FM = model.Generator().type(dtype)

    weight_loc = 'epoch_' + str(args.epoch)
    G_MF.load_state_dict(torch.load('pretrained/' + weight_loc + '/G_MF.pkl'))
    G_FM.load_state_dict(torch.load('pretrained/' + weight_loc + '/G_FM.pkl'))

    if data_num % args.batch_size == 0:
        total_num = data_num / args.batch_size
    else:
        total_num = data_num / args.batch_size + 1
    count =0
    for idx, (img, label) in enumerate(dloader):

        print('Processing : [%d / %d]' %(idx + 1, total_num))
        img = img.type(dtype)
        label = label.type(dtype)
        
        img = Variable(img)
        save_result(args.result_path, G_MF, G_FM, img, label, args.nrow, str(idx),count)
        count = count+1
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='data',
                        help='Root directory of input images')
    parser.add_argument('--sub_dir', type=str, default='test',
                        help='Subdirectory of images. ex) data/test/0/male_images.png, then subdir is test')
    parser.add_argument('--result_path', type=str, default='result',
                        help='Path of generated images')
    parser.add_argument('--epoch', type=int, default=20, 
                        help='Generating images with weight of epoch X')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Test batch size')
    parser.add_argument('--nrow', type=int, default=16,
                        help='How many images in a row')
    args = parser.parse_args()
    main(args)
    
