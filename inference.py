import random
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

from model import ResNet, IRBlock, ResNet_Final

def cos_sim(a, b):
    return F.cosine_similarity(a, b)

def infer():
    #초기화
    seed_num = 123456
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

    cuda = torch.cuda.is_available()

    device = torch.device('cuda' if cuda else 'cpu')

    #model 정의
    num_classes = 42711 
    res_model = ResNet(IRBlock, [3, 4, 6, 3], use_se=False, im_size=112)
    net = nn.Sequential(nn.Linear(512, num_classes))

    model = ResNet_Final(res_model, net)
    model.load_state_dict(torch.load("./pth_file/model_best.pth"))
    model = model.to(device)

    #data 불러오기
    submission = pd.read_csv("./inha_data/sample_submission.csv")

    left_test_paths = list()
    right_test_paths = list()

    for i in range(len(submission)):
        left_test_paths.append(submission['face_images'][i].split()[0])
        right_test_paths.append(submission['face_images'][i].split()[1])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    #왼쪽 이미지 
    left_test = list()

    for left_test_path in left_test_paths:
        
        img = Image.open("/home/jhjeong/jiho_deep/inha_dacon/inha_data/test/" + left_test_path + '.jpg').convert("RGB")# 경로 설정 유의(ex .inha/test)
        img = data_transform(img) # 이미지 데이터 전처리
        left_test.append(img) 
    
    left_test = torch.stack(left_test)

    left_infer_result_list = list()

    with torch.no_grad():
        '''
        메모리 부족으로 6,000개 (배치) 한번에 입력으로 넣지 않고 1,000개 씩 입력으로 줌
        '''
        batch_size = 1000
        for i in range(0, 6):
            i = i * batch_size
            tmp_left_input = left_test[i:i+batch_size]
            #print(tmp_input.size()) # torch.Size([1000, 3, 112, 112])
            _, left_infer_result = model(tmp_left_input.to(device))
            #print(left_infer_result.size()) # torch.Size([1000, 512])
            left_infer_result_list.append(left_infer_result)

        left_infer_result_list = torch.stack(left_infer_result_list, dim=0).view(-1, 512)

    #오른쪽 이미지 
    right_test = list()
    for right_test_path in right_test_paths:
        img = Image.open("/home/jhjeong/jiho_deep/inha_dacon/inha_data/test/" + right_test_path + '.jpg').convert("RGB") # 경로 설정 유의 (ex. inha/test)
        img = data_transform(img)# 이미지 데이터 전처리
        right_test.append(img)
    right_test = torch.stack(right_test)
    #print(right_test.size()) # torch.Size([6000, 3, 112, 112])

    right_infer_result_list = list()
    with torch.no_grad():
        '''
        메모리 부족으로 6,000개 (배치) 한번에 입력으로 넣지 않고 1,000개 씩 입력으로 줌
        '''
        batch_size = 1000
        for i in range(0, 6):
            i = i * batch_size
            tmp_right_input = right_test[i:i+batch_size]
            #print(tmp_input.size()) # torch.Size([1000, 3, 112, 112])
            _, right_infer_result = model(tmp_right_input.to(device))
            #print(left_infer_result.size()) # torch.Size([1000, 512])
            right_infer_result_list.append(right_infer_result)

        right_infer_result_list = torch.stack(right_infer_result_list, dim=0).view(-1, 512)
        #print(right_infer_result_list.size()) # torch.Size([6000, 512])

    cosin_similarity = cos_sim(left_infer_result_list, right_infer_result_list)
    
    # 최종
    submission = pd.read_csv("./inha_data/sample_submission.csv") 
    submission['answer'] = cosin_similarity.tolist()
    #submission.loc['answer'] = submission['answer']
    submission.to_csv('./inha_data/submission.csv', index=False)

if __name__ == '__main__':
    infer()