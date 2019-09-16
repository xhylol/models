import PIL.Image as Image
import requests
import os
import io

IMAGE_COLUMN=5
IMAGE_ROW=2
IMAGE_WIDTH=480
IMAGE_HEIGHT=270

def read_cover_dict(cover_file):
    id_url = {}
    with open(cover_file) as f:
        for eachline in f:
            line = eachline.replace('\n','').split(',')
            id_url[line[0]] = "https://img.yilan.tv/" + line[1]
    return id_url

def read_sim_videos(sim_file):
    videos_list = []
    with open(sim_file) as f:
        for eachline in f:
            line = eachline.replace('\n','').split(',')
            videos_list.append([data.split(':')[0] for data in line])
    return videos_list

def image_compose(img_url_list,save_path):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_WIDTH, IMAGE_ROW * IMAGE_HEIGHT))
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            img_url = img_url_list[(x-1)+(y-1)*IMAGE_COLUMN]
            r = requests.get(img_url,timeout=(5, 10))
            from_image = Image.open(io.BytesIO(r.content)).resize((IMAGE_WIDTH, IMAGE_HEIGHT),Image.BILINEAR)
            to_image.paste(from_image, ((x - 1) * IMAGE_WIDTH, (y - 1) * IMAGE_HEIGHT))
    return to_image.resize((IMAGE_WIDTH*IMAGE_COLUMN/4, IMAGE_HEIGHT*IMAGE_ROW/4)).save(save_path)

def main():
    id_url = read_cover_dict('./img_data.csv')
    videos_list = read_sim_videos('./sim_result.csv')
    index = 0
    for videos in videos_list:
        image_compose([id_url[video] for video in videos],'./video_compose_{}.png'.format(index))
        index += 1

if __name__ == '__main__':
    main()
